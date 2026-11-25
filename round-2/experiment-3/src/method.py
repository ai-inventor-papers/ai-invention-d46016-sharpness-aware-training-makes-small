#!/usr/bin/env python3
"""Retuned SAM vs SGD calibration check on Fashion-MNIST.

Stages (run one at a time; the GPU is shared):
  smoke    two short runs (SGD and SAM) to check the code path and timing
  tune     validation-only learning-rate / rho sweep (tuning seed 101), with the
           time guard and the 0.0003 extension rule; writes tuning_table.csv
  final    five paired seeds (0-4) for SGD and SAM at the frozen selection,
           plus the determinism rerun (SGD seed 0)
  analyze  temperature scaling on validation, test metrics, paired t-intervals
           with Holm correction, pooled reliability bins, placebo, method_out.json

Outputs go under --out-root (default: the workspace).
"""

import argparse
import csv
import gc
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy import optimize, stats

WS = Path(__file__).resolve().parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = WS / "data_cache"

TUNE_SEED = 101
FINAL_SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 40
SAVE_EPOCHS = {5, 15, 40}
BATCH = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BN_MOMENTUM = 0.1
SGD_GRID = [0.001, 0.003, 0.01, 0.03]
SAM_LR_GRID = [0.001, 0.003, 0.01, 0.03]
SAM_RHO_GRID = [0.05, 0.1, 0.2]
REDUCED_LR_GRID = [0.003, 0.01, 0.03]
EXT_LR = 0.0003
PROJECTED_LIMIT_S = 4.5 * 3600  # time guard from the plan
RUN_LIMIT_S = 600               # 10 minutes per 40-epoch run
N_BINS = 15
T_CRIT_DF4 = 2.776              # two-sided 95% t critical value, df = 4

RAM_CAP_BYTES = 12 * 1024**3


def start_ram_watchdog(cap_bytes: int) -> None:
    """macOS cannot cap RLIMIT_AS; watch resident memory instead (aii-use-hardware)."""
    me = psutil.Process()

    def _watch() -> None:
        while True:
            rss = sum(p.memory_info().rss for p in [me, *me.children(recursive=True)])
            if rss > cap_bytes:
                print(f"RAM cap hit: {rss / 1024**3:.1f} GB", file=sys.stderr, flush=True)
                import os
                os._exit(137)
            time.sleep(1)

    threading.Thread(target=_watch, daemon=True).start()


def setup_logging(root: Path) -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logger.add(root / "logs" / "run.log", rotation="30 MB", level="DEBUG")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_data(root: Path) -> dict:
    from torchvision import datasets

    tr = datasets.FashionMNIST(str(DATA_DIR), train=True, download=True)
    te = datasets.FashionMNIST(str(DATA_DIR), train=False, download=True)
    X = tr.data.float().div(255.0).unsqueeze(1)
    y = tr.targets.long()
    Xte = te.data.float().div(255.0).unsqueeze(1)
    yte = te.targets.long()
    assert X.shape == (60000, 1, 28, 28), X.shape
    assert Xte.shape == (10000, 1, 28, 28), Xte.shape

    rng = np.random.default_rng(0)
    perm = rng.permutation(60000)
    val_idx = perm[:10000]
    train_idx = perm[10000:]
    assert len(np.intersect1d(val_idx, train_idx)) == 0
    assert len(train_idx) == 50000 and len(val_idx) == 10000

    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "split_seed0.json").write_text(json.dumps({
        "description": "Fashion-MNIST 60k train images permuted with numpy default_rng(0); "
                       "first 10000 = validation, rest = training (50000). Test = official 10k.",
        "numpy_seed": 0,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(Xte.shape[0]),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
    }))

    Xtr_raw = X[torch.from_numpy(train_idx)]
    mu = Xtr_raw.mean()
    sd = Xtr_raw.std()
    logger.info(f"Normalisation from the 50k training split: mean={mu.item():.4f} std={sd.item():.4f}")

    def norm(t: torch.Tensor) -> torch.Tensor:
        return ((t - mu) / sd).contiguous()

    data = {
        "Xtr": norm(Xtr_raw).to(DEVICE),
        "ytr": y[torch.from_numpy(train_idx)].to(DEVICE),
        "Xva": norm(X[torch.from_numpy(val_idx)]).to(DEVICE),
        "Xte": norm(Xte).to(DEVICE),
        "yva_np": y[torch.from_numpy(val_idx)].numpy(),
        "yte_np": yte.numpy(),
    }
    del X, Xte, Xtr_raw
    gc.collect()
    logger.info(f"Data ready on {DEVICE}: train {tuple(data['Xtr'].shape)}, "
                f"val {tuple(data['Xva'].shape)}, test {tuple(data['Xte'].shape)}")
    return data


# ----------------------------------------------------------------------------
# Model and training
# ----------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


def run_name(kind: str, lr: float, rho: float, seed: int, tag: str = "") -> str:
    name = f"{kind}_lr{lr:g}_rho{rho:g}_seed{seed}"
    return f"{name}_{tag}" if tag else name


@torch.no_grad()
def evaluate_logits(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, X.shape[0], 1000):
        outs.append(model(X[i:i + 1000]).float().cpu())
    model.train()
    return torch.cat(outs).numpy().astype(np.float32)


def train_run(data: dict, *, kind: str, lr: float, rho: float, seed: int,
              epochs: int, root: Path, tag: str = "") -> dict:
    """Train one model. kind is 'sgd' or 'sam'. Resumable: skips a finished run."""
    assert kind in ("sgd", "sam")
    name = run_name(kind, lr, rho, seed, tag)
    runs_dir = root / "results" / "runs"
    logits_dir = root / "results" / "logits"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logits_dir.mkdir(parents=True, exist_ok=True)
    rec_path = runs_dir / f"{name}.json"
    if rec_path.exists():
        logger.info(f"skip finished run {name}")
        return json.loads(rec_path.read_text())

    logger.info(f"START {name} epochs={epochs} device={DEVICE}")
    torch.manual_seed(seed)
    model = Net().to(DEVICE)
    params = list(model.parameters())
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    base = torch.optim.SGD(params, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    N = data["Xtr"].shape[0]
    steps_per_epoch = math.ceil(N / BATCH)
    total_steps = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        base, lambda t: 0.5 * (1.0 + math.cos(math.pi * min(t, total_steps) / total_steps)))
    gen = torch.Generator().manual_seed(seed)  # shared data order across kinds
    save_eps = SAVE_EPOCHS if epochs == EPOCHS else {epochs}

    epoch_loss: list[float] = []
    epoch_wall: list[float] = []
    val_logits = test_logits = None
    t_run0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(N, generator=gen)
        loss_sum = torch.zeros((), device=DEVICE)
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH].to(DEVICE)
            xb = data["Xtr"][idx]
            yb = data["ytr"][idx]
            loss = F.cross_entropy(model(xb), yb)  # unperturbed first pass
            loss_sum += loss.detach() * yb.shape[0]
            loss.backward()
            if kind == "sam":
                with torch.no_grad():
                    gn = torch.sqrt(torch.stack([p.grad.pow(2).sum() for p in params]).sum())
                    eps = [rho * p.grad / (gn + 1e-12) for p in params]
                    for p, e in zip(params, eps):
                        p.add_(e)
                base.zero_grad()
                for m in bns:
                    m.momentum = 0.0  # freeze BN running stats on the perturbed pass
                F.cross_entropy(model(xb), yb).backward()
                for m in bns:
                    m.momentum = BN_MOMENTUM
                with torch.no_grad():
                    for p, e in zip(params, eps):
                        p.sub_(e)
            base.step()
            base.zero_grad()
            sched.step()
        ep_loss = float(loss_sum.item()) / N
        epoch_loss.append(ep_loss)
        dt = time.perf_counter() - t0
        epoch_wall.append(dt)
        if not math.isfinite(ep_loss):
            logger.warning(f"{name} epoch {ep}: non-finite training loss")
        logger.info(f"{name} ep {ep}/{epochs} train_loss={ep_loss:.4f} epoch_s={dt:.1f}")
        if ep in save_eps:
            val_logits = evaluate_logits(model, data["Xva"])
            test_logits = evaluate_logits(model, data["Xte"])
            np.savez(logits_dir / f"{name}_ep{ep}.npz", val=val_logits, test=test_logits)

    wall = time.perf_counter() - t_run0
    val_m = metrics(val_logits, data["yva_np"])
    rec = {
        "name": name, "kind": kind, "lr": lr, "rho": rho if kind == "sam" else 0.0,
        "seed": seed, "tag": tag, "epochs": epochs, "device": str(DEVICE),
        "batch_size": BATCH, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
        "final_train_loss": epoch_loss[-1], "epoch_train_loss": epoch_loss,
        "epoch_wall_s": epoch_wall, "wall_time_s": wall,
        "val_nll": val_m["nll"], "val_ece": val_m["ece"], "val_acc": val_m["acc"],
        "val_brier": val_m["brier"], "val_ace": val_m["ace"],
        "logits_finite": bool(np.isfinite(val_logits).all()),
    }
    rec_path.write_text(json.dumps(rec, indent=2))
    with (root / "logs" / "timings.jsonl").open("a") as fh:
        fh.write(json.dumps({"name": name, "epochs": epochs, "wall_time_s": wall,
                             "mean_epoch_s": float(np.mean(epoch_wall)),
                             "device": str(DEVICE)}) + "\n")
    logger.info(f"DONE {name} wall={wall:.1f}s val_nll={val_m['nll']:.4f} "
                f"val_acc={val_m['acc']:.4f} final_train_loss={epoch_loss[-1]:.4f}")
    del model, base, sched, params, bns
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return rec


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def log_softmax_np(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def nll_of(logits: np.ndarray, y: np.ndarray) -> float:
    lp = log_softmax_np(logits)
    return float(-lp[np.arange(len(y)), y].mean())


def ece_equal_width(conf: np.ndarray, correct: np.ndarray, nb: int = N_BINS) -> float:
    """ECE with nb equal-width bins on max-softmax confidence (Guo et al. 2017)."""
    b = np.clip(np.ceil(conf * nb).astype(int) - 1, 0, nb - 1)
    n = len(conf)
    ece = 0.0
    for k in range(nb):
        m = b == k
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def ace_equal_count(conf: np.ndarray, correct: np.ndarray, nb: int = N_BINS) -> float:
    """Top-label adaptive calibration error: mean |acc - conf| over nb equal-count bins
    (quantile bins of max-softmax confidence). Simplified from Nixon et al. (2019),
    which is per-class; the report states this."""
    order = np.argsort(conf, kind="stable")
    gaps = []
    for g in np.array_split(order, nb):
        if len(g) > 0:
            gaps.append(abs(correct[g].mean() - conf[g].mean()))
    return float(np.mean(gaps))


def metrics(logits: np.ndarray | None, y: np.ndarray) -> dict:
    nan = float("nan")
    if logits is None or not np.isfinite(logits).all():
        return {"acc": nan, "nll": nan, "brier": nan, "ece": nan, "ace": nan}
    lp = log_softmax_np(logits)
    p = np.exp(lp)
    pred = logits.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    conf = p.max(axis=1)
    onehot = np.eye(10)[y]
    return {
        "acc": float(correct.mean()),
        "nll": float(-lp[np.arange(len(y)), y].mean()),
        "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
        "ece": ece_equal_width(conf, correct),
        "ace": ace_equal_count(conf, correct),
    }


def fit_temperature(val_logits: np.ndarray, yva: np.ndarray) -> float:
    res = optimize.minimize_scalar(lambda t: nll_of(val_logits / t, yva),
                                   bounds=(0.05, 10.0), method="bounded",
                                   options={"xatol": 1e-6})
    return float(res.x)


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------

def stage_smoke(data: dict, root: Path, epochs: int) -> None:
    logger.info(f"SMOKE: {epochs} epochs, SGD and SAM at lr 0.01 rho 0.1")
    sgd = train_run(data, kind="sgd", lr=0.01, rho=0.0, seed=TUNE_SEED,
                    epochs=epochs, root=root, tag="smoke")
    sam = train_run(data, kind="sam", lr=0.01, rho=0.1, seed=TUNE_SEED,
                    epochs=epochs, root=root, tag="smoke")
    for r in (sgd, sam):
        assert all(math.isfinite(x) for x in r["epoch_train_loss"]), f"NaN in {r['name']}"
        assert r["epoch_train_loss"][-1] < r["epoch_train_loss"][0], f"loss did not drop in {r['name']}"
    logger.info(f"SMOKE OK: sgd epoch_s={np.mean(sgd['epoch_wall_s']):.1f}, "
                f"sam epoch_s={np.mean(sam['epoch_wall_s']):.1f}")


def select_best(rows: list[dict], kind: str) -> dict:
    cand = [r for r in rows if r["kind"] == kind]
    if not cand:
        raise RuntimeError(f"no {kind} configurations")

    def key(r: dict) -> float:
        v = r["val_nll"]
        return v if (v is not None and math.isfinite(v)) else float("inf")

    best = min(cand, key=key)
    if not math.isfinite(key(best)):
        raise RuntimeError(f"every {kind} configuration diverged; nothing can be selected")
    return best


def write_tuning_table(rows: list[dict], root: Path) -> None:
    out = root / "results" / "tuning_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["kind", "lr", "rho", "seed", "epochs", "val_nll", "val_ece", "val_acc",
            "final_train_loss", "wall_time_s", "device", "name"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["kind"], r["lr"], r["rho"])):
            w.writerow({
                "kind": r["kind"], "lr": r["lr"], "rho": r["rho"], "seed": r["seed"],
                "epochs": r["epochs"], "val_nll": r["val_nll"], "val_ece": r["val_ece"],
                "val_acc": r["val_acc"], "final_train_loss": r["final_train_loss"],
                "wall_time_s": r["wall_time_s"], "device": r["device"], "name": r["name"],
            })
    logger.info(f"Wrote {out} ({len(rows)} rows)")


def load_runs(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((root / "results" / "runs").glob("*.json"))]


def stage_tune(data: dict, root: Path) -> None:
    tune_rows: list[dict] = []

    def go(kind: str, lr: float, rho: float) -> None:
        r = train_run(data, kind=kind, lr=lr, rho=rho, seed=TUNE_SEED,
                      epochs=EPOCHS, root=root, tag="tune")
        tune_rows.append(r)
        write_tuning_table(tune_rows, root)

    # Timing configs first (they are also grid members): time guard before the sweep.
    go("sgd", 0.01, 0.0)
    go("sam", 0.01, 0.1)
    t_sgd = next(r for r in tune_rows if r["kind"] == "sgd")["wall_time_s"]
    t_sam = next(r for r in tune_rows if r["kind"] == "sam")["wall_time_s"]

    # Plan: sgd 4 + sam 12, extension up to 2 sgd + 3 sam, finals 5 + 5, determinism 1.
    n_sgd_left = len(SGD_GRID) - 1 + 2 + 5 + 1
    n_sam_left = len(SAM_LR_GRID) * len(SAM_RHO_GRID) - 1 + 3 + 5
    projected = t_sgd * n_sgd_left + t_sam * n_sam_left
    too_slow = max(t_sgd, t_sam) > RUN_LIMIT_S
    reduced = projected > PROJECTED_LIMIT_S or too_slow
    guard = {
        "t_sgd_40ep_s": t_sgd, "t_sam_40ep_s": t_sam,
        "projected_total_s": projected, "limit_s": PROJECTED_LIMIT_S,
        "run_limit_s": RUN_LIMIT_S, "reduced_grid": reduced,
        "reason": ("a 40-epoch run exceeded 10 minutes" if too_slow else
                   "projected total exceeds 4.5 h" if projected > PROJECTED_LIMIT_S else
                   "full grid fits the time budget"),
    }
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "time_guard.json").write_text(json.dumps(guard, indent=2))
    logger.info(f"TIME GUARD: {guard}")

    sgd_grid = REDUCED_LR_GRID if reduced else SGD_GRID
    sam_lr_grid = REDUCED_LR_GRID if reduced else SAM_LR_GRID
    for lr in sgd_grid:
        if lr != 0.01:
            go("sgd", lr, 0.0)
    for lr in sam_lr_grid:
        for rho in SAM_RHO_GRID:
            if (lr, rho) != (0.01, 0.1):
                go("sam", lr, rho)

    # Extension rule: lowest grid value selected -> also run 0.0003, then re-select.
    best_sgd = select_best(tune_rows, "sgd")
    if best_sgd["lr"] == 0.001:
        logger.info("SGD selected lr is 0.001: running extension lr 0.0003")
        go("sgd", EXT_LR, 0.0)
        best_sgd = select_best(tune_rows, "sgd")
    best_sam = select_best(tune_rows, "sam")
    if best_sam["lr"] == 0.001:
        logger.info("SAM selected lr is 0.001: running extension lr 0.0003 with all rho")
        for rho in SAM_RHO_GRID:
            go("sam", EXT_LR, rho)
        best_sam = select_best(tune_rows, "sam")

    sel = {
        "tuning_seed": TUNE_SEED,
        "selection_criterion": "final-epoch validation NLL (raw logits), lowest wins",
        "sgd": {"lr": best_sgd["lr"], "rho": 0.0, "val_nll": best_sgd["val_nll"]},
        "sam": {"lr": best_sam["lr"], "rho": best_sam["rho"], "val_nll": best_sam["val_nll"]},
        "sgd_at_lower_grid_edge_of_final_grid": best_sgd["lr"] == min(sgd_grid + [EXT_LR]),
        "sam_at_lower_grid_edge_of_final_grid": best_sam["lr"] == min(sam_lr_grid + [EXT_LR]),
        "extension_0003_run": any(r["lr"] == EXT_LR for r in tune_rows),
        "reduced_grid": reduced,
    }
    write_tuning_table(tune_rows, root)
    (root / "results" / "selection.json").write_text(json.dumps(sel, indent=2))
    logger.info(f"SELECTED: {sel}")


def stage_final(data: dict, root: Path) -> None:
    sel = json.loads((root / "results" / "selection.json").read_text())
    for seed in FINAL_SEEDS:
        train_run(data, kind="sgd", lr=sel["sgd"]["lr"], rho=0.0, seed=seed,
                  epochs=EPOCHS, root=root)
        train_run(data, kind="sam", lr=sel["sam"]["lr"], rho=sel["sam"]["rho"], seed=seed,
                  epochs=EPOCHS, root=root)

    # Determinism check: rerun SGD seed 0 at the selected lr and compare logits.
    det = train_run(data, kind="sgd", lr=sel["sgd"]["lr"], rho=0.0, seed=0,
                    epochs=EPOCHS, root=root, tag="det")
    logits_dir = root / "results" / "logits"
    orig = np.load(logits_dir / f"{run_name('sgd', sel['sgd']['lr'], 0.0, 0)}_ep40.npz")
    rerun = np.load(logits_dir / f"{run_name('sgd', sel['sgd']['lr'], 0.0, 0, 'det')}_ep40.npz")
    max_diff_val = float(np.abs(orig["val"] - rerun["val"]).max())
    max_diff_test = float(np.abs(orig["test"] - rerun["test"]).max())
    acc_orig = metrics(orig["test"], data["yte_np"])["acc"]
    acc_rerun = metrics(rerun["test"], data["yte_np"])["acc"]
    out = {
        "max_abs_logit_diff_val": max_diff_val,
        "max_abs_logit_diff_test": max_diff_test,
        "test_acc_original": acc_orig, "test_acc_rerun": acc_rerun,
        "device": str(DEVICE),
        "note": "Difference is MPS nondeterminism unless it is large; a large difference "
                "means the shared-init or shared-order claim is broken.",
    }
    (root / "logs" / "determinism.json").write_text(json.dumps(out, indent=2))
    logger.info(f"DETERMINISM: {out}")


def paired_row(name: str, diffs: np.ndarray) -> dict:
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if math.isfinite(t) else float("nan")
    half = T_CRIT_DF4 * se
    return {"row": name, "per_seed_diff": d.tolist(), "mean": mean, "sd": sd,
            "t": t, "df": n - 1, "ci95_low": mean - half, "ci95_high": mean + half,
            "p_two_sided": p}


def holm(rows: list[dict]) -> None:
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i]["p_two_sided"])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * rows[i]["p_two_sided"])
        running = max(running, val)
        adj[i] = running
    for i, r in enumerate(rows):
        r["p_holm"] = adj[i]
        r["significant_holm_0_05"] = bool(adj[i] < 0.05)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, nb: int = N_BINS) -> list[dict]:
    b = np.clip(np.ceil(conf * nb).astype(int) - 1, 0, nb - 1)
    out = []
    for k in range(nb):
        m = b == k
        n = int(m.sum())
        out.append({
            "bin": k, "lower": k / nb, "upper": (k + 1) / nb, "count": n,
            "mean_confidence": float(conf[m].mean()) if n else None,
            "accuracy": float(correct[m].mean()) if n else None,
        })
    return out


def stage_analyze(data: dict, root: Path) -> None:
    sel = json.loads((root / "results" / "selection.json").read_text())
    logits_dir = root / "results" / "logits"
    yva, yte = data["yva_np"], data["yte_np"]

    per_seed = {"sgd_raw": [], "sgd_ts": [], "sam_raw": [], "sam_ts": []}
    temps = {"sgd": [], "sam": []}
    train_loss = {"sgd": [], "sam": []}
    raw_test = {"sgd": [], "sam": []}
    run_records = {}

    for seed in FINAL_SEEDS:
        for kind, lr, rho in (("sgd", sel["sgd"]["lr"], 0.0),
                              ("sam", sel["sam"]["lr"], sel["sam"]["rho"])):
            name = run_name(kind, lr, rho, seed)
            rec = json.loads((root / "results" / "runs" / f"{name}.json").read_text())
            run_records[name] = rec
            train_loss[kind].append(rec["final_train_loss"])
            z = np.load(logits_dir / f"{name}_ep40.npz")
            T = fit_temperature(z["val"], yva)  # validation only
            temps[kind].append(T)
            raw_test[kind].append(z["test"])
            per_seed[f"{kind}_raw"].append(metrics(z["test"], yte))
            per_seed[f"{kind}_ts"].append(metrics(z["test"] / T, yte))
            logger.info(f"{name}: T={T:.4f} test raw={per_seed[f'{kind}_raw'][-1]} "
                        f"ts={per_seed[f'{kind}_ts'][-1]}")

    def col(arm: str, key: str) -> np.ndarray:
        return np.array([m[key] for m in per_seed[arm]])

    family = []
    for key, label in (("acc", "accuracy"), ("ece", "ECE"), ("nll", "NLL"),
                       ("brier", "Brier"), ("ace", "ACE")):
        family.append(paired_row(f"SAM - SGD: {label}", col("sam_raw", key) - col("sgd_raw", key)))
    for key, label in (("acc", "accuracy"), ("ece", "ECE"), ("nll", "NLL"),
                       ("brier", "Brier"), ("ace", "ACE")):
        family.append(paired_row(f"SAM - TS-SGD: {label}", col("sam_ts", key) - col("sgd_ts", key)))
    for key, label in (("ece", "ECE"), ("nll", "NLL")):
        family.append(paired_row(f"TS-SAM - TS-SGD: {label}", col("sam_ts", key) - col("sgd_ts", key)))
    assert len(family) == 12
    holm(family)
    for r in family:
        logger.info(f"{r['row']}: mean={r['mean']:+.5f} CI=[{r['ci95_low']:+.5f},{r['ci95_high']:+.5f}] "
                    f"p={r['p_two_sided']:.4f} holm={r['p_holm']:.4f}")

    # Pooled reliability bins per arm (all seeds pooled), raw and temperature-scaled.
    bins = {}
    for arm, logits_list, T_list in (
            ("sgd_raw", raw_test["sgd"], [1.0] * 5),
            ("sam_raw", raw_test["sam"], [1.0] * 5),
            ("sgd_ts", raw_test["sgd"], temps["sgd"]),
            ("sam_ts", raw_test["sam"], temps["sam"])):
        confs, corrs = [], []
        for z, T in zip(logits_list, T_list):
            lp = log_softmax_np(z / T)
            p = np.exp(lp)
            confs.append(p.max(axis=1))
            corrs.append((z.argmax(axis=1) == yte).astype(np.float64))
        bins[arm] = reliability_bins(np.concatenate(confs), np.concatenate(corrs))
    (root / "results" / "reliability_bins.json").write_text(json.dumps(
        {"n_bins": N_BINS, "binning": "15 equal-width bins on max-softmax confidence, "
         "all five seeds pooled per arm, 10000 test images per seed", "arms": bins}, indent=2))

    # Placebo: evaluate both arms against shuffled test labels. The SAM-minus-SGD
    # difference should not be significant; a significant result would mean the
    # arm difference is not tied to label information.
    shuf = np.random.default_rng(12345).permutation(yte)
    placebo_per = {"sgd": [], "sam": []}
    for kind in ("sgd", "sam"):
        for z in raw_test[kind]:
            placebo_per[kind].append(metrics(z, shuf))
    placebo_rows = []
    for key in ("acc", "nll", "ece", "brier", "ace"):
        d = np.array([placebo_per["sam"][i][key] - placebo_per["sgd"][i][key] for i in range(5)])
        placebo_rows.append(paired_row(f"PLACEBO SAM - SGD (shuffled test labels): {key}", d))
    holm(placebo_rows)
    self_diff = np.zeros(5)  # an arm compared with itself is identically zero
    self_row = paired_row("PLACEBO SGD - SGD (same model)", self_diff)

    tables = {
        "family_of_12_paired": family,
        "placebo": placebo_rows + [self_row],
        "per_seed_metrics": {k: v for k, v in per_seed.items()},
        "temperature_T": temps,
        "final_train_loss": train_loss,
        "run_records": run_records,
    }
    (root / "results" / "analysis_tables.json").write_text(json.dumps(tables, indent=2))

    # method_out.json in the aii-json exp_gen_sol_out shape: one example per test image
    # (all 10000), the true class as output, and one predict_* field per final model and seed.
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    predictions = {}
    for kind in ("sgd", "sam"):
        for i, seed in enumerate(FINAL_SEEDS):
            predictions[f"predict_{kind}_seed{seed}"] = raw_test[kind][i].argmax(axis=1)
    examples = []
    for j in range(len(yte)):
        ex = {
            "input": f"Fashion-MNIST test image {j} (28x28 grayscale). Classify the clothing item.",
            "output": classes[int(yte[j])],
            "metadata_test_index": j,
            "metadata_true_label": int(yte[j]),
        }
        for key, arr in predictions.items():
            ex[key] = classes[int(arr[j])]
        examples.append(ex)
    logger.info(f"method_out examples: {len(examples)} test images, {len(predictions)} predict fields each")
    method_out = {
        "metadata": {
            "method_name": "SAM vs SGD calibration on Fashion-MNIST (retuned)",
            "description": "Five paired seeds, validation-only tuning at seed 101, temperature "
                           "scaling fitted on validation, 12-row paired family with Holm correction.",
            "selection": sel,
            "family_of_12_paired": family,
            "placebo": tables["placebo"],
            "temperature_T": temps,
            "final_train_loss": train_loss,
            "reliability_bins_file": "results/reliability_bins.json",
            "ace_note": "Top-label, equal-count binned variant; not comparable to per-class ACE values.",
        },
        "datasets": [{"dataset": "FashionMNIST_test_10k", "examples": examples}],
    }
    (root / "results" / "method_out.json").write_text(json.dumps(method_out, indent=2))
    logger.info("Wrote results/method_out.json and analysis tables")


@logger.catch(reraise=True)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "tune", "final", "analyze"])
    ap.add_argument("--out-root", type=Path, default=WS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    root = args.out_root.resolve()
    setup_logging(root)
    start_ram_watchdog(RAM_CAP_BYTES)
    logger.info(f"stage={args.stage} root={root} device={DEVICE} torch={torch.__version__}")
    data = load_data(root)
    if args.stage == "smoke":
        stage_smoke(data, root, args.epochs)
    elif args.stage == "tune":
        stage_tune(data, root)
    elif args.stage == "final":
        stage_final(data, root)
    elif args.stage == "analyze":
        stage_analyze(data, root)


if __name__ == "__main__":
    main()

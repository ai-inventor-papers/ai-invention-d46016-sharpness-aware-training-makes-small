#!/usr/bin/env python3
"""SAM vs SGD at matched training loss on Fashion-MNIST (full experiment).

Stages, run in this order by `uv run method.py --stage all`:
  checks   unit checks: ECE, temperature fit, split sizes, rho=0 SAM vs SGD, placebo
  mini     2-epoch end-to-end run on a 5k training subset
  tune     SAM rho in {0.05, 0.1, 0.2}, seed 101, 40 epochs; choose by validation NLL.
           The first run is the timing run (wall-clock is logged to results/timings.json)
  pilot    seed 100: SAM 40 epochs (target loss) and SGD over the cosine-E grid -> E*
  main     seeds 0-4: condition 1 (SAM 40), condition 2 (SGD 40), condition 3 (SGD E*)
  analyze  paired statistics, Holm, matched-loss table, reproduction check, figures
"""

import argparse
import gc
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from loguru import logger
from scipy import stats
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

WORK = Path(__file__).resolve().parent
RESULTS = WORK / "results"
RUNS = RESULTS / "runs"
LOGITS = RESULTS / "logits"
CHECKS = RESULTS / "checks"
FIGS = WORK / "figures"
DATA_ROOT = WORK / "scratch" / "fmnist"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BASE_LR = 0.01
BATCH = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
N_BINS = 15
EPOCHS = 40
SEEDS_MAIN = [0, 1, 2, 3, 4]
SEED_TUNE = 101
SEED_PILOT = 100
RHO_GRID = [0.05, 0.1, 0.2]
E_GRID = [10, 12, 14, 16, 18, 20]
E_EXT_GRID = [22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
MATCH_TOL = 0.01          # |train loss gap| for choosing E*
FLAG_GAP_A = 0.02         # flag for matched-epoch comparison (a)
RAM_CAP_GB = 12.0         # watchdog; pod total is 29 GB
# Reference numbers from the earlier run in this lineage (reproduction check only).
REPORTED = {"accuracy": 0.0015, "ece_raw": -0.0378, "nll_raw": -0.0980, "nll_ts": -0.0209}
REPORTED_TOL = {"accuracy": 0.005, "ece_raw": 0.005, "nll_raw": 0.01, "nll_ts": 0.01}


# ----------------------------------------------------------------------------- setup

def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(WORK / "logs" / "run.log", rotation="30 MB", level="DEBUG")


def ensure_dirs() -> None:
    for d in (RESULTS, RUNS, LOGITS, CHECKS, FIGS, WORK / "logs", WORK / "scratch"):
        d.mkdir(parents=True, exist_ok=True)


def start_ram_watchdog(cap_gb: float = RAM_CAP_GB) -> None:
    """macOS cannot cap RLIMIT_AS; watch the RSS of this process and its children instead."""
    me = psutil.Process()

    def _watch() -> None:
        while True:
            rss = sum(p.memory_info().rss for p in [me, *me.children(recursive=True)])
            if rss > cap_gb * 1024**3:
                logger.error(f"RAM cap hit: {rss / 1024**3:.1f} GB > {cap_gb} GB; exiting")
                os._exit(137)
            time.sleep(1)

    threading.Thread(target=_watch, daemon=True).start()


# ----------------------------------------------------------------------------- data

def load_data(train_subset: int | None = None) -> dict:
    """Fashion-MNIST: 60k train -> 50k train / 10k validation (numpy seed 0); official 10k test.

    Normalised with the scalar mean and std of the 50k training split. No augmentation.
    """
    train = datasets.FashionMNIST(root=str(DATA_ROOT), train=True, download=True)
    test = datasets.FashionMNIST(root=str(DATA_ROOT), train=False, download=True)
    X = train.data.float().div(255.0).unsqueeze(1)
    y = train.targets.long()
    assert X.shape == (60000, 1, 28, 28), X.shape
    Xte = test.data.float().div(255.0).unsqueeze(1)
    yte = test.targets.long()
    assert Xte.shape == (10000, 1, 28, 28), Xte.shape

    perm = np.random.RandomState(0).permutation(60000)
    tr_idx, va_idx = perm[:50000], perm[50000:]
    assert len(tr_idx) == 50000 and len(va_idx) == 10000
    mean = X[tr_idx].mean().item()
    std = X[tr_idx].std().item()
    logger.info(f"Normalisation from 50k train split: mean={mean:.4f} std={std:.4f}")

    if train_subset is not None:
        tr_idx = tr_idx[:train_subset]
    Xtr = ((X[tr_idx] - mean) / std)
    Xva = ((X[va_idx] - mean) / std)
    Xte = ((Xte - mean) / std)
    data = {
        "Xtr": Xtr.to(DEVICE), "ytr": y[tr_idx].to(DEVICE),
        "Xva": Xva.to(DEVICE), "yva": y[va_idx].to(DEVICE),
        "Xte": Xte.to(DEVICE), "yte": yte.to(DEVICE),
        "ytr_np": y[tr_idx].numpy(), "yva_np": y[va_idx].numpy(), "yte_np": yte.numpy(),
    }
    logger.info(f"Split sizes: train={len(tr_idx)} val={len(va_idx)} test={len(yte)} "
                f"device={DEVICE}")
    del X, y
    gc.collect()
    return data


# ----------------------------------------------------------------------------- model

class SmallCNN(nn.Module):
    """Conv(1->32)-BN-ReLU-Pool, Conv(32->64)-BN-ReLU-Pool, Linear(3136->128)-ReLU, Linear(128->10).

    Convolutions have no bias because each is followed by BatchNorm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def lr_at(epoch: int, n_epochs: int, lr_base: float = BASE_LR) -> float:
    """Per-epoch cosine schedule, e = 0..E-1: lr_e = 0.5 * lr_base * (1 + cos(pi * e / E))."""
    return 0.5 * lr_base * (1.0 + math.cos(math.pi * epoch / n_epochs))


def train_step(model: nn.Module, opt: torch.optim.Optimizer, params: list, x: torch.Tensor,
               y: torch.Tensor, rho: float, sam: bool) -> torch.Tensor:
    """One optimiser step. SGD: one forward/backward. SAM: ascent to w + eps, gradient there,
    restore w, then the base SGD-momentum step with that gradient. BN statistics update in both passes."""
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    if sam:
        with torch.no_grad():
            grads = [p.grad for p in params]
            gnorm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
            eps = [rho * g / (gnorm + 1e-12) for g in grads]
            for p, e in zip(params, eps):
                p.add_(e)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        with torch.no_grad():
            for p, e in zip(params, eps):
                p.sub_(e)
    opt.step()
    return loss.detach()


def train_model(*, method: str, rho: float, epochs: int, seed: int, data: dict,
                on_epoch=None, lr_base: float = BASE_LR) -> nn.Module:
    """Train with seed-controlled init and data order. The shuffle generator is seeded with the
    same seed for SGD and SAM, so both see identical batch sequences."""
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    params = list(model.parameters())
    opt = torch.optim.SGD(params, lr=lr_base, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)
    n = data["Xtr"].shape[0]
    sam = method == "sam"
    for e in range(epochs):
        lr = lr_at(e, epochs, lr_base)
        for g in opt.param_groups:
            g["lr"] = lr
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        model.train()
        losses = []
        t0 = time.time()
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            losses.append(train_step(model, opt, params, data["Xtr"][idx], data["ytr"][idx], rho, sam))
        epoch_time = time.time() - t0
        batch_loss = torch.stack(losses).cpu().numpy().astype(np.float64)
        if on_epoch is not None:
            on_epoch(e + 1, lr, batch_loss, epoch_time, model)
    return model


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor, bs: int = 5000) -> np.ndarray:
    model.eval()
    out = [model(X[i:i + bs]).float().cpu() for i in range(0, X.shape[0], bs)]
    model.train()
    return torch.cat(out).numpy().astype(np.float64)


# ----------------------------------------------------------------------------- metrics

def log_softmax_np(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def nll_np(z: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    logp = log_softmax_np(z / T)
    return float(-logp[np.arange(len(y)), y].mean())


def ece_equal_width(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Top-label ECE with equal-width confidence bins (Guo et al. 2017)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    n = len(conf)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def ace_equal_mass(probs: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Adaptive calibration error (Nixon et al. 2019): per class, split the sorted probabilities
    into n_bins equal-count bins; average |accuracy - confidence| over bins and classes."""
    K = probs.shape[1]
    vals = []
    for k in range(K):
        p = probs[:, k]
        ind = (y == k).astype(np.float64)
        order = np.argsort(p, kind="mergesort")
        for chunk in np.array_split(order, n_bins):
            vals.append(abs(ind[chunk].mean() - p[chunk].mean()))
    return float(np.mean(vals))


def metrics_from_logits(z: np.ndarray, y: np.ndarray, T: float = 1.0) -> dict:
    zt = z / T
    logp = log_softmax_np(zt)
    p = np.exp(logp)
    pred = zt.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    conf = p.max(axis=1)
    onehot = np.eye(z.shape[1])[y]
    return {
        "accuracy": float(correct.mean()),
        "ece": ece_equal_width(conf, correct),
        "ace": ace_equal_mass(p, y),
        "nll": float(-logp[np.arange(len(y)), y].mean()),
        "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
    }


def fit_temperature(z_val: np.ndarray, y_val: np.ndarray) -> tuple[float, bool]:
    """Single temperature by minimising validation NLL over log T in [log 0.05, log 10]."""
    lo, hi = math.log(0.05), math.log(10.0)
    res = minimize_scalar(lambda lt: nll_np(z_val, y_val, math.exp(lt)),
                          bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
    at_bound = bool(res.x <= lo + 1e-4 or res.x >= hi - 1e-4)
    return float(math.exp(res.x)), at_bound


def evaluate_logits(z_val: np.ndarray, y_val: np.ndarray, z_test: np.ndarray, y_test: np.ndarray) -> dict:
    T, at_bound = fit_temperature(z_val, y_val)
    raw = metrics_from_logits(z_test, y_test)
    ts = metrics_from_logits(z_test, y_test, T)
    return {
        "val_nll_raw": nll_np(z_val, y_val),
        "temperature": T,
        "temperature_at_bound": at_bound,
        "test_raw": raw,
        "test_ts": ts,
    }


# ----------------------------------------------------------------------------- runs

def run_config(*, name: str, method: str, rho: float, epochs: int, seed: int, data: dict,
               keep_all_logits: bool = True, out_dir: Path = RUNS, use_cache: bool = True,
               lr_base: float = BASE_LR) -> dict:
    """Train one model, evaluating after every epoch. Writes <out_dir>/<name>.json and logits."""
    path = out_dir / f"{name}.json"
    if use_cache and path.exists():
        logger.info(f"cache hit: {name}")
        return json.loads(path.read_text())
    logger.info(f"START {name}: method={method} rho={rho} epochs={epochs} seed={seed}")
    history, val_logits, test_logits = [], [], []
    t_start = time.time()

    def on_epoch(epoch: int, lr: float, batch_loss: np.ndarray, epoch_time: float, model: nn.Module) -> None:
        z_tr = predict_logits(model, data["Xtr"])
        z_va = predict_logits(model, data["Xva"])
        z_te = predict_logits(model, data["Xte"])
        rec = {
            "epoch": epoch,
            "lr": lr,
            "minibatch_loss_mean": float(batch_loss.mean()),
            "train_loss_eval": nll_np(z_tr, data["ytr_np"]),
            "train_acc_eval": float((z_tr.argmax(1) == data["ytr_np"]).mean()),
            "epoch_train_time_s": epoch_time,
        }
        rec.update(evaluate_logits(z_va, data["yva_np"], z_te, data["yte_np"]))
        history.append(rec)
        logger.info(f"  {name} ep{epoch:02d} lr={lr:.5f} batch_loss={rec['minibatch_loss_mean']:.4f} "
                    f"train_loss_eval={rec['train_loss_eval']:.4f} "
                    f"test_acc={rec["test_raw"]["accuracy"]:.4f} T={rec['temperature']:.3f} "
                    f"t={epoch_time:.1f}s")
        val_logits.append(z_va.astype(np.float16))
        test_logits.append(z_te.astype(np.float16))

    train_model(method=method, rho=rho, epochs=epochs, seed=seed, data=data,
                on_epoch=on_epoch, lr_base=lr_base)
    wall = time.time() - t_start
    LOGITS.mkdir(parents=True, exist_ok=True)
    if keep_all_logits:
        np.savez_compressed(LOGITS / f"{name}.npz", val=np.stack(val_logits), test=np.stack(test_logits),
                            epoch=np.arange(1, epochs + 1))
    else:
        np.savez_compressed(LOGITS / f"{name}.npz", val=val_logits[-1][None], test=test_logits[-1][None],
                            epoch=np.array([epochs]))
    rec = {
        "name": name, "method": method, "rho": rho, "epochs": epochs, "seed": seed,
        "lr_base": lr_base, "lr_schedule": "0.5*lr_base*(1+cos(pi*e/E)), e=0..E-1",
        "batch": BATCH, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
        "n_train": int(data["Xtr"].shape[0]), "wall_time_s": wall, "history": history,
        "final": history[-1],
    }
    path.write_text(json.dumps(rec, indent=1))
    logger.info(f"DONE {name}: wall={wall:.0f}s final_train_loss={history[-1]['train_loss_eval']:.4f} "
                f"test_acc={history[-1]['test_raw']['accuracy']:.4f}")
    del val_logits, test_logits, history
    gc.collect()
    return rec


# ----------------------------------------------------------------------------- checks

def stage_checks(data: dict) -> dict:
    out = {}
    rng = np.random.default_rng(0)

    # (a) ECE: perfect calibration on synthetic probabilities is near 0; constant confidence c and
    # accuracy a gives ECE exactly |a - c|.
    K, N = 10, 400000
    z = rng.normal(size=(N, K)) * 2.0
    p = np.exp(log_softmax_np(z))
    conf = p.max(1)
    # Labels drawn from the model's own distribution (inverse CDF), so it is perfectly calibrated.
    cdf = np.cumsum(p, axis=1)
    u = rng.random(N)[:, None]
    y_sample = (u > cdf).sum(axis=1).clip(0, K - 1)
    ece_perfect = ece_equal_width(conf, (p.argmax(1) == y_sample).astype(float))
    out["ece_perfect_calibration_synthetic_N400k"] = ece_perfect
    assert ece_perfect < 0.01, f"ECE on calibrated synthetic too large: {ece_perfect}"
    n_c = 1000
    probs_c = np.full((n_c, K), 0.2 / 9)
    probs_c[:, 0] = 0.8
    y_c = np.zeros(n_c, dtype=int)
    y_c[: int(0.6 * n_c)] = 0
    y_c[int(0.6 * n_c):] = 1
    ece_const = ece_equal_width(probs_c.max(1), (probs_c.argmax(1) == y_c).astype(float))
    out["ece_constant_c0.8_a0.6"] = ece_const
    assert abs(ece_const - 0.2) < 1e-9, ece_const
    logger.info(f"check (a) ECE: calibrated synthetic={ece_perfect:.5f} (expect ~0), "
                f"constant c=0.8 a=0.6 -> {ece_const:.6f} (expect 0.2)")

    # (b) Temperature fit on synthetic calibrated logits returns T ~ 1.
    zt = rng.normal(size=(50000, K)) * 2.0
    pt = np.exp(log_softmax_np(zt))
    cdf_t = np.cumsum(pt, axis=1)
    yt = (rng.random(50000)[:, None] > cdf_t).sum(axis=1).clip(0, K - 1)
    T_fit, _ = fit_temperature(zt, yt)
    out["temperature_fit_calibrated_synthetic"] = T_fit
    assert abs(T_fit - 1.0) < 0.05, T_fit
    logger.info(f"check (b) temperature on calibrated synthetic logits: T={T_fit:.4f} (expect ~1)")

    # (d) Split sizes, test set and logit shapes.
    out["split_sizes"] = {"train": int(data["Xtr"].shape[0]), "val": int(data["Xva"].shape[0]),
                          "test": int(data["Xte"].shape[0])}
    assert data["Xtr"].shape[0] == 50000 and data["Xva"].shape[0] == 10000 and data["Xte"].shape[0] == 10000
    m = SmallCNN().to(DEVICE)
    shape = predict_logits(m, data["Xte"][:64]).shape
    out["logit_shape_check"] = list(shape)
    assert shape == (64, 10)
    logger.info(f"check (d) split sizes {out['split_sizes']}, logit shape {shape}")
    del m

    # (c) rho = 0: SAM follows the SAM code path (two passes) but with zero perturbation. Compare
    # training-batch losses and final weights with SGD after one epoch from the same seed.
    d_sgd = {}
    d_sam0 = {}

    def grab_batches(method: str, store: dict) -> None:
        def cb(epoch, lr, bl, t, model):
            store["batch_loss"] = bl
            store["params"] = [p.detach().cpu().numpy().astype(np.float64) for p in model.parameters()]
            store["test_logits"] = predict_logits(model, data["Xte"])
        train_model(method=method, rho=0.0, epochs=1, seed=7, data=data, on_epoch=cb)

    grab_batches("sgd", d_sgd)
    grab_batches("sam", d_sam0)
    loss_maxdiff = float(np.max(np.abs(d_sgd["batch_loss"] - d_sam0["batch_loss"])))
    param_maxdiff = float(max(np.max(np.abs(a - b)) for a, b in zip(d_sgd["params"], d_sam0["params"])))
    logit_maxdiff = float(np.max(np.abs(d_sgd["test_logits"] - d_sam0["test_logits"])))
    out["rho0_sam_vs_sgd_one_epoch"] = {
        "max_abs_batch_loss_diff": loss_maxdiff,
        "max_abs_param_diff": param_maxdiff,
        "max_abs_test_logit_diff_eval_mode": logit_maxdiff,
        "note": ("Parameters and batch losses are compared. Eval-mode logits can differ because "
                 "BatchNorm running statistics are updated in both SAM passes, as the spec requires."),
    }
    logger.info(f"check (c) rho=0 SAM vs SGD one epoch: max batch-loss diff={loss_maxdiff:.2e}, "
                f"max param diff={param_maxdiff:.2e}, eval logit diff={logit_maxdiff:.2e}")
    del d_sgd, d_sam0

    # (e) Placebo: two SGD runs with the same seed should give identical outputs.
    placebo_a, placebo_b = {}, {}

    def grab_logits(store: dict):
        def cb(epoch, lr, bl, t, model):
            store["test_logits"] = predict_logits(model, data["Xte"])
        return cb

    train_model(method="sgd", rho=0.0, epochs=1, seed=3, data=data, on_epoch=grab_logits(placebo_a))
    train_model(method="sgd", rho=0.0, epochs=1, seed=3, data=data, on_epoch=grab_logits(placebo_b))
    placebo_nll_diff = nll_np(placebo_a["test_logits"], data["yte_np"]) - nll_np(placebo_b["test_logits"], data["yte_np"])
    placebo_logit_diff = float(np.max(np.abs(placebo_a["test_logits"] - placebo_b["test_logits"])))
    out["placebo_same_seed_sgd"] = {"paired_nll_diff": placebo_nll_diff,
                                    "max_abs_logit_diff": placebo_logit_diff}
    logger.info(f"check (e) placebo (SGD vs SGD, same seed): NLL diff={placebo_nll_diff:.3e}, "
                f"max logit diff={placebo_logit_diff:.3e}")
    del placebo_a, placebo_b

    (CHECKS / "checks.json").write_text(json.dumps(out, indent=1))
    return out


def stage_mini() -> None:
    """(f) Two epochs on a 5k training subset: every condition, every metric, end to end."""
    data = load_data(train_subset=5000)
    out_dir = CHECKS
    run_config(name="mini_sam", method="sam", rho=0.1, epochs=2, seed=0, data=data,
               out_dir=out_dir, use_cache=False, keep_all_logits=False)
    run_config(name="mini_sgd", method="sgd", rho=0.0, epochs=2, seed=0, data=data,
               out_dir=out_dir, use_cache=False, keep_all_logits=False)
    run_config(name="mini_sgd_short", method="sgd", rho=0.0, epochs=1, seed=0, data=data,
               out_dir=out_dir, use_cache=False, keep_all_logits=False)
    logger.info("mini pipeline finished (outputs under results/checks)")
    del data
    gc.collect()


# ----------------------------------------------------------------------------- tuning and pilot

def stage_tune(data: dict) -> float:
    records = {}
    timings = {}
    for rho in RHO_GRID:
        name = f"tune_sam_rho{rho}_s{SEED_TUNE}"
        t0 = time.time()
        rec = run_config(name=name, method="sam", rho=rho, epochs=EPOCHS, seed=SEED_TUNE, data=data)
        timings[name] = time.time() - t0
        records[rho] = rec["final"]["val_nll_raw"]
        logger.info(f"tune rho={rho}: final val NLL={rec['final']['val_nll_raw']:.5f}")
        per_run = timings[name]
        logger.info(f"projected remaining tuning time: {per_run * (len(RHO_GRID) - RHO_GRID.index(rho) - 1) / 60:.1f} min")
    best = min(records, key=lambda r: records[r])
    logger.info(f"chosen rho={best} (lowest final validation NLL: {records})")
    (RESULTS / "tuning.json").write_text(json.dumps(
        {"seed": SEED_TUNE, "final_val_nll_raw_by_rho": {str(k): v for k, v in records.items()},
         "chosen_rho": best, "selection_rule": "lowest final-epoch validation NLL (raw); test not used",
         "wall_time_s": timings}, indent=1))
    return best


def stage_pilot(data: dict, rho: float) -> int:
    pilot = run_config(name=f"pilot_sam_rho{rho}_s{SEED_PILOT}", method="sam", rho=rho,
                       epochs=EPOCHS, seed=SEED_PILOT, data=data)
    L = pilot["final"]["train_loss_eval"]
    logger.info(f"Target training loss L_SAM40 = {L:.5f}")
    grid_losses = {}
    for E in E_GRID:
        rec = run_config(name=f"pilot_sgd_E{E}_s{SEED_PILOT}", method="sgd", rho=0.0, epochs=E,
                         seed=SEED_PILOT, data=data)
        grid_losses[E] = rec["final"]["train_loss_eval"]
        logger.info(f"pilot SGD E={E}: final train loss={grid_losses[E]:.5f} gap={grid_losses[E] - L:+.5f}")
    chosen = next((E for E in E_GRID if abs(grid_losses[E] - L) <= MATCH_TOL), None)
    extended = False
    if chosen is None:
        logger.warning("No E in the grid matches within tolerance; extending the grid")
        extended = True
        for E in E_EXT_GRID:
            rec = run_config(name=f"pilot_sgd_E{E}_s{SEED_PILOT}", method="sgd", rho=0.0, epochs=E,
                             seed=SEED_PILOT, data=data)
            grid_losses[E] = rec["final"]["train_loss_eval"]
            logger.info(f"pilot SGD E={E} (extension): loss={grid_losses[E]:.5f} gap={grid_losses[E] - L:+.5f}")
            if abs(grid_losses[E] - L) <= MATCH_TOL:
                chosen = E
                break
    feasible = chosen is not None
    if not feasible:
        logger.error("Matched-loss test infeasible: no E matches within tolerance")
    result = {
        "target_seed": SEED_PILOT, "rho": rho, "L_SAM40": L,
        "tolerance": MATCH_TOL, "grid": {str(k): v for k, v in sorted(grid_losses.items())},
        "extended_grid": extended, "E_star": chosen, "feasible": feasible,
        "rule": "smallest E whose final eval-mode training loss is within tolerance of L_SAM40",
    }
    (RESULTS / "chosen_E.json").write_text(json.dumps(result, indent=1))
    logger.info(f"chosen E*={chosen}")
    return chosen


# ----------------------------------------------------------------------------- main

def stage_main(data: dict, rho: float, E_star: int) -> None:
    for seed in SEEDS_MAIN:
        run_config(name=f"c1_sam_s{seed}", method="sam", rho=rho, epochs=EPOCHS, seed=seed, data=data)
        run_config(name=f"c2_sgd40_s{seed}", method="sgd", rho=0.0, epochs=EPOCHS, seed=seed, data=data)
        run_config(name=f"c3_sgdE{E_star}_s{seed}", method="sgd", rho=0.0, epochs=E_star, seed=seed,
                   data=data, keep_all_logits=False)
        gc.collect()


# ----------------------------------------------------------------------------- driver

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "checks", "mini", "tune", "pilot", "main", "analyze"])
    args = parser.parse_args()
    ensure_dirs()
    setup_logging()
    start_ram_watchdog()
    logger.info(f"Device: {DEVICE}; stage={args.stage}; torch={torch.__version__}")
    stages = ["checks", "mini", "tune", "pilot", "main", "analyze"] if args.stage == "all" else [args.stage]

    data = None
    for stage in stages:
        logger.info(f"===== stage: {stage} =====")
        t0 = time.time()
        if stage == "checks":
            data = load_data()
            stage_checks(data)
        elif stage == "mini":
            stage_mini()
        elif stage == "tune":
            data = data or load_data()
            stage_tune(data)
        elif stage == "pilot":
            data = data or load_data()
            tuning = json.loads((RESULTS / "tuning.json").read_text())
            stage_pilot(data, tuning["chosen_rho"])
        elif stage == "main":
            data = data or load_data()
            tuning = json.loads((RESULTS / "tuning.json").read_text())
            chosen = json.loads((RESULTS / "chosen_E.json").read_text())
            if chosen["feasible"]:
                stage_main(data, tuning["chosen_rho"], chosen["E_star"])
            else:
                # Pilot infeasible: conditions 1 and 2 only (matched-epoch comparison (a) remains).
                logger.error("Pilot infeasible; condition 3 is skipped and reported as infeasible.")
                for seed in SEEDS_MAIN:
                    run_config(name=f"c1_sam_s{seed}", method="sam", rho=tuning["chosen_rho"],
                               epochs=EPOCHS, seed=seed, data=data)
                    run_config(name=f"c2_sgd40_s{seed}", method="sgd", rho=0.0, epochs=EPOCHS,
                               seed=seed, data=data)
        elif stage == "analyze":
            from analysis import run_analysis
            run_analysis()
        logger.info(f"stage {stage} finished in {time.time() - t0:.1f}s")
        gc.collect()


if __name__ == "__main__":
    logger.catch(reraise=True)(main)()

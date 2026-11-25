#!/usr/bin/env python3
"""SAM vs SGD on a small CNN (Fashion-MNIST): training, tuning and final runs.

Modes (run in this order; each mode skips runs whose metrics.json already exists):
  smoke       2-epoch SGD and SAM runs written to scratch/ (pipeline check)
  timing      time one epoch per condition, apply the timing gate, write timings.csv
  tune        validation-only hyperparameter sweep on seed 100 (full 40 epochs)
  final       seeds 0-4, both conditions, selected hyperparameters, 40 epochs
  compressed  seeds 0-4, both conditions, compressed schedules of 5 and 15 epochs
  pairing     SAM(rho=0) == SGD check on CPU (also run by tests/test_checks.py)
  all         (default, no argument) every stage in order, then analyze.py

Test logits are written only by the final and compressed modes and are never used
for selection. Validation logits are the only input to hyperparameter choice and
to temperature fitting.
"""

from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torchvision.datasets import FashionMNIST

import metrics as M

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
LOGS = RESULTS / "logs"
CKPTS = RESULTS / "checkpoints"
DATA_DIR = ROOT / "scratch" / "data"
SMOKE_DIR = ROOT / "scratch" / "smoke"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS_MAIN = 40
SAVE_EPOCHS_MAIN = (5, 15, 40)
COMPRESSED_EPOCHS = (5, 15)
TUNE_SEED = 100
FINAL_SEEDS = (0, 1, 2, 3, 4)
LR_GRID_FULL = [0.01, 0.03, 0.1]
RHO_GRID_FULL = [0.05, 0.1, 0.2]
LR_GRID_CUT = [0.03, 0.1]
RHO_GRID_CUT = [0.05, 0.1]
SAM_RUN_LIMIT_S = 600.0          # single 40-epoch SAM run above this -> cut LR grid
TOTAL_BUDGET_S = 3 * 3600.0      # projected total above this -> cut compressed, then rho grid


def setup_logging() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(LOGS / "run.log", rotation="30 MB", level="DEBUG")


# ---------------------------------------------------------------- data

def split_indices() -> dict[str, np.ndarray]:
    """Fixed 50k/10k split of the 60k official training images (seed 0)."""
    path = RESULTS / "split_indices.npz"
    if path.exists():
        z = np.load(path)
        return {"train": z["train"], "val": z["val"]}
    rng = np.random.default_rng(0)
    perm = rng.permutation(60000)
    val, train = perm[:10000], perm[10000:]
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(path, train=train, val=val)
    logger.info(f"Saved split indices: train={len(train)} val={len(val)}")
    return {"train": train, "val": val}


def load_data() -> dict:
    """Load Fashion-MNIST, split once, normalise with train-split statistics."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tr = FashionMNIST(root=str(DATA_DIR), train=True, download=True)
        te = FashionMNIST(root=str(DATA_DIR), train=False, download=True)
    except (RuntimeError, OSError) as exc:
        logger.error(f"torchvision FashionMNIST download failed: {exc}")
        raise
    x_all = tr.data.float().div(255.0).unsqueeze(1)
    y_all = tr.targets.long()
    x_test = te.data.float().div(255.0).unsqueeze(1)
    y_test = te.targets.long()
    split = split_indices()
    tr_idx = torch.as_tensor(split["train"], dtype=torch.long)
    va_idx = torch.as_tensor(split["val"], dtype=torch.long)
    mean = float(x_all[tr_idx].mean())
    std = float(x_all[tr_idx].std())
    logger.info(f"Train-split normalisation: mean={mean:.4f} std={std:.4f}")

    def norm(x: torch.Tensor) -> torch.Tensor:
        return ((x - mean) / std).contiguous()

    data = {
        "x_train": norm(x_all[tr_idx]).to(DEVICE),
        "y_train": y_all[tr_idx].to(DEVICE),
        "x_val": norm(x_all[va_idx]).to(DEVICE),
        "y_val": y_all[va_idx].to(DEVICE),
        "x_test": norm(x_test).to(DEVICE),
        "y_test": y_test.to(DEVICE),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS / "labels.npz",
        train=y_all[tr_idx].numpy(),
        val=y_all[va_idx].numpy(),
        test=y_test.numpy(),
    )
    logger.info(
        f"Data: train={tuple(data['x_train'].shape)} val={tuple(data['x_val'].shape)} "
        f"test={tuple(data['x_test'].shape)} device={DEVICE}"
    )
    del x_all, y_all, x_test, y_test, tr, te
    gc.collect()
    return data


# ---------------------------------------------------------------- model and steps

class SmallCNN(nn.Module):
    """Conv(1->32)-BN-ReLU-Pool, Conv(32->64)-BN-ReLU-Pool, FC 3136->128->10.

    padding=1 keeps 28 -> 14 -> 7 (an assumption recorded in the README).
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _bn_modules(model: nn.Module) -> list[nn.BatchNorm2d]:
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def sgd_step(model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return float(loss.detach())


def sam_step(model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor,
             rho: float) -> float:
    """One sharpness-aware minimisation step on top of the base optimiser.

    eps = rho * g / ||g|| with g the gradient at the current weights. The second
    gradient is taken at w + eps with BatchNorm running statistics frozen (momentum
    0 for that pass), then eps is removed and the base optimiser steps with the
    perturbed-point gradient.
    """
    params = list(model.parameters())
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in params]
    gnorm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    scale = rho / (gnorm + 1e-12)
    eps = [g * scale for g in grads]
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.add_(e)
    bns = _bn_modules(model)
    saved_mom = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    opt.zero_grad(set_to_none=True)
    loss2 = F.cross_entropy(model(x), y)
    loss2.backward()
    for m, mo in zip(bns, saved_mom):
        m.momentum = mo
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.sub_(e)
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor, bs: int = 2000) -> np.ndarray:
    model.eval()
    out = [model(x[i:i + bs]).float().cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).numpy().astype(np.float32)


@torch.no_grad()
def train_loss_eval(model: nn.Module, x: torch.Tensor, y: torch.Tensor, bs: int = 2000) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(x), bs):
        total += float(F.cross_entropy(model(x[i:i + bs]), y[i:i + bs], reduction="sum"))
    return total / len(x)


def _sync() -> None:
    if DEVICE.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------- one training run

def train_run(*, data: dict, condition: str, lr: float, rho: float | None, seed: int,
              epochs: int, save_epochs: tuple[int, ...], out_dir: Path,
              include_test: bool, checkpoint: bool, max_epochs_timed: int | None = None,
              name: str | None = None) -> dict:
    """Train one model. Saves logits at save_epochs (val always, test only if include_test)."""
    if condition not in ("sgd", "sam"):
        raise ValueError(f"unknown condition {condition!r}")
    if condition == "sam" and rho is None:
        raise ValueError("SAM needs rho")
    name = name or out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logger.bind(run=name)
    sink = logger.add(LOGS / f"{name}.log", level="DEBUG",
                      filter=lambda r: r["extra"].get("run") == name)
    try:
        log.info(f"START {name}: condition={condition} lr={lr} rho={rho} seed={seed} "
                 f"epochs={epochs} save={save_epochs} include_test={include_test}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SmallCNN().to(DEVICE)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM,
                              weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        gen = torch.Generator().manual_seed(seed)
        n = data["x_train"].shape[0]
        epoch_times: list[float] = []
        saved: dict[str, dict] = {}
        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            model.train()
            perm = torch.randperm(n, generator=gen).to(DEVICE)
            batch_losses = []
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                xb = data["x_train"][idx]
                yb = data["y_train"][idx]
                if condition == "sgd":
                    batch_losses.append(sgd_step(model, opt, xb, yb))
                else:
                    batch_losses.append(sam_step(model, opt, xb, yb, rho))
            lr_now = opt.param_groups[0]["lr"]
            sched.step()
            _sync()
            dt = time.perf_counter() - t0
            epoch_times.append(dt)
            log.info(f"epoch {ep}/{epochs} lr={lr_now:.5f} batch_loss_mean="
                     f"{float(np.mean(batch_losses)):.4f} time={dt:.2f}s")
            if ep in save_epochs:
                tl = train_loss_eval(model, data["x_train"], data["y_train"])
                val_logits = predict_logits(model, data["x_val"])
                val_acc = float((val_logits.argmax(1) == data["y_val"].cpu().numpy()).mean())
                val_nll = float(M.nll_per_example(val_logits, data["y_val"].cpu().numpy()).mean())
                entry = {"epoch": ep, "train_loss_eval": tl, "val_acc": val_acc,
                         "val_nll": val_nll}
                arrays = {"val_logits": val_logits}
                if include_test:
                    test_logits = predict_logits(model, data["x_test"])
                    test_acc = float((test_logits.argmax(1)
                                      == data["y_test"].cpu().numpy()).mean())
                    entry["test_acc"] = test_acc
                    arrays["test_logits"] = test_logits
                    log.info(f"EVAL ep={ep} train_loss={tl:.5f} val_acc={val_acc:.4f} "
                             f"val_nll={val_nll:.4f} test_acc={test_acc:.4f}")
                else:
                    log.info(f"EVAL ep={ep} train_loss={tl:.5f} val_acc={val_acc:.4f} "
                             f"val_nll={val_nll:.4f} (no test logits)")
                np.savez(out_dir / f"logits_ep{ep}.npz", **arrays)
                saved[str(ep)] = entry
        if checkpoint:
            CKPTS.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CKPTS / f"{name}.pt")
        final_tl = saved[str(epochs)]["train_loss_eval"]
        summary = {
            "name": name, "condition": condition, "lr": lr, "rho": rho, "seed": seed,
            "epochs": epochs, "batch": BATCH, "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY, "saved": saved,
            "final_train_loss_eval": final_tl,
            "needs_extension_check": bool(final_tl > 0.01),
            "epoch_times_s": epoch_times,
            "mean_epoch_time_s": float(np.mean(epoch_times)),
            "device": str(DEVICE),
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        log.info(f"DONE {name}: final_train_loss={final_tl:.5f} "
                 f"mean_epoch_time={summary['mean_epoch_time_s']:.2f}s")
        del model, opt, sched
        gc.collect()
        return summary
    except (RuntimeError, ValueError) as exc:
        log.error(f"Run {name} failed: {exc}")
        raise
    finally:
        logger.remove(sink)


def run_or_load(**kwargs) -> dict:
    out_dir: Path = kwargs["out_dir"]
    mpath = out_dir / "metrics.json"
    if mpath.exists():
        logger.info(f"Skipping {out_dir.name}: metrics.json exists")
        return json.loads(mpath.read_text())
    return train_run(**kwargs)


# ---------------------------------------------------------------- helpers

def run_name(condition: str, lr: float, rho: float | None, seed: int, epochs: int) -> str:
    rho_part = "" if rho is None else f"_rho{rho}"
    return f"{condition}_lr{lr}{rho_part}_E{epochs}_s{seed}"


def load_chosen() -> dict:
    return json.loads((RESULTS / "chosen_hparams.json").read_text())


def load_grid() -> dict:
    return json.loads((RESULTS / "chosen_grid.json").read_text())


def read_val_nll(run_dir: Path, epoch: int) -> float:
    z = np.load(run_dir / f"logits_ep{epoch}.npz")
    labels = np.load(RESULTS / "labels.npz")["val"]
    return float(M.nll_per_example(z["val_logits"], labels).mean())


# ---------------------------------------------------------------- modes

def mode_smoke(data: dict) -> None:
    logger.info("SMOKE: 2 epochs SGD and SAM at lr=0.1, rho=0.1 (scratch only)")
    for cond, rho in (("sgd", None), ("sam", 0.1)):
        name = run_name(cond, 0.1, rho, 0, 2) + "_smoke"
        s = train_run(data=data, condition=cond, lr=0.1, rho=rho, seed=0, epochs=2,
                      save_epochs=(2,), out_dir=SMOKE_DIR / name, include_test=True,
                      checkpoint=False, name=name)
        logger.info(f"smoke {name}: {s['saved']['2']}")


def mode_timing(data: dict) -> None:
    """Time one epoch per condition, apply the timing gate, write timings.csv."""
    logger.info("TIMING: warm-up epoch (SGD, discarded), then timed SGD and SAM epochs")
    rows = []
    timings = {}
    for cond, rho in (("sgd", None), ("sam", 0.1), ("sgd", None), ("sam", 0.1)):
        name = f"timing_{cond}"
        s = train_run(data=data, condition=cond, lr=0.1, rho=rho, seed=0, epochs=1,
                      save_epochs=(1,), out_dir=SMOKE_DIR / f"timing_{cond}_{len(timings.get(cond, []))}",
                      include_test=False, checkpoint=False, name=name)
        t = s["epoch_times_s"][0]
        rows.append({"kind": "epoch_timing", "condition": cond, "name": name,
                     "seconds": round(t, 3), "note": "1-epoch measurement"})
        timings.setdefault(cond, []).append(t)
    # warm-up = first pair; timed = second pair
    t_sgd = timings["sgd"][-1]
    t_sam = timings["sam"][-1]
    logger.info(f"Per-epoch time: SGD={t_sgd:.2f}s SAM={t_sam:.2f}s (after warm-up)")

    grid = {"lr": LR_GRID_FULL, "rho": RHO_GRID_FULL, "compressed": True,
            "decisions": []}
    def projected(lr_grid, rho_grid, compressed):
        n_sgd = len(lr_grid)
        n_sam = len(lr_grid) * len(rho_grid)
        tune = (n_sgd * t_sgd + n_sam * t_sam) * EPOCHS_MAIN
        final = 5 * EPOCHS_MAIN * (t_sgd + t_sam)
        comp = 5 * sum(COMPRESSED_EPOCHS) * (t_sgd + t_sam) if compressed else 0.0
        return tune + final + comp

    sam_run_s = t_sam * EPOCHS_MAIN
    if sam_run_s > SAM_RUN_LIMIT_S:
        grid["lr"] = LR_GRID_CUT
        grid["decisions"].append(
            f"single 40-epoch SAM run projected {sam_run_s:.0f}s > {SAM_RUN_LIMIT_S:.0f}s: "
            f"LR grid cut to {LR_GRID_CUT}")
    total = projected(grid["lr"], grid["rho"], True)
    if total > TOTAL_BUDGET_S:
        grid["compressed"] = False
        grid["decisions"].append(
            f"projected total {total / 3600:.2f} h > 3 h: compressed-schedule runs dropped")
        total = projected(grid["lr"], grid["rho"], False)
    if total > TOTAL_BUDGET_S:
        grid["rho"] = RHO_GRID_CUT
        grid["decisions"].append(
            f"projected total {total / 3600:.2f} h still > 3 h: rho grid cut to {RHO_GRID_CUT}")
        total = projected(grid["lr"], grid["rho"], False)
    if not grid["decisions"]:
        grid["decisions"].append("no cut: full grid fits the timing gate")
    grid["projected_total_h"] = total / 3600
    grid["t_sgd_epoch_s"] = t_sgd
    grid["t_sam_epoch_s"] = t_sam
    grid["projected_sam_40ep_run_s"] = sam_run_s
    rows.append({"kind": "gate_decision", "condition": "", "name": "",
                 "seconds": round(total, 1),
                 "note": " | ".join(grid["decisions"])})
    rows.append({"kind": "projection", "condition": "", "name": "",
                 "seconds": round(total, 1),
                 "note": f"projected total GPU seconds; lr={grid['lr']} rho={grid['rho']} "
                         f"compressed={grid['compressed']}"})
    (RESULTS / "chosen_grid.json").write_text(json.dumps(grid, indent=2))
    with (RESULTS / "timings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "condition", "name", "seconds", "note"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Timing gate: {grid['decisions']} projected total {total / 3600:.2f} h")


def mode_tune(data: dict) -> None:
    grid = load_grid()
    rows = []
    cfgs = [("sgd", lr, None) for lr in grid["lr"]]
    cfgs += [("sam", lr, rho) for lr in grid["lr"] for rho in grid["rho"]]
    for cond, lr, rho in cfgs:
        name = run_name(cond, lr, rho, TUNE_SEED, EPOCHS_MAIN)
        out = RUNS / f"tune_{name}"
        s = run_or_load(data=data, condition=cond, lr=lr, rho=rho, seed=TUNE_SEED,
                        epochs=EPOCHS_MAIN, save_epochs=(EPOCHS_MAIN,), out_dir=out,
                        include_test=False, checkpoint=False, name=f"tune_{name}")
        e = s["saved"][str(EPOCHS_MAIN)]
        rows.append({"condition": cond, "lr": lr, "rho": "" if rho is None else rho,
                     "seed": TUNE_SEED, "epochs": EPOCHS_MAIN,
                     "val_nll_final": round(e["val_nll"], 6),
                     "val_acc_final": round(e["val_acc"], 6),
                     "train_loss_final": round(e["train_loss_eval"], 6)})
        logger.info(f"tune {cond} lr={lr} rho={rho}: val_nll={e['val_nll']:.4f} "
                    f"val_acc={e['val_acc']:.4f}")
    with (RESULTS / "tuning_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sgd_rows = [r for r in rows if r["condition"] == "sgd"]
    sam_rows = [r for r in rows if r["condition"] == "sam"]
    best_sgd = min(sgd_rows, key=lambda r: r["val_nll_final"])
    best_sam = min(sam_rows, key=lambda r: r["val_nll_final"])
    chosen = {
        "selection_rule": "minimum validation NLL of the final-epoch model, seed 100",
        "sgd": {"lr": best_sgd["lr"], "val_nll_final": best_sgd["val_nll_final"]},
        "sam": {"lr": best_sam["lr"], "rho": best_sam["rho"],
                "val_nll_final": best_sam["val_nll_final"]},
        "grid": grid,
        "tuning_table": "results/tuning_table.csv",
    }
    (RESULTS / "chosen_hparams.json").write_text(json.dumps(chosen, indent=2))
    logger.info(f"Chosen: SGD lr={chosen['sgd']['lr']}; SAM lr={chosen['sam']['lr']} "
                f"rho={chosen['sam']['rho']}")


def mode_final(data: dict) -> None:
    ch = load_chosen()
    lr_sgd = float(ch["sgd"]["lr"])
    lr_sam, rho = float(ch["sam"]["lr"]), float(ch["sam"]["rho"])
    for seed in FINAL_SEEDS:
        for cond, lr, r in (("sgd", lr_sgd, None), ("sam", lr_sam, rho)):
            name = run_name(cond, lr, r, seed, EPOCHS_MAIN)
            out = RUNS / f"final_{name}"
            s = run_or_load(data=data, condition=cond, lr=lr, rho=r, seed=seed,
                            epochs=EPOCHS_MAIN, save_epochs=SAVE_EPOCHS_MAIN, out_dir=out,
                            include_test=True, checkpoint=True, name=f"final_{name}")
            val_y = np.load(RESULTS / "labels.npz")["val"]
            z = np.load(out / f"logits_ep{EPOCHS_MAIN}.npz")
            temp = M.fit_temperature(z["val_logits"], val_y)
            T = temp["temperature"]
            np.savez(out / "ts_test_logits.npz",
                     test_logits=(z["test_logits"].astype(np.float64) / T).astype(np.float32))
            ts = {"seed": seed, "condition": cond, **temp}
            (out / "temperature.json").write_text(json.dumps(ts, indent=2))
            logger.info(f"final {name}: T={T:.4f} at_bound={temp['at_bound']} "
                        f"final_train_loss={s['final_train_loss_eval']:.5f} "
                        f"test_acc@40={s['saved']['40']['test_acc']:.4f}")
            if s["needs_extension_check"]:
                logger.warning(f"final {name}: eval train loss "
                               f"{s['final_train_loss_eval']:.5f} > 0.01; recorded, not extended")


def mode_compressed(data: dict) -> None:
    ch = load_chosen()
    lr_sgd = float(ch["sgd"]["lr"])
    lr_sam, rho = float(ch["sam"]["lr"]), float(ch["sam"]["rho"])
    for seed in FINAL_SEEDS:
        for E in COMPRESSED_EPOCHS:
            for cond, lr, r in (("sgd", lr_sgd, None), ("sam", lr_sam, rho)):
                name = run_name(cond, lr, r, seed, E)
                out = RUNS / f"compressed_{name}"
                run_or_load(data=data, condition=cond, lr=lr, rho=r, seed=seed, epochs=E,
                            save_epochs=(E,), out_dir=out, include_test=True,
                            checkpoint=False, name=f"compressed_{name}")
                logger.info(f"compressed done: {name}")


def mode_pairing() -> None:
    """SAM(rho=0) must reproduce SGD exactly on the same seed (CPU, a few steps)."""
    dev = torch.device("cpu")
    x = torch.randn(256, 1, 28, 28)
    y = torch.randint(0, 10, (256,))
    results = {}
    for cond in ("sgd", "sam"):
        torch.manual_seed(7)
        m = SmallCNN().to(dev)
        opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        g = torch.Generator().manual_seed(7)
        m.train()
        for _ in range(3):
            perm = torch.randperm(256, generator=g)
            for i in range(0, 256, BATCH):
                idx = perm[i:i + BATCH]
                if cond == "sgd":
                    sgd_step(m, opt, x[idx], y[idx])
                else:
                    sam_step(m, opt, x[idx], y[idx], 0.0)
        results[cond] = predict_logits(m, x)
    diff = float(np.abs(results["sgd"] - results["sam"]).max())
    logger.info(f"Pairing check: max |SGD - SAM(rho=0)| logits = {diff:.3e}")
    if diff > 1e-4:
        raise AssertionError(f"pairing check failed: max diff {diff}")
    return None


def main() -> None:
    setup_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    logger.info(f"method.py mode={mode} device={DEVICE} torch={torch.__version__}")
    if mode == "pairing":
        mode_pairing()
        return
    data = load_data()
    dispatch = {
        "smoke": mode_smoke,
        "timing": mode_timing,
        "tune": mode_tune,
        "final": mode_final,
        "compressed": mode_compressed,
    }
    if mode in ("", "all"):
        # Default: every stage in order. Finished runs are skipped (metrics.json exists),
        # the timing gate runs only when chosen_grid.json is missing, then analysis writes
        # method_out.json and its full/mini/preview variants at the workspace root.
        if not (RESULTS / "chosen_grid.json").exists():
            mode_timing(data)
        for stage in ("tune", "final", "compressed"):
            dispatch[stage](data)
        import analyze
        analyze.main()
        return
    if mode not in dispatch:
        raise SystemExit(f"unknown mode {mode!r}; choose from {sorted(dispatch) + ['pairing', 'all']}")
    dispatch[mode](data)


if __name__ == "__main__":
    main()

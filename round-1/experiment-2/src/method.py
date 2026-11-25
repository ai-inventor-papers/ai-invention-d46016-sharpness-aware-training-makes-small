#!/usr/bin/env python3
"""Does SAM's calibration gain track overfitting? (Fashion-MNIST, small CNN, SGD vs SAM).

Subcommands
  train    : train SGD and SAM runs (per seed), log per-epoch metrics, save logits at logit epochs.
  analyze  : test metrics, temperature-scaling control, matched-training-loss comparison,
             descriptive statistics, figures and results/method_out.json + results/notes.md.
  unit     : sanity checks for ECE, adaptive calibration error, temperature scaling,
             power iteration and SAM(rho=0) == SGD.

Run:  uv run method.py train --out <dir>
      uv run method.py analyze
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.optimize import minimize_scalar

WS = Path(__file__).resolve().parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- pre-specified configuration (artifact plan) ----
LR = 0.03
MOMENTUM = 0.9
WD = 5e-4
RHO = 0.1
EPOCHS = 60
BATCH = 128
N_BINS = 15
HESS_ITERS = 20
HESS_N = 2000
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_EVAL_EPOCHS = [1, 2, 5] + list(range(10, 61, 5))
DEFAULT_LOGIT_EPOCHS = [5, 60]
N_TRAIN = 50000
N_VAL = 10000


# --------------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {message}")
    logger.add(log_dir / "run.log", level="DEBUG", rotation="50 MB")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax_np(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))


# ------------------------------------------------------------------- calibration metrics
def ece_score(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error, equal-width bins on top-label confidence (Guo et al. 2017)."""
    n = len(conf)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def ace_score(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Adaptive calibration error: equal-mass bins on top-label confidence (Nixon et al. 2019)."""
    order = np.argsort(conf, kind="stable")
    chunks = np.array_split(order, n_bins)
    gaps = [abs(correct[c].mean() - conf[c].mean()) for c in chunks if len(c) > 0]
    return float(np.mean(gaps))


def calib_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    logits = logits.astype(np.float64)
    logp = log_softmax_np(logits)
    p = np.exp(logp)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    nll = float(-np.mean(logp[np.arange(len(y)), y]))
    onehot = np.eye(logits.shape[1])[y]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    return {
        "acc": float(correct.mean()),
        "ece": ece_score(conf, correct),
        "nll": nll,
        "brier": brier,
        "ace": ace_score(conf, correct),
    }


# ------------------------------------------------------------------- sharpness
def top_hessian_eig(model: nn.Module, Xh: torch.Tensor, yh: torch.Tensor, iters: int) -> tuple[float, float]:
    """Top Hessian eigenvalue of mean CE on a fixed batch by power iteration (Rayleigh quotient).

    Fixed start vector (CPU generator seed 0). Returns (eigenvalue, |change| in last iteration).
    """
    params = list(model.parameters())
    g = torch.Generator().manual_seed(0)
    v_flat = torch.randn(sum(p.numel() for p in params), generator=g)
    v = _split_like(v_flat, params)
    v = _scale(v, 1.0 / _norm(v))
    eig_prev, eig = None, float("nan")
    for _ in range(iters):
        loss = F.cross_entropy(model(Xh), yh)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((gi * vi).sum() for gi, vi in zip(grads, v))
        Hv = [h.detach() for h in torch.autograd.grad(gv, params)]
        eig_prev, eig = eig, float(sum((vi * h).sum() for vi, h in zip(v, Hv)))
        nHv = _norm(Hv)
        v = _scale(Hv, 1.0 / (nHv + 1e-30))
        del grads, gv, Hv, loss
    delta = float("nan") if eig_prev is None else abs(eig - eig_prev)
    return eig, delta


def _split_like(flat: torch.Tensor, params: list) -> list:
    out, i = [], 0
    for p in params:
        n = p.numel()
        out.append(flat[i:i + n].view_as(p).to(p.device))
        i += n
    return out


def _norm(ts: list) -> float:
    return float(math.sqrt(sum(float((t * t).sum()) for t in ts)))


def _scale(ts: list, c: float) -> list:
    return [t * c for t in ts]


def rho_ball_increase(model: nn.Module, Xh: torch.Tensor, yh: torch.Tensor, rho: float) -> float:
    """L(w + e(w)) - L(w) on the fixed batch, e(w) = rho * g / ||g||."""
    params = list(model.parameters())
    with torch.no_grad():
        L0 = float(F.cross_entropy(model(Xh), yh))
    loss = F.cross_entropy(model(Xh), yh)
    grads = [g.detach() for g in torch.autograd.grad(loss, params)]
    gn = _norm(grads)
    es = [rho * g / (gn + 1e-12) for g in grads]
    with torch.no_grad():
        for p, e in zip(params, es):
            p.add_(e)
        L1 = float(F.cross_entropy(model(Xh), yh))
        for p, e in zip(params, es):
            p.sub_(e)
    return L1 - L0


# ------------------------------------------------------------------- model / optim
class MaxPool2(nn.Module):
    """2x2 max pooling via max_pool2d(return_indices=True), the form that supports double backward
    (needed for Hessian-vector products). Forward values are identical to a 2x2 MaxPool."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, 2, return_indices=True)[0]


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), MaxPool2(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), MaxPool2(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sgd_step(model, opt, xb, yb):
    opt.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    opt.step()
    return loss.detach(), logits.detach()


def sam_step(model, base, xb, yb, rho: float):
    """SAM (Foret et al. 2021), non-adaptive, base optimiser = SGD with momentum and wd."""
    params = list(model.parameters())
    base.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    grads = [p.grad.detach().clone() for p in params]
    gn = _norm(grads)
    es = [rho * g / (gn + 1e-12) for g in grads]
    with torch.no_grad():
        for p, e in zip(params, es):
            p.add_(e)
    base.zero_grad(set_to_none=True)
    F.cross_entropy(model(xb), yb).backward()
    with torch.no_grad():
        for p, e in zip(params, es):
            p.sub_(e)
    base.step()
    return loss.detach(), logits.detach()


# ------------------------------------------------------------------- data
def load_data(out: Path, hess_n: int) -> dict:
    from torchvision import datasets

    root = WS / "scratch" / "data"
    root.mkdir(parents=True, exist_ok=True)
    tr = datasets.FashionMNIST(root=str(root), train=True, download=True)
    te = datasets.FashionMNIST(root=str(root), train=False, download=True)
    X_all = tr.data.float().div(255.0).unsqueeze(1)
    y_all = tr.targets.long()
    X_test = te.data.float().div(255.0).unsqueeze(1)
    y_test = te.targets.long()

    perm = np.random.default_rng(12345).permutation(60000)
    tr_idx, va_idx = perm[:N_TRAIN], perm[N_TRAIN:]
    mean = float(X_all[tr_idx].mean())
    std = float(X_all[tr_idx].std())
    X_all = (X_all - mean) / std
    X_test = (X_test - mean) / std

    hess_idx = tr_idx[np.random.default_rng(0).permutation(N_TRAIN)[:hess_n]]

    out_res = out / "results"
    out_res.mkdir(parents=True, exist_ok=True)
    (out_res / "split_indices.json").write_text(json.dumps({
        "train": tr_idx.tolist(), "validation": va_idx.tolist(),
        "sharpness_batch": hess_idx.tolist(),
        "normalisation": {"mean": mean, "std": std, "computed_on": "50k train split"},
    }))
    np.savez(out_res / "labels.npz", val=y_all[va_idx].numpy(), test=y_test.numpy())

    d = {
        "Xtr": X_all[tr_idx].to(DEVICE), "ytr": y_all[tr_idx].to(DEVICE),
        "Xva": X_all[va_idx].to(DEVICE), "yva_np": y_all[va_idx].numpy(),
        "Xte": X_test.to(DEVICE), "yte_np": y_test.numpy(),
        "Xh": X_all[hess_idx].to(DEVICE), "yh": y_all[hess_idx].to(DEVICE),
    }
    logger.info(f"data: train {len(tr_idx)}, val {len(va_idx)}, test {len(y_test)}, sharpness batch {len(hess_idx)}")
    return d


# ------------------------------------------------------------------- run
def run_one(opt_name: str, seed: int, data: dict, out: Path, epochs: int,
            eval_epochs: set, logit_epochs: set, hess_iters: int, hess_n: int) -> list:
    res_dir = out / "results"
    ck_dir = out / "checkpoints"
    (res_dir / "logits").mkdir(parents=True, exist_ok=True)
    (res_dir / "runs").mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{opt_name}_seed{seed}"
    run_file = res_dir / "runs" / f"{tag}.json"
    resume_file = ck_dir / f"{tag}_resume.pt"

    set_seed(seed)  # same seed -> identical initialisation for SGD and SAM
    model = SmallCNN().to(DEVICE)
    params = list(model.parameters())
    base = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=EPOCHS)
    records: list = []
    start = 1
    if resume_file.exists():
        st = torch.load(resume_file, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"])
        base.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        records = st["records"]
        start = st["epoch"] + 1
        logger.warning(f"{tag}: resuming from epoch {start}")
        if start > epochs:
            return records
    n = N_TRAIN
    status = "ok"
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        g = torch.Generator().manual_seed(1000 * seed + epoch)  # same order for SGD and SAM
        perm = torch.randperm(n, generator=g).to(DEVICE)
        loss_sum = torch.zeros((), device=DEVICE)
        correct_sum = torch.zeros((), device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = data["Xtr"][idx], data["ytr"][idx]
            if opt_name == "sgd":
                loss, logits = sgd_step(model, base, xb, yb)
            else:
                loss, logits = sam_step(model, base, xb, yb, RHO)
            loss_sum += loss * len(idx)
            correct_sum += (logits.argmax(1) == yb).sum()
        sched.step()
        train_loss = float(loss_sum) / n
        train_acc = float(correct_sum) / n
        rec = {"optimizer": opt_name, "seed": seed, "epoch": epoch,
               "train_loss": train_loss, "train_acc": train_acc, "lr": base.param_groups[0]["lr"],
               "val_ece": None, "val_nll": None, "val_brier": None, "val_ace": None, "val_acc": None,
               "top_hessian_eig": None, "hessian_last_change": None, "rho_ball_increase": None}
        if not math.isfinite(train_loss):
            rec["status"] = "diverged"
            records.append(rec)
            status = f"diverged at epoch {epoch}"
            logger.error(f"{tag}: loss not finite at epoch {epoch}; stopping run")
            break
        rec["status"] = "ok"
        if epoch in eval_epochs:
            model.eval()
            with torch.no_grad():
                va_logits = torch.cat([model(data["Xva"][i:i + 1000]) for i in range(0, N_VAL, 1000)])
                va_np = va_logits.cpu().numpy()
            m = calib_metrics(va_np, data["yva_np"])
            eig, delta = top_hessian_eig(model, data["Xh"], data["yh"], hess_iters)
            rec.update({"val_ece": m["ece"], "val_nll": m["nll"], "val_brier": m["brier"],
                        "val_ace": m["ace"], "val_acc": m["acc"],
                        "top_hessian_eig": eig, "hessian_last_change": delta})
            if opt_name == "sam":
                rec["rho_ball_increase"] = rho_ball_increase(model, data["Xh"], data["yh"], RHO)
            if epoch in logit_epochs:
                np.save(res_dir / "logits" / f"{tag}_epoch{epoch}_val.npy", va_np.astype(np.float32))
                with torch.no_grad():
                    te_logits = torch.cat([model(data["Xte"][i:i + 1000]) for i in range(0, len(data["Xte"]), 1000)])
                np.save(res_dir / "logits" / f"{tag}_epoch{epoch}_test.npy", te_logits.cpu().numpy().astype(np.float32))
                torch.save(model.state_dict(), ck_dir / f"{tag}_epoch{epoch}.pt")
            if not math.isfinite(eig):
                rec["status"] = "eig_nonfinite"
            model.train()
            logger.info(f"{tag} ep{epoch:3d} loss {train_loss:.4f} acc {train_acc:.4f} "
                        f"valECE {m['ece']:.4f} valNLL {m['nll']:.4f} eig {eig:.3f} "
                        f"({time.time() - t0:.1f}s)")
        else:
            logger.info(f"{tag} ep{epoch:3d} loss {train_loss:.4f} acc {train_acc:.4f} ({time.time() - t0:.1f}s)")
        records.append(rec)
        torch.save({"model": model.state_dict(), "opt": base.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "records": records}, resume_file)

    if status == "ok" and records and records[-1]["epoch"] != epochs:
        status = "incomplete"
    run_file.write_text(json.dumps({"optimizer": opt_name, "seed": seed, "status": status,
                                    "records": records}, indent=1))
    logger.info(f"{tag}: finished with status '{status}'")
    del model, base, sched
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return records


def write_per_epoch(out: Path) -> None:
    res_dir = out / "results"
    all_recs = []
    for f in sorted((res_dir / "runs").glob("*.json")):
        all_recs += json.loads(f.read_text())["records"]
    (res_dir / "per_epoch.json").write_text(json.dumps(all_recs, indent=1))


def cmd_train(args) -> None:
    out = Path(args.out)
    setup_logging(out / "logs")
    logger.info(f"device {DEVICE}; torch {torch.__version__}")
    eval_epochs = set(args.eval_epochs)
    logit_epochs = set(args.logit_epochs)
    data = load_data(out, args.hess_n)
    order = [(o, s) for s in args.seeds for o in ("sgd", "sam")]
    cfg = {"lr": LR, "momentum": MOMENTUM, "weight_decay": WD, "rho": RHO, "epochs": args.epochs,
           "batch": BATCH, "schedule": "cosine per epoch (T_max=epochs)", "hessian_iters": args.hess_iters,
           "hessian_batch": args.hess_n, "eval_epochs": sorted(eval_epochs),
           "logit_epochs": sorted(logit_epochs), "seeds": args.seeds, "device": str(DEVICE)}
    (out / "results" / "run_config.json").write_text(json.dumps(cfg, indent=1))
    for opt_name, seed in order:
        t0 = time.time()
        run_one(opt_name, seed, data, out, args.epochs, eval_epochs, logit_epochs,
                args.hess_iters, args.hess_n)
        write_per_epoch(out)
        logger.info(f"run {opt_name} seed {seed} took {(time.time() - t0) / 60:.1f} min")
    logger.info("training done")


# ------------------------------------------------------------------- unit checks
def cmd_unit(_args) -> None:
    setup_logging(WS / "logs")
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        logger.info(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    # (a) ECE: perfectly calibrated -> ~0; confidence 0.9 accuracy 0.5 -> 0.4
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 200000)
    correct = (rng.uniform(size=conf.size) < conf).astype(float)
    e_cal = ece_score(conf, correct)
    check("ECE calibrated ~0", e_cal < 0.005, f"{e_cal:.5f}")
    c = np.full(1000, 0.9)
    k = (np.arange(1000) % 2 == 0).astype(float)  # accuracy exactly 0.5
    check("ECE const 0.9 acc 0.5 = 0.4", abs(ece_score(c, k) - 0.4) < 1e-9, f"{ece_score(c, k):.6f}")
    # (b) ACE on the same sets
    check("ACE calibrated ~0", ace_score(conf, correct) < 0.01, f"{ace_score(conf, correct):.5f}")
    check("ACE const 0.9 acc 0.5 = 0.4", abs(ace_score(c, k) - 0.4) < 1e-9, f"{ace_score(c, k):.6f}")
    # (c) temperature scaling recovers the known generating temperature.
    # Labels drawn from softmax(z / T_true): the NLL-optimal T for z must be T_true.
    T_true = 2.5
    z = rng.normal(size=(20000, 10)) * 2.0
    yl = np.array([rng.choice(10, p=p) for p in softmax_np(z / T_true)])
    T_fit = fit_temperature(z, yl)
    check("TS recovers known T=2.5 (n=20000, tol 0.1 for sampling noise)", abs(T_fit - T_true) < 0.1,
          f"T_fit={T_fit:.4f}")
    # Exact check with soft targets softmax(z/T_true): minimiser must be T_true
    soft = softmax_np(z / T_true)

    def soft_nll(logT):
        lp = log_softmax_np(z / math.exp(logT))
        return -float(np.mean(np.sum(soft * lp, axis=1)))
    r = minimize_scalar(soft_nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded",
                        options={"xatol": 1e-8})
    check("TS exact soft-target recovery of T=2.5 within 0.01", abs(math.exp(r.x) - T_true) < 0.01,
          f"T={math.exp(r.x):.5f}")
    # (d) power iteration on quadratic with eigenvalues 1..10
    class Quad(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(10))
    # Well-separated spectrum (top 10, next 5): gap ratio 0.5 -> 20 iterations give ~1e-6 error.
    # The plain 1..10 spectrum has ratio 0.9 and 20 iterations only reach ~5e-2 error; this is logged below.
    Hdiag = torch.tensor([10.0, 5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6])

    def power_on_quadratic(iters):
        m = Quad()
        params = list(m.parameters())
        g = torch.Generator().manual_seed(0)
        v = _split_like(torch.randn(10, generator=g), params)
        v = _scale(v, 1.0 / _norm(v))
        eig = None
        for _ in range(iters):
            Hv = [Hdiag * v[0]]
            eig = float((v[0] * Hv[0]).sum())
            v = _scale(Hv, 1.0 / _norm(Hv))
        return eig
    eig10 = power_on_quadratic(20)
    check("power iteration returns 10 within 1e-3 (separated spectrum)", abs(eig10 - 10.0) < 1e-3, f"{eig10:.6f}")
    Hdiag = torch.arange(1, 11, dtype=torch.float32)
    eig_dense = power_on_quadratic(20)
    logger.info(f"info: plain 1..10 spectrum, 20 iters -> {eig_dense:.4f} (gap ratio 0.9; not a pass/fail check)")
    # (e) SAM with rho=0 equals SGD for 20 steps
    x = torch.randn(256, 1, 28, 28, device=DEVICE)
    y = torch.randint(0, 10, (256,), device=DEVICE)
    set_seed(0)
    m1 = SmallCNN().to(DEVICE)
    set_seed(0)
    m2 = SmallCNN().to(DEVICE)
    o1 = torch.optim.SGD(m1.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    o2 = torch.optim.SGD(m2.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    maxdiff = 0.0
    for s in range(20):
        idx = slice((s * 64) % 192, (s * 64) % 192 + 64)
        sgd_step(m1, o1, x[idx], y[idx])
        sam_step(m2, o2, x[idx], y[idx], 0.0)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        maxdiff = max(maxdiff, float((p1.detach() - p2.detach()).abs().max()))
    check("SAM(rho=0) == SGD over 20 steps", maxdiff < 1e-5, f"max param diff {maxdiff:.2e}")
    logger.info("unit checks " + ("ALL PASSED" if ok else "SOME FAILED"))
    if not ok:
        sys.exit(1)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    logits = logits.astype(np.float64)

    def nll(logT):
        T = math.exp(logT)
        lp = log_softmax_np(logits / T)
        return -float(np.mean(lp[np.arange(len(y)), y]))
    res = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded",
                          options={"xatol": 1e-6})
    return float(math.exp(res.x))


# ------------------------------------------------------------------- analysis
def cmd_analyze(args) -> None:
    from analysis import run_analysis  # separate module: metrics, stats, figures
    run_analysis(WS, args)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train")
    pt.add_argument("--out", default=str(WS))
    pt.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    pt.add_argument("--epochs", type=int, default=EPOCHS)
    pt.add_argument("--eval-epochs", type=int, nargs="+", default=DEFAULT_EVAL_EPOCHS)
    pt.add_argument("--logit-epochs", type=int, nargs="+", default=DEFAULT_LOGIT_EPOCHS)
    pt.add_argument("--hess-iters", type=int, default=HESS_ITERS)
    pt.add_argument("--hess-n", type=int, default=HESS_N)
    sub.add_parser("unit")
    pa = sub.add_parser("analyze")
    pa.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = p.parse_args()
    {"train": cmd_train, "unit": cmd_unit, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()

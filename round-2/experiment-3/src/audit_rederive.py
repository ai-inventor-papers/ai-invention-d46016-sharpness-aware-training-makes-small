#!/usr/bin/env python3
"""Independent re-derivation of the headline numbers from raw files.

Deliberately different code path from method.py:
  - reads only results/logits/*.npz (raw logits) and labels from the official FashionMNIST
    cache via torchvision; the validation labels come from splits/split_seed0.json;
  - NLL via torch.nn.functional.cross_entropy (method.py uses numpy log_softmax);
  - temperature by a log-spaced grid search (method.py uses scipy minimize_scalar);
  - ECE via np.digitize on explicit bin edges (method.py uses ceil arithmetic);
  - paired t-test via scipy.stats.ttest_1samp (method.py computes t by hand);
  - placebo: the same SAM-minus-SGD comparison against a fixed shuffle of test labels.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torchvision import datasets

WS = Path(__file__).resolve().parent
LOG = WS / "results" / "logits"
RUNS = WS / "results" / "runs"
SEEDS = [0, 1, 2, 3, 4]
SEL = json.loads((WS / "results" / "selection.json").read_text())
SGD = f"sgd_lr{SEL['sgd']['lr']:g}_rho0_seed"
SAM = f"sam_lr{SEL['sam']['lr']:g}_rho{SEL['sam']['rho']:g}_seed"

split = json.loads((WS / "splits" / "split_seed0.json").read_text())
val_idx = np.array(split["val_idx"])
te = datasets.FashionMNIST(str(WS / "data_cache"), train=False, download=False)
yte = te.targets.numpy()
tr = datasets.FashionMNIST(str(WS / "data_cache"), train=True, download=False)
yva = tr.targets.numpy()[val_idx]


def nll_t(z: np.ndarray, y: np.ndarray) -> float:
    return float(torch.nn.functional.cross_entropy(torch.tensor(z, dtype=torch.float64),
                                                   torch.tensor(y)).item())


def fit_T_grid(zv: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 2000))
    losses = [nll_t(zv / t, y) for t in grid]
    return float(grid[int(np.argmin(losses))])


def ece_bins(z: np.ndarray, y: np.ndarray, nb: int = 15) -> float:
    p = torch.softmax(torch.tensor(z, dtype=torch.float64), dim=1).numpy()
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, nb + 1)
    idx = np.digitize(conf, edges[1:-1], right=True)  # bins (lo, hi]
    e = 0.0
    for k in range(nb):
        m = idx == k
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def brier(z: np.ndarray, y: np.ndarray) -> float:
    p = torch.softmax(torch.tensor(z, dtype=torch.float64), dim=1).numpy()
    oh = np.eye(10)[y]
    return float(((p - oh) ** 2).sum(1).mean())


def load(prefix: str, seed: int):
    z = np.load(LOG / f"{prefix}{seed}_ep40.npz")
    return z["val"].astype(np.float64), z["test"].astype(np.float64)


def arm_stats(prefix: str):
    out = {"test_nll": [], "ts_test_nll": [], "ece": [], "ts_ece": [], "T": [], "train_loss": []}
    for s in SEEDS:
        zv, zt = load(prefix, s)
        T = fit_T_grid(zv, yva)
        out["T"].append(T)
        out["test_nll"].append(nll_t(zt, yte))
        out["ts_test_nll"].append(nll_t(zt / T, yte))
        out["ece"].append(ece_bins(zt, yte))
        out["ts_ece"].append(ece_bins(zt / T, yte))
        out["train_loss"].append(json.loads((RUNS / f"{prefix}{s}.json").read_text())["final_train_loss"])
    return {k: np.array(v) for k, v in out.items()}


def report_row(label: str, d: np.ndarray) -> dict:
    t, p = stats.ttest_1samp(d, 0.0)
    half = stats.t.ppf(0.975, df=len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"row": label, "mean": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
            "p_two_sided": float(p)}


def main() -> None:
    sgd = arm_stats(SGD)
    sam = arm_stats(SAM)
    rows = {
        "SAM - TS-SGD: NLL": report_row("SAM - TS-SGD: NLL", sam["ts_test_nll"] - sgd["ts_test_nll"]),
        "SAM - SGD: NLL": report_row("SAM - SGD: NLL", sam["test_nll"] - sgd["test_nll"]),
        "SAM - TS-SGD: ECE": report_row("SAM - TS-SGD: ECE", sam["ts_ece"] - sgd["ts_ece"]),
        "SAM - SGD: ECE": report_row("SAM - SGD: ECE", sam["ece"] - sgd["ece"]),
    }
    print("Temperatures SGD:", np.round(sgd["T"], 3).tolist())
    print("Temperatures SAM:", np.round(sam["T"], 3).tolist())
    print("Final train loss SGD mean:", round(float(sgd["train_loss"].mean()), 4),
          "SAM mean:", round(float(sam["train_loss"].mean()), 4))
    for r in rows.values():
        print(f"{r['row']}: mean={r['mean']:+.5f} CI=[{r['ci95'][0]:+.5f},{r['ci95'][1]:+.5f}] p={r['p_two_sided']:.4f}")

    # Placebo: the same comparison against a fixed shuffle of the test labels.
    shuf = np.random.default_rng(12345).permutation(yte)
    placebo = {}
    sgd_raw_shuf = []
    sam_raw_shuf = []
    for s in SEEDS:
        _, zt_s = load(SGD, s)
        _, zt_m = load(SAM, s)
        sgd_raw_shuf.append(nll_t(zt_s, shuf))
        sam_raw_shuf.append(nll_t(zt_m, shuf))
    placebo["SAM - SGD: NLL, shuffled labels"] = report_row(
        "SAM - SGD: NLL, shuffled labels", np.array(sam_raw_shuf) - np.array(sgd_raw_shuf))
    print("PLACEBO", json.dumps(placebo, indent=1))

    out = {"headline": rows, "placebo": placebo,
           "T_sgd": sgd["T"].tolist(), "T_sam": sam["T"].tolist()}
    (WS / "results" / "audit_rederivation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

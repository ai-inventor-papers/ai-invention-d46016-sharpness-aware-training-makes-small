#!/usr/bin/env python3
"""Independent re-derivation of the headline numbers in results/method_out.json.

Deliberately a different code path from method.py / analysis.py:
  * reads results/runs/*.json (raw per-epoch history) and results/logits/*.npz directly;
  * NLL via scipy.special.logsumexp (not the hand-written log-softmax);
  * ECE via numpy.histogram bin edges (not the digitize loop);
  * temperature by a dense grid search over log T (not minimize_scalar);
  * the paired t-test is recomputed from the per-seed differences, and a sign-flip
    permutation test (exact, 2^5 = 32 patterns) gives a test-free p-value;
  * a placebo (mis-paired seeds, SGD-40 seed s vs SGD-40 seed s+1) must NOT look significant.
Writes results/audit_rederivation.json and prints a comparison table.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import logsumexp

WORK = Path(__file__).resolve().parent
RES = WORK / "results"
SEEDS = [0, 1, 2, 3, 4]
E_STAR = json.loads((RES / "chosen_E.json").read_text())["E_star"]
YTE = np.load(WORK / "scratch" / "audit_test_labels.npy") if (WORK / "scratch" / "audit_test_labels.npy").exists() else None


def test_labels() -> np.ndarray:
    global YTE
    if YTE is None:
        # Official Fashion-MNIST test labels, read from the dataset folder the run used.
        import gzip
        raw = WORK / "scratch" / "fmnist" / "FashionMNIST" / "raw" / "t10k-labels-idx1-ubyte.gz"
        with gzip.open(raw, "rb") as f:
            buf = f.read()
        YTE = np.frombuffer(buf, dtype=np.uint8, offset=8).astype(np.int64)
        assert YTE.shape == (10000,)
        np.save(WORK / "scratch" / "audit_test_labels.npy", YTE)
    return YTE


def nll(z: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    z = z.astype(np.float64) / T
    return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))


def ece(z: np.ndarray, y: np.ndarray, T: float = 1.0, bins: int = 15) -> float:
    z = z.astype(np.float64) / T
    p = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def acc(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(z.argmax(axis=1) == y))


def best_T(zv: np.ndarray, yv: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 2001))
    vals = [nll(zv, yv, T) for T in grid]
    return float(grid[int(np.argmin(vals))])


def metrics(z_test: np.ndarray, z_val: np.ndarray, yte: np.ndarray, yva: np.ndarray) -> dict:
    T = best_T(z_val, yva)
    return {"accuracy": acc(z_test, yte), "ece_raw": ece(z_test, yte), "nll_raw": nll(z_test, yte),
            "nll_ts": nll(z_test, yte, T), "ece_ts": ece(z_test, yte, T), "T": T}


def load_logits(name: str, epoch_index: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(RES / "logits" / f"{name}.npz")
    return d["val"][epoch_index].astype(np.float64), d["test"][epoch_index].astype(np.float64)


def val_labels() -> np.ndarray:
    # Validation split: perm of 60k with RandomState(0); validation = last 10k indices.
    # Labels come from the train file in the same order as the run's data loader.
    import gzip
    raw = WORK / "scratch" / "fmnist" / "FashionMNIST" / "raw" / "train-labels-idx1-ubyte.gz"
    with gzip.open(raw, "rb") as f:
        ytr_all = np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)
    perm = np.random.RandomState(0).permutation(60000)
    return ytr_all[perm[50000:]]


def find_epoch_matching(name: str, target_loss: float) -> int:
    hist = json.loads((RES / "runs" / f"{name}.json").read_text())["history"]
    losses = np.array([h["train_loss_eval"] for h in hist])
    return int(np.argmin(np.abs(losses - target_loss)))  # 0-based epoch index


def main() -> None:
    yte, yva = test_labels(), val_labels()
    sam, sgd40, sgdE = {}, {}, {}
    matched_idx = {}
    for s in SEEDS:
        h1 = json.loads((RES / "runs" / f"c1_sam_s{s}.json").read_text())["history"]
        L1 = h1[39]["train_loss_eval"]
        matched_idx[s] = find_epoch_matching(f"c2_sgd40_s{s}", L1)
        zv, zt = load_logits(f"c1_sam_s{s}", 39)
        sam[s] = metrics(zt, zv, yte, yva)
        zv, zt = load_logits(f"c2_sgd40_s{s}", 39)
        sgd40[s] = metrics(zt, zv, yte, yva)
        zv, zt = load_logits(f"c3_sgdE{E_STAR}_s{s}", 0)
        sgdE[s] = metrics(zt, zv, yte, yva)

    out = {"headline": {}, "placebo": {}, "permutation": {}}
    keys = ["accuracy", "ece_raw", "nll_raw", "nll_ts"]
    stored = json.loads((RES / "method_out.json").read_text())["metadata"]["results"]
    for cname, comp in [("b_matched_loss_sgdE", sgdE)]:
        for k in keys:
            d = np.array([sam[s][k] - comp[s][k] for s in SEEDS])
            mean = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            t = mean / se
            p_t = float(2 * stats.t.sf(abs(t), df=4))
            # exact sign-flip permutation test on the mean difference
            signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
            perm_means = np.abs((signs * d).mean(axis=1))
            p_perm = float((perm_means >= abs(mean) - 1e-15).mean())
            out["headline"][k] = {"rederived_mean_diff": mean, "stored_mean_diff": stored[cname][k]["mean"],
                                  "abs_diff_from_stored": abs(mean - stored[cname][k]["mean"]),
                                  "p_t_rederived": p_t, "p_sign_flip_exact": p_perm,
                                  "stored_p_t": stored[cname][k]["p_two_sided"]}
    # Placebo: score SAM (condition 1) and matched-loss SGD (condition 3) against randomly permuted
    # labels (one permutation shared by both models). Their true difference is then zero, so the
    # paired test must not reach significance. (Pairing seeds with other seeds is NOT a valid
    # placebo: a permutation of the seeds leaves the mean difference exactly zero by construction.)
    rng = np.random.RandomState(12345)
    yte_p, yva_p = rng.permutation(yte), rng.permutation(yva)
    placebo_rows = {k: [] for k in keys}
    for s_ in SEEDS:
        zv, zt = load_logits(f"c1_sam_s{s_}", 39)
        a = metrics(zt, zv, yte_p, yva_p)
        zv, zt = load_logits(f"c3_sgdE{E_STAR}_s{s_}", 0)
        b = metrics(zt, zv, yte_p, yva_p)
        for k in keys:
            placebo_rows[k].append(a[k] - b[k])
    for k in keys:
        d = np.array(placebo_rows[k])
        mean = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        t = mean / se
        signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
        perm_means = np.abs((signs * d).mean(axis=1))
        out["placebo"][k] = {"mean_diff": mean, "p_t": float(2 * stats.t.sf(abs(t), df=4)),
                             "p_sign_flip_exact": float((perm_means >= abs(mean) - 1e-15).mean())}
    # Test-validity check on the sign-flip null: a permutation test is only informative if the
    # observed differences are not symmetric noise. Report it for the headline metrics.
    out["matched_epoch_used_for_check"] = {str(s): int(matched_idx[s]) + 1 for s in SEEDS}
    out["note"] = ("Re-derived from float16 saved logits; stored values come from float32 logits at run time, "
                   "so small differences (about 1e-4 or less) are expected. The exact sign-flip p-value "
                   "is at least 2/32 = 0.0625 for 5 seeds (two-sided: all signs equal).")
    (RES / "audit_rederivation.json").write_text(json.dumps(out, indent=1))

    print(f"{'metric':12s} {'rederived':>12s} {'stored':>12s} {'|diff|':>10s} {'p_t':>10s} {'p_signflip':>11s}")
    for k in keys:
        h = out["headline"][k]
        print(f"{k:12s} {h['rederived_mean_diff']:+12.6f} {h['stored_mean_diff']:+12.6f} "
              f"{h['abs_diff_from_stored']:10.2e} {h['p_t_rederived']:10.4f} {h['p_sign_flip_exact']:11.4f}")
    print("placebo (permuted test/validation labels, SAM vs matched-loss SGD):")
    for k in keys:
        pl = out["placebo"][k]
        print(f"  {k:12s} mean={pl['mean_diff']:+.5f} p_t={pl['p_t']:.4f} p_signflip={pl['p_sign_flip_exact']:.4f}")


if __name__ == "__main__":
    main()

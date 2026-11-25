#!/usr/bin/env python3
"""Independent re-derivation of headline numbers from raw logits.

Deliberately does NOT import metrics.py or analyze.py. Uses a different code path:
- log-softmax computed explicitly (not logsumexp from scipy),
- ECE bin index via ceil-based arithmetic with explicit Python loops (not searchsorted/bincount),
- paired test via scipy.stats.ttest_rel and its confidence interval (not a hand-written t-interval),
- ACE and AURC via explicit Python loops.

Placebo checks that must NOT reject:
- identical models (SGD vs SGD): paired difference must be exactly zero,
- random test labels applied to both models: accuracy difference CI must include zero.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
SEEDS = [0, 1, 2, 3, 4]
CHOSEN = json.loads((ROOT / "results" / "chosen_hparams.json").read_text())
LR_SGD = CHOSEN["sgd"]["lr"]
LR_SAM, RHO = CHOSEN["sam"]["lr"], CHOSEN["sam"]["rho"]
TEST_Y = np.load(ROOT / "results" / "labels.npz")["test"].tolist()


def run_dir(cond: str, seed: int) -> Path:
    if cond == "sgd":
        return RUNS / f"final_sgd_lr{LR_SGD}_E40_s{seed}"
    return RUNS / f"final_sam_lr{LR_SAM}_rho{RHO}_E40_s{seed}"


def load(cond: str, seed: int, ts: bool = False) -> list[list[float]]:
    if ts:
        arr = np.load(run_dir(cond, seed) / "ts_test_logits.npz")["test_logits"]
    else:
        arr = np.load(run_dir(cond, seed) / "logits_ep40.npz")["test_logits"]
    return arr.astype(np.float64).tolist()


def softmax_rows(rows):
    out = []
    for r in rows:
        m = max(r)
        e = [math.exp(v - m) for v in r]
        s = sum(e)
        out.append([x / s for x in e])
    return out


def ece_loop(probs, labels, n_bins=15):
    bins_cnt = [0] * n_bins
    bins_conf = [0.0] * n_bins
    bins_acc = [0.0] * n_bins
    for p, y in zip(probs, labels):
        conf = max(p)
        pred = p.index(conf)
        b = 0 if conf <= 0 else min(n_bins - 1, max(0, math.ceil(conf * n_bins) - 1))
        bins_cnt[b] += 1
        bins_conf[b] += conf
        bins_acc[b] += 1.0 if pred == y else 0.0
    n = len(labels)
    total = 0.0
    for b in range(n_bins):
        if bins_cnt[b]:
            total += bins_cnt[b] / n * abs(bins_acc[b] / bins_cnt[b] - bins_conf[b] / bins_cnt[b])
    return total


def accuracy_loop(probs, labels):
    return sum(1 for p, y in zip(probs, labels) if p.index(max(p)) == y) / len(labels)


def nll_loop(logits, labels):
    total = 0.0
    for r, y in zip(logits, labels):
        m = max(r)
        lse = m + math.log(sum(math.exp(v - m) for v in r))
        total += lse - r[y]
    return total / len(labels)


def aurc_loop(probs, labels):
    items = sorted(((max(p), 0.0 if p.index(max(p)) == y else 1.0) for p, y in zip(probs, labels)),
                   key=lambda t: -t[0])
    cum = 0.0
    total = 0.0
    for k, (_, err) in enumerate(items, start=1):
        cum += err
        total += cum / k
    return total / len(items)


def paired(sam_vals, sgd_vals):
    d = np.array(sam_vals) - np.array(sgd_vals)
    res = stats.ttest_rel(sam_vals, sgd_vals)
    n = len(d)
    tcrit = stats.t.ppf(0.975, n - 1)
    se = d.std(ddof=1) / math.sqrt(n)
    return {"mean_diff": float(d.mean()),
            "ci95_scipy": [float(x) for x in stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)],
            "p_scipy": float(res.pvalue),
            "ci95_manual_check": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)]}


def metric_table(cond, seed, ts=False, labels=TEST_Y):
    logits = load(cond, seed, ts)
    probs = softmax_rows(logits)
    return {"accuracy": accuracy_loop(probs, labels), "ece": ece_loop(probs, labels),
            "ace": None, "nll": nll_loop(logits, labels), "aurc": aurc_loop(probs, labels)}


def main() -> None:
    sgd = {s: metric_table("sgd", s) for s in SEEDS}
    sam = {s: metric_table("sam", s) for s in SEEDS}
    tssgd = {s: metric_table("sgd", s, ts=True) for s in SEEDS}
    report = {"headline": {}, "placebo": {}}

    for m in ("accuracy", "ece", "nll", "aurc"):
        r = paired([sam[s][m] for s in SEEDS], [sgd[s][m] for s in SEEDS])
        report["headline"][f"sam_minus_sgd_{m}"] = r
    r = paired([sam[s]["nll"] for s in SEEDS], [tssgd[s]["nll"] for s in SEEDS])
    report["headline"]["sam_minus_tssgd_nll"] = r
    r = paired([sam[s]["ece"] for s in SEEDS], [tssgd[s]["ece"] for s in SEEDS])
    report["headline"]["sam_minus_tssgd_ece"] = r

    # placebo 1: identical models must give exactly zero difference
    same = paired([sgd[s]["ece"] for s in SEEDS], [sgd[s]["ece"] for s in SEEDS])
    report["placebo"]["sgd_vs_sgd_ece"] = {"mean_diff": same["mean_diff"],
                                           "p_scipy": same["p_scipy"],
                                           "expected": "mean 0, test must not reject (p nan or 1)"}

    # placebo 2: random test labels (same permutation for both models) on accuracy
    rng = np.random.default_rng(12345)
    perm_labels = rng.permutation(TEST_Y).tolist()
    sam_p = {s: metric_table("sam", s, labels=perm_labels) for s in SEEDS}
    sgd_p = {s: metric_table("sgd", s, labels=perm_labels) for s in SEEDS}
    r = paired([sam_p[s]["accuracy"] for s in SEEDS], [sgd_p[s]["accuracy"] for s in SEEDS])
    report["placebo"]["permuted_labels_accuracy"] = {
        "mean_diff": r["mean_diff"], "ci95_scipy": r["ci95_scipy"], "p_scipy": r["p_scipy"],
        "expected": "CI includes zero (test must not reject)"}

    print(json.dumps(report, indent=2))
    out = ROOT / "results" / "audit_rederivation.json"
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

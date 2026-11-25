#!/usr/bin/env python3
"""Independent re-derivation of the headline numbers in results/caveats.md.

Reads the raw per-run and per-epoch records from inputs/ directly, with its own code path:
plain Python lists, a hand-written sample standard deviation, and a Student-t critical value
from scipy. It does not import eval.py and does not read gap_table.csv or eval_out.json.

It then runs the same paired t-test on placebo input (each seed's SAM/SGD labels swapped at
random, all 2^n assignments enumerated) and reports how often it rejects, plus the exact
sign-flip permutation p-value that is the smallest achievable for n seeds.
"""
import itertools
import json
import math
from pathlib import Path

from scipy import stats

WS = Path(__file__).resolve().parent.parent
D1 = WS / "inputs" / "dep1_sam_fashion_5seed" / "full_method_out.json"
D2 = WS / "inputs" / "dep2_sam_trajectory_3seed" / "full_method_out.json"


def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def t_interval(d):
    n = len(d)
    m = mean(d)
    se = sd(d) / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    t = m / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    return m, m - tc * se, m + tc * se, p


def sign_flip_p(d):
    """Exact two-sided sign-flip permutation p-value for paired differences d."""
    obs = abs(mean(d))
    n = len(d)
    count = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=n):
        total += 1
        if abs(mean([s * v for s, v in zip(signs, d)])) >= obs - 1e-15:
            count += 1
    return count / total


def load_src1():
    rows = json.loads(D1.read_text())["datasets"]
    rec = next(x for x in rows if x["dataset"] == "fashion_mnist_test_per_model_run")["examples"]
    out = {}
    for r in rec:
        m = r["metadata_metrics"]
        if not isinstance(m, dict):
            m = eval(m, {"__builtins__": {}})  # literal dict string written by the producer
        out[(r["metadata_model"], r["metadata_schedule"], int(r["metadata_epoch"]), int(r["metadata_seed"]))] = m
    return out


def load_src2():
    rows = json.loads(D2.read_text())["datasets"]
    ep = next(x for x in rows if x["dataset"] == "Fashion-MNIST per-epoch training records")["examples"]
    return {(r["metadata_optimizer"], int(r["metadata_seed"]), int(r["metadata_epoch"])): r for r in ep}


def main():
    s1 = load_src1()
    seeds1 = [0, 1, 2, 3, 4]
    lines = []
    ece40 = [s1[("SAM", "40-epoch run", 40, s)]["ece"] - s1[("SGD", "40-epoch run", 40, s)]["ece"] for s in seeds1]
    m, lo, hi, p = t_interval(ece40)
    lines.append(f"S1 test ECE gap ep40: mean {m:+.5f} CI [{lo:+.5f},{hi:+.5f}] t-p {p:.2e}; exact sign-flip p {sign_flip_p(ece40):.4f}")
    nll40 = [s1[("SAM", "40-epoch run", 40, s)]["nll"] - s1[("SGD", "40-epoch run", 40, s)]["nll"] for s in seeds1]
    m, lo, hi, p = t_interval(nll40)
    lines.append(f"S1 test NLL gap ep40: mean {m:+.5f} CI [{lo:+.5f},{hi:+.5f}]")
    acc40 = [s1[("SAM", "40-epoch run", 40, s)]["accuracy"] - s1[("SGD", "40-epoch run", 40, s)]["accuracy"] for s in seeds1]
    m, lo, hi, p = t_interval(acc40)
    lines.append(f"S1 accuracy gap ep40: mean {m:+.5f} CI [{lo:+.5f},{hi:+.5f}]")
    ece5 = [s1[("SAM", "compressed", 5, s)]["ece"] - s1[("SGD", "compressed", 5, s)]["ece"] for s in seeds1]
    m, lo, hi, p = t_interval(ece5)
    lines.append(f"S1 test ECE gap ep5 (compressed): mean {m:+.5f} CI [{lo:+.5f},{hi:+.5f}] t-p {p:.4f}")
    ts_nll = [s1[("SAM", "40-epoch run", 40, s)]["nll"] - s1[("TS-SGD", "40-epoch run", 40, s)]["nll"] for s in seeds1]
    m, lo, hi, p = t_interval(ts_nll)
    lines.append(f"S1 SAM minus TS-SGD NLL ep40: mean {m:+.5f} CI [{lo:+.5f},{hi:+.5f}]")

    # Placebo for the source-1 ECE gap at 40: swap labels per seed at random, enumerate all 2^5 assignments
    base = [s1[("SAM", "40-epoch run", 40, s)]["ece"] for s in seeds1]
    other = [s1[("SGD", "40-epoch run", 40, s)]["ece"] for s in seeds1]
    rejects = 0
    combos = 0
    for swap in itertools.product((0, 1), repeat=len(seeds1)):
        d = [(o - b) if sw else (b - o) for b, o, sw in zip(base, other, swap)]
        combos += 1
        if t_interval(d)[3] < 0.05:
            rejects += 1
    lines.append(f"Placebo S1 ECE (all 32 label swaps): t-test rejects at 0.05 in {rejects}/{combos}")

    s2 = load_src2()
    seeds2 = [0, 1, 2]
    def gap(metric, ep):
        return [float(s2[("SAM", s, ep)]["metadata_" + metric]) - float(s2[("SGD", s, ep)]["metadata_" + metric]) for s in seeds2]
    v60 = gap("val_ece", 60)
    m, lo, hi, p = t_interval(v60)
    lines.append(f"S2 val ECE gap ep60: mean {m:+.5f} CI [{lo:+.5f},{hi:+.5f}]; exact sign-flip min p {sign_flip_p(v60):.3f}")
    nll_w = [a - b for a, b in zip(gap("val_nll", 60), gap("val_nll", 1))]
    m, lo, hi, p = t_interval(nll_w)
    lines.append(f"S2 val NLL widening 60 minus 1: mean {m:+.5f} CI [{lo:+.5f},{hi:+.5f}] t-p {p:.4f}; exact sign-flip p {sign_flip_p(nll_w):.3f}")
    ece_w = [a - b for a, b in zip(gap("val_ece", 60), gap("val_ece", 1))]
    m, lo, hi, p = t_interval(ece_w)
    lines.append(f"S2 val ECE widening 60 minus 1: mean {m:+.5f} CI [{lo:+.5f},{hi:+.5f}] t-p {p:.4f}")

    # Placebo for the source-2 NLL widening test: all 2^3 label swaps
    rejects = 0
    combos = 0
    for swap in itertools.product((0, 1), repeat=3):
        d = [(-x if sw else x) for x, sw in zip(nll_w, swap)]
        combos += 1
        if t_interval(d)[3] < 0.05:
            rejects += 1
    lines.append(f"Placebo S2 NLL widening (8 sign assignments): t-test rejects at 0.05 in {rejects}/{combos}")

    # Peak of the mean gap, from raw records
    means = {e: mean(gap("val_ece", e)) for e in sorted({k[2] for k in s2 if k[2] in (1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)})}
    e_min = min(means, key=means.get)
    lines.append(f"S2 val ECE mean gap extremum: epoch {e_min} value {means[e_min]:+.5f}; epoch 60 value {means[60]:+.5f}")

    out = WS / "checks" / "rederive_headlines_out.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

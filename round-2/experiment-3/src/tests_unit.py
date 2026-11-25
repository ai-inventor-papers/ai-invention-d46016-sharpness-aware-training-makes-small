#!/usr/bin/env python3
"""Stage 0 unit checks (CPU). Run: .venv/bin/python tests_unit.py"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import method as M  # noqa: E402

WS = Path(__file__).resolve().parent


def test_sam_two_pass_matches_hand_computation() -> None:
    """Toy linear model f(x) = w*x, loss = 0.5*(w*x - y)^2 (one sample), SAM with rho.
    Hand: g = (w x - y) x; eps = rho g/|g|; g2 = (w+eps x - y) x; w_new = w - lr*g2."""
    torch.manual_seed(0)
    w = nn.Parameter(torch.tensor([0.7], dtype=torch.float64))
    x = torch.tensor([1.5], dtype=torch.float64)
    y = torch.tensor([2.0], dtype=torch.float64)
    lr, rho = 0.1, 0.2
    params = [w]
    base = torch.optim.SGD(params, lr=lr, momentum=0.0)

    # Same operations as train_run's SAM branch on one batch.
    pred = w * x
    loss = 0.5 * (pred - y).pow(2).sum()
    loss.backward()
    with torch.no_grad():
        gn = torch.sqrt(torch.stack([p.grad.pow(2).sum() for p in params]).sum())
        eps = [rho * p.grad / (gn + 1e-12) for p in params]
        for p, e in zip(params, eps):
            p.add_(e)
    base.zero_grad()
    loss2 = 0.5 * (w * x - y).pow(2).sum()
    loss2.backward()
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.sub_(e)
    base.step()

    g0 = (0.7 * 1.5 - 2.0) * 1.5
    eps_h = rho * g0 / abs(g0)
    g2 = (0.7 + eps_h) * 1.5 * 1.5 - 2.0 * 1.5
    expected = 0.7 - lr * g2
    assert abs(w.item() - expected) < 1e-9, (w.item(), expected)
    print(f"PASS SAM two-pass update: {w.item():.10f} == {expected:.10f}")


def test_bn_running_stats_frozen_on_perturbed_pass() -> None:
    bn = nn.BatchNorm2d(4)
    bn.train()
    x = torch.randn(16, 4, 5, 5)
    bn(x)  # first pass updates stats normally
    before_mean = bn.running_mean.clone()
    before_var = bn.running_var.clone()
    bn.momentum = 0.0
    bn(torch.randn(16, 4, 5, 5) * 3 + 2)  # perturbed pass
    bn.momentum = M.BN_MOMENTUM
    assert torch.equal(bn.running_mean, before_mean)
    assert torch.equal(bn.running_var, before_var)
    print("PASS BatchNorm running stats unchanged during the perturbed pass")


def test_temperature_one_is_identity() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=(500, 10)).astype(np.float32)
    y = rng.integers(0, 10, 500)
    m1 = M.metrics(z, y)
    m2 = M.metrics(z / 1.0, y)
    assert m1 == m2
    assert abs(M.nll_of(z / 1.0, y) - m1["nll"]) < 1e-12
    print("PASS temperature T=1 reproduces raw metrics")


def test_ece_calibrated_vs_overconfident() -> None:
    rng = np.random.default_rng(2)
    n = 200000
    p = rng.uniform(0.1, 1.0, n)  # confidence
    correct = rng.uniform(size=n) < p  # perfectly calibrated
    ece_cal = M.ece_equal_width(p, correct.astype(float))
    assert ece_cal < 0.01, ece_cal
    conf_over = np.clip(p + 0.3, 0, 1)
    ece_over = M.ece_equal_width(conf_over, correct.astype(float))
    assert ece_over > 0.2, ece_over
    print(f"PASS ECE calibrated={ece_cal:.4f} (near 0), overconfident={ece_over:.4f} (clearly positive)")


def test_splits() -> None:
    s = json.loads((WS / "splits" / "split_seed0.json").read_text())
    tr, va = set(s["train_idx"]), set(s["val_idx"])
    assert len(s["train_idx"]) == 50000 and len(s["val_idx"]) == 10000
    assert len(tr & va) == 0
    assert len(tr | va) == 60000
    assert s["n_test"] == 10000
    print("PASS split sizes 50000 / 10000 / 10000, no train-val overlap")


def test_paired_t_matches_scipy() -> None:
    d = np.array([0.3, -0.1, 0.2, 0.25, 0.05])
    row = M.paired_row("toy", d)
    t_ref, p_ref = stats.ttest_rel(d, np.zeros_like(d))
    assert abs(row["t"] - t_ref) < 1e-10, (row["t"], t_ref)
    assert abs(row["p_two_sided"] - p_ref) < 1e-10
    half = 2.776 * d.std(ddof=1) / math.sqrt(5)
    assert abs((row["ci95_high"] - row["ci95_low"]) / 2 - half) < 1e-12
    print("PASS paired t, p and 95% interval match scipy.stats.ttest_rel")


def test_holm() -> None:
    rows = [{"p_two_sided": p} for p in (0.01, 0.04, 0.03, 0.2)]
    M.holm(rows)
    # sorted p: 0.01 (x4=0.04), 0.03 (x3=0.09), 0.04 (x2=0.08 -> monotone 0.09), 0.2 (x1)
    expected = [0.04, 0.09, 0.09, 0.2]
    for r, e in zip(rows, expected):
        assert abs(r["p_holm"] - e) < 1e-12, (r["p_holm"], e)
    print("PASS Holm adjustment")


if __name__ == "__main__":
    test_sam_two_pass_matches_hand_computation()
    test_bn_running_stats_frozen_on_perturbed_pass()
    test_temperature_one_is_identity()
    test_ece_calibrated_vs_overconfident()
    test_splits()
    test_paired_t_matches_scipy()
    test_holm()
    print("ALL UNIT CHECKS PASSED")

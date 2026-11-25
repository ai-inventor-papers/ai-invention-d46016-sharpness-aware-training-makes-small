#!/usr/bin/env python3
"""Unit and pairing checks from the testing plan (steps 1, 2 and 8).

Run: .venv/bin/python tests/test_checks.py   (exit non-zero on any failure)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import metrics as M  # noqa: E402
import method as METHOD  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def test_ece() -> bool:
    perfect_conf = np.full(1000, 0.7)
    rng = np.random.default_rng(0)
    perfect_correct = (rng.random(1000) < 0.7).astype(float)
    # perfectly calibrated in expectation is noisy; use exactly calibrated bins instead:
    conf = np.array([0.5] * 4 + [1.0] * 4)
    correct = np.array([1, 0, 1, 0, 1, 1, 1, 1], dtype=float)  # 0.5 bin acc 0.5, 1.0 bin acc 1.0
    ok1 = check("ECE zero for exactly calibrated bins", abs(M.ece(conf, correct)) < 1e-12,
                f"got {M.ece(conf, correct):.6f}")
    ok2 = check("ECE hand-built value = 0.1", abs(M.ece_known_example() - 0.1) < 1e-12,
                f"got {M.ece_known_example():.6f}")
    _ = perfect_conf, perfect_correct
    return ok1 and ok2


def test_aurc() -> bool:
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1.0, 0.0, 1.0, 1.0])
    expected = (0.0 + 0.5 + 1.0 / 3.0 + 0.25) / 4.0
    got = M.aurc(conf, correct)
    return check("AURC matches hand-computed value", abs(got - expected) < 1e-12,
                 f"got {got:.6f} expected {expected:.6f}")


def test_ace_nll_brier() -> bool:
    logits = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]))
    labels = np.array([0, 1])
    nll = M.nll_per_example(logits, labels)
    ok1 = check("NLL matches -log p", np.allclose(nll, -np.log([0.7, 0.8])))
    probs = M.softmax64(logits)
    b = M.brier_per_example(probs, labels)
    exp_b = [(0.7 - 1) ** 2 + 0.2 ** 2 + 0.1 ** 2, 0.1 ** 2 + (0.8 - 1) ** 2 + 0.1 ** 2]
    ok2 = check("Brier sums over classes", np.allclose(b, exp_b))
    conf = np.linspace(0.1, 0.9, 50)
    correct = (np.arange(50) % 2).astype(float)
    a = M.ace(conf, correct)
    ok3 = check("ACE finite and in [0,1]", np.isfinite(a) and 0 <= a <= 1, f"got {a:.4f}")
    return ok1 and ok2 and ok3


def test_temperature_recovery() -> bool:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 3.0, size=(20000, 10))
    t_gen = 2.0
    p = M.softmax64(base / t_gen)
    labels = np.array([rng.choice(10, p=row) for row in p[:20000]])
    res = M.fit_temperature(base, labels)
    ok = abs(res["temperature"] - t_gen) < 0.05
    return check("temperature scaling recovers generating T within 0.05", ok,
                 f"got {res['temperature']:.4f} (grid {res['grid_best_T']:.4f})")


def test_pairing() -> bool:
    try:
        METHOD.mode_pairing()
    except AssertionError as exc:
        return check("SAM(rho=0) reproduces SGD exactly (CPU)", False, str(exc))
    return check("SAM(rho=0) reproduces SGD exactly (CPU)", True)


def test_holm_and_t() -> bool:
    adj = M.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    ok1 = check("Holm smallest p multiplied by m", abs(adj["a"] - 0.03) < 1e-12, f"{adj}")
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = M.paired_t_interval(d)
    ok2 = check("paired t df = 4", t["df"] == 4 and abs(t["mean"] - 3.0) < 1e-12)
    return ok1 and ok2


def main() -> int:
    results = [test_ece(), test_aurc(), test_ace_nll_brier(), test_temperature_recovery(),
               test_holm_and_t(), test_pairing()]
    print(f"{sum(results)}/{len(results)} groups passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

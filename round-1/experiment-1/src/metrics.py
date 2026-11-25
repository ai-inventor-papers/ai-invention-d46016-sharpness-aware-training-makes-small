"""Calibration and accuracy metrics computed from saved logits (numpy only).

Conventions
- Confidence is the maximum softmax probability (top-label confidence).
- ECE: 15 equal-width bins on confidence, bins are (lo, hi]; first bin includes 0.
- ACE: adaptive calibration error, 15 equal-mass bins (sorted by confidence),
  unweighted mean over bins of |accuracy - mean confidence|.
- NLL: mean negative log-likelihood of the true class.
- Brier: multiclass Brier score, sum over classes, mean over examples.
- AURC: area under the risk-coverage curve; examples ranked by confidence
  (descending), risk at coverage k = mean error among the k most confident.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

N_BINS = 15
TEMP_LO, TEMP_HI = 0.05, 20.0


def softmax64(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll_per_example(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    lse = logsumexp(z, axis=1)
    return lse - z[np.arange(len(z)), labels]


def confidence_and_correct(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0):
    probs = softmax64(logits / temperature)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    return conf, correct, probs


def brier_per_example(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    onehot = np.eye(probs.shape[1])[labels]
    return np.sum((probs - onehot) ** 2, axis=1)


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error with equal-width bins over (0, 1]."""
    n = len(conf)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_acc = np.bincount(idx, weights=correct, minlength=n_bins)
    nz = cnt > 0
    return float(np.abs(sum_acc[nz] - sum_conf[nz]).sum() / n)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Per-bin counts, mean confidence and accuracy (equal-width bins)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_acc = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(cnt > 0, sum_conf / np.maximum(cnt, 1), np.nan)
        acc = np.where(cnt > 0, sum_acc / np.maximum(cnt, 1), np.nan)
    return {
        "edges": edges.tolist(),
        "count": cnt.tolist(),
        "mean_confidence": mean_conf.tolist(),
        "accuracy": acc.tolist(),
    }


def ace(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Adaptive calibration error: equal-mass bins, unweighted mean over bins."""
    order = np.argsort(conf, kind="stable")
    groups = np.array_split(order, n_bins)
    vals = [abs(correct[g].mean() - conf[g].mean()) for g in groups if len(g) > 0]
    return float(np.mean(vals))


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (ranking by confidence, descending)."""
    order = np.argsort(-conf, kind="stable")
    err = (1.0 - correct)[order]
    cum = np.cumsum(err)
    risk = cum / np.arange(1, len(err) + 1)
    return float(risk.mean())


def all_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:
    conf, correct, probs = confidence_and_correct(logits, labels, temperature)
    return {
        "accuracy": float(correct.mean()),
        "ece": ece(conf, correct),
        "ace": ace(conf, correct),
        "nll": float(nll_per_example(logits, labels, temperature).mean()),
        "brier": float(brier_per_example(probs, labels).mean()),
        "aurc": aurc(conf, correct),
    }


def bootstrap_ci(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0,
                 n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile bootstrap CI over test examples, one resample per replicate."""
    conf, correct, probs = confidence_and_correct(logits, labels, temperature)
    nll_pe = nll_per_example(logits, labels, temperature)
    brier_pe = brier_per_example(probs, labels)
    rng = np.random.default_rng(seed)
    n = len(labels)
    samples = {k: np.empty(n_boot) for k in ("accuracy", "ece", "ace", "nll", "brier", "aurc")}
    for b in range(n_boot):
        i = rng.integers(0, n, size=n)
        c, r = conf[i], correct[i]
        samples["accuracy"][b] = r.mean()
        samples["ece"][b] = ece(c, r)
        samples["ace"][b] = ace(c, r)
        samples["nll"][b] = nll_pe[i].mean()
        samples["brier"][b] = brier_pe[i].mean()
        samples["aurc"][b] = aurc(c, r)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in samples.items()}


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Fit one temperature on validation logits by minimising validation NLL.

    Bounded scalar minimisation over [0.05, 20]. A coarse log-grid is also
    evaluated so that a non-global bounded-search optimum is visible.
    """
    def obj(t: float) -> float:
        return float(nll_per_example(logits, labels, t).mean())

    res = minimize_scalar(obj, bounds=(TEMP_LO, TEMP_HI), method="bounded",
                          options={"xatol": 1e-6})
    grid = np.exp(np.linspace(np.log(TEMP_LO), np.log(TEMP_HI), 400))
    grid_vals = np.array([obj(t) for t in grid])
    best_grid = float(grid[np.argmin(grid_vals)])
    t_hat = float(res.x)
    at_bound = bool(t_hat <= TEMP_LO * 1.001 or t_hat >= TEMP_HI * 0.999)
    return {
        "temperature": t_hat,
        "val_nll_at_T": obj(t_hat),
        "val_nll_at_bounds": {str(TEMP_LO): obj(TEMP_LO), str(TEMP_HI): obj(TEMP_HI)},
        "grid_best_T": best_grid,
        "grid_best_val_nll": float(grid_vals.min()),
        "at_bound": at_bound,
    }


# ---------------------------------------------------------------- statistics

def paired_t_interval(d: np.ndarray, alpha: float = 0.05) -> dict:
    """Mean of paired differences with a t-interval (df = n - 1) and two-sided p."""
    from scipy import stats

    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    if se == 0.0:
        p = 1.0 if mean == 0.0 else 0.0
        t_stat = float("inf") if mean != 0.0 else 0.0
    else:
        t_stat = mean / se
        p = float(2 * stats.t.sf(abs(t_stat), n - 1))
    return {
        "n": n,
        "df": n - 1,
        "mean": mean,
        "sd": sd,
        "ci95": [mean - tcrit * se, mean + tcrit * se],
        "t": float(t_stat),
        "p": p,
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values over a family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adj[name] = running
    return adj


# ---------------------------------------------------------------- self-test helpers

def ece_known_example() -> float:
    """Hand-built case with known ECE = 0.1.

    Four examples at confidence 0.9 with 3 correct: bin accuracy 0.75, gap 0.15,
    weight 4/6, contribution 0.1. Two examples at confidence 0.5 with 1 correct:
    bin accuracy 0.5 equals confidence, gap 0. Total ECE = 0.1.
    """
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.5, 0.5])
    correct = np.array([1, 1, 1, 0, 1, 0], dtype=np.float64)
    return ece(conf, correct)

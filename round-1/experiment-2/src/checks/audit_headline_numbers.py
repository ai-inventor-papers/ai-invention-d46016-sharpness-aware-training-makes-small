"""Independent re-derivation of headline numbers from raw logits (not from method_out.json aggregates).
Different code paths: np.histogram binning for ECE, torch.nn.functional.cross_entropy for NLL,
scipy interp1d for matched-loss interpolation, exhaustive sign-flip placebo for the 3-pair comparison."""
import json, itertools, numpy as np, torch
from scipy.interpolate import interp1d
from pathlib import Path
R = str(Path(__file__).resolve().parents[1] / "results")
lab = np.load(f"{R}/labels.npz"); yt, yv = lab["test"], lab["val"]
doc = json.load(open(f"{R}/method_out.json"))
out = {**doc["metadata"], **doc["metadata"]["aggregate_tables"]}
def ece_hist(logits, y, bins=15):
    t = torch.from_numpy(logits).double()
    p = torch.softmax(t, 1).numpy(); conf = p.max(1); corr = (p.argmax(1) == y).astype(float)
    counts, edges = np.histogram(conf, bins=bins, range=(0.0, 1.0))
    ids = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = ids == b
        if m.sum(): e += m.sum() / len(y) * abs(corr[m].mean() - conf[m].mean())
    return e
def nll_torch(logits, y):
    return float(torch.nn.functional.cross_entropy(torch.from_numpy(logits).double(), torch.from_numpy(y).long()))
rows = {}
for opt in ("sgd", "sam"):
    for s in (0, 1, 2):
        te = np.load(f"{R}/logits/{opt}_seed{s}_epoch60_test.npy")
        rows[(opt, s)] = dict(ece=ece_hist(te, yt), nll=nll_torch(te, yt),
                              acc=float((te.argmax(1) == yt).mean()))
        ref = [r for r in out["test_metrics_epochs_5_and_60"] if r["optimizer"].lower() == opt and r["seed"] == s and r["epoch"] == 60][0]
        print(f"{opt} s{s} ep60: ECE indep {rows[(opt,s)]['ece']:.6f} vs reported {ref['test_ece']:.6f} | "
              f"NLL indep {rows[(opt,s)]['nll']:.6f} vs {ref['test_nll']:.6f} | acc {rows[(opt,s)]['acc']:.4f} vs {ref['test_acc']:.4f}")
d_ece = np.array([rows[("sam", s)]["ece"] - rows[("sgd", s)]["ece"] for s in (0, 1, 2)])
d_nll = np.array([rows[("sam", s)]["nll"] - rows[("sgd", s)]["nll"] for s in (0, 1, 2)])
print("indep SAM-SGD test ECE per seed", d_ece.round(5), "mean", d_ece.mean().round(5))
print("indep SAM-SGD test NLL per seed", d_nll.round(5), "mean", d_nll.mean().round(5))
# placebo: sign-flip null over the 8 sign patterns of the 3 per-seed differences
for name, d in (("ECE", d_ece), ("NLL", d_nll)):
    obs = abs(d.mean()); null = [abs((np.array(sg) * d).mean()) for sg in itertools.product((-1, 1), repeat=3)]
    print(f"sign-flip placebo {name}: fraction of 8 sign patterns with |mean| >= observed = {np.mean(np.array(null) >= obs - 1e-15):.3f}")
# shuffled-label placebo: a metric that is not vacuous should move when labels are scrambled
rng = np.random.default_rng(0)
te = np.load(f"{R}/logits/sam_seed0_epoch60_test.npy")
print("ECE real labels", round(ece_hist(te, yt), 4), "| shuffled labels", round(ece_hist(te, rng.permutation(yt)), 4))
# matched-loss eigenvalue difference via a separate interpolation path (fine log grid, interp1d)
per = json.load(open(f"{R}/per_epoch.json"))
for s in (0, 1, 2):
    def pts(opt):
        rs = [r for r in per if r["optimizer"] == opt and r["seed"] == s and r["top_hessian_eig"] is not None]
        return np.log([r["train_loss"] for r in rs]), np.array([r["top_hessian_eig"] for r in rs])
    xs, es = pts("sgd"); xa, ea = pts("sam")
    lo, hi = max(xs.min(), xa.min()), min(xs.max(), xa.max())
    g = np.linspace(lo, hi, 200)
    fs = interp1d(*zip(*sorted(zip(xs, es))), bounds_error=False, fill_value="extrapolate")
    fa = interp1d(*zip(*sorted(zip(xa, ea))), bounds_error=False, fill_value="extrapolate")
    print(f"matched eig seed {s}: mean(SAM-SGD) over log-grid = {np.mean(fa(g) - fs(g)):.2f} "
          f"(reported {out['matched_training_loss'][s]['mean_eig_diff']:.2f})")
# headline TS numbers
ts = {(r["optimizer"], r["seed"]): r for r in out["temperature_scaling"] if r["epoch"] == 60}
print("SGD+TS mean ECE", round(np.mean([ts[("SGD", s)]["test_ece_TS"] for s in (0,1,2)]), 4),
      "SAM+TS mean ECE", round(np.mean([ts[("SAM", s)]["test_ece_TS"] for s in (0,1,2)]), 4))

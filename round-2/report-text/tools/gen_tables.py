"""Render the iteration-2 tables and figure numbers from the artifacts' own output files.

Writes scratch/tables.md (markdown tables) and scratch/numbers.json (figure inputs).
Read-only on the artifact workspaces.
"""
import csv
import json
import math
import os
import statistics as st

import numpy as np
from scipy import stats

ROOT = "/Volumes/aii-s20029/work/run_S_3Y_l0tX7Xe/3_invention_loop"
I1 = f"{ROOT}/iter_1/gen_art"
I2 = f"{ROOT}/iter_2/gen_art"
HERE = os.path.dirname(os.path.abspath(__file__))
TC = stats.t.ppf(0.975, 4)
out = []
nums = {}


def add(s=""):
    out.append(s)


def f(x, d=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{d}f}" if d else f"{x:.0f}"


def pv(p):
    if p is None:
        return "–"
    if p < 0.0001:
        return f"{p:.1e}"
    return f"{p:.4f}"


def ci_str(lo, hi, d=4):
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def paired(a, b):
    diffs = np.array(a) - np.array(b)
    mu = diffs.mean()
    se = diffs.std(ddof=1) / math.sqrt(len(diffs))
    t, p = stats.ttest_rel(a, b)
    return mu, mu - TC * se, mu + TC * se, p, diffs


# ---------- Experiment 3: tuned headline re-run ----------
a3 = json.load(open(f"{I2}/gen_art_experiment_3/results/analysis_tables.json"))

add("### T-A. Tuning on seed 101, 40 epochs, validation NLL (Experiment 3)")
add()
add("| Condition | Learning rate | rho | Validation NLL | Validation accuracy | Final training loss |")
add("|---|---|---|---|---|---|")
with open(f"{I2}/gen_art_experiment_3/results/tuning_table.csv") as fh:
    tune = list(csv.DictReader(fh))
tune.sort(key=lambda r: (r["kind"] != "sgd", float(r["lr"]), float(r["rho"])))
for r in tune:
    rho = "none" if r["kind"] == "sgd" else r["rho"]
    add(f"| {r['kind'].upper()} | {r['lr']} | {rho} | {float(r['val_nll']):.5f} | {float(r['val_acc']):.4f} | {float(r['final_train_loss']):.5f} |")
add()

add("### T-B. Paired contrasts, five seeds, 40 epochs (Experiment 3)")
add()
add("| Contrast | Mean difference | 95% CI | p (two-sided) | Holm p | Holm, 0.05 |")
add("|---|---|---|---|---|---|")
for r in a3["family_of_12_paired"]:
    add(f"| {r['row']} | {r['mean']:+.5f} | {ci_str(r['ci95_low'], r['ci95_high'], 5)} | {pv(r['p_two_sided'])} | {pv(r.get('p_holm', r['p_two_sided']))} | {'yes' if r['significant_holm_0_05'] else 'no'} |")
add()

add("### T-C. Placebo: test labels shuffled (Experiment 3)")
add()
add("| Contrast | Mean difference | 95% CI | p (two-sided) | Holm p |")
add("|---|---|---|---|---|")
for r in a3["placebo"]:
    name = r["row"].replace("PLACEBO SAM - SGD (shuffled test labels): ", "")
    add(f"| {name} | {r['mean']:+.5f} | {ci_str(r['ci95_low'], r['ci95_high'], 5)} | {pv(r['p_two_sided'])} | {pv(r.get('p_holm'))} |")
add()

add("### T-D. Per-condition means over five seeds, test set (Experiment 3)")
add()
add("| Condition | Accuracy | ECE | ACE | NLL | Brier | Mean T | Mean final training loss |")
add("|---|---|---|---|---|---|---|---|")
pm = a3["per_seed_metrics"]
Tm = a3["temperature_T"]
tl = a3["final_train_loss"]
for key, label, tkey in [("sgd_raw", "SGD, raw", "sgd"), ("sgd_ts", "SGD, temperature-scaled", "sgd"),
                         ("sam_raw", "SAM, raw", "sam"), ("sam_ts", "SAM, temperature-scaled", "sam")]:
    rows = pm[key]
    m = {k: st.mean(r[k] for r in rows) for k in ["acc", "ece", "ace", "nll", "brier"]}
    Tcol = f"{st.mean(Tm[tkey]):.3f}" if key.endswith("raw") else "–"
    tcol = f"{st.mean(tl[tkey]):.4f}" if key.endswith("raw") else "–"
    add(f"| {label} | {m['acc']:.4f} | {m['ece']:.4f} | {m['ace']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} | {Tcol} | {tcol} |")
add()
nums["exp3_T"] = {"sgd": Tm["sgd"], "sam": Tm["sam"]}

# ---------- Experiment 4: matched training loss ----------
R4 = f"{I2}/gen_art_experiment_4/results/runs"
def run(pref, s):
    return json.load(open(f"{R4}/{pref}_s{s}.json"))["final"]
conds = {"c1_sam": "SAM 40 epochs", "c2_sgd40": "SGD 40 epochs", "c3_sgdE10": "SGD 10 epochs (matched loss)"}
cdata = {p: [run(p, s) for s in range(5)] for p in conds}

def ccontrast(a, b, split, m):
    x = [cdata[a][s][split][m] for s in range(5)]
    y = [cdata[b][s][split][m] for s in range(5)]
    return paired(x, y)

# Holm p values reported in the artifact's README for the primary family (7 rows)
readme_holm = {"accuracy": 0.055, "ece_raw": 0.0018, "ece_ts": 0.97, "nll_raw": 0.0019, "nll_ts": 0.0094, "brier": 0.0010, "ace_raw": 0.0018}

add("### T-E. Primary comparison: SAM 40 epochs minus SGD 10 epochs at matched training loss (Experiment 4), recomputed from the run records")
add()
add("| Metric | Mean difference | 95% CI | p (two-sided, recomputed) | Holm p (artifact README) |")
add("|---|---|---|---|---|")
prim = [("Accuracy", "test_raw", "accuracy", "accuracy"), ("ECE, raw (15 bins)", "test_raw", "ece", "ece_raw"),
        ("ECE, temperature-scaled", "test_ts", "ece", "ece_ts"), ("NLL, raw", "test_raw", "nll", "nll_raw"),
        ("NLL, temperature-scaled", "test_ts", "nll", "nll_ts"), ("Brier, raw", "test_raw", "brier", "brier"),
        ("ACE, raw", "test_raw", "ace", "ace_raw")]
primnums = {}
for label, split, m, hk in prim:
    mu, lo, hi, p, _ = ccontrast("c1_sam", "c3_sgdE10", split, m)
    primnums[hk] = (mu, lo, hi)
    add(f"| {label} | {mu:+.5f} | {ci_str(lo, hi, 5)} | {pv(p)} | {readme_holm[hk]} |")
add()
nums["exp4_primary"] = primnums

add("### T-F. Reference contrast: SAM 40 epochs minus SGD 40 epochs, final models (Experiment 4), recomputed")
add()
add("| Metric | Mean difference | 95% CI | p (two-sided) |")
add("|---|---|---|---|")
for label, split, m in [("Accuracy", "test_raw", "accuracy"), ("ECE, raw", "test_raw", "ece"), ("ECE, temperature-scaled", "test_ts", "ece"), ("NLL, raw", "test_raw", "nll"), ("NLL, temperature-scaled", "test_ts", "nll")]:
    mu, lo, hi, p, _ = ccontrast("c1_sam", "c2_sgd40", split, m)
    add(f"| {label} | {mu:+.5f} | {ci_str(lo, hi, 5)} | {pv(p)} |")
add()

add("### T-G. Matched training loss: per-seed values and the selection gap (Experiment 4)")
add()
ml = json.load(open(f"{I2}/gen_art_experiment_4/results/matched_loss_table.json"))
add("| Seed | SAM 40-epoch training loss | SGD 10-epoch training loss | Gap, SAM minus SGD 10 | SGD 40-epoch training loss |")
add("|---|---|---|---|---|")
for r in ml["rows"]:
    add(f"| {r['seed']} | {r['sam40_train_loss']:.5f} | {r['sgd_matched_loss_train_loss']:.5f} | {r['gap_matched_loss_b']:+.5f} | {r['sgd40_train_loss']:.5f} |")
add()

add("### T-H. Re-derivation of the Experiment 4 headline numbers from saved logits")
add()
ad = json.load(open(f"{I2}/gen_art_experiment_4/results/audit_rederivation.json"))["headline"]
add("| Quantity | Re-derived mean difference | Stored mean difference | Absolute difference |")
add("|---|---|---|---|")
for k in ["accuracy", "ece_raw", "nll_raw", "nll_ts"]:
    v = ad[k]
    add(f"| {k} | {v['rederived_mean_diff']:+.6f} | {v['stored_mean_diff']:+.6f} | {v['abs_diff_from_stored']:.2e} |")
add()

# ---------- Experiment 1: TS-vs-TS row (reviewer item) ----------
rows1 = list(csv.DictReader(open(f"{I1}/gen_art_experiment_1/results/per_run_metrics.csv")))
def g1(model, s, m):
    return float([r for r in rows1 if r["model"] == model and r["epoch"] == "40" and r["seed"] == str(s) and r["schedule"] == "40-epoch run"][0][m])
add("### T-I. Experiment 1 calibrated comparison, epoch 40, five seeds, recomputed from per-run metrics (TS-SAM minus TS-SGD)")
add()
add("| Metric | Per-seed differences (seeds 0-4) | Mean difference | 95% CI | p (two-sided) |")
add("|---|---|---|---|---|")
ts_nums = {}
for m in ["ece", "nll", "accuracy"]:
    a = [g1("TS-SAM", s, m) for s in range(5)]
    b = [g1("TS-SGD", s, m) for s in range(5)]
    mu, lo, hi, p, d = paired(a, b)
    ts_nums[m] = (mu, lo, hi, p)
    per = ", ".join(f"{x:+.5f}" for x in d)
    add(f"| {m} | {per} | {mu:+.5f} | {ci_str(lo, hi, 5)} | {pv(p)} |")
add()
nums["exp1_ts_ece"] = ts_nums["ece"][:3]

# ---------- Experiment 1 pooled reliability bins ----------
rb = json.load(open(f"{I1}/gen_art_experiment_1/results/reliability_bins_pooled.json"))
add("### T-J. Pooled reliability bins, test set, Experiment 1 artifact (`reliability_bins_pooled.json`)")
add()
add("| Bin | SGD count | SGD confidence | SGD accuracy | TS-SGD count | TS-SGD confidence | TS-SGD accuracy | SAM count | SAM confidence | SAM accuracy |")
add("|---|---|---|---|---|---|---|---|---|---|")
edges = rb["SGD"]["edges"]
def cell(d, i):
    c = d["count"][i]
    mc = d["mean_confidence"][i]
    ac = d["accuracy"][i]
    if c == 0:
        return "0", "–", "–"
    return f"{c}", f"{mc:.4f}", f"{ac:.4f}"
for i in range(len(rb["SGD"]["count"])):
    lo, hi = edges[i], edges[i + 1]
    cells = []
    for k in ["SGD", "TS-SGD", "SAM"]:
        cells += list(cell(rb[k], i))
    add(f"| {lo:.3f}–{hi:.3f} | " + " | ".join(cells) + " |")
add()

# ---------- Experiment 2: rho-ball and Hessian per logged epoch ----------
pe = json.load(open(f"{I1}/gen_art_experiment_2/results/per_epoch.json"))
by = {}
for r in pe:
    by.setdefault((r["optimizer"], r["epoch"]), []).append(r)
add("### T-K. Top Hessian eigenvalue and SAM rho-ball loss increase, per logged epoch, mean over three seeds (Experiment 2, `per_epoch.json`)")
add()
add("| Epoch | SGD top Hessian eigenvalue | SAM top Hessian eigenvalue | Ratio SGD / SAM | SAM rho-ball loss increase |")
add("|---|---|---|---|---|")
ratios = {}
for ep in [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60]:
    sg = st.mean(r["top_hessian_eig"] for r in by[("sgd", ep)])
    sm = st.mean(r["top_hessian_eig"] for r in by[("sam", ep)])
    rho = st.mean(r["rho_ball_increase"] for r in by[("sam", ep)])
    ratios[ep] = sg / sm
    add(f"| {ep} | {sg:.2f} | {sm:.2f} | {sg/sm:.2f} | {rho:.4f} |")
add()
nums["exp2_ratio"] = ratios

# ---------- Experiment 5 (evaluation): full gap table ----------
add("### T-L. Gap table, SAM minus SGD, two trajectories reported separately (Evaluation 1, `gap_table.csv`)")
add()
add("| Source | Comparison | Timepoint | n | Mean gap | 95% t CI | p (unadj.) | Holm p | Per-seed signs |")
add("|---|---|---|---|---|---|---|---|---|")
with open(f"{ROOT}/round-2/evaluation-1/src/results/gap_table.csv") as fh:
    for r in csv.DictReader(fh):
        if r["n"] == "0":
            continue
        ci = ci_str(float(r["ci_lo"]), float(r["ci_hi"]), 5) if r["ci_lo"] else "–"
        p = pv(float(r["p_unadjusted"])) if r["p_unadjusted"] else "–"
        hp = pv(float(r["holm_p"])) if r["holm_p"] else "–"
        mean = f"{float(r['mean']):+.5f}" if r["mean"] else "–"
        add(f"| {r['source']} | {r['comparison']} | {r['timepoint']} | {r['n']} | {mean} | {ci} | {p} | {hp} | {r['per_seed_signs']} |")
add()

with open(os.path.join(HERE, "tables.md"), "w") as fh:
    fh.write("\n".join(out))
with open(os.path.join(HERE, "numbers.json"), "w") as fh:
    json.dump(nums, fh, indent=1, default=float)
print("ok", len(out))

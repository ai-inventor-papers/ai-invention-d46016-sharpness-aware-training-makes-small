#!/usr/bin/env python3
"""Analysis for SAM vs SGD at matched training loss.

Reads results/runs/*.json and results/logits/*.npz written by method.py and produces:
  results/method_out.json        exp_gen_sol_out format (datasets/examples) with full analysis in metadata
  results/matched_loss_table.json per seed and condition: final train loss and gap to SAM epoch 40
  results/timings.json           wall-clock per run and total
  results/figure_data/*.npz      raw top-label confidences and correctness behind each figure
  figures/*.pdf, *.png           rendered with aii-data-fig-gen
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats

WORK = Path(__file__).resolve().parent
RESULTS = WORK / "results"
RUNS = RESULTS / "runs"
LOGITS = RESULTS / "logits"
FIGS = WORK / "figures"
FIG_DATA = RESULTS / "figure_data"
SPECS = WORK / "figures" / "specs"
import os
CHART_GEN = Path(os.environ.get("AII_DATA_FIG_GEN", "chart_gen.py"))  # path to aii-data-fig-gen/scripts/chart_gen.py

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 40
N_BINS = 15
MATCH_TOL = 0.01
FLAG_GAP_A = 0.02
METRICS = {  # name -> (block, key, lower_is_better)
    "accuracy": ("test_raw", "accuracy", False),
    "ece_raw": ("test_raw", "ece", True),
    "ece_ts": ("test_ts", "ece", True),
    "nll_raw": ("test_raw", "nll", True),
    "nll_ts": ("test_ts", "nll", True),
    "brier": ("test_raw", "brier", True),
    "ace": ("test_raw", "ace", True),
}
REPORTED = {"accuracy": 0.0015, "ece_raw": -0.0378, "nll_raw": -0.0980, "nll_ts": -0.0209}
REPORTED_TOL = {"accuracy": 0.005, "ece_raw": 0.005, "nll_raw": 0.01, "nll_ts": 0.01}


def load_run(name: str) -> dict:
    return json.loads((RUNS / f"{name}.json").read_text())


def metric_at(epoch_rec: dict, metric: str) -> float:
    block, key, _ = METRICS[metric]
    return float(epoch_rec[block][key])


def paired_stats(d: np.ndarray) -> dict:
    n = len(d)
    df = n - 1
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df))
    if se > 0:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), df))
    else:
        t, p = float("nan"), float("nan")
    return {"mean": mean, "sd": sd, "t": t, "df": df, "t_crit": tcrit,
            "ci95": [mean - tcrit * se, mean + tcrit * se], "p_two_sided": p,
            "per_seed": [float(x) for x in d]}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i] if not np.isnan(p[i]) else np.nan
        running = max(running, val) if not np.isnan(val) else running
        adj[i] = min(1.0, running) if not np.isnan(p[i]) else np.nan
    return [float(x) for x in adj]


def run_analysis() -> None:
    logger.info("Analysis: loading runs")
    FIG_DATA.mkdir(parents=True, exist_ok=True)
    SPECS.mkdir(parents=True, exist_ok=True)
    chosen = json.loads((RESULTS / "chosen_E.json").read_text())
    tuning = json.loads((RESULTS / "tuning.json").read_text())
    E_star = chosen["E_star"]
    if not chosen["feasible"] or E_star is None:
        raise RuntimeError("Matched-loss test infeasible; analysis of condition 3 cannot run")

    per_seed = []
    matched_rows = []
    for s in SEEDS:
        c1 = load_run(f"c1_sam_s{s}")
        c2 = load_run(f"c2_sgd40_s{s}")
        c3 = load_run(f"c3_sgdE{E_star}_s{s}")
        L1 = c1["history"][EPOCHS - 1]["train_loss_eval"]
        gaps = [abs(h["train_loss_eval"] - L1) for h in c2["history"]]
        e_s = int(np.argmin(gaps)) + 1
        c2_match = c2["history"][e_s - 1]
        gap_a = c2_match["train_loss_eval"] - L1
        L3 = c3["final"]["train_loss_eval"]
        gap_b = L3 - L1
        per_seed.append({"seed": s, "c1": c1["history"][EPOCHS - 1], "c2_final": c2["history"][EPOCHS - 1],
                         "c2_matched": c2_match, "e_s": e_s, "c3": c3["final"]})
        matched_rows.append({
            "seed": s,
            "sam40_train_loss": L1,
            "sgd40_train_loss": c2["history"][EPOCHS - 1]["train_loss_eval"],
            "matched_epoch_e_s": e_s,
            "sgd_matched_epoch_train_loss": c2_match["train_loss_eval"],
            "gap_matched_epoch_a": gap_a,
            "flag_gap_a_over_0.02": bool(abs(gap_a) > FLAG_GAP_A),
            "sgd_matched_loss_E_star": E_star,
            "sgd_matched_loss_train_loss": L3,
            "gap_matched_loss_b": gap_b,
            "flag_gap_b_over_0.01": bool(abs(gap_b) > MATCH_TOL),
        })
        logger.info(f"seed {s}: SAM40 loss={L1:.4f}; matched-epoch e_s={e_s} gap_a={gap_a:+.4f}; "
                    f"E*={E_star} gap_b={gap_b:+.4f}")

    # Paired comparisons: d_s = SAM (condition 1, epoch 40) minus comparator.
    comparisons = {"a_matched_epoch_sgd40": "c2 at epoch e_s", "b_matched_loss_sgdE": f"c3 SGD E={E_star} final"}
    results = {}
    pvals_flat = []
    keys_flat = []
    for cname in ["a_matched_epoch_sgd40", "b_matched_loss_sgdE"]:
        results[cname] = {}
        for m in METRICS:
            sam_vals = np.array([metric_at(p["c1"], m) for p in per_seed])
            if cname.startswith("a"):
                comp_vals = np.array([metric_at(p["c2_matched"], m) for p in per_seed])
            else:
                comp_vals = np.array([metric_at(p["c3"], m) for p in per_seed])
            st = paired_stats(sam_vals - comp_vals)
            st["sam_mean"] = float(sam_vals.mean())
            st["comparator_mean"] = float(comp_vals.mean())
            st["lower_is_better"] = METRICS[m][2]
            results[cname][m] = st
            pvals_flat.append(st["p_two_sided"])
            keys_flat.append((cname, m))
    holm_adj = holm(pvals_flat)
    for (cname, m), adj in zip(keys_flat, holm_adj):
        results[cname][m]["p_holm"] = adj
    logger.info("Holm-adjusted p-values across the 14-test family:")
    for (cname, m) in keys_flat:
        st = results[cname][m]
        logger.info(f"  {cname:24s} {m:9s} d={st['mean']:+.5f} CI=[{st['ci95'][0]:+.5f},{st['ci95'][1]:+.5f}] "
                    f"p={st['p_two_sided']:.4f} holm={st['p_holm']:.4f}")

    # Reproduction check: condition 1 minus condition 2 at epoch 40 against the reported headline.
    repro = {}
    for m in REPORTED:
        vals = np.array([metric_at(p["c1"], m) - metric_at(p["c2_final"], m) for p in per_seed])
        observed = float(vals.mean())
        diff = observed - REPORTED[m]
        repro[m] = {"observed_sam_minus_sgd40_mean": observed, "reported": REPORTED[m],
                    "abs_difference": diff, "tolerance": REPORTED_TOL[m],
                    "within_tolerance": bool(abs(diff) <= REPORTED_TOL[m])}
        logger.info(f"repro {m}: observed={observed:+.5f} reported={REPORTED[m]:+.4f} "
                    f"within={repro[m]['within_tolerance']}")
    repro_note = (f"Tuned rho={tuning['chosen_rho']} on seed {tuning['seed']}. The earlier run's rho is not "
                  "recorded in this run's inputs, so this comparison is not exact.")

    # Matched-loss table
    (RESULTS / "matched_loss_table.json").write_text(json.dumps({
        "target": "SAM 40-epoch eval-mode training loss (condition 1)",
        "E_star": E_star, "tolerance_b": MATCH_TOL, "flag_a_over": FLAG_GAP_A,
        "rows": matched_rows}, indent=1))

    # Timings
    timings = {}
    for f in sorted(RUNS.glob("*.json")):
        d = json.loads(f.read_text())
        timings[d["name"]] = d["wall_time_s"]
    (RESULTS / "timings.json").write_text(json.dumps({
        "per_run_wall_time_s": timings,
        "total_run_wall_time_s": float(sum(timings.values())),
        "hardware": "Apple silicon, MPS device", "note": "wall-clock of training plus per-epoch evaluation",
    }, indent=1))

    # Figures: raw data behind each claim
    def top_label(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = z.astype(np.float64)
        zs = z - z.max(axis=1, keepdims=True)
        p = np.exp(zs) / np.exp(zs).sum(axis=1, keepdims=True)
        return p.max(axis=1), (p.argmax(axis=1) == y).astype(float)

    from torchvision import datasets  # test labels from the official test set
    yte = datasets.FashionMNIST(root=str(WORK / "scratch" / "fmnist"), train=False, download=False).targets.numpy()

    pooled = {"sam_final": ([], []), "sgd40_matched_epoch": ([], []), f"sgd_E{E_star}_final": ([], []),
              "sgd40_final": ([], [])}
    for s in SEEDS:
        n1 = np.load(LOGITS / f"c1_sam_s{s}.npz")
        n2 = np.load(LOGITS / f"c2_sgd40_s{s}.npz")
        n3 = np.load(LOGITS / f"c3_sgdE{E_star}_s{s}.npz")
        e_s = per_seed[SEEDS.index(s)]["e_s"]
        conf, corr = top_label(n1["test"][-1], yte)
        pooled["sam_final"][0].append(conf); pooled["sam_final"][1].append(corr)
        conf, corr = top_label(n2["test"][e_s - 1], yte)
        pooled["sgd40_matched_epoch"][0].append(conf); pooled["sgd40_matched_epoch"][1].append(corr)
        conf, corr = top_label(n3["test"][-1], yte)
        pooled[f"sgd_E{E_star}_final"][0].append(conf); pooled[f"sgd_E{E_star}_final"][1].append(corr)
        conf, corr = top_label(n2["test"][-1], yte)
        pooled["sgd40_final"][0].append(conf); pooled["sgd40_final"][1].append(corr)
    pooled_arr = {}
    for k, (cs, rs) in pooled.items():
        pooled_arr[k + "_conf"] = np.concatenate(cs)
        pooled_arr[k + "_correct"] = np.concatenate(rs)
    np.savez_compressed(FIG_DATA / "top_label_conf_correct.npz", **pooled_arr)

    make_figures(pooled_arr, E_star)

    # method_out.json (exp_gen_sol_out format): one example per official test image and seed.
    # Each predict_* field is the argmax class of the saved float16 test logits for one condition.
    examples = []
    for p in per_seed:
        s_ = p["seed"]
        n1 = np.load(LOGITS / f"c1_sam_s{s_}.npz")
        n2 = np.load(LOGITS / f"c2_sgd40_s{s_}.npz")
        n3 = np.load(LOGITS / f"c3_sgdE{E_star}_s{s_}.npz")
        pred_sam = n1["test"][-1].astype(np.float64).argmax(axis=1)
        pred_sgd40 = n2["test"][-1].astype(np.float64).argmax(axis=1)
        pred_sgd_me = n2["test"][p["e_s"] - 1].astype(np.float64).argmax(axis=1)
        pred_sgd_ml = n3["test"][-1].astype(np.float64).argmax(axis=1)
        for i in range(len(yte)):
            examples.append({
                "input": f"Classify Fashion-MNIST official test image {i} (seed {s_}) into one of 10 classes",
                "output": str(int(yte[i])),
                "metadata_seed": int(s_),
                "metadata_test_index": int(i),
                "predict_sam_40ep": str(int(pred_sam[i])),
                "predict_sgd_40ep": str(int(pred_sgd40[i])),
                "predict_sgd_matched_epoch": str(int(pred_sgd_me[i])),
                "predict_sgd_matched_loss": str(int(pred_sgd_ml[i])),
            })
    logger.info(f"method_out examples: {len(examples)} (test images x seeds)")
    assert len(examples) == 5 * len(yte)

    out = {
        "metadata": {
            "method_name": "SAM vs SGD at matched training loss (Fashion-MNIST small CNN)",
            "description": ("Paired seeds 0-4. Primary test: SAM 40 epochs vs SGD with E* epochs (matched "
                            "eval-mode train loss). Secondary: SAM epoch 40 vs SGD 40-epoch snapshot at the "
                            "matched epoch. Every model temperature-scaled on validation logits."),
            "hyperparameters": {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4, "batch": 128,
                                "schedule": "per-epoch cosine 0.5*lr*(1+cos(pi e/E))", "rho_tuned": tuning["chosen_rho"],
                                "E_star": E_star, "bins": N_BINS},
            "predict_fields": {"predict_sam_40ep": "SAM, rho tuned, epoch 40 (condition 1)",
                               "predict_sgd_40ep": "SGD, epoch 40 (condition 2)",
                               "predict_sgd_matched_epoch": "SGD 40-epoch run at matched epoch e_s (comparison a)",
                               "predict_sgd_matched_loss": "SGD, E* epochs, final (condition 3, primary comparison)"},
            "tuning": tuning,
            "pilot": chosen,
            "comparisons": comparisons,
            "results": results,
            "holm_family": "2 comparisons x 7 metrics = 14 tests",
            "reproduction_check": {"values": repro, "note": repro_note},
            "matched_loss_table": matched_rows,
            "assumptions": [
                "No data augmentation (not specified in the plan).",
                "Convolutions have no bias (followed by BatchNorm).",
                "BatchNorm running statistics update in both SAM passes (spec); eval-mode logits at rho=0 "
                "therefore differ from SGD.",
                "Metrics use float32 logits; saved logits are float16 for audit.",
                "ACE: Nixon et al. multiclass form, 15 equal-mass bins per class, mean over classes and bins.",
                "Brier: multiclass sum over classes, averaged over samples, raw probabilities.",
            ],
        },
        "datasets": [{"dataset": "Fashion-MNIST official 10k test set, 5 paired seeds",
                      "examples": examples}],
    }
    (RESULTS / "method_out.json").write_text(json.dumps(out, indent=1))
    logger.info("Wrote results/method_out.json")


def make_figures(pooled: dict, E_star: int) -> None:
    def calib_spec(title: str, series_keys: list[tuple[str, str]], xlabel: str) -> dict:
        series = []
        for label, key in series_keys:
            series.append({"label": label,
                           "labels": pooled[key + "_correct"].astype(int).tolist(),
                           "probabilities": pooled[key + "_conf"].round(6).tolist()})
        return {"type": "calibration", "title": title, "xlabel": xlabel, "ylabel": "Observed accuracy",
                "aspect": "1:1", "width_in": 3.4, "bins": N_BINS, "strategy": "uniform",
                "series": series}

    specs = {
        "fig_reliability_sam_vs_matched_loss_sgd": calib_spec(
            "Matched training loss (primary)",
            [("SAM, 40 epochs", "sam_final"), (f"SGD, {E_star} epochs", f"sgd_E{E_star}_final")],
            "Top-label confidence"),
        "fig_reliability_sam_vs_matched_epoch_sgd": calib_spec(
            "Matched epoch (secondary)",
            [("SAM, 40 epochs", "sam_final"), ("SGD, matched epoch", "sgd40_matched_epoch")],
            "Top-label confidence"),
    }
    hist_spec = {
        "type": "hist", "title": "Top-label confidence distribution (test set, pooled over seeds)",
        "xlabel": "Top-label confidence", "ylabel": "Density", "aspect": "16:9", "width_in": 6.5,
        "bins": 20, "series": [
            {"label": "SAM, 40 epochs", "values": pooled["sam_final_conf"].round(6).tolist()},
            {"label": f"SGD, {E_star} epochs", "values": pooled[f"sgd_E{E_star}_final_conf"].round(6).tolist()},
            {"label": "SGD, 40 epochs", "values": pooled["sgd40_final_conf"].round(6).tolist()},
        ],
    }
    specs["fig_confidence_histogram"] = hist_spec
    SPECS.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
        spec_path = SPECS / f"{name}.json"
        spec_path.write_text(json.dumps(spec))
        cmd = [sys.executable, str(CHART_GEN), "--spec", str(spec_path), "--out", str(FIGS / name)]
        logger.info(f"render {name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"chart_gen failed for {name}: {proc.stderr[-2000:]}")
            raise RuntimeError(f"figure {name} refused: {proc.stderr[-500:]}")
        logger.info(proc.stdout.strip()[-300:])


if __name__ == "__main__":
    from loguru import logger as _l
    _l.remove()
    _l.add(sys.stdout, level="INFO")
    run_analysis()

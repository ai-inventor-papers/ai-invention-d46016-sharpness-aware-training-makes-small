#!/usr/bin/env python3
"""Analysis of saved logits: per-run metrics, paired statistics, Holm correction,
verdicts, figures, and results/method_out.json (exp_gen_sol_out format).

Inputs: results/runs/* (logits, temperature.json, metrics.json), results/labels.npz,
results/chosen_hparams.json. Test logits are read only here, after all selection.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger

import method as METHOD
import metrics as M
from method import FINAL_SEEDS, RESULTS, RUNS, ROOT, run_name

FIG = ROOT / "figures"
SPECS = RESULTS / "figure_specs"
# Path to the aii-data-fig-gen chart_gen.py script; set AII_CHART_GEN to its location.
CHART_GEN = Path(os.environ.get("AII_CHART_GEN", "chart_gen.py"))
N_BOOT = 1000
SCHED_40 = "40-epoch run"
SCHED_COMP = "compressed"
PRIMARY = ["accuracy", "ece", "ace", "nll", "brier", "aurc"]
HOLM_FAMILY = ["accuracy", "ece", "ace", "nll", "brier", "aurc",
               "sam_vs_tssgd_nll", "sam_vs_tssgd_ece"]


def setup_logging() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(RESULTS / "logs" / "analysis.log", level="DEBUG")


def run_dirs(ch: dict) -> dict:
    lr_sgd = float(ch["sgd"]["lr"])
    lr_sam, rho = float(ch["sam"]["lr"]), float(ch["sam"]["rho"])
    cfg = {"sgd": (lr_sgd, None), "sam": (lr_sam, rho)}
    return cfg


def collect(ch: dict, labels_test: np.ndarray) -> tuple[dict, dict]:
    """Return (records, metrics). records[key] = logits; key = (label, schedule, epoch, seed)."""
    cfg = run_dirs(ch)
    logits: dict[tuple, np.ndarray] = {}
    for seed in FINAL_SEEDS:
        for cond, (lr, rho) in cfg.items():
            label = cond.upper()
            d40 = RUNS / ("final_" + run_name(cond, lr, rho, seed, 40))
            if not (d40 / "logits_ep40.npz").exists():
                raise FileNotFoundError(f"missing final run {d40}")
            for ep in (5, 15, 40):
                z = np.load(d40 / f"logits_ep{ep}.npz")
                logits[(label, SCHED_40, ep, seed)] = z["test_logits"]
            ts = np.load(d40 / "ts_test_logits.npz")["test_logits"]
            logits[("TS-" + label, SCHED_40, 40, seed)] = ts
            for E in (5, 15):
                dc = RUNS / ("compressed_" + run_name(cond, lr, rho, seed, E))
                if (dc / f"logits_ep{E}.npz").exists():
                    logits[(label, SCHED_COMP, E, seed)] = np.load(dc / f"logits_ep{E}.npz")["test_logits"]
    return logits, {}


def compute_metrics(logits: dict, labels: np.ndarray) -> tuple[dict, dict]:
    met, cis = {}, {}
    for key, lg in logits.items():
        met[key] = M.all_metrics(lg, labels)
        cis[key] = M.bootstrap_ci(lg, labels, n_boot=N_BOOT, seed=int(key[3]))
        logger.info(f"metrics {key}: acc={met[key]['accuracy']:.4f} ece={met[key]['ece']:.4f} "
                    f"nll={met[key]['nll']:.4f}")
    return met, cis


def arr(met: dict, label: str, sched: str, ep: int, metric: str) -> np.ndarray:
    return np.array([met[(label, sched, ep, s)][metric] for s in FINAL_SEEDS])


def seed_bootstrap_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict_diff(ci: list[float], lower_is_better: bool) -> str:
    """Verdict for a paired difference d = SAM - comparator, judged by its 95% CI."""
    lo, hi = ci
    if lo > 0:
        direction, favours_sam = "SAM higher", not lower_is_better
    elif hi < 0:
        direction, favours_sam = "SAM lower", lower_is_better
    else:
        return "no detectable difference (CI includes zero)"
    side = "favours SAM" if favours_sam else "favours SGD"
    return f"{direction}: {side} (CI excludes zero)"


def _length_verdict(ci: list[float]) -> str:
    """Advantage = SGD ECE - SAM ECE at E=40 minus at E=5; positive = advantage grows."""
    lo, hi = ci
    if lo > 0:
        return "advantage grows with training length (CI excludes zero)"
    if hi < 0:
        return "advantage shrinks with training length (CI excludes zero)"
    return "no detectable change in advantage with training length (CI includes zero)"


def main() -> None:
    setup_logging()
    ch = METHOD.load_chosen()
    labels_test = np.load(RESULTS / "labels.npz")["test"]
    logits, _ = collect(ch, labels_test)
    met, cis = compute_metrics(logits, labels_test)
    comp_present = all((lab, SCHED_COMP, E, s) in met
                       for lab in ("SGD", "SAM") for E in (5, 15) for s in FINAL_SEEDS)
    logger.info(f"compressed runs present for both conditions: {comp_present}")

    # ---------- per-run table
    rows = []
    for key in sorted(met):
        label, sched, ep, seed = key
        row = {"model": label, "schedule": sched, "epoch": ep, "seed": seed}
        for m in PRIMARY:
            row[m] = round(met[key][m], 6)
            row[m + "_ci_lo"] = round(cis[key][m][0], 6)
            row[m + "_ci_hi"] = round(cis[key][m][1], 6)
        rows.append(row)
    with (RESULTS / "per_run_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- paired primary contrasts (final 40-epoch runs, seeds 0-4)
    tests: dict[str, dict] = {}
    for m in PRIMARY:
        d = arr(met, "SAM", SCHED_40, 40, m) - arr(met, "SGD", SCHED_40, 40, m)
        tests[m] = {"description": f"SAM - SGD, {m}, epoch 40", "d": d.tolist(),
                    **M.paired_t_interval(d), "seed_bootstrap_ci95": seed_bootstrap_ci(d)}
    for m in ("nll", "ece"):
        d = arr(met, "SAM", SCHED_40, 40, m) - arr(met, "TS-SGD", SCHED_40, 40, m)
        key = f"sam_vs_tssgd_{m}"
        tests[key] = {"description": f"SAM - TS-SGD, {m}, epoch 40 (SAM raw vs temperature-scaled SGD)",
                      "d": d.tolist(), **M.paired_t_interval(d),
                      "seed_bootstrap_ci95": seed_bootstrap_ci(d)}
    d = arr(met, "SAM", SCHED_40, 40, "aurc") - arr(met, "TS-SGD", SCHED_40, 40, "aurc")
    tests["aurc_sam_vs_tssgd_extra"] = {
        "description": "SAM - TS-SGD, AURC (ranking; not in Holm family)",
        "d": d.tolist(), **M.paired_t_interval(d), "seed_bootstrap_ci95": seed_bootstrap_ci(d)}

    holm_p = M.holm({k: tests[k]["p"] for k in HOLM_FAMILY})
    for k in HOLM_FAMILY:
        tests[k]["holm_p"] = holm_p[k]
    logger.info("Holm-adjusted p-values: " + ", ".join(f"{k}={holm_p[k]:.4f}" for k in HOLM_FAMILY))

    # ---------- training-length contrast: advantage(SGD - SAM) at E=40 minus E=5 / 15
    gap = {}
    if comp_present:
        for m in ("ece", "nll"):
            adv = {}
            for E, sched in ((5, SCHED_COMP), (15, SCHED_COMP), (40, SCHED_40)):
                adv[E] = arr(met, "SGD", sched, E, m) - arr(met, "SAM", sched, E, m)
            for E in (5, 15):
                dd = adv[40] - adv[E]
                gap[f"{m}_adv40_minus_adv{E}"] = {
                    "description": f"({m}: SGD - SAM) at E=40 minus at E={E}; positive = advantage grows with length",
                    "adv_at_E": {str(e): adv[e].tolist() for e in (5, 15, 40)},
                    "d": dd.tolist(), **M.paired_t_interval(dd),
                    "seed_bootstrap_ci95": seed_bootstrap_ci(dd)}
    else:
        logger.warning("compressed runs missing: training-length contrast untested")

    # ---------- verdicts
    def ci_of(k):
        return tests[k]["ci95"]

    verdicts = {
        "a_accuracy": {
            "criterion": "SAM test accuracy differs from SGD at epoch 40 (paired, seeds 0-4)",
            "verdict": verdict_diff(ci_of("accuracy"), lower_is_better=False),
            "mean_diff": tests["accuracy"]["mean"], "ci95": ci_of("accuracy"),
            "holm_p": tests["accuracy"]["holm_p"]},
        "b_calibration_vs_sgd": {
            "criterion": "SAM ECE is lower than SGD ECE (paired, seeds 0-4)",
            "verdict": verdict_diff(ci_of("ece"), lower_is_better=True),
            "mean_diff": tests["ece"]["mean"], "ci95": ci_of("ece"),
            "holm_p": tests["ece"]["holm_p"]},
        "c_survives_temperature_scaling": {
            "criterion": "SAM NLL is lower than temperature-scaled SGD NLL (paired, seeds 0-4)",
            "verdict": verdict_diff(ci_of("sam_vs_tssgd_nll"), lower_is_better=True),
            "mean_diff": tests["sam_vs_tssgd_nll"]["mean"], "ci95": ci_of("sam_vs_tssgd_nll"),
            "holm_p": tests["sam_vs_tssgd_nll"]["holm_p"]},
        "c2_ece_vs_temperature_scaling": {
            "criterion": "SAM ECE is lower than temperature-scaled SGD ECE (paired, seeds 0-4)",
            "verdict": verdict_diff(ci_of("sam_vs_tssgd_ece"), lower_is_better=True),
            "mean_diff": tests["sam_vs_tssgd_ece"]["mean"], "ci95": ci_of("sam_vs_tssgd_ece"),
            "holm_p": tests["sam_vs_tssgd_ece"]["holm_p"]},
        "d_training_length": {
            "criterion": "SGD-minus-SAM ECE advantage grows from E=5 to E=40",
            "verdict": ("untested (compressed runs not run)" if not gap else
                        _length_verdict(gap["ece_adv40_minus_adv5"]["ci95"])),
            "mean_diff": gap["ece_adv40_minus_adv5"]["mean"] if gap else None,
            "ci95": gap["ece_adv40_minus_adv5"]["ci95"] if gap else None},
    }

    # ---------- reliability bins and figures
    labels_rel = labels_test
    pooled = {}
    for label, key_fn in (("SGD", ("SGD", SCHED_40, 40)), ("TS-SGD", ("TS-SGD", SCHED_40, 40)),
                          ("SAM", ("SAM", SCHED_40, 40))):
        confs, corrs = [], []
        for seed in FINAL_SEEDS:
            key = key_fn + (seed,)
            conf, corr, _ = M.confidence_and_correct(logits[key], labels_rel)
            rb = M.reliability_bins(conf, corr)
            assert sum(rb["count"]) == len(labels_rel), "bin counts must sum to 10000 per model"
            confs.append(conf)
            corrs.append(corr)
        conf_all = np.concatenate(confs)
        corr_all = np.concatenate(corrs)
        pooled[label] = {"conf": conf_all, "corr": corr_all}
        pooled[label]["bins"] = M.reliability_bins(conf_all, corr_all)
        assert sum(pooled[label]["bins"]["count"]) == 5 * len(labels_rel)

    (RESULTS / "reliability_bins_pooled.json").write_text(json.dumps(
        {k: v["bins"] for k, v in pooled.items()}, indent=2))
    render_figures(pooled)

    # ---------- method_out.json
    out = build_method_out(ch, met, cis, tests, holm_p, gap, verdicts, comp_present, logits)
    write_outputs(out)
    logger.info(f"Wrote method_out.json and variants ({len(out['datasets'][0]['examples'])} run examples)")
    for k, v in verdicts.items():
        logger.info(f"VERDICT {k}: {v['verdict']} mean_diff={v['mean_diff']} ci95={v['ci95']}")


def _truncate(v):
    if isinstance(v, str):
        return v if len(v) <= 200 else v[:200] + "..."
    if isinstance(v, list):
        return [_truncate(x) for x in v]
    if isinstance(v, dict):
        return {k: _truncate(x) for k, x in v.items()}
    return v


def write_outputs(out: dict) -> None:
    """method_out.json (full), mini (first 3 examples per dataset), preview (mini, strings cut to 200)."""
    full = ROOT / "method_out.json"
    full.write_text(json.dumps(out, indent=2))
    mini = json.loads(json.dumps(out))
    for ds in mini["datasets"]:
        ds["examples"] = ds["examples"][:3]
    (ROOT / "full_method_out.json").write_text(json.dumps(out, indent=2))
    (ROOT / "mini_method_out.json").write_text(json.dumps(mini, indent=2))
    (ROOT / "preview_method_out.json").write_text(json.dumps(_truncate(mini), indent=2))


def render_figures(pooled: dict) -> None:
    SPECS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    titles = {"SGD": "SGD (raw)", "TS-SGD": "SGD, temperature-scaled", "SAM": "SAM"}
    cal_panels = []
    for label in ("SGD", "TS-SGD", "SAM"):
        cal_panels.append({
            "type": "calibration", "title": titles[label],
            "xlabel": "Confidence", "ylabel": "Accuracy",
            "aspect": "1:1", "bins": 15, "strategy": "uniform",
            "series": [{"label": label,
                        "probabilities": np.round(pooled[label]["conf"], 6).tolist(),
                        "labels": pooled[label]["corr"].astype(int).tolist()}],
        })
    spec_rel = {"type": "panel", "ncols": 1, "aspect": "2:5", "panels": cal_panels}
    hist_panels = []
    for label in ("SGD", "SAM"):
        hist_panels.append({
            "type": "hist", "title": titles[label],
            "xlabel": "Confidence (max probability)", "ylabel": "Test examples",
            "aspect": "4:3", "bins": 15, "xlim": [0.0, 1.0],
            "series": [{"label": f"{label}, 5 seeds pooled",
                        "values": np.round(pooled[label]["conf"], 6).tolist()}],
        })
    spec_hist = {"type": "panel", "ncols": 2, "panels": hist_panels}
    for name, spec in (("fig_reliability_diagrams", spec_rel), ("fig_confidence_histograms", spec_hist)):
        p = SPECS / f"{name}.json"
        p.write_text(json.dumps(spec))
        if not CHART_GEN.exists():
            logger.warning(f"chart renderer not found at {CHART_GEN} (set AII_CHART_GEN); "
                           f"{name} not re-rendered, spec written to {p}")
            continue
        cmd = [sys.executable, str(CHART_GEN), "--spec", str(p), "--out", str(FIG / name)]
        logger.info("render: " + " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(res.stdout.strip()[-500:])
        if res.returncode != 0:
            logger.error(res.stderr[-2000:])
            raise RuntimeError(f"figure render failed: {name}")


CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def build_prediction_examples(logits: dict) -> list[dict]:
    """One example per official test image: true class and each model's predicted class.

    predict_<model>_seed<k> fields hold the epoch-40 predicted class name for seed k.
    """
    labels = np.load(RESULTS / "labels.npz")["test"]
    pred_keys = [("SGD", "sgd"), ("SAM", "sam"), ("TS-SGD", "ts_sgd")]
    preds = {}
    for lab, short in pred_keys:
        for seed in FINAL_SEEDS:
            lg = logits[(lab, SCHED_40, 40, seed)]
            preds[(short, seed)] = lg.argmax(axis=1)
    examples = []
    for i, y in enumerate(labels.tolist()):
        ex = {
            "input": f"Fashion-MNIST official test image index {i} (10k test set)",
            "output": f"true class: {CLASS_NAMES[y]}",
            "metadata_test_index": i,
            "metadata_true_label": int(y),
        }
        for (short, seed), arr in preds.items():
            ex[f"predict_{short}_seed{seed}"] = CLASS_NAMES[int(arr[i])]
        examples.append(ex)
    return examples


def build_method_out(ch, met, cis, tests, holm_p, gap, verdicts, comp_present, logits) -> dict:
    ex40 = []
    for key in sorted(met):
        label, sched, ep, seed = key
        m = met[key]
        ex40.append({
            "input": f"{label} evaluated on the 10k Fashion-MNIST test set; training schedule: {sched}; "
                     f"checkpoint epoch {ep}; seed {seed}",
            "output": (f"accuracy {m['accuracy']:.4f} (95% CI {cis[key]['accuracy'][0]:.4f}-"
                       f"{cis[key]['accuracy'][1]:.4f}); ECE {m['ece']:.4f}; ACE {m['ace']:.4f}; "
                       f"NLL {m['nll']:.4f}; Brier {m['brier']:.4f}; AURC {m['aurc']:.4f}"),
            "metadata_model": label,
            "metadata_schedule": sched,
            "metadata_epoch": ep,
            "metadata_seed": seed,
            "metadata_metrics": {k: round(v, 6) for k, v in m.items()},
            "metadata_ci95": {k: [round(a, 6), round(b, 6)] for k, (a, b) in cis[key].items()},
        })
    contrast_ex = []
    for k, t in tests.items():
        contrast_ex.append({
            "input": t["description"],
            "output": (f"mean paired difference {t['mean']:.5f}, 95% t-interval "
                       f"[{t['ci95'][0]:.5f}, {t['ci95'][1]:.5f}] (df={t['df']}), p={t['p']:.4f}"
                       + (f", Holm-adjusted p={t['holm_p']:.4f}" if "holm_p" in t else "")),
            "metadata_contrast": k,
            "metadata_mean_diff": round(t["mean"], 6),
            "metadata_ci95": [round(t["ci95"][0], 6), round(t["ci95"][1], 6)],
            "metadata_p": round(t["p"], 6),
            "metadata_per_seed_diff": [round(x, 6) for x in t["d"]],
        })
    for k, g in gap.items():
        contrast_ex.append({
            "input": g["description"],
            "output": (f"mean difference {g['mean']:.5f}, 95% t-interval "
                       f"[{g['ci95'][0]:.5f}, {g['ci95'][1]:.5f}] (df={g['df']}), p={g['p']:.4f}"),
            "metadata_contrast": k,
            "metadata_mean_diff": round(g["mean"], 6),
            "metadata_ci95": [round(g["ci95"][0], 6), round(g["ci95"][1], 6)],
            "metadata_p": round(g["p"], 6),
            "metadata_per_seed_diff": [round(x, 6) for x in g["d"]],
        })
    verdict_ex = [{
        "input": v["criterion"],
        "output": f"{v['verdict']}" + (f"; mean difference {v['mean_diff']:.5f}, 95% CI "
                                       f"[{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]" if v["ci95"] else ""),
        "metadata_criterion_id": k,
    } for k, v in verdicts.items()]
    timing = {}
    tpath = RESULTS / "chosen_grid.json"
    if tpath.exists():
        timing = json.loads(tpath.read_text())
    return {
        "metadata": {
            "method_name": "SAM vs SGD calibration on a small CNN (Fashion-MNIST)",
            "description": ("Plain SGD (momentum 0.9) and sharpness-aware minimisation (SAM) on the same "
                            "base optimiser, matched architecture, data split, batch size, schedule and "
                            "seeds. Hyperparameters chosen on validation only. Reports accuracy, ECE (15 "
                            "bins), ACE, NLL, Brier, AURC, temperature-scaled SGD control, paired "
                            "t-intervals over 5 seeds with Holm correction, and a compressed-schedule "
                            "training-length check."),
            "chosen_hparams": ch,
            "split": "60k official training images: fixed 50k train / 10k validation (seed 0); "
                     "10k official test set used only in final evaluation",
            "architecture": "Conv(1,32,3,p1)-BN-ReLU-Pool, Conv(32,64,3,p1)-BN-ReLU-Pool, FC 3136-128-10",
            "training": {"batch_size": 128, "epochs": 40, "compressed_epochs": [5, 15],
                         "scheduler": "CosineAnnealingLR(T_max=E) stepped per epoch",
                         "momentum": 0.9, "weight_decay": 5e-4, "augmentation": "none",
                         "normalisation": "x/255 then train-split mean/std"},
            "timing_gate": timing,
            "verdicts": verdicts,
            "holm_family": HOLM_FAMILY,
            "holm_adjusted_p": holm_p,
            "paired_tests": tests,
            "training_length_contrast": gap,
            "compressed_runs_present": comp_present,
            "deviations": [
                "Post-epoch-40 extension to 60 epochs is not applied; the eval-mode training loss is recorded "
                "in each run's metrics.json (needs_extension_check) and reported as an under-convergence flag.",
                "Holm correction is applied over the plan's family only; the training-length contrast and the "
                "AURC SAM vs TS-SGD contrast are reported uncorrected.",
                "Seed bootstrap is a check only (5 seeds give 3125 unique resamples).",
                "MPS kernels are not bit-deterministic; seeds fix initialisation and batch order, not floats.",
            ],
            "limitations": [
                "One dataset (Fashion-MNIST), one small CNN, no ImageNet-scale or CSAM runs (Tan et al. 2025 scope).",
                "Five paired seeds: low power; a non-significant result is weak evidence of no difference.",
                "Hyperparameters tuned on one seed (100) and the final-epoch validation NLL only.",
            ],
        },
        "datasets": [
            {"dataset": "fashion_mnist_test_predictions",
             "examples": build_prediction_examples(logits)},
            {"dataset": "fashion_mnist_test_per_model_run", "examples": ex40},
            {"dataset": "paired_contrasts_and_verdicts",
             "examples": contrast_ex + verdict_ex},
        ],
    }


if __name__ == "__main__":
    main()

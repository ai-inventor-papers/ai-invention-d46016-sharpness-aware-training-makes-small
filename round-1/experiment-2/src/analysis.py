"""Analysis for the SAM vs SGD calibration-mechanism experiment.

Reads results/per_epoch.json and the saved logits, then writes:
  results/method_out.json   test metrics (epochs 5, 60), temperature-scaling control,
                            matched-training-loss table, descriptive statistics, run status
  results/notes.md          divergence / resume / sign notes and deviations
  figures/*.json            chart specs (aii-data-fig-gen), rendered to figures/*.pdf and *.png
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger

import os
# Location of the aii-data-fig-gen renderer. Set AII_CHART_GEN to its chart_gen.py path (not bundled in this repository).
CHART_GEN = Path(os.environ["AII_CHART_GEN"]) if os.environ.get("AII_CHART_GEN") else None
TEST_EPOCHS = (5, 60)
N_BINS = 15
BOOT_N = 10000
OPT_LABEL = {"sgd": "SGD", "sam": "SAM"}
COLOR_MAP = {"SGD": "#0173B2", "SAM": "#DE8F05", "SGD+TS": "#0173B2", "SAM+TS": "#DE8F05",
             "SGD unscaled": "#0173B2", "SAM unscaled": "#DE8F05"}


def _metrics_module():
    import method  # noqa: E402  (same directory)
    return method


def _load_logits(res: Path, tag: str, epoch: int, split: str) -> np.ndarray:
    return np.load(res / "logits" / f"{tag}_epoch{epoch}_{split}.npy").astype(np.float64)


def _render(spec: dict, name: str, figdir: Path) -> None:
    if CHART_GEN is None:
        raise SystemExit("set AII_CHART_GEN to the path of aii-data-fig-gen/scripts/chart_gen.py to render figures")
    figdir.mkdir(parents=True, exist_ok=True)
    spec_path = figdir / f"{name}.json"
    spec_path.write_text(json.dumps(spec, indent=1))
    cmd = [sys.executable, str(CHART_GEN), "--spec", str(spec_path), "--out", str(figdir / name)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(f"figure {name} refused:\n{proc.stderr}")
        raise SystemExit(f"figure {name} failed")
    logger.info(f"rendered figures/{name}.pdf / .png")


def run_analysis(ws: Path, args) -> None:
    from loguru import logger as lg  # noqa: F401
    from method import calib_metrics, fit_temperature, log_softmax_np  # noqa: E402
    res = ws / "results"
    figdir = ws / "figures"
    logs = ws / "logs"
    logs.mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {message}")
    logger.add(logs / "analysis.log", level="DEBUG")

    seeds = list(args.seeds)
    recs = json.loads((res / "per_epoch.json").read_text())
    cfg = json.loads((res / "run_config.json").read_text())
    labels = np.load(res / "labels.npz")
    y_val, y_test = labels["val"], labels["test"]
    by_key = {}
    for r in recs:
        by_key.setdefault((r["optimizer"], r["seed"]), []).append(r)

    def rec_at(opt, seed, epoch):
        for r in by_key[(opt, seed)]:
            if r["epoch"] == epoch:
                return r
        return None

    # ---- run status
    run_status = {f"{o}_seed{s}": (by_key[(o, s)][-1]["status"], by_key[(o, s)][-1]["epoch"])
                  for o in ("sgd", "sam") for s in seeds if (o, s) in by_key}
    notes = ["# Notes on runs", ""]
    for k, (st, ep) in run_status.items():
        notes.append(f"- {k}: last recorded epoch {ep}, final status `{st}`")
    notes.append("")

    # ---- test metrics and temperature scaling
    test_rows = []
    ts_rows = []
    for opt in ("sgd", "sam"):
        for s in seeds:
            tag = f"{opt}_seed{s}"
            for ep in TEST_EPOCHS:
                try:
                    va = _load_logits(res, tag, ep, "val")
                    te = _load_logits(res, tag, ep, "test")
                except FileNotFoundError:
                    logger.warning(f"missing logits for {tag} epoch {ep}")
                    continue
                m_te = calib_metrics(te, y_test)
                m_va = calib_metrics(va, y_val)
                test_rows.append({"optimizer": OPT_LABEL[opt], "seed": s, "epoch": ep,
                                  "test_acc": m_te["acc"], "test_ece": m_te["ece"], "test_nll": m_te["nll"],
                                  "test_brier": m_te["brier"], "test_ace": m_te["ace"],
                                  "val_ece": m_va["ece"], "val_nll": m_va["nll"]})
                T = fit_temperature(va, y_val)
                ts_te = te / T
                m_ts = calib_metrics(ts_te, y_test)
                argmax_same = bool(np.array_equal(te.argmax(1), ts_te.argmax(1)))
                at_bound = bool(T < 0.05 * 1.001 or T > 20 / 1.001)
                ts_rows.append({"optimizer": OPT_LABEL[opt], "seed": s, "epoch": ep,
                                "T": T, "T_at_bound": at_bound, "argmax_unchanged": argmax_same,
                                "test_acc": m_ts["acc"], "test_ece_TS": m_ts["ece"],
                                "test_nll_TS": m_ts["nll"], "test_brier_TS": m_ts["brier"],
                                "test_ece_unscaled": m_te["ece"], "test_nll_unscaled": m_te["nll"]})
                if not argmax_same:
                    notes.append(f"- WARNING: argmax changed under TS for {tag} epoch {ep}")

    # ---- matched training loss comparison (per seed; eval epochs only)
    matched = []
    for s in seeds:
        sg = [r for r in by_key.get(("sgd", s), []) if r["top_hessian_eig"] is not None and r["status"] == "ok"]
        sa = [r for r in by_key.get(("sam", s), []) if r["top_hessian_eig"] is not None and r["status"] == "ok"]
        if not sg or not sa:
            matched.append({"seed": s, "note": "missing eval records"})
            continue
        lo = max(min(r["train_loss"] for r in sg), min(r["train_loss"] for r in sa))
        hi = min(max(r["train_loss"] for r in sg), max(r["train_loss"] for r in sa))
        if not lo < hi:
            matched.append({"seed": s, "note": "no overlap of training-loss ranges; step skipped"})
            continue
        grid = np.exp(np.linspace(math.log(lo), math.log(hi), 8))

        def interp(rs, key):
            x = np.log([r["train_loss"] for r in rs])
            y = np.array([r[key] for r in rs])
            o = np.argsort(x)
            return np.interp(np.log(grid), x[o], y[o])
        eig_s, eig_a = interp(sg, "top_hessian_eig"), interp(sa, "top_hessian_eig")
        ece_s, ece_a = interp(sg, "val_ece"), interp(sa, "val_ece")
        matched.append({"seed": s, "train_loss_grid": grid.tolist(),
                        "eig_sgd": eig_s.tolist(), "eig_sam": eig_a.tolist(),
                        "eig_sam_minus_sgd": (eig_a - eig_s).tolist(),
                        "val_ece_sgd": ece_s.tolist(), "val_ece_sam": ece_a.tolist(),
                        "val_ece_sam_minus_sgd": (ece_a - ece_s).tolist(),
                        "mean_eig_diff": float(np.mean(eig_a - eig_s)),
                        "mean_val_ece_diff": float(np.mean(ece_a - ece_s))})

    # ---- statistics: SAM minus SGD, per seed, with seed-level bootstrap
    def diffs(metric_fn):
        out = []
        for s in seeds:
            a, b = metric_fn("sam", s), metric_fn("sgd", s)
            out.append(None if a is None or b is None else a - b)
        return out

    def get_test(opt, s, ep, key):
        for r in test_rows:
            if r["optimizer"] == OPT_LABEL[opt] and r["seed"] == s and r["epoch"] == ep:
                return r[key]
        return None

    def get_val_ece(opt, s, ep):
        r = rec_at(opt, s, ep)
        return None if r is None else r["val_ece"]

    stat_defs = {
        "test_ece_ep60": lambda o, s: get_test(o, s, 60, "test_ece"),
        "test_nll_ep60": lambda o, s: get_test(o, s, 60, "test_nll"),
        "test_acc_ep60": lambda o, s: get_test(o, s, 60, "test_acc"),
        "val_ece_ep5": lambda o, s: get_val_ece(o, s, 5),
    }
    rng = np.random.default_rng(0)
    stats = {}
    for name, fn in stat_defs.items():
        d = diffs(fn)
        valid = [x for x in d if x is not None]
        if len(valid) == 0:
            stats[name] = {"note": "no data"}
            continue
        v = np.array(valid)
        boots = rng.choice(v, size=(BOOT_N, len(v)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        stats[name] = {
            "per_seed_SAM_minus_SGD": d, "mean": float(v.mean()),
            "std_across_seeds": float(v.std(ddof=1)) if len(v) > 1 else None,
            "bootstrap95_seed_level": [float(lo), float(hi)],
            "sign_pattern": "".join("+" if x > 0 else ("-" if x < 0 else "0") for x in v),
            "n_seeds": len(v),
        }
    stats["note"] = ("n=3 seed pairs: no test can reach conventional significance on its own. "
                     "No p-values are reported, so no Holm correction is applied; intervals are descriptive.")

    # ---- Hessian sign check
    neg = [(r["optimizer"], r["seed"], r["epoch"], r["top_hessian_eig"]) for r in recs
           if r["top_hessian_eig"] is not None and r["top_hessian_eig"] < 0]
    n_eig = sum(1 for r in recs if r["top_hessian_eig"] is not None)
    notes.append(f"- Top Hessian eigenvalue negative at {len(neg)} of {n_eig} checkpoints"
                 + (": " + ", ".join(f"{o} s{s} ep{e}={v:.3f}" for o, s, e, v in neg[:20]) if neg else "."))
    sgd60 = [r["train_loss"] for r in recs if r["optimizer"] == "sgd" and r["epoch"] == 60]
    if sgd60 and min(sgd60) > 0.05:
        notes.append(f"- SGD training loss at epoch 60 is {sgd60} (above 0.05): the overfitting regime may not be fully reached.")
    notes.append("- Hessian power iteration: top eigenvalue by magnitude, so a negative value would mean the "
                 "largest-magnitude eigenvalue is negative.")
    if args.__dict__.get("note_extra"):
        notes.append(args.note_extra)

    # ---- figures (chart specs)
    per_opt_series = []
    for opt in ("sgd", "sam"):
        for s in seeds:
            per_opt_series.append((opt, s))

    def color_label(opt, s):
        return f"{OPT_LABEL[opt]} seed {s}"

    fig_a = {"type": "line", "title": "Training loss", "xlabel": "Epoch", "ylabel": "Training loss (clean)",
             "aspect": "4:3", "logy": True, "color_map": COLOR_MAP,
             "series": [{"label": color_label(o, s),
                         "x": [r["epoch"] for r in by_key[(o, s)]],
                         "values": [r["train_loss"] for r in by_key[(o, s)]]} for o, s in per_opt_series]}
    eval_series_eig = [{"label": color_label(o, s),
                        "x": [r["epoch"] for r in by_key[(o, s)] if r["top_hessian_eig"] is not None],
                        "values": [r["top_hessian_eig"] for r in by_key[(o, s)] if r["top_hessian_eig"] is not None]}
                       for o, s in per_opt_series]
    fig_c = {"type": "line", "title": "Top Hessian eigenvalue vs epoch", "xlabel": "Epoch",
             "ylabel": "Top Hessian eigenvalue", "aspect": "4:3", "color_map": COLOR_MAP,
             "series": eval_series_eig}
    eval_pts = []
    for o, s in per_opt_series:
        rs = [r for r in by_key[(o, s)] if r["val_ece"] is not None]
        eval_pts.append({"label": color_label(o, s), "x": [r["train_loss"] for r in rs],
                         "values": [r["val_ece"] for r in rs]})
    fig_b = {"type": "scatter", "title": "Validation ECE vs training loss", "xlabel": "Training loss (log scale)",
             "ylabel": "Validation ECE (15 bins)", "aspect": "4:3", "logx": True, "color_map": COLOR_MAP,
             "series": eval_pts}
    rho_series = [{"label": color_label("sam", s), "x": [r["epoch"] for r in by_key[("sam", s)]
                   if r["rho_ball_increase"] is not None],
                   "values": [r["rho_ball_increase"] for r in by_key[("sam", s)] if r["rho_ball_increase"] is not None]}
                  for s in seeds]
    fig_d = {"type": "line", "title": "SAM rho-ball loss increase", "xlabel": "Epoch",
             "ylabel": "L(w + e) - L(w) on fixed batch", "aspect": "4:3", "color_map": COLOR_MAP,
             "series": rho_series}
    _render(fig_a, "fig_a_train_loss", figdir)
    _render(fig_b, "fig_b_val_ece_vs_train_loss", figdir)
    _render(fig_c, "fig_c_top_hessian_eig", figdir)
    _render(fig_d, "fig_d_rho_ball_increase", figdir)

    # reliability diagrams at epoch 60 (test set), pooled across seeds, one per optimiser
    rel_series = []
    for opt in ("sgd", "sam"):
        confs, corrs = [], []
        for s in seeds:
            try:
                te = _load_logits(res, f"{opt}_seed{s}", 60, "test")
            except FileNotFoundError:
                continue
            p = np.exp(log_softmax_np(te))
            confs.append(p.max(1))
            corrs.append((p.argmax(1) == y_test).astype(float))
        if confs:
            rel_series.append({"label": OPT_LABEL[opt], "probabilities": np.concatenate(confs).round(6).tolist(),
                               "labels": np.concatenate(corrs).astype(int).tolist()})
    if rel_series:
        fig_e = {"type": "calibration", "title": "Reliability at epoch 60 (test)", "xlabel": "Confidence",
                 "ylabel": "Accuracy", "aspect": "1:1", "bins": N_BINS, "strategy": "uniform",
                 "color_map": COLOR_MAP, "series": rel_series}
        _render(fig_e, "fig_e_reliability_ep60", figdir)

    ts_bars = []
    cats = []
    vals_u, vals_t = {}, {}
    for r in ts_rows:
        if r["epoch"] != 60:
            continue
        vals_u.setdefault(r["optimizer"], []).append(r["test_ece_unscaled"])
        vals_t.setdefault(r["optimizer"], []).append(r["test_ece_TS"])
    if vals_u:
        cats = [o for o in ("SGD", "SAM") if o in vals_u]
        fig_f = {"type": "bar", "title": "Test ECE: unscaled vs temperature-scaled (epoch 60)",
                 "xlabel": "Optimiser", "ylabel": "Test ECE (15 bins)", "aspect": "4:3",
                 "categories": cats, "color_map": {"unscaled": "#0173B2", "temperature-scaled": "#DE8F05"},
                 "series": [{"label": "unscaled", "values": [float(np.mean(vals_u[c])) for c in cats],
                             "errors": [float(np.std(vals_u[c], ddof=1)) if len(vals_u[c]) > 1 else 0.0 for c in cats]},
                            {"label": "temperature-scaled", "values": [float(np.mean(vals_t[c])) for c in cats],
                             "errors": [float(np.std(vals_t[c], ddof=1)) if len(vals_t[c]) > 1 else 0.0 for c in cats]}]}
        _render(fig_f, "fig_f_ts_vs_unscaled_ece", figdir)

    # ---- output in the exp_gen_sol_out format: datasets -> examples (input, output, metadata_*, predict_*)
    def fmt(x, nd=6):
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    def clean(x):
        """JSON-safe copy: numpy scalars and arrays to Python values, NaN to None."""
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return [clean(v) for v in x.tolist()]
        if isinstance(x, (np.floating, float)):
            return None if not math.isfinite(float(x)) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        return x

    datasets = []
    # 1. every per-epoch record (6 runs x 60 epochs) — the training-dynamics examples
    ex_epoch = []
    for r in recs:
        ev = r["top_hessian_eig"] is not None
        ex_epoch.append({
            "input": f"optimizer={OPT_LABEL[r['optimizer']]}, seed={r['seed']}, epoch={r['epoch']}: per-epoch training record",
            "output": (f"train_loss={fmt(r['train_loss'])}, train_acc={fmt(r['train_acc'])}, status={r['status']}"
                       + (f", val_ece={fmt(r['val_ece'])}, val_nll={fmt(r['val_nll'])}, "
                          f"top_hessian_eig={fmt(r['top_hessian_eig'], 4)}" if ev else ", no evaluation at this epoch")),
            "metadata_optimizer": OPT_LABEL[r["optimizer"]],
            "metadata_seed": r["seed"],
            "metadata_epoch": r["epoch"],
            "metadata_train_loss": clean(r["train_loss"]),
            "metadata_train_acc": clean(r["train_acc"]),
            "metadata_evaluated": ev,
            "metadata_val_ece": clean(r["val_ece"]),
            "metadata_val_nll": clean(r["val_nll"]),
            "metadata_val_brier": clean(r["val_brier"]),
            "metadata_val_ace": clean(r["val_ace"]),
            "metadata_top_hessian_eig": clean(r["top_hessian_eig"]),
            "metadata_rho_ball_increase": clean(r["rho_ball_increase"]),
            "metadata_status": r["status"],
        })
    datasets.append({"dataset": "Fashion-MNIST per-epoch training records", "examples": ex_epoch})

    # 2. test-set metrics at epochs 5 and 60
    ex_test = []
    for t in test_rows:
        ex_test.append({
            "input": f"optimizer={t['optimizer']}, seed={t['seed']}, epoch={t['epoch']}: test metrics from saved test logits",
            "output": (f"test_acc={fmt(t['test_acc'])}, test_ece={fmt(t['test_ece'])}, test_nll={fmt(t['test_nll'])}, "
                       f"test_brier={fmt(t['test_brier'])}, test_ace={fmt(t['test_ace'])}"),
            **{f"metadata_{k}": clean(v) for k, v in t.items() if k not in ("optimizer",)},
            "metadata_optimizer": t["optimizer"],
        })
    datasets.append({"dataset": "Fashion-MNIST test-set metrics (epochs 5 and 60)", "examples": ex_test})

    # 3. temperature-scaling control
    ex_ts = []
    for t in ts_rows:
        ex_ts.append({
            "input": f"optimizer={t['optimizer']}, seed={t['seed']}, epoch={t['epoch']}: temperature scaling fitted on validation",
            "output": (f"T={fmt(t['T'])}, test_ece_unscaled={fmt(t['test_ece_unscaled'])}, "
                       f"test_ece_TS={fmt(t['test_ece_TS'])}, argmax_unchanged={t['argmax_unchanged']}"),
            **{f"metadata_{k}": clean(v) for k, v in t.items() if k not in ("optimizer",)},
            "metadata_optimizer": t["optimizer"],
        })
    datasets.append({"dataset": "Fashion-MNIST temperature scaling control", "examples": ex_ts})

    # 4. matched training loss comparison, one example per seed and grid point
    ex_match = []
    for m in matched:
        if "train_loss_grid" not in m:
            continue
        for j, g in enumerate(m["train_loss_grid"]):
            ex_match.append({
                "input": f"seed={m['seed']}, matched training loss grid point {j + 1} of {len(m['train_loss_grid'])}",
                "output": (f"train_loss={fmt(g, 4)}, top_hessian_eig_sgd={fmt(m['eig_sgd'][j], 3)}, "
                           f"top_hessian_eig_sam={fmt(m['eig_sam'][j], 3)}, val_ece_sgd={fmt(m['val_ece_sgd'][j], 5)}, "
                           f"val_ece_sam={fmt(m['val_ece_sam'][j], 5)}"),
                "metadata_seed": m["seed"],
                "metadata_train_loss": clean(g),
                "metadata_eig_sgd": clean(m["eig_sgd"][j]),
                "metadata_eig_sam": clean(m["eig_sam"][j]),
                "metadata_val_ece_sgd": clean(m["val_ece_sgd"][j]),
                "metadata_val_ece_sam": clean(m["val_ece_sam"][j]),
            })
    if ex_match:
        datasets.append({"dataset": "Fashion-MNIST matched training loss comparison", "examples": ex_match})

    # 5. seed-level statistics, SAM minus SGD
    ex_stat = []
    for name, v in stats.items():
        if not isinstance(v, dict) or "mean" not in v:
            continue
        lo, hi = v["bootstrap95_seed_level"]
        ex_stat.append({
            "input": f"SAM minus SGD, {name}, per seed over 3 seeds",
            "output": (f"mean={fmt(v['mean'])}, std={fmt(v['std_across_seeds'])}, "
                       f"bootstrap95=[{fmt(lo)}, {fmt(hi)}], sign_pattern={v['sign_pattern']}"),
            "metadata_metric": name,
            "metadata_mean": clean(v["mean"]),
            "metadata_std_across_seeds": clean(v["std_across_seeds"]),
            "metadata_bootstrap95_seed_level": clean(v["bootstrap95_seed_level"]),
            "metadata_per_seed_SAM_minus_SGD": clean(v["per_seed_SAM_minus_SGD"]),
            "metadata_sign_pattern": v["sign_pattern"],
        })
    datasets.append({"dataset": "Fashion-MNIST seed-level statistics (SAM minus SGD)", "examples": ex_stat})

    # each example carries the method's prediction under a predict_<method> key (exp_gen_sol_out rule)
    for d in datasets:
        for ex in d["examples"]:
            if "metadata_optimizer" in ex:
                ex[f"predict_{ex['metadata_optimizer'].lower()}"] = ex["output"]
            elif "metadata_metric" in ex:
                ex["predict_sam_minus_sgd"] = ex["output"]
            else:
                ex["predict_sgd_and_sam"] = ex["output"]

    full = {
        "metadata": {
            "method_name": "SGD vs SAM calibration-mechanism experiment",
            "description": ("Does SAM's calibration gain track overfitting? Small CNN on Fashion-MNIST, "
                            "plain SGD vs SAM (rho 0.1), seeds 0-2, 60 epochs. Descriptive at n=3 seed pairs."),
            "run_configuration": clean(cfg),
            "run_status": {k: list(v) for k, v in run_status.items()},
            "hessian_negative_checkpoints": clean(neg),
            "statistics_note": stats.get("note"),
            "matched_training_loss_notes": [m.get("note") for m in matched if "note" in m],
            "figures": ["figures/fig_a_train_loss.pdf", "figures/fig_b_val_ece_vs_train_loss.pdf",
                        "figures/fig_c_top_hessian_eig.pdf", "figures/fig_d_rho_ball_increase.pdf",
                        "figures/fig_e_reliability_ep60.pdf", "figures/fig_f_ts_vs_unscaled_ece.pdf"],
            "notes_file": "results/notes.md",
            "aggregate_tables": {
                "test_metrics_epochs_5_and_60": clean(test_rows),
                "temperature_scaling": clean(ts_rows),
                "matched_training_loss": clean(matched),
                "statistics_SAM_minus_SGD": clean(stats),
                "hessian_negative_checkpoints": clean(neg),
            },
        },
        "datasets": datasets,
    }
    full = clean(full)

    def truncated(n):
        """Same schema; each dataset keeps its first n examples (mini n=5, preview n=2)."""
        return {"metadata": full["metadata"],
                "datasets": [{"dataset": d["dataset"], "examples": d["examples"][:n]} for d in full["datasets"]]}

    (res / "method_out.json").write_text(json.dumps(full, indent=1))
    (res / "full_method_out.json").write_text(json.dumps(full, indent=1))
    (res / "mini_method_out.json").write_text(json.dumps(truncated(5), indent=1))
    (res / "preview_method_out.json").write_text(json.dumps(truncated(2), indent=1))
    (res / "notes.md").write_text("\n".join(notes) + "\n")
    n_examples = sum(len(d["examples"]) for d in datasets)
    logger.info(f"analysis written: {n_examples} examples in {len(datasets)} datasets "
                f"(full, mini, preview) to results/")

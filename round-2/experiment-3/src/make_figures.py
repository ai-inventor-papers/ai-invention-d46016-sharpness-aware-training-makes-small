#!/usr/bin/env python3
"""Render the reliability figure (aii-data-fig-gen spec) and a post-hoc placebo on the
temperature-scaled arms. The post-hoc placebo is NOT part of the pre-registered 12-row
family; it is reported separately and labelled as post hoc."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS))
import method as M  # noqa: E402

FIG_DIR = WS / "figures"
FIG_DIR.mkdir(exist_ok=True)
# Path to the aii-data-fig-gen skill's chart_gen.py, supplied through an environment
# variable so no machine-specific path is written into the repository.
CHART_GEN = Path(os.environ["AII_DATA_FIG_GEN"]) if os.environ.get("AII_DATA_FIG_GEN") else None


def main() -> None:
    bins = json.loads((WS / "results" / "reliability_bins.json").read_text())["arms"]

    def series_for(arm: str, label: str) -> dict:
        xs, ys = [], []
        for b in bins[arm]:
            if b["count"] > 0:
                xs.append(round(b["mean_confidence"], 4))
                ys.append(round(b["accuracy"], 4))
        return {"label": label, "x": xs, "values": ys}

    spec = {
        "type": "line",
        "title": "Reliability diagram, pooled test set (5 seeds)",
        "show_title": False,
        "xlabel": "Mean confidence in bin (max softmax, temperature-scaled)",
        "ylabel": "Accuracy in bin",
        "aspect": "4:3",
        "series": [
            {"label": "Ideal", "x": [0.0, 1.0], "values": [0.0, 1.0], "linestyle": "--", "marker": "none"},
            series_for("sgd_ts", "SGD + temperature scaling"),
            series_for("sam_ts", "SAM + temperature scaling"),
        ],
        "xlim": [0.0, 1.0],
        "ylim": [0.0, 1.0],
    }
    spec_path = FIG_DIR / "reliability_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    if CHART_GEN is None:
        raise SystemExit("Set AII_DATA_FIG_GEN to the aii-data-fig-gen chart_gen.py path to render the figure.")
    cmd = [sys.executable, str(CHART_GEN), "--spec", str(spec_path), "--out", str(FIG_DIR / "reliability")]
    subprocess.run(cmd, check=True)

    # Post-hoc placebo on the temperature-scaled arms (not in the 12-row family).
    tables = json.loads((WS / "results" / "analysis_tables.json").read_text())
    from torchvision import datasets
    te = datasets.FashionMNIST(str(M.DATA_DIR), train=False, download=False)
    yte = te.targets.numpy()
    shuf = np.random.default_rng(12345).permutation(yte)
    sel = json.loads((WS / "results" / "selection.json").read_text())
    per = {"sgd": [], "sam": []}
    for seed in M.FINAL_SEEDS:
        for kind, lr, rho in (("sgd", sel["sgd"]["lr"], 0.0), ("sam", sel["sam"]["lr"], sel["sam"]["rho"])):
            z = np.load(WS / "results" / "logits" / f"{M.run_name(kind, lr, rho, seed)}_ep40.npz")
            T = tables["temperature_T"][kind][M.FINAL_SEEDS.index(seed)]
            per[kind].append(M.metrics(z["test"] / T, shuf))
    rows = {}
    for key in ("acc", "nll", "ece", "brier", "ace"):
        d = np.array([per["sam"][i][key] - per["sgd"][i][key] for i in range(5)])
        r = M.paired_row(f"POSTHOC PLACEBO TS SAM - TS-SGD (shuffled labels): {key}", d)
        rows[key] = r
    rows_list = list(rows.values())
    M.holm(rows_list)
    out = {"note": "Post hoc. Not part of the pre-registered 12-row family; reported separately. "
                   "Both temperature-scaled arms evaluated against a fixed shuffle of the test labels "
                   "(numpy default_rng(12345)).",
           "rows": rows_list}
    (WS / "results" / "posthoc_placebo_ts.json").write_text(json.dumps(out, indent=2))
    for r in rows_list:
        print(f"{r['row']}: mean={r['mean']:+.5f} p={r['p_two_sided']:.4f} holm5={r['p_holm']:.4f}")


if __name__ == "__main__":
    main()

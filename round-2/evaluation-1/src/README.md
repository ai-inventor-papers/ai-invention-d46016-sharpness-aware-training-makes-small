# Does the SAM calibration gap grow? (evaluation of saved outputs)

Evaluation only. No training and no new logits. The script reads two existing SAM-vs-SGD trajectories on Fashion-MNIST and tests whether the SAM minus SGD gap in test and validation ECE and NLL is near zero early and widens with training. The two trajectories are reported separately and never pooled.

Headline reading (details in `results/caveats.md`):
- Accuracy: no detectable difference at 40 epochs (source 1) or 60 epochs (source 2).
- Source 2, 3 seeds, 60 epochs: validation ECE and NLL gaps are near zero at epochs 1 to 5, then SAM is better. The largest mean advantage is at epochs 25 to 30, and the gap narrows afterwards. The NLL widening from epoch 1 to 60 is -0.10 on the mean, but its t-test p (0.006) is parametric; the exact minimum p with 3 seeds is 0.25, so the widening is descriptive only (see `results/caveats.md`).
- Source 1, 5 seeds: the test ECE gap moves from +0.006 at 5 epochs to -0.038 at 40 epochs. The 5 and 15 epoch points come from separate compressed schedules, so this mixes length with schedule shape.
- Temperature scaling of SGD removes most of the ECE advantage. The NLL advantage survives.

## Layout

- `eval.py`: the whole analysis. Reads the inputs, computes paired gaps, t intervals, Holm correction, exact seed-pair bootstrap, widening tests and gap-versus-training-loss slopes, writes the outputs, validates the eval JSON against the `exp_eval_sol_out` schema, and renders the figure.
- `pyproject.toml`: dependencies (numpy, scipy, matplotlib, loguru, jsonschema).
- `inputs/dep1_sam_fashion_5seed/`: copies of the declared `full_method_out.json`, `mini_` and `preview_` files from artifact `art_SL9gPfE09HlN` (gen_art_experiment_1).
- `inputs/dep2_sam_trajectory_3seed/`: copies of the declared files from artifact `art_WOFinDmS2PoX` (gen_art_experiment_2). Per-epoch values are read from its `full_method_out.json`, not from its `results/per_epoch.json`, which is not in the declared file list.
- `results/gap_table.csv` and `results/gap_table.md`: the one table, 99 rows, each with mean gap, t interval, bootstrap interval, unadjusted and Holm p-values, per-seed signs and a note. Missing source 1 validation metrics and training loss are listed as MISSING.
- `full_eval_out.json`: the full evaluation output, with per-seed gaps in `datasets` and headline numbers in `metrics_agg`. Validated against `exp_eval_sol_out`.
- `mini_eval_out.json`, `preview_eval_out.json`: first three examples per dataset (preview also truncates strings to 200 characters).
- `results/figure_spec.json`: the chart spec passed to the house-style renderer.
- `results/caveats.md`: scope limits and the written reading.
- `figures/fig_gap_vs_epoch_and_training_loss.pdf` and `.png`: six panels. (a) and (b) are source 2 validation gaps by epoch; (c) and (d) are source 2 ECE gap against SGD and SAM training loss, per seed; (e) and (f) are source 1 test gaps at 5, 15 and 40 epochs.
- `logs/run.log`: run log.
- `.venv/`: local Python environment. Not published; it is restored with the command below.

## How to run

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python numpy scipy matplotlib loguru jsonschema
.venv/bin/python eval.py
```

The figure is rendered by the `aii-data-fig-gen` skill's `chart_gen.py`. Set `AII_DATA_FIG_GEN` to its path before running (for example `export AII_DATA_FIG_GEN=/path/to/aii-data-fig-gen/scripts/chart_gen.py`). The JSON schema is vendored at `schemas/exp_eval_sol_out.json`.

## Restoring removed files

- `.venv/` (regenerable): `uv venv .venv --python=3.12 && uv pip install --python .venv/bin/python numpy scipy matplotlib loguru jsonschema`

Nothing else was removed. The inputs are small enough to keep in the repository.

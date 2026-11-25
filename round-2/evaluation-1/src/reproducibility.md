# Reproducing the SAM minus SGD gap evaluation

This artifact is evaluation only. It trains nothing and recomputes no logits. It reads the declared output files of two earlier artifacts and reports the calibration gap between SAM and SGD over training.

## 1. Getting the artifact

The workspace is published as one folder of a public GitHub repository. Clone the repository and `cd` into this artifact's folder (`gen_art_evaluation_1`). Every path below is relative to that folder.

Inputs, all inside the repository:

- `inputs/dep1_sam_fashion_5seed/` holds copies of `full_method_out.json`, `mini_method_out.json` and `preview_method_out.json` from artifact `art_SL9gPfE09HlN` (gen_art_experiment_1). They are copied, not linked. The sibling folder `gen_art_experiment_1` in the repository is the original.
- `inputs/dep2_sam_trajectory_3seed/` holds the same kind of copies from artifact `art_WOFinDmS2PoX` (gen_art_experiment_2). The per-epoch values are read from its `full_method_out.json`. Its `results/per_epoch.json` is not used.
- `schemas/exp_eval_sol_out.json` is a vendored copy of the output schema from the aii-json skill.
- No user-uploaded files are used. The uploads folder was not present in the run.

## 2. System packages, Python and environment

- Python 3.12 (the environment was built with CPython 3.12.x). The `uv` tool is required.
- No system packages beyond Python are needed.

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python \
  numpy==2.5.5 scipy==1.18.1 matplotlib==3.11.2 loguru==0.8.0 jsonschema==4.26.0
```

The exact versions installed in the run's `.venv` are the pins above, and they match `pyproject.toml`. Transitive packages used in the run were attrs 26.1.0, contourpy 1.4.0, cycler 0.12.1, fonttools 4.68.0, jsonschema-specifications 2025.9.1, kiwisolver 1.5.1, packaging 26.4, pillow 13.0.0, pyparsing 3.3.3, python-dateutil 4.0.0, referencing 0.37.0, rpds-py 2026.9.1, typing-extensions 4.16.0 and tzdata 2026.6. `uv.lock` records the resolved set.

## 3. Downloads, environment variables and keys

- No dataset or model is downloaded. No API keys are used.
- One environment variable is needed for the figure step: `AII_DATA_FIG_GEN`, the path to the `chart_gen.py` script of the `aii-data-fig-gen` skill. Set it to that script's path on your machine. The figure is the only output that needs it; the tables and JSON files do not.

## 4. Commands, in the order run

```bash
export AII_DATA_FIG_GEN=/path/to/aii-data-fig-gen/scripts/chart_gen.py
.venv/bin/python eval.py                              # tables, JSON outputs, figure; runs in about 2 seconds
.venv/bin/python checks/rederive_headlines.py          # independent re-derivation and placebo checks
```

- `eval.py` writes `results/gap_table.csv`, `results/gap_table.md`, `full_eval_out.json`, `mini_eval_out.json`, `preview_eval_out.json`, `results/figure_spec.json`, `figures/fig_gap_vs_epoch_and_training_loss.pdf` and `.png`, and `logs/run.log`.
- Mini and preview variants are produced by `eval.py` itself, not by the aii-json formatter. That formatter accepts only a top-level JSON array, and this output is an object with `datasets`. Mini keeps the first three examples per dataset. Preview does the same and truncates strings to 200 characters.
- `checks/rederive_headlines.py` reads the raw records directly, with its own code path. It writes `checks/rederive_headlines_out.txt`.
- Hardware: CPU only. The run took about 2 seconds on an Apple silicon laptop. No GPU is used.
- Seeds: none. The analysis is deterministic. The bootstrap enumerates all n^n resamples instead of sampling.

## 5. Outputs and where they appear

- `full_eval_out.json`: the complete evaluation. `metrics_agg` holds the headline numbers. `datasets` holds per-seed gaps for source 1 (5 seeds), per-epoch gaps for source 2 (3 seeds, 14 logged epochs), and test endpoints for source 2.
- `results/gap_table.md` and `.csv`: the single table of 99 rows. Each row gives the mean gap, the 95% t interval, the seed-pair bootstrap interval, the unadjusted and Holm p-values, per-seed signs and a note. Source 1 validation metrics and training loss are listed as MISSING because they are not in the source file.
- `figures/fig_gap_vs_epoch_and_training_loss.pdf`: the gap-versus-epoch and gap-versus-training-loss figure, six panels.
- `results/caveats.md`: the scope limits and the written reading. It includes a correction on p-values, which are parametric and cannot reach significance at n = 3 (see section 6).
- `checks/rederive_headlines_out.txt`: the re-derived headline values and the placebo results.

## 6. Known limits a reader must check

- The p-values in the table are parametric t-test values. For source 2 (3 seeds) the smallest exact sign-flip p-value is 0.25. For source 1 (5 seeds) it is 0.0625. Read these as descriptive, not as evidence.
- The placebo test does not fail cleanly. For source 2 the t-test rejects 2 of 8 label assignments at 0.05. Those two are the all-same-sign assignments, which is the design floor. The check is reported as it is.
- Source 1 gaps at 5 and 15 epochs come from separate compressed schedules, so the change with length mixes length with schedule shape.
- The source 2 gap is largest near epoch 25 to 30 and narrows afterwards. The epoch-60 endpoint is not the largest advantage.
- The figure needs the `aii-data-fig-gen` skill; it cannot be rebuilt from this folder alone.

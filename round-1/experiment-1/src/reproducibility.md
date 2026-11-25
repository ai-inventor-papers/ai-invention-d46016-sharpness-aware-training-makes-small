# Reproducing the SAM vs SGD calibration results

This describes what was actually run to produce the published results. The original runs used macOS on Apple silicon (M3 Max, Metal Performance Shaders through `torch.device("mps")`), not Ubuntu. The code selects `mps` when available and otherwise `cpu`. A CUDA device is never used. On Ubuntu the code runs on CPU, so runtime will be longer and floating-point results will differ slightly. Seeds fix initialisation and batch order, not floating-point values, so expect small differences from the published numbers.

## 1. Getting the artifact

The workspace is published as one folder of a public GitHub repository. Clone the repository, then `cd` into this folder (`gen_art_experiment_1`). All paths below are relative to that folder.

## 2. System packages and Python

- Python 3.12 (the environment was built with CPython 3.12.14).
- `uv` for environment management (the environment was created with `uv 0.6.14`).
- No system packages beyond the Python environment are needed. On Ubuntu, install `python3.12` and `uv` (for example from `https://astral.sh/uv`).

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python -r pyproject.toml
```

`pyproject.toml` pins every package to the version installed in the run's `.venv`. These are the exact versions used:

- torch `2.16.0.dev20261116+aii1` and torchvision `0.30.0.dev20261116`. These are development builds from the run's package mirror. They are not standard public PyPI releases, so on Ubuntu install a torch 2.x release, then match the other pins. Use `torch` 2.x on CPU (`--index-url https://download.pytorch.org/whl/cpu` on Ubuntu without a GPU).
- numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, loguru 0.8.0, psutil 7.2.2, jsonschema 4.26.0 (validation only), and the transitive pins in `pyproject.toml`.

## 3. Data, models, and keys

- **Data**: Fashion-MNIST, downloaded automatically by torchvision into `scratch/data/` (about 30 MB). Source: `torchvision.datasets.FashionMNIST`. Mirror: HuggingFace `zalando-datasets/fashion_mnist`. Not used in the run: the mirror was not needed.
- **No model downloads**: the network is trained from scratch.
- **API keys**: none are required for the training or analysis.
- **Optional environment variable**: `AII_CHART_GEN` gives the path to the `chart_gen.py` script from the `aii-data-fig-gen` skill, used to render the figures. If it is unset, the default `chart_gen.py` must be on the path. Figures can be rebuilt from `results/figure_specs/*.json` with any renderer.
- **Input the user uploaded**: none. The run used no private inputs.

## 4. Exact commands, in order

Hardware used for the original runs: Apple M3 Max (virtual), MPS device, unified memory, 29 GB limit. Total runtime was about 40 minutes for the full pipeline (timing gate projected 0.6 GPU-hours, which included the warm-up and sweep).

```bash
# 1. Unit checks and pairing check (SAM with rho = 0 reproduces SGD exactly on CPU)
.venv/bin/python tests/test_checks.py

# 2. Full pipeline, one command: timing gate -> tuning (seed 100) -> final (seeds 0-4)
#    -> compressed (5 and 15 epochs) -> analysis (method_out.json and its variants).
#    Finished runs are skipped, so re-running only redoes the analysis.
uv run method.py            # equivalent: .venv/bin/python method.py

# 3. Independent re-derivation of headline numbers from raw logits
.venv/bin/python audit_rederive.py
```

`run_all.sh` runs the same stages with separate calls. Figures are rendered with the `aii-data-fig-gen` renderer when `AII_CHART_GEN` points to its `chart_gen.py`. Without it, the figure specs are written to `results/figure_specs/` and the existing figures are kept.

Individual stages, if needed:

```bash
.venv/bin/python method.py smoke       # 2-epoch smoke runs in scratch/ (pipeline check)
.venv/bin/python method.py timing      # timing gate, writes results/timings.csv and chosen_grid.json
.venv/bin/python method.py tune        # seed 100, validation only, writes tuning_table.csv and chosen_hparams.json
.venv/bin/python method.py final       # seeds 0-4, 40 epochs, logits at epochs 5/15/40, checkpoints
.venv/bin/python method.py compressed  # seeds 0-4, 5 and 15 epochs
.venv/bin/python analyze.py            # metrics, paired statistics, Holm, verdicts, figures, method_out.json
```

Fixed settings in `method.py`: batch size 128, SGD momentum 0.9, weight decay 5e-4, cosine schedule with `T_max` equal to the number of epochs, 40 epochs for the main runs, 5 and 15 epochs for compressed runs, learning rate grid {0.01, 0.03, 0.1}, rho grid {0.05, 0.1, 0.2}, tuning seed 100, final seeds 0-4. The split is seed 0 (`np.random.default_rng(0)`), giving 50k train, 10k validation, and the official 10k test set.

Selected hyperparameters (from `results/chosen_hparams.json`): SGD lr = 0.01; SAM lr = 0.01, rho = 0.1. Both selections are at the lower edge of the learning-rate grid.

## 5. Outputs and where they appear

- `method_out.json` (workspace root): per-run metrics with bootstrap CIs, paired differences with t-intervals, Holm-adjusted p-values, and verdicts. Schema-validated against `exp_gen_sol_out`. `full_method_out.json` (identical copy: 10,000 per-test-image prediction rows, 60 per-run rows, 18 contrast and verdict rows), `mini_method_out.json` (first 3 examples per dataset), and `preview_method_out.json` (mini with strings cut to 200 characters) are written at the root by the same run.
- `results/per_run_metrics.csv`: one row per model, schedule, epoch, and seed.
- `results/tuning_table.csv`: validation NLL for all 12 tuning configurations. One configuration (SAM lr 0.03, rho 0.2) collapsed to chance (training loss 2.302, validation accuracy 0.13). It was not selected.
- `results/audit_rederivation.json`: independent recomputation of the headline numbers and placebo checks.
- `figures/fig_reliability_diagrams` and `figures/fig_confidence_histograms` (PDF and PNG; rendered from `results/figure_specs/`).
- `results/checkpoints/`: ten final-run state dicts. These stay on the run's storage volume and are not published (see `.aii/manifest.yaml`).

Headline numbers (paired over 5 seeds, df = 4, epoch 40, test set):

| Contrast | Mean difference | 95% CI | Holm p |
|---|---|---|---|
| Accuracy, SAM − SGD | +0.0015 | [−0.0007, +0.0036] | 0.13 |
| ECE, SAM − SGD | −0.0378 | [−0.0414, −0.0341] | 0.0001 |
| NLL, SAM − SGD | −0.0980 | [−0.1092, −0.0867] | 0.0001 |
| NLL, SAM − temperature-scaled SGD | −0.0209 | [−0.0266, −0.0151] | 0.0027 |
| ECE, SAM − temperature-scaled SGD | −0.0031 | [−0.0059, −0.0002] | 0.079 |

Read the table with the caveats in `README.md`: SAM is under-fit at epoch 40 (training loss 0.107 against 0.008 for SGD), and the selected learning rates sit on the grid edge.

## 6. Checking the reproduction

A correct reproduction should give:
- Test accuracy near 92% for SGD and SAM at epoch 40.
- SGD fitted temperatures between 1.8 and 1.9 (overconfident), and SAM temperatures near 0.95.
- Paired differences with signs and rough magnitudes as in the table above. Exact values will vary in the third or fourth decimal place on a different device.
- `tests/test_checks.py` passes all six groups, and the pairing check gives a maximum logit difference of zero on CPU.

# Reproducing the SAM vs SGD calibration results

**Important: where this was run.** The results were produced on **macOS 26.4 on an Apple M3 Max (virtualised)**, using the Apple GPU through PyTorch's `mps` backend. They were **not** produced on Ubuntu. The Ubuntu steps below are the closest equivalent; they run on CPU (or CUDA if you adapt the device line), and the numbers are expected to agree only approximately, not bit-for-bit. The exact-logit determinism check (see step 6) was done on MPS.

## 1. Get the artifact

The workspace is published as one folder of a public GitHub repository. Clone the repository and enter the folder for this artifact:

```
git clone <repository-url>
cd <repository>/<this-artifact-folder>
```

Every path in this document is relative to that folder.

## 2. System packages and Python

- Python **3.12.x** (we used 3.12.14). Install it with your system package manager or `uv python install 3.12`.
- `uv` (we used 0.6.14).
- No system libraries beyond the Python wheels are needed.

## 3. Virtual environment and exact library versions

The exact pinned set is in `pyproject.toml` (30 packages). Create the environment as follows:

```
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python -r pyproject.toml
```

Caveat: `torch` and `torchvision` are pinned to **nightly development builds** (`torch==2.16.0.dev20261116+aii1`, `torchvision==0.30.0.dev20261116`). The `+aii1` build is a guest MPS build that is not on the public index. On Ubuntu, install the nearest nightly build from the PyTorch index, for example:

```
uv pip install --python .venv/bin/python --pre torch==2.16.0.dev20261116 torchvision==0.30.0.dev20261116 --index-url https://download.pytorch.org/whl/nightly/cpu
```

If that exact version is not available, the closest nightly will run the code, but the logits will differ slightly from the published ones.

No API keys, tokens, or environment variables are needed for the experiment itself. The figure step needs one environment variable (step 5).

## 4. Data

The data is FashionMNIST from torchvision. The first run of `method.py --stage tune` downloads it into `data_cache/` (about 83 MB). The download is not published (it is listed in the manifest as redownloadable). If you need it earlier:

```
.venv/bin/python -c "from torchvision import datasets; datasets.FashionMNIST('data_cache', train=True, download=True); datasets.FashionMNIST('data_cache', train=False, download=True)"
```

The validation split is fixed by `numpy.random.default_rng(0)`; the indices are in `splits/split_seed0.json`. No user-uploaded data was used.

## 5. Commands we ran, in order

Hardware: Apple M3 Max (virtualised), MPS device, 29 GB memory cap in the guest. The GPU was used one process at a time.

| Step | Command | Runtime (observed) |
|---|---|---|
| a | `.venv/bin/python tests_unit.py` | under 1 minute (CPU) |
| b | `.venv/bin/python method.py --stage smoke --out-root scratch/smoke --epochs 2` (scratch output removed afterwards) | about 10 s |
| c | `.venv/bin/python method.py --stage tune` | about 25 min (17 configurations); 40-epoch SGD about 50 to 76 s, SAM about 88 to 121 s |
| d | `.venv/bin/python method.py --stage final` | about 14 min (5 SGD, 5 SAM, 1 determinism rerun) |
| e | `.venv/bin/python method.py --stage analyze` | a few seconds |
| f | `AII_DATA_FIG_GEN=<path to aii-data-fig-gen>/scripts/chart_gen.py .venv/bin/python make_figures.py` | a few seconds |
| g | `.venv/bin/python audit_rederive.py` | a few seconds |
| h | `.venv/bin/python make_mini_preview.py` | under 1 second |

Stage `tune` ran the full grid (the time guard did not trigger the reduced grid). Its selection was SGD lr 0.001 (validation NLL 0.214; the extension 0.0003 was run and was worse) and SAM lr 0.01, rho 0.1 (validation NLL 0.182). Seeds: tuning seed 101; final seeds 0 to 4; the placebo labels use `default_rng(12345)`.

Stage `final` was run once and then, after a fix to a path-building bug in the determinism comparison, re-run with `--stage final`; finished runs are loaded from `results/runs/`, so only the comparison was recomputed.

## 6. Expected outputs and where they appear

| File | What it contains |
|---|---|
| `results/method_out.json`, `full_method_out.json` | headline tables and one example per test image (10000) with the true class and ten `predict_*` fields, in the exp_gen_sol_out schema (validated); `mini_method_out.json` and `preview_method_out.json` hold the first 3 examples |
| `results/analysis_tables.json` | per-seed test metrics, temperatures, train losses, the 12-row paired family, placebo rows |
| `results/tuning_table.csv`, `results/selection.json` | the validation sweep and the frozen selection |
| `results/reliability_bins.json` | pooled 15-bin reliability data per arm |
| `results/posthoc_placebo_ts.json` | post-hoc placebo on temperature-scaled arms (not in the 12-row family) |
| `results/audit_rederivation.json` | independent re-derivation of headline numbers |
| `results/runs/*.json` | one record per training run |
| `results/logits/*.npz` | float32 validation and test logits at epochs 5, 15, 40 per run |
| `logs/determinism.json` | max logit difference of the SGD seed-0 rerun (0.0 on MPS) |
| `logs/time_guard.json`, `logs/timings.jsonl` | time guard decision and per-run timings |
| `figures/reliability.pdf` | reliability diagram |

Headline numbers and where they appear in the paper draft: SAM − TS-SGD NLL −0.0326 (95% CI −0.0396 to −0.0256, Holm-adjusted p = 0.0018), accuracy +0.0096, SAM − SGD ECE −0.0050 (raw) but +0.0020 after temperature scaling (not significant). Temperatures: SGD 1.03 to 1.06, SAM 0.92 to 0.95. The paper should report the placebo failure for NLL and Brier alongside these.

## 7. Notes for a reader

- The NLL and Brier shuffled-label placebos are significant. This means the placebo does not isolate label information for confidence-sensitive metrics. Do not present it as a clean null.
- SAM's final training loss is lower than SGD's in this round (0.110 vs 0.122). The fit confound is not controlled; see the README.
- Figures use `aii-data-fig-gen`; the environment variable above points to its `chart_gen.py`.

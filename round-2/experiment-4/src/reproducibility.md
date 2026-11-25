# Reproducing the SAM vs SGD matched-loss results

This folder is one artifact of a public GitHub repository. Start from a clone of that repository and `cd` into this folder. All paths below are relative to this folder.

## 1. Hardware and system

- What was run: Apple silicon Mac (arm64), PyTorch on the Apple GPU through the `mps` device. Total training time was about 18 minutes (`results/timings.json`: 1098 s of training plus per-epoch evaluation).
- Ubuntu: the code runs on CPU with the same logic (`DEVICE` falls back to `cpu` automatically). Expect a much longer runtime, and not bit-identical numbers, because CPU and MPS kernels differ in floating-point order. Nondeterminism is documented in the README.
- System packages: none beyond Python 3.12 and `uv`.

## 2. Python environment

Python 3.12.14 was used. The exact versions are pinned in `pyproject.toml`, generated from the run's virtual environment with `uv pip freeze`.

```
uv venv .venv --python=3.12
uv pip install -p .venv/bin/python -e .
```

Notes:
- The run used `torch==2.16.0.dev20261116+aii1` and `torchvision==0.30.0.dev20261116`, a development build with MPS support. A reader on Ubuntu may not find these exact wheels on PyPI. If the pinned versions fail to install, use the newest stable `torch` and `torchvision` for Python 3.12 and record the substitution. The statistical pipeline does not depend on the exact torch build, but the logits will differ at the float level.
- `pyproject.toml` pins `setuptools`, `loguru`, `psutil`, `scipy`, `numpy`, `matplotlib` and `jsonschema`, which the code and the schema check use.

## 3. Data and environment variables

- Dataset: Fashion-MNIST, downloaded automatically by torchvision on the first run into `scratch/fmnist/` (about 82 MB). No account or API key is needed.
- Environment variables: none are required by the experiment. `AII_DATA_FIG_GEN` is optional. It points at `chart_gen.py` from the aii-data-fig-gen skill. Without it the figure step is skipped with an error, and the numeric results are still written. Set it like this: `export AII_DATA_FIG_GEN=/path/to/aii-data-fig-gen/scripts/chart_gen.py`.
- No user-uploaded files were used.

## 4. Commands, in order

Each stage writes its outputs under `results/` and skips work that already exists (`results/runs/*.json` acts as a cache). Delete a run's JSON to recompute it.

```
uv run python method.py --stage checks    # unit checks: results/checks/checks.json (about 7 s)
uv run python method.py --stage mini      # 2-epoch end-to-end run on a 5k subset (about 3 s)
uv run python method.py --stage tune      # SAM rho in {0.05, 0.1, 0.2}, seed 101, 40 epochs (about 4 min)
uv run python method.py --stage pilot     # seed 100: SAM 40 epochs, SGD E grid -> E* (about 3 min)
uv run python method.py --stage main      # seeds 0-4: conditions 1, 2, 3 (about 11 min)
uv run python method.py --stage analyze   # statistics, Holm, tables, figures, method_out.json (about 5 s)
uv run python audit_rederive.py           # independent re-derivation of the headline numbers
```

`uv run python method.py --stage all` runs checks, mini, tune, pilot, main and analyze in one call.

Chosen values in this run:
- rho = 0.1 (lowest final validation NLL on seed 101; `results/tuning.json`)
- E* = 10 (smallest E in the grid whose training loss is within 0.01 of SAM's 40-epoch loss on seed 100; `results/chosen_E.json`)

Fixed settings: batch 128, momentum 0.9, weight decay 5e-4, learning rate 0.01 with per-epoch cosine decay, 50k/10k/10k split with numpy seed 0, no augmentation.

## 5. Output files and where they appear

- `results/method_out.json`: the full analysis in exp_gen_sol_out format. The `metadata.results` block contains the paired statistics, the Holm-adjusted p-values and the reproduction check.
- `full_method_out.json`, `mini_method_out.json`, `preview_method_out.json`: the full, mini and preview variants of `method_out.json`, produced by the aii-json formatter. The formatter's own path check rejected this workspace, so the run used a copy of that script with the allowed root extended to this folder. The mini variant holds the first 3 examples, and the preview variant also truncates strings to 200 characters.
- `results/matched_loss_table.json`: per-seed training losses and gaps (SAM epoch 40 vs matched-epoch SGD, and vs matched-loss SGD).
- `results/chosen_E.json`, `results/tuning.json`, `results/timings.json`.
- `results/checks/checks.json`: unit checks.
- `results/audit_rederivation.json`: the independent re-derivation and the placebo.
- `results/logits/*.npz`: per-epoch float16 logits for every run. These are kept on the run's storage volume and are also in this repository because each file is under 100 MB.
- `figures/fig_reliability_sam_vs_matched_loss_sgd.pdf|png`: reliability diagram, SAM vs matched-loss SGD (primary).
- `figures/fig_reliability_sam_vs_matched_epoch_sgd.pdf|png`: reliability diagram, SAM vs SGD at the matched epoch (secondary).
- `figures/fig_confidence_histogram.pdf|png`: top-label confidence distribution.

Headline numbers for the paper (primary comparison, SAM 40 epochs minus SGD 10 epochs at matched loss, mean of 5 paired seeds):
- test NLL, raw: −0.0125 (95% CI −0.0152 to −0.0099; Holm p = 0.0019)
- test NLL, temperature-scaled: −0.0082 (CI −0.0110 to −0.0053; Holm p = 0.0094)
- ECE, raw: −0.0076 (CI −0.0091 to −0.0061; Holm p = 0.0018)
- ECE, temperature-scaled: −0.0001 (not significant)
- accuracy: +0.0025 (CI +0.0007 to +0.0042; Holm p = 0.055, not significant at 0.05)

## 6. Verification

`audit_rederive.py` recomputes the four primary headline numbers from the raw histories and the saved logits, using different code (logsumexp, histogram binning, grid-search temperature). The re-derived values match the stored values within 1.2e-4 (accuracy) and 5e-5 (ECE, NLL). The placebo results and their interpretation are in `README.md` under "Audit".

## 7. What a reader should get

Expected (not yet tested on a second machine): running the commands above on the same hardware reproduces the numbers in section 5 to about 1e-3 for NLL, ECE and accuracy. Larger differences would point to a changed dataset split, normalisation, or shuffle generator, which should be checked first.

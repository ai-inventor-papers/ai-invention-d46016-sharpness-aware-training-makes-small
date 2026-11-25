# Reproducing the SGD vs SAM calibration-mechanism results

This folder is one artifact of a repository. Start from a clone of the repository and `cd` into this folder
(`sam-calibration-mechanism/`, or the artifact folder name in the repository). Every path below is relative to this folder.

## 1. What was actually run, and where

All results in `results/` were produced on an Apple silicon Mac (macOS, arm64) using PyTorch's `mps` device,
with Python 3.12.14. The Apple GPU was the only accelerator; the run did not use CUDA. The pinned PyTorch build
(`2.16.0.dev20261116+aii1`) is an Apple-silicon development build and is not on the public PyPI index.
Section 5 explains how to reproduce on Ubuntu.

## 2. System requirements and environment

- macOS on Apple silicon (for the exact-device run) or Ubuntu (CPU run, see Section 5).
- Python 3.12 (the run used 3.12.14). Install `uv` (https://docs.astral.sh/uv/).
- No system packages beyond Python are needed. torchvision downloads Fashion-MNIST automatically on first use.
- No API keys, no environment variables are required for the training and checks.
- One optional environment variable for figures only: `AII_CHART_GEN` (see Section 4).

## 3. Environment setup (exact versions)

```bash
uv venv .venv --python 3.12
```

Install the exact pins from `pyproject.toml` (all 25 packages, including `setuptools==84.2.1`):

```bash
uv pip install --python .venv/bin/python \
  "contourpy==1.4.0" "cycler==0.12.1" "filelock==4.3.1" "fonttools==4.68.0" "fsspec==2026.9.0" \
  "jinja2==3.1.6" "kiwisolver==1.5.1" "loguru==0.8.0" "markupsafe==3.0.4" "matplotlib==3.11.2" \
  "mpmath==1.3.0" "networkx==3.7" "numpy==2.5.5" "packaging==26.4" "pillow==13.0.0" \
  "pyparsing==3.3.3" "python-dateutil==4.0.0" "scipy==1.18.1" "setuptools==84.2.1" "sympy==1.14.0" \
  "typing-extensions==4.16.0" "tzdata==2026.6" "torch==2.16.0.dev20261116+aii1" "torchvision==0.30.0.dev20261116"
```

`torch` and `torchvision` are the Apple-silicon development builds used in the run. On a machine where these
wheels are unavailable, install the nearest available PyTorch and torchvision (see Section 5).

## 4. Exact commands, in the order they were run

All commands are run from the artifact folder. Plain `uv run method.py` cannot resolve the Apple-only torch pin from PyPI,
so the commands use `uv run --no-sync --python .venv/bin/python method.py ...` against the venv built in Section 3.

```bash
# 4.1 Sanity checks: ECE, adaptive calibration error, temperature scaling, power iteration, SAM(rho=0) == SGD
.venv/bin/python method.py unit
#   Result: all checks pass. Runtime about 3 s on Apple GPU.

# 4.2 Smoke run (2 epochs, Hessian on 200 images, 5 power iterations). Output to scratch/ (not kept).
.venv/bin/python method.py train --out scratch/smoke --epochs 2 --eval-epochs 1 2 --logit-epochs 1 2 \
  --hess-iters 5 --hess-n 200 --seeds 0
#   Runtime about 17 s.

# 4.3 Determinism check: repeat the 2-epoch SGD run and compare epoch training losses.
.venv/bin/python method.py train --out scratch/determinism --epochs 2 --eval-epochs 1 2 --logit-epochs 1 2 \
  --hess-iters 5 --hess-n 200 --seeds 0
#   Epoch training losses matched exactly (max difference 0.0) on Apple GPU.

# 4.4 Full pre-specified run: 6 runs (seeds 0,1,2 x {sgd, sam}), 60 epochs, Hessian on 2000 images, 20 iterations.
#   Defaults: lr 0.03, momentum 0.9, weight decay 5e-4, rho 0.1, batch 128, cosine schedule,
#   evaluation at epochs 1,2,5,10,15,...,60, logits and checkpoints at epochs 5 and 60.
.venv/bin/python method.py train --out . --seeds 0 1 2
#   Runtime on Apple GPU: SGD about 1.8-1.9 min per run, SAM about 7.5-11.6 min per run, about 36 min total.

# 4.5 Analysis: test metrics at epochs 5 and 60, temperature scaling, matched-loss table, statistics, figures.
export AII_CHART_GEN=/path/to/aii-data-fig-gen/scripts/chart_gen.py   # see Section 4 note below
export AII_CHART_GEN=/path/to/chart_gen.py
.venv/bin/python method.py analyze --seeds 0 1 2
#   Runtime about 10 s. Writes results/method_out.json, results/notes.md, figures/*.{json,pdf,png}.

# 4.6 (the analyze step above also writes the full, mini and preview variants; there is no separate step.)

# 4.7 Independent re-derivation of headline numbers from the raw logits, plus placebo checks.
.venv/bin/python checks/audit_headline_numbers.py
```

Note on 4.5: `analysis.py` renders figures by calling the `aii-data-fig-gen` renderer (`chart_gen.py`), which is
not part of this repository. Set `AII_CHART_GEN` to its path. Without it, `analyze` stops before writing
`results/method_out.json` and `results/notes.md`, so the variable must be set. The figure specs are kept in
`figures/*.json`, so the figures can be re-rendered with any renderer that reads the same spec.

Note on the JSON variants: the `aii-json` formatter in the pipeline refuses files outside its own project root,
so the variants were generated with `checks/make_variants.py`, which applies the same full/mini/preview logic
full = method_out plus all 360 per-epoch records; mini keeps the first 5 items of every list, preview the first 2, with a `_truncated` marker.

## 5. Reproducing on Ubuntu (CPU)

The code selects `mps` if available and otherwise `cpu`, so on Ubuntu it runs on CPU without changes.
A CPU run is expected to be several times slower than the Apple GPU run (the plan estimated 3 to 5 times). This was not measured on Ubuntu.
The eigenvalue and calibration numbers can differ slightly from the Apple GPU run, because floating-point
results depend on the backend. The per-epoch training losses of the smoke run are the first check to compare.

Use a CPU PyTorch build for your platform, for example:

```bash
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining pins from Section 3 (without the two Apple-specific torch lines).

## 6. Expected outputs

| File | What a reader should get | Where it appears |
|---|---|---|
| `results/per_epoch.json` | 6 runs x 60 epochs of training loss and accuracy; validation ECE, NLL, Brier, ACE, Hessian eigenvalue, rho-ball increase at evaluation epochs | Sections on training dynamics and sharpness |
| `results/method_out.json` (+ `full_`, `mini_`, `preview_` variants) | All 412 examples in exp_gen_sol_out format; aggregate tables (test metrics, temperature scaling, matched loss, statistics) are in `metadata.aggregate_tables` | Main results tables |
| `results/notes.md` | No divergence; all 6 runs complete; no negative Hessian eigenvalue in 84 checkpoints | Run-status paragraph |
| `figures/fig_a..fig_f` | Training loss, validation ECE vs training loss (mechanism), Hessian eigenvalue, rho-ball increase, reliability at epoch 60, temperature-scaled vs unscaled ECE | Figures |
| `results/logits/*.npy` | Validation and test logits at epochs 5 and 60 (float32, shape 10000 x 10) | Input to all test metrics |
| `checks/audit_headline_numbers.py` output | Recomputed test ECE and NLL agree with the reported numbers to 1e-6 | Verification paragraph |

Headline values (seeds 0-2, epoch 60): test ECE SGD 0.045, SAM 0.028; SAM minus SGD test ECE -0.0171, test NLL -0.076;
temperature-scaled test ECE SGD 0.0094, SAM 0.0070.

## 7. Inputs that are not published

- Fashion-MNIST is downloaded by torchvision to `scratch/data/` on first use. It is not published, and `scratch/` is
  deleted by the manifest after the round, so the download happens again on a rerun.
- No user-uploaded files were used.

## 8. Files kept on the run volume (not in the published repository)

`checkpoints/` (model weights for the six runs and resumption states, about 44 MB) and `results/logits/` are kept
on the run's storage volume, and the repository publishes their paths only. A reader who needs them reruns
Section 4.4 (about 36 min on Apple GPU).

# Does SAM's calibration gain track overfitting?

Mechanism sub-experiment. A small CNN is trained on Fashion-MNIST with plain SGD and with
Sharpness-Aware Minimization (SAM), 3 seeds each, under one pre-specified configuration
(lr 0.03, momentum 0.9, weight decay 5e-4, rho 0.1, batch 128, 60 epochs, cosine schedule,
no augmentation). Both arms share the initialisation and the data order for each seed.

Every number here is descriptive: with 3 seed pairs no test can reach conventional
significance on its own, so no p-values are reported and the bootstrap intervals are unstable.

## Layout

- `method.py` — data split, model, SGD and SAM steps, per-epoch training and evaluation
  (validation ECE, NLL, Brier, adaptive calibration error, top Hessian eigenvalue by power
  iteration, rho-ball loss increase), unit checks (`unit`), training (`train`), analysis (`analyze`).
- `analysis.py` — test metrics at epochs 5 and 60, temperature-scaling control, matched-training-loss
  comparison, seed-level statistics with 10,000-resample bootstrap, and figure specs.
- `figures/` — chart specs (`*.json`) and rendered figures (`*.pdf`, `*.png`):
  - `fig_a_train_loss` — training loss vs epoch, seeds as lines
  - `fig_b_val_ece_vs_train_loss` — validation ECE vs training loss (log x), the mechanism plot
  - `fig_c_top_hessian_eig` — top Hessian eigenvalue vs epoch
  - `fig_d_rho_ball_increase` — SAM rho-ball loss increase vs epoch
  - `fig_e_reliability_ep60` — reliability diagram, test set, epoch 60, pooled over seeds
  - `fig_f_ts_vs_unscaled_ece` — test ECE, unscaled vs temperature-scaled, epoch 60
- `results/per_epoch.json` — one record per epoch per run (train loss/accuracy each epoch; validation
  metrics, Hessian eigenvalue and rho-ball increase at evaluation epochs).
- `results/method_out.json` — run configuration, test metric table (epochs 5 and 60), temperature
  scaling table, matched-loss table, statistics, Hessian sign check.
- `results/notes.md` — run status and warnings.
- `results/runs/` — per-run JSON; `results/split_indices.json` — train/validation/sharpness-batch indices.
- `results/logits/` — validation and test logits at epochs 5 and 60 (float32), kept on the run volume.
- `checkpoints/` — state dicts at epochs 5 and 60 and resume states, kept on the run volume.
- `logs/` — training and analysis logs.

## How to run

See `reproducibility.md` for exact versions, every command run, runtimes and expected outputs.
Headline numbers were re-derived from raw logits by `checks/audit_headline_numbers.py`.
Figures need the `aii-data-fig-gen` renderer: set `AII_CHART_GEN` to its `chart_gen.py` path.


```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch torchvision numpy scipy matplotlib loguru
.venv/bin/python method.py unit                     # sanity checks
.venv/bin/python method.py train --seeds 0 1 2      # six pre-specified runs (about 36 min on Apple GPU)
.venv/bin/python method.py analyze --seeds 0 1 2    # metrics, statistics, figures
```

## Results (seeds 0–2, epoch 60 unless stated)

- Training loss reaches 0.0045 (SGD) vs 0.030 (SAM): SGD reaches the overfitting regime; SAM stays near 0.03.
- Test ECE: SGD 0.045 mean, SAM 0.028 mean (SAM minus SGD −0.017; per-seed signs all negative).
- Test NLL: SAM lower by 0.076 on average (all three seeds). Test accuracy difference is +0.001 (sign pattern +−+).
- Epoch 5 validation ECE: no consistent direction (+−−), so the calibration gap is not near zero
  at epoch 5 for every seed, and it does not establish a growth with training length on its own.
- Temperature scaling (fitted on validation only, argmax unchanged): SGD+TS test ECE mean 0.0094,
  SAM+TS mean 0.0070. Most of the epoch-60 gap disappears after scaling, so much of the SAM
  advantage is a global confidence shift (fitted T 1.86–1.90 for SGD vs 1.36–1.40 for SAM), not ranking quality.
- Sharpness: top Hessian eigenvalue is about 18–22 for SGD and 4–8 for SAM from epoch 5 onward.
  At matched training loss (interpolated over 0.03–0.46), SAM's eigenvalue is about 12–16 lower and its
  validation ECE about 0.010–0.014 lower on average. No negative eigenvalue was found in 84 checkpoints.

This is a descriptive test of the mechanism at one untuned setting (lr 0.03, rho 0.1); it does not
establish that overfitting causes the calibration gain.

## Deviations and implementation choices

- Three seeds (0–2) are the pre-specified primary set. The optional seeds 3 and 4 extension was not run.
- Max pooling is computed with `max_pool2d(return_indices=True)` because the default form does not
  support double backward for the Hessian-vector products. Forward values are identical.
- The Hessian and rho-ball evaluations use the model in eval mode (BatchNorm running statistics), so
  the eigenvalue estimate is comparable across checkpoints.
- The SAM second forward pass also updates BatchNorm running statistics.
- The power-iteration unit check uses a well-separated spectrum (top eigenvalue 10, next 5). With the
  plain 1..10 spectrum (gap ratio 0.9) 20 iterations give 9.94, which is logged but not a pass criterion.
- Temperature-scaling bound check: no fitted T hit the search bounds [0.05, 20].
- Figures are rendered from JSON specs with `aii-data-fig-gen`'s `chart_gen.py`.

## Restoring removed files

- `scratch/` (FashionMNIST download and smoke outputs): run
  `uv run --no-sync --python .venv/bin/python method.py unit`, which downloads FashionMNIST into `scratch/data`, and
  `uv run --no-sync --python .venv/bin/python method.py train --out scratch/smoke --epochs 2 --eval-epochs 1 2 --logit-epochs 1 2 --hess-iters 5 --hess-n 200 --seeds 0`.
- `.venv/`: `uv venv .venv --python 3.12`, then install the pins listed in `pyproject.toml`, for example
  `uv pip install --python .venv/bin/python torch==2.16.0.dev20261116+aii1 torchvision==0.30.0.dev20261116 numpy==2.5.5 scipy==1.18.1 matplotlib==3.11.2 loguru==0.8.0`
  (the Apple-silicon torch build is not on PyPI; see `reproducibility.md`, Section 5, for a CPU alternative).
- `__pycache__/`: created automatically by `uv run --no-sync --python .venv/bin/python method.py unit`.

# SAM vs SGD calibration on a small CNN (Fashion-MNIST)

Question: does sharpness-aware minimisation (SAM) improve test accuracy and calibration over plain SGD for a small convolutional network on Fashion-MNIST?

## Result (5 paired seeds, epoch 40, 10k official test set)

| Contrast (paired, seeds 0-4) | Mean difference | 95% t-interval (df=4) | Holm-adjusted p |
|---|---|---|---|
| Accuracy, SAM − SGD | +0.0015 | [−0.0007, +0.0036] | 0.13 (no detectable difference) |
| ECE (15 bins), SAM − SGD | −0.0378 | [−0.0414, −0.0341] | 0.0001 |
| ACE, SAM − SGD | −0.0386 | [−0.0428, −0.0344] | 0.0001 |
| NLL, SAM − SGD | −0.0980 | [−0.1092, −0.0867] | 0.0001 |
| Brier, SAM − SGD | −0.0136 | [−0.0179, −0.0093] | 0.0037 |
| AURC, SAM − SGD | −0.0015 | [−0.0021, −0.0009] | 0.0064 |
| NLL, SAM − temperature-scaled SGD | −0.0209 | [−0.0266, −0.0151] | 0.0027 |
| ECE, SAM − temperature-scaled SGD | −0.0031 | [−0.0059, −0.0002] | 0.079 (not significant after Holm) |
| Training-length: ECE advantage at E=40 minus at E=5 | +0.0435 | [+0.0376, +0.0494] | not in Holm family |

Reading: SAM does not change test accuracy detectably at this scale. Its ECE is clearly lower than raw SGD. Temperature scaling of SGD removes most of that ECE gap, and SAM keeps a lower NLL than temperature-scaled SGD. The ECE gap against temperature-scaled SGD does not survive Holm correction.

## Caveats that matter for interpretation

- **SAM is under-fit at epoch 40.** Eval-mode training loss is 0.107 for SAM against 0.008 for SGD. The planned 60-epoch extension triggers at 0.01 and was not run (see Deviations). Part of SAM's calibration advantage may reflect a less-fit model rather than the sharpness mechanism.
- **Learning-rate grid boundary.** Both conditions selected lr = 0.01, the smallest value in the grid. A smaller learning rate could change the selected configuration.
- **One tuning config collapsed.** SAM at lr 0.03, rho 0.2 reached a validation NLL of 2.30 (chance level, training loss 2.302) without producing NaN. Fallback 4 (halve the LR for NaN divergence) was not applied to it. It was not selected, so it does not affect the reported comparison.
- **Temperatures are well above 1.** Fitted SGD temperatures are 1.82–1.88 (overconfident), and SAM's are 0.94–0.95. No fitted T is at a bound (0.05 or 20).

## Layout

- `method.py` — data split and normalisation, SmallCNN, SGD and SAM steps, timing gate, tuning, final and compressed runs (modes: `smoke`, `timing`, `tune`, `final`, `compressed`, `pairing`).
- `metrics.py` — ECE, ACE, NLL, Brier, AURC, bootstrap CIs, temperature fitting, paired t-intervals, Holm correction.
- `analyze.py` — per-run metrics, paired statistics, verdicts, figures, `method_out.json` (root).
- `tests/test_checks.py` — unit checks (known ECE and AURC values, temperature recovery, Holm, t df) and the SAM(ρ=0)-equals-SGD pairing check. All 6 groups pass.
- `run_all.sh` — runs timing, tune, final, compressed, then analyze.
- `results/` (kept) — `method_out.json` (exp_gen_sol_out format, validated), `per_run_metrics.csv`, `tuning_table.csv`, `chosen_hparams.json`, `chosen_grid.json`, `timings.csv`, `split_indices.npz`, `labels.npz`, `reliability_bins_pooled.json`, `runs/` (logits per run, temperature JSON), `logs/` (training logs per run).
- `results/checkpoints/` (kept, stays on the run's volume) — 10 final state_dicts.
- `figures/` (kept) — `fig_reliability_diagrams` and `fig_confidence_histograms` (PDF + PNG), rendered by aii-data-fig-gen from specs in `results/figure_specs/`.
- `scratch/` (deleted after the round) — Fashion-MNIST download and smoke and timing runs.

## How to run

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python torch torchvision numpy scipy matplotlib loguru psutil jsonschema
.venv/bin/python tests/test_checks.py
./run_all.sh          # ~40 min on Apple M3 Max (MPS)
```

## Method notes

- Data: torchvision FashionMNIST (60k train, 10k test). Fixed 50k/10k split of the training images (seed 0). Normalised with train-split mean and std. No augmentation.
- Model: Conv(1→32, 3, padding=1)-BN-ReLU-Pool, Conv(32→64, 3, padding=1)-BN-ReLU-Pool, FC 3136→128→10. padding=1 is an assumption that keeps 28→14→7.
- Training: batch 128, SGD momentum 0.9, weight decay 5e-4, CosineAnnealingLR(T_max=E) stepped per epoch. Shared initialisation and batch order per seed.
- SAM: ρ-sized ascent step along the normalised gradient, second gradient with BatchNorm running statistics frozen, then the base SGD step.
- Selection: validation NLL of the final-epoch model at seed 100. Test logits are written only by final and compressed runs and are never used for selection.
- Temperature: one per model, fitted on validation logits (bounded scalar minimisation over [0.05, 20]).
- Statistics: paired differences over 5 seeds; primary inference is the paired t-interval (df=4). A seed bootstrap is reported as a check only.

## Deviations from the plan

- The post-epoch-40 extension to 60 epochs was not run. SGD (0.008) is below the 0.01 threshold; SAM (0.107) is above it. The under-fit is recorded in `metrics.json` and reported above.
- Holm correction is over the plan's family only. The training-length contrast and the AURC SAM vs TS-SGD contrast are reported uncorrected.
- MPS kernels are not bit-deterministic. Seeds fix initialisation and batch order, not floating-point results.
- The timing gate kept the full grids (projected total ≈ 0.6 GPU-hours).

## Restoring removed files

- `scratch/` (FashionMNIST download and smoke runs): `.venv/bin/python -c "from torchvision.datasets import FashionMNIST; FashionMNIST('scratch/data', train=True, download=True); FashionMNIST('scratch/data', train=False, download=True)"`. HF mirror fallback: `zalando-datasets/fashion_mnist`. Smoke runs: `.venv/bin/python method.py smoke`.
- `.venv/`: `uv venv .venv --python=3.12 && uv pip install --python .venv/bin/python -r pyproject.toml`.
- `__pycache__/`: `uv run method.py` (Python recreates the bytecode cache on import).

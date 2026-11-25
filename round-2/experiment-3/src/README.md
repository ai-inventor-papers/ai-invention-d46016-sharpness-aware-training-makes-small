# Retuned SAM vs SGD calibration check (Fashion-MNIST)

Question: does sharpness-aware minimization (SAM) improve test accuracy and calibration over plain SGD for a small CNN on Fashion-MNIST, once neither optimiser's learning rate sits at the grid edge?

Scope: one dataset (Fashion-MNIST), one two-block CNN, five paired seeds. The claim is restricted to that setting.

## Results (test set, 10,000 images, five paired seeds)

Final models: SGD lr 0.001 (momentum 0.9, weight decay 5e-4); SAM lr 0.01, rho 0.1. Both 40 epochs, per-iteration cosine schedule, batch 128.

Paired differences, arm minus arm (mean, paired 95% t-interval, df = 4, Holm over the 12-row family):

| Row | Mean | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| SAM - SGD: accuracy | +0.0096 | [+0.0065, +0.0127] | 0.0010 | 0.0070 |
| SAM - SGD: ECE | -0.0050 | [-0.0079, -0.0021] | 0.0084 | 0.0336 |
| SAM - SGD: NLL | -0.0356 | [-0.0428, -0.0285] | 0.0002 | 0.0016 |
| SAM - SGD: Brier | -0.0158 | [-0.0185, -0.0130] | 0.0001 | 0.0010 |
| SAM - SGD: ACE | -0.0060 | [-0.0089, -0.0031] | 0.0048 | 0.0240 |
| SAM - TS-SGD: NLL | -0.0326 | [-0.0396, -0.0256] | 0.0002 | 0.0018 |
| SAM - TS-SGD: Brier | -0.0152 | [-0.0179, -0.0126] | 0.0001 | 0.0010 |
| SAM - TS-SGD: ECE | +0.0020 | [-0.0003, +0.0043] | 0.0718 | 0.2154 |
| SAM - TS-SGD: ACE | +0.0013 | [-0.0019, +0.0044] | 0.3208 | 0.3208 |
| TS-SAM - TS-SGD: ECE | +0.0020 | [-0.0003, +0.0043] | 0.0718 | 0.2154 |
| TS-SAM - TS-SGD: NLL | -0.0326 | [-0.0396, -0.0256] | 0.0002 | 0.0018 |

The SAM - TS-SGD accuracy row is identical to the SAM - SGD accuracy row, because temperature does not change the argmax (12 rows in total; full per-seed values in `results/analysis_tables.json`).

Reading:
- **Accuracy**: SAM is higher by about 1.0 point (92.3% vs 91.4% mean). The interval is a bound on a five-seed estimate, not an equivalence claim.
- **NLL after temperature scaling**: the NLL lead over temperature-scaled SGD survives the Holm correction (-0.033, interval excludes zero). This is the headline claim.
- **ECE and ACE after temperature scaling**: no significant difference. The raw ECE lead of SAM largely disappears once SGD is temperature-scaled.
- **Temperatures**: SGD fitted T is 1.03 to 1.06 (raw SGD is over-confident). SAM fitted T is 0.92 to 0.95, so **raw SAM is under-confident**; this is the reverse of the usual pattern and is stated because it changes how the raw ECE row should be read.
- **Final training loss**: SGD 0.122 (mean), SAM 0.110. In this round SAM is not under-fit relative to SGD, unlike round 1 (training loss 0.107 vs 0.008). The fit confound is therefore smaller than the round-1 one, but it is still not controlled by matching training loss (see limitations).

## Tuning (validation only, seed 101, selection by final-epoch validation NLL)

`results/tuning_table.csv` holds every configuration; `results/selection.json` the choice.

- SGD: lr 0.0003 (val NLL 0.250), **0.001 (0.214, selected)**, 0.003 (0.238), 0.01 (0.273), 0.03 (0.267). The lower extension was run because 0.001 was the lowest grid value; 0.0003 was worse, so the optimum is interior.
- SAM: 12 grid configurations (lr 0.001, 0.003, 0.01, 0.03 by rho 0.05, 0.1, 0.2). **lr 0.01, rho 0.1 (0.182, selected)**; the next best was lr 0.03, rho 0.1 (0.187). No SAM configuration was at the lower grid edge, so the 0.0003 SAM extension was not run.
- The round-1 collapse at SAM lr 0.03, rho 0.2 did **not** reproduce under this recipe and tuning seed: that configuration reached val accuracy 0.929 (val NLL 0.194).

## Controls and checks

- Same seed gives the same initialisation and the same data order for both optimisers.
- Temperature fitted on the validation split only (bounded scalar minimisation of NLL over T in [0.05, 10]).
- Determinism: rerunning SGD seed 0 gave logits identical to the original run (max abs difference 0.0; `logs/determinism.json`).
- Unit checks (`tests_unit.py`, CPU): SAM two-pass update equals the hand-computed value; BatchNorm running statistics are unchanged on the perturbed pass; T = 1 reproduces raw metrics; ECE is near 0 for a calibrated synthetic sample and clearly positive for an over-confident one; split sizes are 50000 / 10000 / 10000 with no overlap; the paired t and p match `scipy.stats.ttest_rel`; the Holm adjustment matches a hand calculation.

## Placebo checks (read carefully)

- **Self-comparison**: SGD against itself gives a zero difference by construction.
- **Pre-registered shuffled-label placebo** (SAM - SGD, both evaluated against one fixed shuffle of test labels): accuracy difference is null (p = 0.63). NLL (p = 0.009, Holm 0.036) and Brier (p = 0.002, Holm 0.010) differences are **significant** under random labels. This is because NLL and Brier respond to confidence sharpness even without label information, and SAM is the sharper model under wrong labels. So the shuffled-label placebo does not work as a clean null for these metrics: part of the raw NLL and Brier lead is a confidence-level effect, not evidence of label information alone. The accuracy row behaves as a null.
- **Post-hoc placebo on the temperature-scaled arms** (not part of the 12-row family; `results/posthoc_placebo_ts.json`): the NLL and ECE differences are again significant under shuffled labels (p = 0.0002 and 0.0001). Temperature scaling removes the average confidence offset on the real labels but not the sharpness sensitivity under shuffled ones. This placebo therefore does not separate calibration quality from confidence sharpness. It is reported, not used to change the conclusions above.

## Decisions and deviations from the plan

- Tuning ran the full grid; the time guard (`logs/time_guard.json`) projected about 40 minutes of tuning, far below the 4.5 h limit, so the reduced grid was not used. Each 40-epoch run took 50 to 120 s on MPS.
- Five seeds (0 to 4) for finals, one tuning seed (101). Holm family size 12. t critical value 2.776 (df = 4).
- ACE is the top-label, equal-count binned variant over max-softmax confidence, not per-class ACE of Nixon et al. (2019). It is not directly comparable to published ACE values.
- No model weights are saved. Logits at epochs 5, 15 and 40 are saved for every run; training-length conclusions are not drawn (criterion c deferred).
- Per-epoch evaluations are at epochs 5, 15 and 40 only.
- The determinism check reused the finals' SGD seed 0 at the selected lr.
- The post-hoc TS placebo was added after seeing the results. It is labelled as post hoc.

## Limitations

- Single dataset, single small architecture, five seeds: low power for effects smaller than about 0.5 points of accuracy or 0.005 of ECE. The accuracy claim is a bound.
- Learning rate and rho were chosen on one tuning seed; a different seed could select differently.
- NLL/Brier leads are not separated from confidence-sharpness effects by the placebo checks above; a matched-training-loss comparison was not run and is the next question.
- The raw ECE lead of SAM is not robust to temperature scaling.

## Layout

- `method.py` — full pipeline: data split, CNN, SGD and SAM training, tuning with the time guard and 0.0003 extension, finals, determinism check, analysis (temperature scaling, paired family, Holm, reliability bins, placebo), output writing. Stages: `smoke`, `tune`, `final`, `analyze`.
- `tests_unit.py` — stage-0 unit checks on CPU.
- `make_figures.py` — renders the reliability figure and the post-hoc placebo.
- `splits/split_seed0.json` — train, validation and test index definition (seed 0).
- `results/method_out.json` — headline tables in the exp_gen_sol_out schema (validated).
- `results/analysis_tables.json` — full per-seed metrics, temperatures, train losses, family rows, placebo.
- `results/tuning_table.csv`, `results/selection.json` — tuning sweep and selection.
- `results/reliability_bins.json` — pooled 15-bin reliability data per arm (raw and temperature-scaled).
- `results/posthoc_placebo_ts.json` — post-hoc placebo on temperature-scaled arms.
- `results/runs/*.json` — one record per training run (config, epoch losses, timings, validation metrics).
- `results/logits/*.npz` — float32 validation and test logits at epochs 5, 15, 40 per run (84 files, about 65 MB in total; each file is under 1 MB and is published with the repository).
- `results/audit_rederivation.json` — independent re-derivation of the headline numbers (see Audit below).
- `full_method_out.json` (all 10000 test images as examples; each has `input`, `output` = true class, `metadata_test_index`, `metadata_true_label` and ten `predict_*` fields, one per final model and seed), `mini_method_out.json` (first 3 examples), `preview_method_out.json` (mini with strings truncated to 200 characters). Made by `make_mini_preview.py`. Validated against exp_gen_sol_out.
- `audit_rederive.py` — re-derivation script (separate code path, see below).
- `make_mini_preview.py` — writes the three method-output variants.
- `figures/reliability.pdf|png` — reliability diagram from `figures/reliability_spec.json` via aii-data-fig-gen (`make_figures.py` needs `AII_DATA_FIG_GEN` set to that skill's `chart_gen.py`).
- `logs/` — run log, per-run timings, time guard, determinism check, stage stdout.

## Audit

`audit_rederive.py` recomputes the headline numbers from the raw logits with a different code path (torch cross-entropy instead of numpy log-softmax; grid-search temperature instead of `minimize_scalar`; `np.digitize` ECE binning; `scipy.stats.ttest_1samp` instead of the hand-computed t). Results match `method.py` for NLL (SAM − TS-SGD −0.03259, p = 0.0002) and for the raw NLL and ECE rows. The ECE row SAM − TS-SGD differs slightly (p = 0.064 here vs 0.072 in `method.py`) because of tie handling at bin edges; both are non-significant.

The shuffled-label placebo was run in the audit as well. **It does not behave as a null for NLL**: SAM − SGD NLL against shuffled labels is significant (p = 0.009). The placebo therefore does not validate the NLL test; see the placebo section above.

## Run

```
uv venv .venv --python=3.12
uv pip install torch torchvision numpy scipy loguru psutil matplotlib
.venv/bin/python tests_unit.py
.venv/bin/python method.py --stage tune
.venv/bin/python method.py --stage final
.venv/bin/python method.py --stage analyze
AII_DATA_FIG_GEN=/path/to/aii-data-fig-gen/scripts/chart_gen.py .venv/bin/python make_figures.py
.venv/bin/python audit_rederive.py
.venv/bin/python make_mini_preview.py
```

Exact versions and the environment used are in `pyproject.toml` and `reproducibility.md`.

Tuning and finals are resumable: a finished run is loaded from `results/runs/` instead of retrained. Device: Apple MPS when available, otherwise CPU.

## Restoring removed files

- `data_cache/` (FashionMNIST, about 83 MB; deleted after the round):
  ```
  uv run python -c "from torchvision import datasets; datasets.FashionMNIST('data_cache', train=True, download=True); datasets.FashionMNIST('data_cache', train=False, download=True)"
  ```
- `.venv/` (regenerable environment):
  ```
  uv venv .venv --python=3.12 && uv pip install torch torchvision numpy scipy loguru psutil matplotlib
  ```

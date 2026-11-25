# SAM vs SGD at Matched Training Loss (Fashion-MNIST)

Question: does sharpness-aware minimisation (SAM) improve test accuracy and calibration over plain SGD for a small CNN on Fashion-MNIST, once the comparison is made at matched training loss?

## What was done

- Data: Fashion-MNIST. The 60k training images are split (numpy seed 0) into 50k train and 10k validation. The official 10k test set is used for reporting. Normalised with the 50k train mean and std. No augmentation (not specified in the plan; an assumption).
- Model: Conv(1→32)-BN-ReLU-Pool, Conv(32→64)-BN-ReLU-Pool, Linear(3136→128)-ReLU, Linear(128→10). Convolutions have no bias because they are followed by BatchNorm.
- Training: batch 128, momentum 0.9, weight decay 5e-4, cross-entropy, per-epoch cosine schedule `0.5·0.01·(1+cos(πe/E))`. Learning rate 0.01 for both optimisers. Initialisation and shuffle order are seeded identically for SGD and SAM.
- SAM: one ascent step with radius ρ, then the base SGD-momentum step at the perturbed gradient. BatchNorm statistics update in both passes (as the spec requires).
- Conditions (paired by seed 0–4):
  1. SAM, 40 epochs, ρ = 0.1 (tuned)
  2. SGD, 40 epochs
  3. SGD, E\* = 10 epochs (matched loss)
- Matching:
  - ρ was chosen from {0.05, 0.1, 0.2} on seed 101 by lowest final validation NLL. Test was not used. ρ = 0.1 won (validation NLL 0.1998 vs 0.2104 and 0.2299).
  - E\* was chosen on seed 100. SAM 40-epoch eval-mode training loss L = 0.1121. The SGD grid E ∈ {10,…,20} gave the first E within 0.01 at **E = 10** (gap −0.0009). This is the lower edge of the grid, so the matched comparator is a 10-epoch schedule.
- Comparisons (each as SAM minus comparator, paired by seed, 95% t-intervals with df = 4):
  - (a) secondary: SAM epoch 40 vs condition 2 at the epoch whose training loss is closest to SAM's (e_s = 10 or 11 per seed)
  - (b) **primary**: SAM epoch 40 vs condition 3 final epoch
- Metrics on test logits: accuracy, ECE (15 equal-width bins), ACE (15 equal-mass bins, Nixon et al. multiclass form), NLL, Brier. Each raw and, where relevant, temperature-scaled (T fitted on validation NLL with a bounded scalar search over log T ∈ [log 0.05, log 10]).
- Holm correction over the 14-test family (2 comparisons × 7 metrics).

## Results (5 paired seeds)

Primary comparison (b): SAM 40 epochs minus SGD 10 epochs at matched training loss. Negative values favour SAM for every metric here.

| Metric | Mean diff | 95% CI | p (raw) | p (Holm) |
|---|---|---|---|---|
| Accuracy | +0.0025 | [+0.0007, +0.0042] | 0.018 | 0.055 |
| ECE (raw) | −0.0076 | [−0.0091, −0.0061] | 0.0001 | 0.0018 |
| ECE (temp-scaled) | −0.0001 | [−0.0016, +0.0013] | 0.82 | 0.97 |
| NLL (raw) | −0.0125 | [−0.0152, −0.0099] | 0.0002 | 0.0019 |
| NLL (temp-scaled) | −0.0082 | [−0.0110, −0.0053] | 0.0013 | 0.0094 |
| Brier | −0.0054 | [−0.0063, −0.0045] | 0.0001 | 0.0010 |
| ACE | −0.0011 | [−0.0013, −0.0009] | 0.0001 | 0.0018 |

Secondary comparison (a): SAM 40 epochs minus SGD 40-epoch snapshot at the matched epoch. Accuracy +0.0102; ECE raw −0.0188; ECE temp-scaled −0.0007 (n.s.); NLL raw −0.0414; NLL temp-scaled −0.0275; Brier −0.0171; ACE −0.0045. All except temperature-scaled ECE survive Holm. This comparison is weaker because mid-schedule snapshots are not finished models at a high learning rate.

Reading the primary result: SAM's raw NLL, raw ECE and Brier advantages persist at matched training loss. The temperature-scaled ECE difference disappears, which means much of the raw ECE advantage is a confidence-scale effect that a single temperature removes. The temperature-scaled NLL advantage remains (about −0.008) and is significant after Holm. Accuracy improves by about 0.25 points, but the Holm-adjusted p = 0.055 does not reach significance.

Matched-loss table (`results/matched_loss_table.json`): per-seed SAM loss 0.099–0.110. The gap to the 10-epoch SGD loss is +0.0004 to +0.0055 (all under the 0.01 flag). The gap at the matched epoch for (a) is −0.0046 to +0.0085 (under the 0.02 flag).

Reproduction check (condition 1 minus condition 2 at epoch 40, mean of 5 seeds), against the earlier reported headline:

| Quantity | Observed | Reported | Within tolerance |
|---|---|---|---|
| Accuracy | +0.0024 | +0.0015 | yes (0.005) |
| ECE raw | −0.0373 | −0.0378 | yes (0.005) |
| NLL raw | −0.1009 | −0.0980 | yes (0.01) |
| NLL temp-scaled | −0.0210 | −0.0209 | yes (0.01) |

The earlier tuned ρ is not recorded in this run's inputs, so this check is not exact.

## Caveats

- Five seeds give low power. Intervals that exclude zero are reported, but a non-significant result is not evidence of no difference.
- One dataset and one small architecture. Claims are scoped to that setting.
- The SGD learning rate is fixed at 0.01 for both optimisers and is not re-tuned. A reviewer could call the SGD baseline under-tuned. The plan's optional lr = 0.003 extra row was not run.
- The matched-loss SGD comparator reaches its training loss at the lower edge of the epoch grid (E\* = 10). A shorter schedule might match more closely; this was not explored.
- The rho = 0 check: SAM with ρ = 0 reproduces SGD exactly in parameters and per-batch losses (max difference 0). Eval-mode logits differ by up to 0.146, because the spec's double BatchNorm update changes running statistics. This is the expected effect of that choice, not an optimisation difference.
- The placebo (SGD vs SGD, same seed) gave an exactly zero difference on MPS in this run. MPS does not guarantee bitwise determinism in general.
- Figures use top-label confidence and correctness pooled over seeds, computed from the float16 logits saved for audit. Reported metrics use float32 logits, so pooled figure ECE values (e.g. 0.006 vs 0.015 in the reliability legend) differ from per-seed means.


## Audit

An independent re-derivation (`audit_rederive.py`, output `results/audit_rederivation.json`) recomputes the four primary headline numbers from the raw per-epoch histories and the saved float16 logits. It uses different code paths: logsumexp for NLL, histogram binning for ECE, and a grid search for the temperature.

| Metric (primary, SAM minus matched-loss SGD) | Re-derived | Stored | Abs. difference | Sign-flip p (exact) |
|---|---|---|---|---|
| Accuracy | +0.00258 | +0.00246 | 1.2e-4 | 0.0625 |
| ECE raw | −0.00766 | −0.00761 | 5.1e-5 | 0.0625 |
| NLL raw | −0.01253 | −0.01254 | 3.7e-6 | 0.0625 |
| NLL temp-scaled | −0.00817 | −0.00818 | 4.5e-6 | 0.0625 |

Three findings from the audit that limit how strongly the headline can be read:

1. **The exact sign-flip test cannot reach p < 0.05.** With 5 paired seeds, all five differences share a sign in every headline metric, so the smallest achievable two-sided p is 2/32 = 0.0625. The t-test results (in the main table) depend on the five differences being roughly normal.
2. **The label-permutation placebo passes only for accuracy.** SAM and matched-loss SGD were scored against randomly permuted test and validation labels (one permutation shared by both). Accuracy gives no difference (p = 0.52; sign-flip p = 0.44), as a null should. NLL and ECE do not: they still differ significantly (raw NLL −0.78, raw ECE −0.0071, p ≈ 0.0001), because these metrics reward the less-confident model even when labels carry no information. Temperature-scaled NLL is not a valid placebo at all, since the temperature is fitted on permuted validation labels. So the NLL and ECE advantages are not shown to be free of a confidence-scale effect by this placebo. Temperature scaling and the matched-loss design are the controls that address it, and the temperature-scaled NLL advantage (−0.0082) is what survives them.
3. **A placebo built by re-pairing seeds is not valid.** A permutation of seeds leaves the mean difference exactly zero by construction. This was tried first and replaced.

Not re-derived in the audit: the secondary comparison (a), the Holm-adjusted p-values, the ACE and Brier values, the reproduction-check values, and the matched-loss gaps. They come from the same `analysis.py` code path as the primary numbers and were not separately recomputed.

## Layout

- `method.py` — full pipeline and entry point (`--stage checks|mini|tune|pilot|main|analyze|all`). Contains data, model, training, metrics and the run logic.
- `audit_rederive.py` — independent re-derivation of the headline numbers and placebo (see Audit).
- `pyproject.toml` — exact pinned dependencies (from `uv pip freeze`).
- `reproducibility.md` — step-by-step reproduction instructions.
- `analysis.py` — paired statistics, Holm correction, matched-loss table, reproduction check, figure specs and `method_out.json`.
- `results/method_out.json` — exp_gen_sol_out format (validated against the schema), with the full analysis under `metadata`.
- `results/matched_loss_table.json` — per-seed training loss and gaps.
- `results/chosen_E.json` — pilot target loss, SGD E grid, chosen E\*.
- `results/tuning.json` — ρ tuning on seed 101.
- `results/timings.json` — wall-clock per run (about 75 s per SAM 40-epoch run and about 20 s per SGD 10–20-epoch run on MPS).
- `results/checks/checks.json` — unit checks (ECE, temperature fit, split sizes, ρ = 0, placebo) and mini-pipeline outputs.
- `results/runs/*.json` — per-epoch history for every run (train loss, metrics, temperature).
- `results/logits/*.npz` — per-epoch float16 validation and test logits (kept, on the run volume).
- `results/figure_data/top_label_conf_correct.npz` — raw confidences and correctness behind the figures.
- `figures/` — reliability diagrams (primary and matched-epoch) and confidence histogram, rendered with aii-data-fig-gen (PDF and PNG).
- `logs/` — run logs.

## Running

```
uv venv .venv --python=3.12
uv pip install -p .venv/bin/python torch torchvision numpy scipy matplotlib loguru psutil jsonschema
uv run method.py --stage all       # checks, mini, tune, pilot, main, analyze
uv run method.py --stage analyze   # re-run the analysis from saved results
```

## Restoring removed files

The manifest marks two paths for deletion. Both are rebuilt by the commands below.

- `scratch/` (Fashion-MNIST download cache, about 82 MB): redownloaded by torchvision.
  ```
  uv run python -c "from torchvision import datasets; datasets.FashionMNIST(root='scratch/fmnist', train=True, download=True); datasets.FashionMNIST(root='scratch/fmnist', train=False, download=True)"
  ```
- `.venv/` (Python environment): recreated with
  ```
  uv venv .venv --python=3.12 && uv pip install -p .venv/bin/python torch torchvision numpy scipy matplotlib loguru psutil jsonschema
  ```
- `__pycache__/`: regenerated automatically on the next run.

The logits (`results/logits/`) are kept on the run's storage volume, not in the published repository. Re-running the full pipeline regenerates them.

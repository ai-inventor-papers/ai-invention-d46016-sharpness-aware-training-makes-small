# Does sharpness-aware minimization improve calibration of a small CNN on Fashion-MNIST?

## Framing

The run asks whether sharpness-aware minimization (SAM) [1] improves test accuracy and test calibration over standard stochastic gradient descent (SGD) for a small convolutional network trained on Fashion-MNIST [6]. The question matters because SAM roughly doubles the cost of each step, and its accuracy gains have been shown mostly on large models. Tan et al. argue that SAM counteracts overconfidence by implicitly maximising predictive entropy, and they report the effect at larger scale [4]. Guo et al. showed that temperature scaling removes most of the miscalibration of overconfident networks without changing accuracy [2]. Nixon et al. showed that expected calibration error (ECE) can read as low for a poorly calibrated model, because binning hides errors [3]. The hypothesis under test therefore asks for three quantities: a lower test ECE and negative log-likelihood (NLL) for SAM, an NLL advantage that remains after temperature scaling of the SGD baseline, and a gap that grows with training length.

The hypothesis has three explicit criteria. (a) SAM's mean test ECE and NLL are lower than tuned SGD's, with a paired 95% interval that excludes zero. (b) The SAM advantage in NLL persists after temperature scaling of SGD. (c) The calibration gap is larger at full training than at short training. Accuracy is reported either way.

Two artifacts were run in this iteration. Their findings are recorded below with the numbers read from their outputs. The feedback that motivated the two artifacts is not in the material this step received, so this section does not record what it objected to. The third direction, a check of prior work, did not run (see the record of designed work under Iteration 1).

# Iteration 1

## Strategy

The run had no previous results to build on, so this iteration set out to get a controlled measurement of the main comparison first, and then a mechanism measurement. Two artifacts were produced.

The first artifact, a controlled comparison at fixed hyperparameters chosen on validation data, measures accuracy, ECE, adaptive calibration error (ACE), NLL, Brier score and area under the risk-coverage curve (AURC) for SGD and SAM, with a temperature-scaled SGD control fitted on validation logits [ARTIFACT:art_SL9gPfE09HlN]. It uses five paired seeds, a 40-epoch main schedule, and 5- and 15-epoch schedules, which we call compressed schedules, for the training-length question. Learning rates and the SAM radius were selected on a tuning seed (seed 100) using validation NLL only.

The second artifact is a mechanism study on three seed pairs with one setting fixed in advance (learning rate 0.03, SAM radius 0.1, 60 epochs, cosine schedule) [ARTIFACT:art_WOFinDmS2PoX]. It tracks per-epoch training loss, validation ECE and the top Hessian eigenvalue, and it compares the two optimisers at matched training loss. Its temperature-scaling control uses the same reasoning as the first artifact.

The two artifacts differ in design, and the record should keep that difference clear. The first chose its learning rates by a validation search. The second fixed its learning rate in advance and did not tune it. The first has five seeds; the second has three.

## Artifact 1: controlled comparison at fixed hyperparameters

**Setup.** The network has two convolution blocks (Conv 1 to 32 filters, Conv 32 to 64 filters, each with batch normalisation, ReLU and 2x2 max pooling) followed by a fully connected layer of width 128 and a 10-way output. The 60,000 standard training images are split once into 50,000 for training and 10,000 for validation, with the standard 10,000 test images used only in final evaluation. Training uses batch size 128, momentum 0.9, weight decay 5e-4 and a cosine schedule over the stated number of epochs, with no augmentation [ARTIFACT:art_SL9gPfE09HlN].

**Tuning.** The grid was learning rate in {0.01, 0.03, 0.1} and radius in {0.05, 0.1, 0.2}. Table 1 gives validation NLL for all twelve settings on the tuning seed. Selection took the lowest final validation NLL. SGD selected learning rate 0.01, and SAM selected learning rate 0.01 with radius 0.1. Both selections lie at the lower edge of the learning-rate grid. One SAM setting (learning rate 0.03, radius 0.2) collapsed to random accuracy (validation accuracy 0.1305, training loss 2.302) and was not selected.

Table 1. Tuning on seed 100, 40 epochs, final-epoch values, read from `results/tuning_table.csv`. Selection rule: lowest validation NLL. This table shows which hyperparameters enter the final runs.

| Condition | Learning rate | rho | Validation NLL | Validation accuracy | Training loss |
|---|---|---|---|---|---|
| SGD | 0.01 | none | 0.26497 | 0.9286 | 0.008858 |
| SGD | 0.03 | none | 0.270721 | 0.9326 | 0.005446 |
| SGD | 0.1 | none | 0.274023 | 0.929 | 0.009904 |
| SAM | 0.01 | 0.05 | 0.190537 | 0.933 | 0.051861 |
| SAM | 0.01 | 0.1 | 0.18404 | 0.932 | 0.11747 |
| SAM | 0.01 | 0.2 | 0.217258 | 0.9218 | 0.201134 |
| SAM | 0.03 | 0.05 | 0.211529 | 0.9309 | 0.02717 |
| SAM | 0.03 | 0.1 | 0.187573 | 0.9319 | 0.073706 |
| SAM | 0.03 | 0.2 | 2.301991 | 0.1305 | 2.301912 |
| SAM | 0.1 | 0.05 | 0.209597 | 0.9321 | 0.037403 |
| SAM | 0.1 | 0.1 | 0.193233 | 0.9303 | 0.062919 |
| SAM | 0.1 | 0.2 | 0.208249 | 0.9234 | 0.175019 |

**Final runs.** Five seeds (0 to 4) were trained for 40 epochs with the selected settings, and separately for 5 and 15 epochs as compressed schedules. Each compressed schedule is its own full training run with a cosine schedule over that many epochs, not a run stopped early from the 40-epoch run. Table 2 gives the means over the five seeds at each condition. Per-run values are in the artifact's per-run table.

Table 2. Test-set metrics, mean over five seeds, read from `results/per_run_metrics.csv`. Columns are the condition, the schedule and the epoch at which the model was evaluated. TS-SGD is SGD with a single temperature fitted on validation logits. TS-SAM is included because the artifact evaluated it at epoch 40. Lower is better for ECE, ACE, NLL, Brier and AURC.

| Condition | Schedule | Epoch | Accuracy | ECE | ACE | NLL | Brier | AURC |
|---|---|---|---|---|---|---|---|---|
| SGD | 40-epoch run | 5 | 0.8954 | 0.0254 | 0.0246 | 0.2874 | 0.1506 | 0.0166 |
| SGD | 40-epoch run | 15 | 0.9158 | 0.0353 | 0.0351 | 0.2725 | 0.1291 | 0.0122 |
| SGD | 40-epoch run | 40 | 0.9222 | 0.0441 | 0.0440 | 0.3095 | 0.1244 | 0.0111 |
| SAM | 40-epoch run | 5 | 0.8988 | 0.0121 | 0.0109 | 0.2770 | 0.1449 | 0.0158 |
| SAM | 40-epoch run | 15 | 0.9167 | 0.0063 | 0.0060 | 0.2280 | 0.1198 | 0.0110 |
| SAM | 40-epoch run | 40 | 0.9236 | 0.0063 | 0.0054 | 0.2115 | 0.1108 | 0.0095 |
| SGD | compressed | 5 | 0.9141 | 0.0092 | 0.0071 | 0.2396 | 0.1253 | 0.0120 |
| SGD | compressed | 15 | 0.9228 | 0.0227 | 0.0223 | 0.2316 | 0.1156 | 0.0102 |
| SAM | compressed | 5 | 0.8958 | 0.0149 | 0.0137 | 0.2847 | 0.1489 | 0.0165 |
| SAM | compressed | 15 | 0.9130 | 0.0095 | 0.0093 | 0.2381 | 0.1248 | 0.0118 |
| TS-SGD | 40-epoch run | 40 | 0.9222 | 0.0094 | 0.0078 | 0.2324 | 0.1157 | 0.0110 |
| TS-SAM | 40-epoch run | 40 | 0.9236 | 0.0099 | 0.0087 | 0.2128 | 0.1110 | 0.0095 |

Two features of Table 2 matter for the hypothesis. First, at the 40-epoch run's final epoch SAM's ECE is 0.0063 against 0.0441 for SGD, and the gap is large relative to seed variance. Second, the SAM model at epoch 40 has a training loss of 0.107 against 0.008 for SGD (reported by the artifact [ARTIFACT:art_SL9gPfE09HlN]), so SAM is less fitted at the end of the main schedule. That confound is taken up below.

**Paired contrasts.** Each SAM run is compared with the SGD run that shares its seed. Table 3 gives the paired mean differences with t-intervals at df = 4, and the p-values after Holm correction over the family of eight contrasts. The family covers accuracy, ECE, ACE, NLL, Brier, AURC and the two comparisons against temperature-scaled SGD. Figure 1 draws the same intervals.

Table 3. Paired differences, SAM minus the baseline, at epoch 40, five seeds, df = 4. Values are in the metric's own units; "x100" marks the rescaling used in Figure 1. Holm p is over the eight-contrast family. Read from `full_method_out.json`.

| Contrast | Mean difference | 95% CI | Holm p | Verdict |
|---|---|---|---|---|
| Accuracy, SAM minus SGD | +0.00146 | [-0.00071, +0.00363] | 0.135 | no detectable difference |
| ECE, SAM minus SGD | -0.03779 | [-0.04145, -0.03412] | 7.1e-5 | SAM lower |
| ACE, SAM minus SGD | -0.03861 | [-0.04282, -0.03440] | 9.9e-5 | SAM lower |
| NLL, SAM minus SGD | -0.09796 | [-0.10918, -0.08673] | 1.0e-4 | SAM lower |
| Brier, SAM minus SGD | -0.01362 | [-0.01792, -0.00933] | 0.0037 | SAM lower |
| AURC, SAM minus SGD | -0.00152 | [-0.00212, -0.00092] | 0.0064 | SAM lower |
| NLL, SAM minus TS-SGD | -0.02088 | [-0.02662, -0.01514] | 0.0027 | SAM lower |
| ECE, SAM minus TS-SGD | -0.00307 | [-0.00590, -0.00024] | 0.079 | not significant after Holm |
| TS-SAM minus TS-SGD, ECE (added in Iteration 2) | +0.00054 | [-0.00291, +0.00399] | unadjusted p = 0.68 | no difference |
| TS-SAM minus TS-SGD, NLL (added in Iteration 2) | -0.01963 | [-0.02569, -0.01357] | unadjusted p = 0.0008 | SAM lower |

**Correction (Iteration 2).** The ECE row above compares uncalibrated SAM with temperature-scaled SGD, so it is not a like-for-like comparison. The two rows added beneath it compare temperature-scaled SAM with temperature-scaled SGD, computed from the per-run metrics (Table 20). The ECE difference is +0.00054 with an interval that includes zero, so the calibrated ECE advantage is not supported. The NLL difference holds. The Holm p-values are not computed again over the larger family. The raw p-value for ECE is 0.68, which no correction makes significant.

[FIGURE:fig1]

Figure 1. Paired differences, SAM minus the baseline, at epoch 40 over five seeds. Points are mean differences and bars are 95% t-intervals (df = 4), in x100 metric units: percentage points for accuracy, ECE, ACE, Brier and AURC, and x100 NLL units for NLL. Left: against SGD. Right: against temperature-scaled SGD. The accuracy interval crosses zero, and the NLL gap against temperature-scaled SGD excludes zero.

Table 4 gives the per-seed paired differences for the two main metrics at epoch 40. Each row is one seed, so the variance of the effect shows directly.

Table 4. Per-seed paired differences at epoch 40, SAM minus SGD, read from the paired-test records. The sign is the same for every seed on both metrics.

| Seed | ECE difference | NLL difference |
|---|---|---|
| 0 | -0.03631 | -0.09233 |
| 1 | -0.04284 | -0.11242 |
| 2 | -0.03537 | -0.09602 |
| 3 | -0.03670 | -0.08911 |
| 4 | -0.03770 | -0.09991 |

**Training length.** The artifact tested whether the ECE advantage grows with training length by comparing the SGD-minus-SAM ECE advantage at three lengths: the 5-epoch compressed schedule, the 15-epoch compressed schedule and the 40-epoch run. Table 5 gives those advantages. A positive value means SGD is worse than SAM.

Table 5. SGD-minus-SAM ECE advantage by training length, mean over five seeds, from the per-seed values of the compressed and 40-epoch schedules. Positive means SAM is better. The 5-epoch row is a compressed schedule, not an early stop of the 40-epoch run.

| Training length | Mean advantage (SGD minus SAM ECE) |
|---|---|
| 5 epochs (compressed) | -0.0057 |
| 15 epochs (compressed) | +0.0132 |
| 40 epochs | +0.0378 |

The growth from 5 to 40 epochs is significant in the artifact's paired test (difference 0.0435, 95% CI [0.0376, 0.0494]). The 5-epoch row is negative, so at the 5-epoch schedule SAM's ECE is higher than SGD's. That reverses the early-stopping prediction in criterion (c) of the hypothesis, and the record should show it as a reversal. The reversal is small in NLL terms but present in ECE, with SAM's compressed-5 ECE at 0.0149 against SGD's 0.0092.

**Check.** The artifact computed again the main paired differences and intervals from the saved test logits with separate code, and its control checks did not reject. Table 6 reproduces the independent computation, read from its check output. The artifact did not compute again the ECE binning loop or the ACE, Brier and AURC values for every run, and it did not retrain the model. Scope, clarified in Iteration 2: the check covers the primary paired contrasts and the two control checks only.

Table 6. Independent computation of the main contrasts from saved test logits, epoch 40. The SciPy and independent intervals agree to the reported digits.

| Contrast | Mean difference | 95% CI (manual check) |
|---|---|---|
| Accuracy, SAM minus SGD | +0.00146 | [-0.00071, +0.00363] |
| ECE, SAM minus SGD | -0.03779 | [-0.04145, -0.03412] |
| NLL, SAM minus SGD | -0.09796 | [-0.10918, -0.08673] |
| AURC, SAM minus SGD | -0.00152 | [-0.00212, -0.00092] |
| NLL, SAM minus TS-SGD | -0.02088 | [-0.02662, -0.01514] |
| ECE, SAM minus TS-SGD | -0.00307 | [-0.00590, -0.00024] |

Control checks: comparing SGD with itself gives a mean ECE difference of exactly 0, and shuffling the test labels gives an accuracy difference of -0.00036 with a 95% interval of [-0.00235, +0.00163] that includes zero.

**Reading of Artifact 1.** Criterion (a) holds for ECE and NLL with intervals well clear of zero. The accuracy result is no significant difference, with a point estimate of +0.15 percentage points and an interval that crosses zero. Criterion (b) holds for NLL: SAM's NLL is lower than temperature-scaled SGD's by 0.021 with an interval excluding zero, and the Holm p is 0.0027. For ECE the advantage over temperature-scaled SGD is 0.003 with an interval just excluding zero, and it is not significant after Holm correction (p = 0.079). The shift that temperature scaling removes is therefore most of the ECE gap but not all of the NLL gap. Criterion (c) holds in the direction of the hypothesis at 15 and 40 epochs and reverses at 5 epochs.

Two caveats limit how far these results apply. First, SAM is under-fit at epoch 40, so part of its calibration advantage may come from a less-fitted model. Second, the selected learning rate lies at the lower edge of the grid, so a lower learning rate might change the SGD baseline. The second artifact addresses the first caveat directly.

## Artifact 2: mechanism study at fixed setting

**Setup.** The second artifact uses the same network family, with a 10,000-image validation split and the standard test set. Both optimisers share the initialisation and data order for each seed. The setting was fixed before the runs: learning rate 0.03, momentum 0.9, weight decay 5e-4, batch 128, 60 epochs, cosine schedule, SAM radius 0.1 [ARTIFACT:art_WOFinDmS2PoX]. The learning rate was not tuned for this artifact. That choice differs from Artifact 1, and it means the two artifacts do not share a baseline. The artifact reports summary statistics only at n = 3 seed pairs, with no p-values, so no test can be significant at the conventional level.

**Test metrics at epochs 5 and 60.** Table 7 gives the test metrics for each seed at the two evaluated epochs, read from the artifact's average table.

Table 7. Test-set metrics per seed at epochs 5 and 60, read from `results/method_out.json`. ECE uses 15 equal-width bins. Accuracy is a fraction.

| Optimiser | Seed | Epoch | Test accuracy | Test ECE | Test NLL | Test Brier | Test ACE | Validation ECE | Validation NLL |
|---|---|---|---|---|---|---|---|---|---|
| SGD | 0 | 5 | 0.9088 | 0.0105 | 0.2541 | 0.1321 | 0.0090 | 0.0064 | 0.2262 |
| SGD | 1 | 5 | 0.8978 | 0.0143 | 0.2754 | 0.1472 | 0.0128 | 0.0116 | 0.2559 |
| SGD | 2 | 5 | 0.8986 | 0.0239 | 0.2791 | 0.1459 | 0.0238 | 0.0179 | 0.2581 |
| SGD | 0 | 60 | 0.9226 | 0.0466 | 0.3186 | 0.1239 | 0.0466 | 0.0452 | 0.2901 |
| SGD | 1 | 60 | 0.9239 | 0.0439 | 0.3162 | 0.1246 | 0.0438 | 0.0419 | 0.2803 |
| SGD | 2 | 60 | 0.9251 | 0.0432 | 0.3105 | 0.1222 | 0.0432 | 0.0429 | 0.2883 |
| SAM | 0 | 5 | 0.9048 | 0.0167 | 0.2682 | 0.1389 | 0.0170 | 0.0222 | 0.2427 |
| SAM | 1 | 5 | 0.9054 | 0.0067 | 0.2589 | 0.1361 | 0.0072 | 0.0105 | 0.2412 |
| SAM | 2 | 5 | 0.8987 | 0.0128 | 0.2746 | 0.1444 | 0.0120 | 0.0098 | 0.2541 |
| SAM | 0 | 60 | 0.9256 | 0.0282 | 0.2405 | 0.1131 | 0.0282 | 0.0249 | 0.2111 |
| SAM | 1 | 60 | 0.9233 | 0.0284 | 0.2350 | 0.1131 | 0.0284 | 0.0256 | 0.2224 |
| SAM | 2 | 60 | 0.9263 | 0.0258 | 0.2415 | 0.1142 | 0.0240 | 0.0236 | 0.2181 |

At epoch 60 the per-seed paired differences in test ECE (SAM minus SGD) are -0.0184, -0.0155 and -0.0174, all negative. The test NLL differences are -0.0781, -0.0812 and -0.0690, also all negative. The test accuracy differences are +0.0030, -0.0006 and +0.0012, which have mixed signs. At epoch 5 the test ECE differences have no consistent sign.

**Temperature scaling.** Table 8 gives the temperature-scaled values for each seed. The temperature for each model was fitted on the validation split. Scaling changes no predicted class, so accuracy is unchanged by design.

Table 8. Temperature-scaled test metrics per seed, read from the artifact's temperature-scaling record. T is the fitted temperature, and "unscaled" reproduces the test ECE and NLL from Table 7 for comparison. Fitted temperatures for SGD at epoch 60 lie near 1.9, and for SAM near 1.4.

| Optimiser | Seed | Epoch | T | Test ECE, scaled | Test ECE, unscaled | Test NLL, scaled | Test NLL, unscaled |
|---|---|---|---|---|---|---|---|
| SGD | 0 | 5 | 1.0161 | 0.0094 | 0.0105 | 0.2537 | 0.2541 |
| SGD | 1 | 5 | 1.0464 | 0.0110 | 0.0143 | 0.2744 | 0.2754 |
| SGD | 2 | 5 | 1.1326 | 0.0118 | 0.0239 | 0.2743 | 0.2791 |
| SGD | 0 | 60 | 1.8979 | 0.0088 | 0.0466 | 0.2364 | 0.3186 |
| SGD | 1 | 60 | 1.8546 | 0.0104 | 0.0439 | 0.2373 | 0.3162 |
| SGD | 2 | 60 | 1.8936 | 0.0091 | 0.0432 | 0.2330 | 0.3105 |
| SAM | 0 | 5 | 0.8015 | 0.0140 | 0.0167 | 0.2651 | 0.2682 |
| SAM | 1 | 5 | 0.9085 | 0.0074 | 0.0067 | 0.2587 | 0.2589 |
| SAM | 2 | 5 | 1.0211 | 0.0123 | 0.0128 | 0.2743 | 0.2746 |
| SAM | 0 | 60 | 1.3787 | 0.0070 | 0.0282 | 0.2190 | 0.2405 |
| SAM | 1 | 60 | 1.3947 | 0.0052 | 0.0284 | 0.2160 | 0.2350 |
| SAM | 2 | 60 | 1.3556 | 0.0089 | 0.0258 | 0.2224 | 0.2415 |

Averaged over seeds at epoch 60, scaling moves SGD's test ECE from 0.0446 to 0.0094 and SAM's from 0.0275 to 0.0070. The SGD model's fitted temperature is well above 1, which matches its overconfidence. The SAM model's fitted temperature is also above 1 at epoch 60, so SAM is still overconfident, but less so. Averaged over seeds, scaling moves SGD's test NLL from 0.3151 to 0.2356 and SAM's from 0.2390 to 0.2192. The SAM advantage in NLL therefore decreases from 0.076 to 0.016 NLL units after scaling. Most of the unscaled NLL gap is a global confidence shift, and a residual remains.

[FIGURE:fig2]

Figure 2. Test ECE (left) and test NLL (right) at epoch 60 for SGD and SAM, with and without temperature scaling. Bars are means over three seeds and error bars are sample standard deviations across seeds. The scaled SAM and SGD ECE values are close, and the NLL advantage for SAM remains after scaling.

**Training loss and calibration over training.** Table 9 gives the per-epoch means at selected epochs, averaged over the three seeds, read from the per-epoch record. The evaluation grid is every epoch from 1 to 60; the rows shown are a subset of it. The top Hessian eigenvalue is estimated by 20 power iterations on a fixed 2,000-image batch.

Table 9. Per-epoch means over three seeds at selected epochs, read from `results/per_epoch.json`. Training loss is the mean over the epoch. Validation ECE is the mean of the three seeds. Top Hessian eigenvalue is the mean of the three seeds at evaluated epochs. The SAM loss increase within the radius is in the same record and is reported at the logged epochs in Table 22 (Iteration 2).

| Epoch | SGD training loss | SGD validation ECE | SGD top Hessian eigenvalue | SAM training loss | SAM validation ECE | SAM top Hessian eigenvalue |
|---|---|---|---|---|---|---|
| 1 | 0.4662 | 0.0160 | 24.95 | 0.5477 | 0.0116 | 9.18 |
| 2 | 0.2873 | 0.0106 | 20.69 | 0.3167 | 0.0144 | 6.80 |
| 5 | 0.1996 | 0.0119 | 19.09 | 0.2256 | 0.0141 | 6.21 |
| 10 | 0.1314 | 0.0219 | 15.72 | 0.1724 | 0.0094 | 6.60 |
| 15 | 0.0925 | 0.0313 | 20.58 | 0.1399 | 0.0064 | 6.33 |
| 20 | 0.0610 | 0.0368 | 19.19 | 0.1131 | 0.0092 | 6.94 |
| 30 | 0.0188 | 0.0457 | 25.02 | 0.0751 | 0.0136 | 8.01 |
| 40 | 0.0050 | 0.0440 | 14.55 | 0.0486 | 0.0180 | 7.50 |
| 50 | 0.0046 | 0.0434 | 19.89 | 0.0343 | 0.0243 | 7.43 |
| 60 | 0.0045 | 0.0433 | 20.20 | 0.0306 | 0.0247 | 7.13 |

Correction (Iteration 2): the SGD top Hessian eigenvalue is between 1.9 and 3.35 times the SAM value across the logged epochs (2.7 at epoch 1, 3.35 at epoch 25, 1.9 at epoch 40). The previous claim, two to three times at every evaluated epoch, was too broad. The ratio at each logged epoch is in Table 22 of Iteration 2. SGD's validation ECE rises from 0.012 at epoch 5 to 0.043 at epoch 60, while its training loss falls to 0.005. SAM's validation ECE falls to a minimum of 0.0064 at epoch 15 and then rises to 0.025 at epoch 60, while its training loss stays near 0.03. Both optimisers become less calibrated as training continues past the point where their validation ECE is lowest, and the rise in SGD's validation ECE is larger. The per-epoch means do not show a gap at epoch 5, which matches the sign-mixed comparison in Table 7.

**Matched training loss.** The artifact compared the two optimisers at equal training loss, by interpolating each run's per-epoch values on a common grid of eight training-loss levels. Table 10 gives the per-seed means of the differences over that grid. A negative value means SAM is lower.

Table 10. Matched training-loss comparison, SAM minus SGD, per seed, averaged over the eight-level grid. Read from the matched-loss record in `results/method_out.json`.

| Seed | Mean top Hessian eigenvalue difference | Mean validation ECE difference |
|---|---|---|
| 0 | -15.62 | -0.0108 |
| 1 | -13.34 | -0.0096 |
| 2 | -12.16 | -0.0141 |
| Mean | -13.71 | -0.0115 |

At matched training loss, SAM's top Hessian eigenvalue is about 12 to 16 lower, and its validation ECE is about 0.010 to 0.014 lower. The grid for seed 0 shows that the validation ECE gap is largest at the lower training losses and decreases to about zero, and for some levels reverses sign, at the highest training losses. The matched comparison therefore does not show the gap is constant across training losses.

**Statistics for Artifact 2.** Table 11 gives the per-seed contrasts at epoch 60 and at epoch 5 read from the artifact's statistics record. The artifact reports no p-values, and the signs are the evidence.

Table 11. SAM minus SGD per seed, read from the artifact's statistics record. Sign pattern is written in seed order, "-" for negative, "+" for positive.

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sign pattern |
|---|---|---|---|---|---|
| Test ECE, epoch 60 | -0.0184 | -0.0155 | -0.0174 | -0.0171 | --- |
| Test NLL, epoch 60 | -0.0781 | -0.0812 | -0.0690 | -0.0761 | --- |
| Test accuracy, epoch 60 | +0.0030 | -0.0006 | +0.0012 | +0.0012 | +-+ |
| Validation ECE, epoch 5 | +0.0158 | -0.0011 | -0.0081 | +0.0022 | +-- |

The Hessian sign check found no negative top eigenvalue in any of the 84 checkpoints, so the power iteration returned the dominant curvature in each case.

## What the iteration established

On the controlled comparison, SAM at a validation-selected learning rate and radius has a lower test ECE and a lower test NLL than SGD, with paired intervals excluding zero and a consistent sign across all five seeds [ARTIFACT:art_SL9gPfE09HlN]. Its test accuracy is the same within the seed variance. The NLL advantage remains against temperature-scaled SGD, with Holm p = 0.0027. The ECE advantage over the same baseline is small and not significant after correction.

On the mechanism study, the ECE and NLL differences at epoch 60 keep their sign across three seeds, and the matched-loss comparison finds a lower top Hessian eigenvalue and a lower validation ECE for SAM [ARTIFACT:art_WOFinDmS2PoX]. Temperature scaling removes most of the test ECE and NLL gap at epoch 60, and what remains of the NLL gap is about one-fifth of the unscaled gap.

Against the hypothesis:

1. Criterion (a), lower test ECE and NLL than tuned SGD: supported by Artifact 1 with five seeds, and by Artifact 2 as summary statistics at three seeds.
2. Criterion (b), NLL advantage after temperature scaling: supported by Artifact 1 (Holm p = 0.0027). Artifact 2 shows a residual scaled NLL advantage of 0.016 NLL units on average, without a test.
3. Criterion (c), gap larger at full training than at short training: supported at 15 and 40 epochs in Artifact 1. Reversed at 5 epochs under compressed training, where SAM's ECE is higher. Artifact 2 shows no consistent sign at epoch 5.
4. Accuracy: no significant difference in either artifact, recorded as such.

Two caveats apply to every reading. SAM is less fitted than SGD at the end of both main schedules, so the calibration advantage may in part reflect a less-fitted model; the matched-loss comparison in Artifact 2 addresses this only at the training losses the runs reached. And the two artifacts use different learning rates, one tuned and one fixed, so their numbers are not comparable.

**Correction (Iteration 2).** Three statements in this section are changed by the Iteration 2 measurements. (i) The calibrated ECE advantage was computed as uncalibrated SAM against temperature-scaled SGD. Like-for-like, the paired difference is +0.00054 with an interval of -0.00291 to +0.00399 (Table 20), so no calibrated ECE advantage is supported. (ii) The conclusion on criterion (a) is conditional. It holds in direction in Iteration 2 (Tables 13 and 17), but the size of the accuracy difference depends on the SGD learning rate. At the validation-selected learning rate of 0.001 the accuracy difference is +0.96 points (Table 13). (iii) The reversal at 5 epochs came from separately trained compressed schedules. Within one training run the test ECE gap at 5 epochs is -0.0133 (Table 23).

## Failed directions and dropped work

1. SAM at learning rate 0.03 with radius 0.2 failed to train (training loss 2.302, validation accuracy 0.1305) and was excluded from selection. This is a failed setting, not a finding about SAM's calibration.
2. The 60-epoch extension of Artifact 1 was not run. The artifact reports that SAM is under-fit at epoch 40 and that this extension would test it. The question stays open.
3. The early-stopping test of criterion (c) was replaced by compressed schedules, which are separate full training runs. The record keeps this change. The 40-epoch run's own epoch 5 and epoch 15 values are in Table 2 and differ from the compressed runs at the same epoch count.

## Designed but not run

1. Check of prior work (the third direction set for this iteration): designed to confirm whether any prior work runs SAM against temperature-scaled SGD on a small CNN, which baselines and radius ranges would be expected, and which calibration metric definitions to use. It was dropped at the artifact stage: the artifact stage returned no artifact for its design. This is not an experimental result, and it records no null or failed-direction finding. It remains open.

# Iteration 2

## Why this iteration ran what it ran

The review feedback on the Iteration 1 report had five major points and three further points. Four of them decided what this iteration measured. The first was that the calibrated ECE comparison compared uncalibrated SAM with temperature-scaled SGD, so the paired temperature-scaled comparison was missing. The second was that the primary conclusion on criterion (a) was not scoped to two confounds: the fit gap between the optimisers at epoch 40, and the selected learning rates, which were at the lower edge of the grid. The third was that criterion (c) rested on separately trained compressed schedules, which do not follow one training trajectory. The fourth was that the related-work comparison and the nearest published results were not named in the report. The points were corrections of wording and of one stated ratio.

The hypothesis update kept the NLL claim, dropped the raw ECE claim as a calibration gain, and named the next test: whether the NLL gain holds at matched training loss, and with learning rates away from the grid edge. The update also named the checks in order: two further runs on the Iteration 1 setting, a matched-loss comparison with a temperature-scaled control, and one per-epoch read of a single 40-epoch trajectory.

Three artifacts were run to address those points. Each design choice is recorded here, because the choice decides what the result can and cannot say.

1. **Artifact 3 (art_v2M5ZOOHpgid) is a re-run of the primary comparison with the learning rates tuned on validation NLL over a grid that extends below the previous edge.** The Iteration 1 grid for the SGD learning rate was {0.01, 0.03, 0.1}, and it selected its lower edge. This artifact used {0.0003, 0.001, 0.003, 0.01, 0.03} for SGD and {0.001, 0.003, 0.01, 0.03} for SAM, so that the selected values lie inside the grid. It keeps five paired seeds, the Iteration 1 architecture and split, and the same temperature-scaled control, and it reports the final training loss beside every calibration row [ARTIFACT:art_v2M5ZOOHpgid].
2. **Artifact 4 (art_7yu4oYgjmi55) matches training loss directly.** Artifact 1 showed that SAM ends with a much higher training loss than SGD, so an NLL gap could be a fit gap. This artifact compares SAM at 40 epochs with SGD at the epoch count whose training loss matches SAM's within 0.01. We call the training loss the mean loss on the full training set, computed after the epoch with batch normalisation in inference mode. The matching epoch was chosen on seed 100, and the SGD learning rate was fixed at 0.01 so that only the training length differs between the conditions [ARTIFACT:art_7yu4oYgjmi55]. The design has a known weakness, stated in its caveats: Artifact 3 found SGD at learning rate 0.01 to be the worst of its five SGD rates on validation NLL, so this baseline is weaker than the one Artifact 3 selected.
3. **Evaluation 1 (art_V0gOM__yRf8b) reads the training-length trajectory from outputs the run already holds.** It trains nothing and computes no logits again. It reports the SAM-minus-SGD gap at each evaluated epoch inside each source, and it keeps the two sources separate [ARTIFACT:art_V0gOM__yRf8b]. The compressed-schedule comparison of Iteration 1 is replaced by its within-run checkpoints, which are read from the same training run.

Every number in this section was read from the artifacts' outputs or computed again from their per-run records, in this iteration. Where a figure is taken from an artifact's own summary and was not computed again, the text says so.

## Artifact 3: the primary comparison with learning rates tuned off the grid edge

**Setup.** The network, data split, batch size, momentum, weight decay and cosine schedule are those of Artifact 1. Each condition is trained for 40 epochs. Learning rates and the SAM radius were selected on tuning seed 101 by lowest final validation NLL, and the test set was not used for selection [ARTIFACT:art_v2M5ZOOHpgid]. Table 12 gives the full tuning grid. The selected settings are SGD at learning rate 0.001 and SAM at learning rate 0.01 with radius 0.1. Neither lies on an edge of its grid.

Table 12. Tuning grid on seed 101, 40 epochs, final-epoch values, validation NLL. Read from `results/tuning_table.csv`. The selection rule is lowest final validation NLL.

| Condition | Learning rate | rho | Validation NLL | Validation accuracy | Final training loss |
|---|---|---|---|---|---|
| SGD | 0.0003 | none | 0.24995 | 0.9113 | 0.23453 |
| SGD | 0.001 | none | 0.21415 | 0.9233 | 0.12430 |
| SGD | 0.003 | none | 0.23818 | 0.9258 | 0.03788 |
| SGD | 0.01 | none | 0.27298 | 0.9288 | 0.00871 |
| SGD | 0.03 | none | 0.26651 | 0.9320 | 0.00608 |
| SAM | 0.001 | 0.05 | 0.22085 | 0.9195 | 0.19937 |
| SAM | 0.001 | 0.1 | 0.24760 | 0.9077 | 0.24128 |
| SAM | 0.001 | 0.2 | 0.30157 | 0.8864 | 0.30399 |
| SAM | 0.003 | 0.05 | 0.18776 | 0.9321 | 0.12601 |
| SAM | 0.003 | 0.1 | 0.20577 | 0.9260 | 0.17906 |
| SAM | 0.003 | 0.2 | 0.24244 | 0.9117 | 0.23560 |
| SAM | 0.01 | 0.05 | 0.19161 | 0.9336 | 0.04923 |
| SAM | 0.01 | 0.1 | 0.18244 | 0.9331 | 0.11087 |
| SAM | 0.01 | 0.2 | 0.20774 | 0.9273 | 0.18431 |
| SAM | 0.03 | 0.05 | 0.21143 | 0.9313 | 0.02896 |
| SAM | 0.03 | 0.1 | 0.18722 | 0.9335 | 0.07043 |
| SAM | 0.03 | 0.2 | 0.19445 | 0.9289 | 0.14705 |

The SGD learning rate that Iteration 1 selected, 0.01, is the fourth of five in this grid, and its validation NLL is 0.273, against 0.214 at 0.001. The Iteration 1 baseline was therefore not the best SGD setting on validation NLL. That is the first finding of this iteration, and it is recorded as a correction to the Iteration 1 reading of the SGD baseline in the corrections section at the end.

**Paired contrasts.** Table 13 gives the twelve-row paired family, with five seeds, paired t-intervals at df = 4, and Holm correction over the family. Figure 3 draws the NLL rows and Figure 5 the ECE rows.

Table 13. Paired differences over five seeds at 40 epochs, SAM minus the baseline, from `results/analysis_tables.json`. The family has twelve contrasts; Holm p is over that family. The two rows marked TS-SAM compare temperature-scaled SAM with temperature-scaled SGD.

| Contrast | Mean difference | 95% CI | p (two-sided) | Holm p | Holm, 0.05 |
|---|---|---|---|---|---|
| SAM  minus  SGD: accuracy | +0.00958 | [+0.00649, +0.01267] | 0.0010 | 0.0070 | yes |
| SAM  minus  SGD: ECE | -0.00500 | [-0.00786, -0.00213] | 0.0084 | 0.0336 | yes |
| SAM  minus  SGD: NLL | -0.03561 | [-0.04275, -0.02848] | 0.0002 | 0.0016 | yes |
| SAM  minus  SGD: Brier | -0.01575 | [-0.01845, -0.01304] | 8.6e-05 | 0.0010 | yes |
| SAM  minus  SGD: ACE | -0.00598 | [-0.00891, -0.00305] | 0.0048 | 0.0240 | yes |
| SAM  minus  TS-SGD: accuracy | +0.00958 | [+0.00649, +0.01267] | 0.0010 | 0.0070 | yes |
| SAM  minus  TS-SGD: ECE | +0.00199 | [-0.00028, +0.00427] | 0.0718 | 0.2154 | no |
| SAM  minus  TS-SGD: NLL | -0.03259 | [-0.03958, -0.02561] | 0.0002 | 0.0018 | yes |
| SAM  minus  TS-SGD: Brier | -0.01524 | [-0.01789, -0.01260] | 8.9e-05 | 0.0010 | yes |
| SAM  minus  TS-SGD: ACE | +0.00128 | [-0.00185, +0.00440] | 0.3208 | 0.3208 | no |
| TS-SAM  minus  TS-SGD: ECE | +0.00199 | [-0.00028, +0.00427] | 0.0718 | 0.2154 | no |
| TS-SAM  minus  TS-SGD: NLL | -0.03259 | [-0.03958, -0.02561] | 0.0002 | 0.0018 | yes |

[FIGURE:fig3]

Figure 3. Paired NLL differences at 40 epochs. Each dot is a mean over five paired seeds and each bar is a 95% t-interval (df = 4). A line marks zero. Row labels give the artifact number: Artifact 1 (art_SL9gPfE09HlN), Artifact 2 (art_WOFinDmS2PoX, its 60-epoch run read in Evaluation 1), Artifact 3 (art_v2M5ZOOHpgid) and Artifact 4 (art_7yu4oYgjmi55). Rows marked TS compare temperature-scaled models.

[FIGURE:fig5]

Figure 5. Paired ECE differences, drawn from the same rows as Figure 3, with the same intervals and the same labels. The ECE rows in Figure 5 are read against the zero line; the row for Artifact 3, temperature-scaled SAM minus temperature-scaled SGD crosses it.

**Shuffled-label control.** Table 14 gives the shuffled-label control run in which the test labels are shuffled. Accuracy does not separate the conditions under shuffling (mean difference -0.00032, p = 0.63). NLL and Brier do separate the conditions under shuffling (NLL shuffled-label control difference -0.22405, p = 0.009, Holm p = 0.036). The shuffled-label control therefore does not isolate label information for the proper scoring rules. The artifact reports this as a limitation, and so does this record: the NLL and Brier gaps are not, by themselves, evidence that SAM uses the labels better.

Table 14. Shuffled-label control, test labels shuffled, SAM minus SGD at 40 epochs, five seeds. Read from `analysis_tables.json` (shuffled-label control rows).

| Contrast | Mean difference | 95% CI | p (two-sided) | Holm p |
|---|---|---|---|---|
| acc | -0.00032 | [-0.00200, +0.00136] | 0.6250 | 0.6250 |
| nll | -0.22405 | [-0.35504, -0.09306] | 0.0090 | 0.0359 |
| ece | +0.00159 | [-0.00058, +0.00376] | 0.1117 | 0.3219 |
| brier | +0.00283 | [+0.00174, +0.00392] | 0.0020 | 0.0099 |
| ace | +0.00160 | [-0.00055, +0.00375] | 0.1073 | 0.3219 |
| PLACEBO SGD  minus  SGD (same model) | +0.00000 | [+0.00000, +0.00000] | nan | n/a |

**Temperature and training loss.** Table 15 gives the per-condition means over the five seeds and the fitted temperatures. The fitted temperature for SGD is above 1, which is the usual overconfidence of a trained network, and the fitted temperature for SAM is below 1, so SAM's confidence falls short of its accuracy on the test set. Both conditions have a final training loss in the same range, about 0.11 to 0.12. The optimiser that fits less is now SGD, which is the reverse of Iteration 1, where SGD reached a training loss of 0.008 and SAM of 0.107. The fit gap therefore depends on the SGD learning rate, not on the optimiser alone.

Table 15. Per-condition means over five seeds on the test set, with the fitted temperatures and final training loss. Means are computed from the per-seed records in `analysis_tables.json`.

| Condition | Accuracy | ECE | ACE | NLL | Brier | Mean T | Mean final training loss |
|---|---|---|---|---|---|---|---|
| SGD, raw | 0.9139 | 0.0125 | 0.0116 | 0.2482 | 0.1269 | 1.049 | 0.1219 |
| SGD, temperature-scaled | 0.9139 | 0.0085 | 0.0076 | 0.2466 | 0.1266 | n/a | n/a |
| SAM, raw | 0.9235 | 0.0075 | 0.0056 | 0.2126 | 0.1111 | 0.938 | 0.1096 |
| SAM, temperature-scaled | 0.9235 | 0.0105 | 0.0089 | 0.2140 | 0.1113 | n/a | n/a |

**Reading.** On the primary family, SAM's accuracy is higher than SGD's by 0.96 points, with an interval of +0.65 to +1.27 points and a Holm p of 0.007. The Iteration 1 accuracy null of +0.15 points was therefore specific to the SGD learning rate of 0.01. Raw ECE is lower for SAM (Holm p = 0.034), and the NLL advantage holds against both raw and temperature-scaled SGD (Holm p = 0.0016 raw, 0.0018 temperature-scaled). The ECE advantage over temperature-scaled SGD is not significant. Its mean is +0.0020 with an interval of -0.0003 to +0.0043, so temperature-scaled SAM is, if anything, slightly worse on ECE. That is the like-for-like result on ECE. In this artifact the rows labelled SAM minus TS-SGD and TS-SAM minus TS-SGD have identical per-seed ECE and NLL differences, so they are one comparison, computed on temperature-scaled SAM. The result matches the like-for-like comparison of Artifact 1 in Table 20: once both conditions are temperature-scaled, the ECE difference is zero.

## Artifact 4: SAM against SGD at matched training loss

**Setup and selection.** Both optimisers use learning rate 0.01, a 0.5 cosine schedule and the same network. The SAM radius 0.1 was chosen on seed 101 by lowest validation NLL, which gave 0.1998 at radius 0.1 against 0.2104 at 0.05 and 0.2299 at 0.2 [ARTIFACT:art_7yu4oYgjmi55]. The matching epoch for SGD was chosen on seed 100. The SAM 40-epoch training loss on that seed was 0.1121. The SGD grid over 10 to 20 epochs gave the first epoch within 0.01 at epoch 10, with a gap of -0.0009, so the matched control is a 10-epoch SGD run at the edge of the grid. Table 16 gives the matched-loss values per seed and Table 17 gives the primary comparison.

Table 16. Matched training loss, per seed. Training losses for SAM at 40 epochs and for SGD at the matched epoch, with the gap between them. Read from `results/matched_loss_table.json`.

| Seed | SAM 40-epoch training loss | SGD 10-epoch training loss | Gap, SAM minus SGD 10 | SGD 40-epoch training loss |
|---|---|---|---|---|
| 0 | 0.10183 | 0.10497 | +0.00314 | 0.00804 |
| 1 | 0.09941 | 0.10439 | +0.00498 | 0.00743 |
| 2 | 0.11010 | 0.11120 | +0.00110 | 0.00852 |
| 3 | 0.10175 | 0.10723 | +0.00548 | 0.00776 |
| 4 | 0.10265 | 0.10302 | +0.00038 | 0.00738 |

**Primary comparison.** Table 17 gives SAM at 40 epochs minus SGD at 10 epochs. The mean differences and intervals are computed again here from the per-run records, and the Holm p values are those the artifact's README reports over its fourteen-test family. The computed again values agree with the artifact to the reported digits.

Table 17. SAM at 40 epochs minus SGD at 10 epochs, five paired seeds, at matched training loss. Mean and 95% t-interval (df = 4) are computed again from `results/runs/`; Holm p is the artifact's value over its fourteen-test family.

| Metric | Mean difference | 95% CI | p (two-sided, recomputed) | Holm p (artifact README) |
|---|---|---|---|---|
| Accuracy | +0.00246 | [+0.00069, +0.00423] | 0.0183 | 0.055 |
| ECE, raw (15 bins) | -0.00761 | [-0.00910, -0.00612] | 0.0001 | 0.0018 |
| ECE, temperature-scaled | -0.00013 | [-0.00159, +0.00134] | 0.8225 | 0.97 |
| NLL, raw | -0.01254 | [-0.01516, -0.00992] | 0.0002 | 0.0019 |
| NLL, temperature-scaled | -0.00818 | [-0.01103, -0.00533] | 0.0013 | 0.0094 |
| Brier, raw | -0.00538 | [-0.00627, -0.00449] | 7.5e-05 | 0.001 |
| ACE, raw | -0.00109 | [-0.00130, -0.00087] | 0.0001 | 0.0018 |

At matched training loss, the raw ECE and NLL advantages of SAM persist, and so does the NLL advantage after temperature scaling. The temperature-scaled ECE difference is zero within its interval. The accuracy advantage is +0.25 points, with an interval of +0.07 to +0.42 points and a Holm p of 0.055, which the artifact reports as not significant after correction. Matching training loss therefore means the fit gap cannot account for the NLL result in this design. It does not remove the dependence on the SGD learning rate, because the SGD condition here is the weaker baseline described above.

**Reference contrast.** Table 18 compares SAM at 40 epochs with SGD at 40 epochs. The SGD condition at 40 epochs reaches a training loss near 0.008, against about 0.10 for SAM and for the matched SGD condition. Its raw ECE is far higher, and its fitted temperature is near 1.9. This pair is the same comparison as Artifact 1 at learning rate 0.01, and it is reported here to show that the matched comparison changes the size of the effect and does not change its direction.

Table 18. SAM at 40 epochs minus SGD at 40 epochs, five paired seeds, computed again from `results/runs/`.

| Metric | Mean difference | 95% CI | p (two-sided) |
|---|---|---|---|
| Accuracy | +0.00238 | [-0.00080, +0.00556] | 0.1061 |
| ECE, raw | -0.03726 | [-0.04147, -0.03304] | 1.6e-05 |
| ECE, temperature-scaled | -0.00217 | [-0.00445, +0.00011] | 0.0575 |
| NLL, raw | -0.10086 | [-0.11040, -0.09132] | 8.0e-06 |
| NLL, temperature-scaled | -0.02102 | [-0.02664, -0.01541] | 0.0005 |

**Secondary comparison.** The artifact also compares SAM at 40 epochs with SGD at the epoch of matched loss, read from SGD's own run at that epoch, rather than with a separate 10-epoch run. The artifact's README reports the accuracy difference as +0.0102, the raw ECE difference as -0.0188 and the raw NLL difference as -0.0414, with the temperature-scaled ECE difference as -0.0007 and not significant. These values are the artifact's own summary, and they were not computed again here. The artifact states that models taken before the schedule ends are not final models at a high learning rate, so this comparison is weaker than the primary one and is not used in the reading.

**Check.** The artifact computed again the primary contrasts from the saved 16-bit logits with separate code. Table 19 gives the agreement. The differences are below 1.2e-4, which the artifact attributes to the 16-bit saved logits against 32-bit computation at run time.

Table 19. Check of the Artifact 4 primary contrasts, SAM at 40 epochs minus SGD at 10 epochs, from saved logits.

| Quantity | Recomputed mean difference | Stored mean difference | Absolute difference |
|---|---|---|---|
| accuracy | +0.002580 | +0.002460 | 1.20e-04 |
| ece_raw | -0.007662 | -0.007610 | 5.14e-05 |
| nll_raw | -0.012533 | -0.012537 | 3.71e-06 |
| nll_ts | -0.008173 | -0.008177 | 4.54e-06 |

**Reading.** The matched-loss comparison supports three conclusions at this learning rate: the raw ECE advantage, the raw NLL advantage, and the temperature-scaled NLL advantage, each with an interval that excludes zero. It does not support a temperature-scaled ECE advantage. Five seeds cannot reach an exact p below 0.0625, so none of these results can be significant at 0.05 on an exact test. The t-tests in the tables are parametric and are reported as such.

## Evaluation 1: the calibration gap across training length

**Setup.** This evaluation reads two existing trajectories and does not pool them. Source 1 is Artifact 1: five paired seeds at 40 epochs, with 5- and 15-epoch compressed schedules, which are separate training runs and not checkpoints of the 40-epoch run. The evaluation also reads Artifact 1's within-run checkpoints at epochs 5 and 15 of the 40-epoch run. Source 2 is Artifact 2: three paired seeds over 60 epochs with per-epoch validation values and a test read at epochs 5 and 60 [ARTIFACT:art_V0gOM__yRf8b]. The two sources use different learning rates (0.01 and 0.03), so their numbers are not comparable across sources.

**Gap table.** The gap table in full follows. It reports the SAM-minus-SGD gap, its t-interval, the raw p-value and the Holm p-value where the artifact computed one, and the sign of the gap for each seed. The artifact's bootstrap intervals and notes are in its own `results/gap_table.csv`, which is kept with the artifact.

Table 23. Gap table, SAM minus SGD. Source 1 is Artifact 1 and source 2 is Artifact 2. Per-seed signs are written in seed order. Read from `results/gap_table.csv`.

| Source | Comparison | Timepoint | n | Mean gap | 95% t CI | p (unadj.) | Holm p | Per-seed signs |
|---|---|---|---|---|---|---|---|---|
| S1 | SAM-SGD test accuracy | epoch 5 (compressed) | 5 | -0.01830 | [-0.02017, -0.01643] | 1.1e-05 | 3.3e-05 | ----- |
| S1 | SAM-SGD test accuracy | epoch 15 (compressed) | 5 | -0.00978 | [-0.01234, -0.00722] | 0.0004 | 0.0009 | ----- |
| S1 | SAM-SGD test accuracy | epoch 40 (40-epoch run) | 5 | +0.00146 | [-0.00071, +0.00363] | 0.1348 | 0.1348 | -++++ |
| S1 | widening: gap(40)  minus  gap(15) test accuracy | 40 vs 15 | 5 | +0.01124 | [+0.00950, +0.01298] | 5.7e-05 | n/a | +++++ |
| S1 | widening: gap(40)  minus  gap(5) test accuracy | 40 vs 5 | 5 | +0.01976 | [+0.01682, +0.02270] | 4.9e-05 | n/a | +++++ |
| S1 | SAM-SGD test ece | epoch 5 (compressed) | 5 | +0.00574 | [+0.00177, +0.00970] | 0.0159 | 0.0159 | +++++ |
| S1 | SAM-SGD test ece | epoch 15 (compressed) | 5 | -0.01317 | [-0.01625, -0.01009] | 0.0003 | 0.0006 | ----- |
| S1 | SAM-SGD test ece | epoch 40 (40-epoch run) | 5 | -0.03778 | [-0.04145, -0.03412] | 8.8e-06 | 2.6e-05 | ----- |
| S1 | widening: gap(40)  minus  gap(15) test ece | 40 vs 15 | 5 | -0.02462 | [-0.02861, -0.02062] | 6.8e-05 | n/a | ----- |
| S1 | widening: gap(40)  minus  gap(5) test ece | 40 vs 5 | 5 | -0.04352 | [-0.04940, -0.03764] | 3.3e-05 | n/a | ----- |
| S1 | SAM-SGD test nll | epoch 5 (compressed) | 5 | +0.04514 | [+0.03915, +0.05113] | 3.1e-05 | 6.2e-05 | +++++ |
| S1 | SAM-SGD test nll | epoch 15 (compressed) | 5 | +0.00650 | [+0.00323, +0.00978] | 0.0053 | 0.0053 | +++++ |
| S1 | SAM-SGD test nll | epoch 40 (40-epoch run) | 5 | -0.09796 | [-0.10918, -0.08673] | 1.7e-05 | 5.2e-05 | ----- |
| S1 | widening: gap(40)  minus  gap(15) test nll | 40 vs 15 | 5 | -0.10446 | [-0.11323, -0.09570] | 5.0e-06 | n/a | ----- |
| S1 | widening: gap(40)  minus  gap(5) test nll | 40 vs 5 | 5 | -0.14310 | [-0.15440, -0.13181] | 3.9e-06 | n/a | ----- |
| S1 | SAM-SGD test brier | epoch 5 (compressed) | 5 | +0.02364 | [+0.02096, +0.02632] | 1.7e-05 | 5.0e-05 | +++++ |
| S1 | SAM-SGD test brier | epoch 15 (compressed) | 5 | +0.00920 | [+0.00751, +0.01088] | 0.0001 | 0.0002 | +++++ |
| S1 | SAM-SGD test brier | epoch 40 (40-epoch run) | 5 | -0.01362 | [-0.01792, -0.00933] | 0.0009 | 0.0009 | ----- |
| S1 | widening: gap(40)  minus  gap(15) test brier | 40 vs 15 | 5 | -0.02282 | [-0.02587, -0.01977] | 3.2e-05 | n/a | ----- |
| S1 | widening: gap(40)  minus  gap(5) test brier | 40 vs 5 | 5 | -0.03726 | [-0.04153, -0.03299] | 1.7e-05 | n/a | ----- |
| S1 | SAM-SGD test ace | epoch 5 (compressed) | 5 | +0.00665 | [+0.00417, +0.00913] | 0.0017 | 0.0017 | +++++ |
| S1 | SAM-SGD test ace | epoch 15 (compressed) | 5 | -0.01303 | [-0.01635, -0.00971] | 0.0004 | 0.0008 | ----- |
| S1 | SAM-SGD test ace | epoch 40 (40-epoch run) | 5 | -0.03861 | [-0.04282, -0.03440] | 1.4e-05 | 4.2e-05 | ----- |
| S1 | widening: gap(40)  minus  gap(15) test ace | 40 vs 15 | 5 | -0.02558 | [-0.03047, -0.02070] | 0.0001 | n/a | ----- |
| S1 | widening: gap(40)  minus  gap(5) test ace | 40 vs 5 | 5 | -0.04527 | [-0.04920, -0.04133] | 5.7e-06 | n/a | ----- |
| S1 | SAM-SGD test aurc | epoch 5 (compressed) | 5 | +0.00449 | [+0.00386, +0.00513] | 3.9e-05 | 0.0001 | +++++ |
| S1 | SAM-SGD test aurc | epoch 15 (compressed) | 5 | +0.00159 | [+0.00130, +0.00189] | 0.0001 | 0.0002 | +++++ |
| S1 | SAM-SGD test aurc | epoch 40 (40-epoch run) | 5 | -0.00152 | [-0.00212, -0.00092] | 0.0021 | 0.0021 | ----- |
| S1 | widening: gap(40)  minus  gap(15) test aurc | 40 vs 15 | 5 | -0.00311 | [-0.00349, -0.00273] | 2.1e-05 | n/a | ----- |
| S1 | widening: gap(40)  minus  gap(5) test aurc | 40 vs 5 | 5 | -0.00601 | [-0.00668, -0.00534] | 1.5e-05 | n/a | ----- |
| S1 | SAM-SGD test ece (within-run checkpoint) | epoch 5 of 40-epoch run | 5 | -0.01333 | [-0.02156, -0.00509] | 0.0109 | n/a | ----- |
| S1 | SAM-SGD test ece (within-run checkpoint) | epoch 15 of 40-epoch run | 5 | -0.02899 | [-0.03561, -0.02238] | 0.0003 | n/a | ----- |
| S1 | SAM-SGD test nll (within-run checkpoint) | epoch 5 of 40-epoch run | 5 | -0.01041 | [-0.02934, +0.00851] | 0.2013 | n/a | --+-- |
| S1 | SAM-SGD test nll (within-run checkpoint) | epoch 15 of 40-epoch run | 5 | -0.04452 | [-0.05455, -0.03449] | 0.0002 | n/a | ----- |
| S1 | SAM  minus  TS-SGD test accuracy | epoch 40 (40-epoch run) | 5 | +0.00146 | [-0.00071, +0.00363] | 0.1348 | n/a | -++++ |
| S1 | SAM  minus  TS-SGD test ece | epoch 40 (40-epoch run) | 5 | -0.00307 | [-0.00590, -0.00024] | 0.0397 | n/a | ----- |
| S1 | SAM  minus  TS-SGD test nll | epoch 40 (40-epoch run) | 5 | -0.02088 | [-0.02662, -0.01514] | 0.0005 | n/a | ----- |
| S1 | SAM  minus  TS-SGD test ace | epoch 40 (40-epoch run) | 5 | -0.00235 | [-0.00606, +0.00135] | 0.1530 | n/a | --+-- |
| S1 | SAM  minus  TS-SGD test brier | epoch 40 (40-epoch run) | 5 | -0.00494 | [-0.00775, -0.00212] | 0.0082 | n/a | ----- |
| S1 | SAM  minus  TS-SGD test aurc | epoch 40 (40-epoch run) | 5 | -0.00151 | [-0.00209, -0.00093] | 0.0019 | n/a | ----- |
| S2 | SAM-SGD val_ece | epoch 1 | 3 | -0.00439 | [-0.02314, +0.01437] | 0.4203 | n/a | -+- |
| S2 | SAM-SGD val_ece | epoch 2 | 3 | +0.00387 | [-0.00924, +0.01698] | 0.3320 | n/a | +++ |
| S2 | SAM-SGD val_ece | epoch 5 | 3 | +0.00220 | [-0.02829, +0.03270] | 0.7855 | n/a | +-- |
| S2 | SAM-SGD val_ece | epoch 10 | 3 | -0.01249 | [-0.02417, -0.00080] | 0.0442 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 15 | 3 | -0.02493 | [-0.03264, -0.01721] | 0.0051 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 20 | 3 | -0.02755 | [-0.02998, -0.02512] | 0.0004 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 25 | 3 | -0.03257 | [-0.04470, -0.02043] | 0.0074 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 30 | 3 | -0.03213 | [-0.03700, -0.02726] | 0.0012 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 35 | 3 | -0.02970 | [-0.03707, -0.02232] | 0.0033 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 40 | 3 | -0.02602 | [-0.02813, -0.02390] | 0.0004 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 45 | 3 | -0.02085 | [-0.02559, -0.01610] | 0.0028 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 50 | 3 | -0.01912 | [-0.02441, -0.01384] | 0.0041 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 55 | 3 | -0.01860 | [-0.02559, -0.01160] | 0.0076 | n/a | --- |
| S2 | SAM-SGD val_ece | epoch 60 | 3 | -0.01861 | [-0.02390, -0.01333] | 0.0043 | n/a | --- |
| S2 | widening: gap(60)  minus  gap(1) val_ece | 60 vs 1 | 3 | -0.01423 | [-0.02891, +0.00046] | 0.0530 | n/a | --- |
| S2 | widening: gap(60)  minus  gap(5) val_ece | 60 vs 5 | 3 | -0.02081 | [-0.05415, +0.01252] | 0.1151 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 1 | 3 | +0.03374 | [+0.00118, +0.06629] | 0.0468 | n/a | +++ |
| S2 | SAM-SGD val_nll | epoch 2 | 3 | +0.01627 | [+0.01156, +0.02098] | 0.0045 | n/a | +++ |
| S2 | SAM-SGD val_nll | epoch 5 | 3 | -0.00073 | [-0.03996, +0.03850] | 0.9435 | n/a | +-- |
| S2 | SAM-SGD val_nll | epoch 10 | 3 | -0.00627 | [-0.01848, +0.00594] | 0.1577 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 15 | 3 | -0.04193 | [-0.08781, +0.00395] | 0.0590 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 20 | 3 | -0.05927 | [-0.06191, -0.05664] | 0.0001 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 25 | 3 | -0.08660 | [-0.14196, -0.03125] | 0.0214 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 30 | 3 | -0.09527 | [-0.12979, -0.06074] | 0.0070 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 35 | 3 | -0.09487 | [-0.13568, -0.05406] | 0.0098 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 40 | 3 | -0.08329 | [-0.10923, -0.05734] | 0.0052 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 45 | 3 | -0.07575 | [-0.09514, -0.05636] | 0.0035 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 50 | 3 | -0.07074 | [-0.09756, -0.04392] | 0.0077 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 55 | 3 | -0.06917 | [-0.09720, -0.04115] | 0.0088 | n/a | --- |
| S2 | SAM-SGD val_nll | epoch 60 | 3 | -0.06903 | [-0.09532, -0.04274] | 0.0077 | n/a | --- |
| S2 | widening: gap(60)  minus  gap(1) val_nll | 60 vs 1 | 3 | -0.10277 | [-0.13743, -0.06810] | 0.0061 | n/a | --- |
| S2 | widening: gap(60)  minus  gap(5) val_nll | 60 vs 5 | 3 | -0.06830 | [-0.13322, -0.00338] | 0.0455 | n/a | --- |
| S2 | val_ece gap vs SGD training loss: per-seed OLS slope | epochs 1-60 (14 logged) | 3 | +0.06531 | [+0.02656, +0.10405] | n/a | n/a | +++ |
| S2 | val_ece gap vs SAM training loss: per-seed OLS slope | epochs 1-60 (14 logged) | 3 | +0.05633 | [+0.02643, +0.08623] | n/a | n/a | +++ |
| S2 | val_nll gap vs SGD training loss: per-seed OLS slope | epochs 1-60 (14 logged) | 3 | +0.28912 | [+0.18641, +0.39183] | n/a | n/a | +++ |
| S2 | val_nll gap vs SAM training loss: per-seed OLS slope | epochs 1-60 (14 logged) | 3 | +0.26532 | [+0.15616, +0.37448] | n/a | n/a | +++ |
| S2 | SAM-SGD test_acc | epoch 5 (test endpoint) | 3 | +0.00123 | [-0.01338, +0.01585] | 0.7513 | n/a | -++ |
| S2 | SAM-SGD test_ece | epoch 5 (test endpoint) | 3 | -0.00417 | [-0.02677, +0.01843] | 0.5106 | n/a | +-- |
| S2 | SAM-SGD test_nll | epoch 5 (test endpoint) | 3 | -0.00233 | [-0.04064, +0.03598] | 0.8181 | n/a | +-- |
| S2 | SAM-SGD test_brier | epoch 5 (test endpoint) | 3 | -0.00194 | [-0.02410, +0.02022] | 0.7422 | n/a | +-- |
| S2 | SAM-SGD test_ace | epoch 5 (test endpoint) | 3 | -0.00311 | [-0.02827, +0.02204] | 0.6479 | n/a | +-- |
| S2 | SAM raw  minus  TS-SGD ece | epoch 5 (test endpoint) | 3 | +0.00132 | [-0.01318, +0.01581] | 0.7331 | n/a | +-+ |
| S2 | SAM raw  minus  TS-SGD nll | epoch 5 (test endpoint) | 3 | -0.00023 | [-0.03752, +0.03706] | 0.9813 | n/a | +-+ |
| S2 | SAM-TS  minus  SGD-TS ECE | epoch 5 (test endpoint) | 3 | +0.00050 | [-0.00984, +0.01084] | 0.8553 | n/a | +-+ |
| S2 | SAM-TS  minus  SGD-TS NLL | epoch 5 (test endpoint) | 3 | -0.00147 | [-0.03530, +0.03236] | 0.8692 | n/a | +-- |
| S2 | SAM-SGD test_acc | epoch 60 (test endpoint) | 3 | +0.00120 | [-0.00327, +0.00567] | 0.3675 | n/a | +-+ |
| S2 | SAM-SGD test_ece | epoch 60 (test endpoint) | 3 | -0.01710 | [-0.02073, -0.01347] | 0.0024 | n/a | --- |
| S2 | SAM-SGD test_nll | epoch 60 (test endpoint) | 3 | -0.07613 | [-0.09189, -0.06037] | 0.0023 | n/a | --- |
| S2 | SAM-SGD test_brier | epoch 60 (test endpoint) | 3 | -0.01009 | [-0.01474, -0.00543] | 0.0113 | n/a | --- |
| S2 | SAM-SGD test_ace | epoch 60 (test endpoint) | 3 | -0.01763 | [-0.02266, -0.01260] | 0.0044 | n/a | --- |
| S2 | SAM raw  minus  TS-SGD ece | epoch 60 (test endpoint) | 3 | +0.01806 | [+0.01459, +0.02152] | 0.0020 | n/a | +++ |
| S2 | SAM raw  minus  TS-SGD nll | epoch 60 (test endpoint) | 3 | +0.00342 | [-0.01002, +0.01685] | 0.3880 | n/a | +-+ |
| S2 | SAM-TS  minus  SGD-TS ECE | epoch 60 (test endpoint) | 3 | -0.00239 | [-0.00861, +0.00382] | 0.2395 | n/a | --- |
| S2 | SAM-TS  minus  SGD-TS NLL | epoch 60 (test endpoint) | 3 | -0.01641 | [-0.02987, -0.00294] | 0.0345 | n/a | --- |
| S2 | SAM-SGD val_ece largest mean advantage (extremum over logged epochs) | epoch 25 | 3 | -0.03257 | n/a | n/a | n/a |  |
| S2 | SAM-SGD val_nll largest mean advantage (extremum over logged epochs) | epoch 30 | 3 | -0.09527 | n/a | n/a | n/a |  |

**Reading of the gap table.** Accuracy shows no difference at 40 epochs in source 1 (+0.00146, interval -0.00071 to +0.00363) and at 60 epochs in source 2 (+0.00120, interval -0.00327 to +0.00567). Source 1's ECE and NLL gaps move from a positive value at 5 epochs in the compressed schedule to a negative value at 40 epochs. That compressed reading is the one the Iteration 1 record gave. The within-run checkpoint reads the opposite sign at 5 epochs. The within-run test ECE gap is -0.01332 (interval -0.02156 to -0.00509) at epoch 5, -0.02899 at epoch 15, and -0.03778 at epoch 40. The compressed schedules therefore do not answer criterion (c). Read along a single trajectory, the gap is already negative at 5 epochs and grows to 40 epochs.

Source 2 is near zero for validation ECE at epochs 1 to 5: -0.00439 at epoch 1, +0.00387 at epoch 2, +0.00220 at epoch 5, all with intervals that include zero. The validation ECE gap reaches its largest mean advantage at epoch 25, -0.03257, and then decreases to -0.01861 at epoch 60. The validation NLL gap follows the same shape, with its largest advantage at epoch 30 (-0.09527). The evaluation's summary describes the NLL gap as near zero over epochs 1 to 5. The gap table shows otherwise at epochs 1 and 2: +0.03374 at epoch 1 (interval +0.00118 to +0.06629, raw p = 0.047) and +0.01627 at epoch 2, with SAM worse on validation NLL early in training. The table is the reading used here.

[FIGURE:fig4]

Figure 4. Validation gaps, SAM minus SGD, by training epoch in Artifact 2 (three paired seeds, learning rate 0.03). Left: ECE. Right: NLL. The line is the mean over seeds and the shaded region is the 95% t-interval (df = 2). The shaded region is wide at the early epochs and smaller from epoch 25 on, and with three seeds the shape of the trajectory is a reading of point estimates only.

**Test endpoints in source 2.** At epoch 60 the test gaps are -0.01710 for ECE (interval -0.02073 to -0.01347) and -0.07613 for NLL (interval -0.09189 to -0.06037). After both models are temperature-scaled, the NLL difference is -0.01641 (interval -0.02987 to -0.00294, raw p = 0.035) and the ECE difference is -0.00239 (interval -0.00861 to +0.00382). Temperature scaling of SGD removes almost all of the ECE gap. The NLL advantage remains, at about a fifth of its raw size, which matches the re-tuned and matched-loss readings below.

**Limits of the evaluation.** The evaluation's own caveats hold and are restated here. With three seeds the smallest exact p-value is 0.25, and with five seeds 0.0625, so the parametric p-values in the table are not evidence. The slopes of gap against training loss reflect relationship within one trajectory, not a test of overfitting, because the two conditions reach different training losses at the same epoch. The evaluation's shuffled-label control check rejects only at the smallest attainable p-value of its test.

## Re-reads of Iteration 1 outputs

Two Iteration 1 outputs were not reported in the Iteration 1 record. Both are reported here, with the previous review taken into account, and with no change to the conclusions drawn at the time.

**Pooled reliability bins, Artifact 1.** The artifact's `reliability_bins_pooled.json` [ARTIFACT:art_SL9gPfE09HlN] gives the test-set reliability bins for SGD, temperature-scaled SGD and SAM. Table 21 gives them with counts, mean confidence and accuracy per bin. The shape of the epoch-40 gap is shown in the bins. Raw SGD places 43,961 of its 50,000 pooled predictions in the top bin (0.933 to 1.000), where its mean confidence is 0.9965 and its accuracy is 0.9669, so its confidence exceeds its accuracy there by about 0.030. SAM places 37,493 predictions in that bin, with a mean confidence of 0.9922 and an accuracy of 0.9903, a gap of about 0.002. Temperature-scaled SGD places 37,858 predictions there, with a mean confidence of 0.9905 and an accuracy of 0.9865. The raw ECE advantage of SAM is therefore concentrated in the top bin: SAM is better matched there, and it places fewer predictions in that bin.

Table 21. Pooled test-set reliability bins for the three Artifact 1 models, 50,000 predictions each (five seeds of 10,000 test images). Read from `results/reliability_bins_pooled.json`.

| Bin | SGD count | SGD confidence | SGD accuracy | TS-SGD count | TS-SGD confidence | TS-SGD accuracy | SAM count | SAM confidence | SAM accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 0.000 to 0.067 | 0 | n/a | n/a | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.067 to 0.133 | 0 | n/a | n/a | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.133 to 0.200 | 0 | n/a | n/a | 0 | n/a | n/a | 1 | 0.1971 | 0.0000 |
| 0.200 to 0.267 | 0 | n/a | n/a | 11 | 0.2414 | 0.3636 | 10 | 0.2444 | 0.1000 |
| 0.267 to 0.333 | 12 | 0.3080 | 0.2500 | 67 | 0.3092 | 0.2388 | 71 | 0.3086 | 0.1972 |
| 0.333 to 0.400 | 36 | 0.3730 | 0.2778 | 183 | 0.3746 | 0.3552 | 194 | 0.3723 | 0.3505 |
| 0.400 to 0.467 | 99 | 0.4385 | 0.3333 | 410 | 0.4358 | 0.4488 | 390 | 0.4379 | 0.4077 |
| 0.467 to 0.533 | 444 | 0.5089 | 0.4505 | 942 | 0.5045 | 0.4841 | 1016 | 0.5046 | 0.4970 |
| 0.533 to 0.600 | 620 | 0.5672 | 0.5032 | 1143 | 0.5663 | 0.5687 | 1254 | 0.5665 | 0.5654 |
| 0.600 to 0.667 | 611 | 0.6338 | 0.4828 | 1163 | 0.6334 | 0.6019 | 1248 | 0.6330 | 0.6250 |
| 0.667 to 0.733 | 711 | 0.7011 | 0.5570 | 1300 | 0.7012 | 0.7085 | 1377 | 0.7011 | 0.7066 |
| 0.733 to 0.800 | 841 | 0.7672 | 0.5981 | 1484 | 0.7682 | 0.7446 | 1637 | 0.7676 | 0.7648 |
| 0.800 to 0.867 | 1054 | 0.8349 | 0.6603 | 2027 | 0.8359 | 0.8254 | 2111 | 0.8354 | 0.8153 |
| 0.867 to 0.933 | 1611 | 0.9024 | 0.7151 | 3412 | 0.9037 | 0.8757 | 3198 | 0.9033 | 0.8971 |
| 0.933 to 1.000 | 43961 | 0.9965 | 0.9669 | 37858 | 0.9905 | 0.9865 | 37493 | 0.9922 | 0.9903 |

**Loss increase and Hessian ratio, Artifact 2.** The artifact's per-epoch record [ARTIFACT:art_WOFinDmS2PoX] also holds the SAM loss increase within the perturbation radius, which is the loss rise inside the perturbation radius of each SAM step. Table 22 gives it at the epochs of Table 9 of Iteration 1, with the top Hessian eigenvalues for both optimisers. The loss increase within the radius is between 0.016 and 0.066 across the logged epochs and falls over training. The ratio of SGD's top Hessian eigenvalue to SAM's varies more than the Iteration 1 record stated. It is 2.7 at epoch 1, 3.35 at epoch 25 and 1.9 at epoch 40.

Table 22. Top Hessian eigenvalue for each optimiser and SAM loss increase within the perturbation radius, mean over three seeds at each logged epoch. Read from `results/per_epoch.json`.

| Epoch | SGD top Hessian eigenvalue | SAM top Hessian eigenvalue | Ratio SGD / SAM | SAM loss increase within the radius |
|---|---|---|---|---|
| 1 | 24.95 | 9.18 | 2.72 | 0.0662 |
| 2 | 20.69 | 6.80 | 3.04 | 0.0389 |
| 5 | 19.09 | 6.21 | 3.07 | 0.0461 |
| 10 | 15.72 | 6.60 | 2.38 | 0.0514 |
| 15 | 20.58 | 6.33 | 3.25 | 0.0354 |
| 20 | 19.19 | 6.94 | 2.76 | 0.0381 |
| 25 | 22.82 | 6.82 | 3.35 | 0.0340 |
| 30 | 25.02 | 8.01 | 3.12 | 0.0484 |
| 40 | 14.55 | 7.50 | 1.94 | 0.0262 |
| 50 | 19.88 | 7.43 | 2.68 | 0.0240 |
| 60 | 20.20 | 7.13 | 2.83 | 0.0164 |

## Reasoning on the criteria after this iteration

**Criterion (a), lower test ECE and NLL than tuned SGD.** NLL is lower for SAM in all three designs [ARTIFACT:art_SL9gPfE09HlN]: Artifact 1 (-0.098 raw against the Iteration 1 SGD), Artifact 3 (-0.036 raw against the re-tuned SGD, Holm p = 0.0016) and Artifact 4 (-0.013 raw against SGD at matched loss, Holm p = 0.0019). Raw ECE is lower in all three designs as well. The conclusion is supported in direction. Its magnitude depends on the SGD learning rate and on the training-loss match. The conclusion is therefore not stated as a single number, and the fit confound is no longer open for the NLL result at matched loss: that gap does not vanish when SGD is trained to the same loss. The confound is still open for the SGD learning rate, because Artifact 4's baseline is not the validation-selected one.

**Criterion (b), NLL advantage after temperature scaling.** The NLL advantage persists after temperature scaling in four readings: Artifact 1 (temperature-scaled SAM minus temperature-scaled SGD, -0.0196, interval excludes zero), Artifact 3 (-0.0326, Holm p = 0.0018), Artifact 4 (-0.0082, Holm p = 0.0094) and Artifact 2 at epoch 60 (-0.0164, raw p = 0.035, three seeds, point-estimate). The result is consistent across designs, and its size is between about 0.008 and 0.033 NLL units. Criterion (b) holds in this sense, and the NLL claim is kept at its original size.

**Criterion (c), a gap larger at full training than at short training.** Read along single trajectories [ARTIFACT:art_WOFinDmS2PoX], the gap grows with training. In Artifact 2 the validation ECE gap is near zero at epochs 1 to 5 and reaches -0.033 at epoch 25. In Artifact 1's within-run checkpoints the test ECE gap grows from -0.013 at epoch 5 to -0.038 at epoch 40. Two caveats limit this reading. Artifact 2's gap decreases after epoch 25 and is no longer largest at the endpoint. Artifact 1's gap is already significantly negative at 5 epochs, so the early gap is near zero in one trajectory and clearly negative in the other. Criterion (c) is therefore supported in part: the gap grows with training in both trajectories up to the point each reaches, and the early-training prediction of a near-zero gap holds in one source only.

**Criterion ruled out.** The raw ECE advantage of SAM against temperature-scaled SGD is not supported. In the like-for-like paired comparison the difference is +0.00054 in Artifact 1, +0.0020 in Artifact 3 and -0.0001 in Artifact 4, each with an interval that includes zero. This is the sub-claim the hypothesis stated in advance as a possible failure. It failed, and the record says so.

**Accuracy.** SAM's accuracy is higher in all designs, and the size depends on the SGD learning rate: +0.15 points at learning rate 0.01 in Artifact 1 (Holm p = 0.135), +0.96 points at the validation-selected 0.001 in Artifact 3 (Holm p = 0.007), and +0.25 points at matched loss in Artifact 4 (Holm p = 0.055). The Iteration 1 statement that accuracy shows no difference is therefore a statement about one SGD setting, and it is corrected in place below.

## What was learned

Four conclusions are now supported that were not before. The NLL advantage of SAM over SGD holds against temperature-scaled SGD in four readings, and it holds at matched training loss. The raw ECE advantage is a confidence-scale effect: temperature scaling removes it in each design, and the like-for-like ECE comparison shows no difference. The accuracy result depends on the SGD learning rate. The SGD baseline that Iteration 1 used was not the validation optimum, and the grid's lower edge was where its selection fell.

Two conclusions were ruled out or reduced. The ECE sub-claim against temperature-scaled SGD is ruled out in the sense the hypothesis defined. The Iteration 1 reading of the early calibration gap as a reversal is reduced: a within-run read of the same trajectory shows the gap already negative at 5 epochs in Artifact 1.

**Failed directions, labelled.**

1. *The SGD learning rate of 0.01 as the tuned baseline.* Killed by Artifact 3's validation search: validation NLL was 0.273 at 0.01 against 0.214 at 0.001, and the 0.01 setting was the Iteration 1 selection on a three-value grid. The Iteration 1 accuracy null depends on this choice. Artifact 4 uses it as a labelled weaker baseline, and that is recorded as a limitation, not as a finding about SAM.
2. *Lower SAM learning rates at radius 0.1.* Killed by Artifact 3's search: validation NLL was 0.248 at learning rate 0.001 and 0.206 at 0.003, against 0.182 at 0.01. Neither was selected.
3. *Compressed schedules as the test of criterion (c).* Confounded: each compressed run is a separate training run with its own cosine schedule. The within-run checkpoints replace it in Evaluation 1, and the compressed reading is kept in the record as what it is.
4. *Test of overfitting through the gap-against-training-loss slope.* Reported by Evaluation 1 as relationship within one trajectory, not a test. It is not used as evidence for a mechanism.

**Caveats that remain.** Artifacts 3 and 4 use five seeds, and the source 2 trajectory of the evaluation uses three, so an exact test cannot reach 0.05 and the parametric p-values are the only tests available. The fit direction changed between designs, so no single fit correction applies. The related-work comparison remains not run.

According to the summaries recorded in the hypothesis, Tan et al. [4] argue that SAM counteracts overconfidence through implicit entropy maximisation and report the effect at larger scale, and Morosini et al. [5] report that calibration error tracks curvature-based sharpness along training under several gradient methods. This record draws on the reference entries and the summaries in the hypothesis, and neither paper was read in full in this iteration. On a small CNN, this run adds a temperature-scaled NLL control at matched training loss (Artifacts 3 and 4), and the matched-loss curvature comparison of Artifact 2 at a learning rate of 0.03. It does not add a larger model, a new optimiser or a claim beyond this dataset. Whether the calibration gain of SAM is a ranking gain at scale is open, and the related-work check that would answer where these results lie against Tan et al. and Morosini et al. has not run.

## What the next iteration takes from this one

The open question is whether the NLL advantage holds at matched training loss when the SGD baseline is the validation-selected one. The next iteration should run the matched-loss comparison with SGD at learning rate 0.001, which Artifact 3 selected, and with the matching epoch chosen on the same procedure. It should also add seeds, because five seeds cannot give an exact p-value below 0.0625, and it should read criterion (c) from per-epoch test metrics on a single 40-epoch trajectory rather than from validation values alone. The related-work check against Tan et al. [4] and Morosini et al. [5] stays open and is still a required step before any positive claim is written up.

## Corrections to previous sections

These corrections are made in place in the Iteration 1 text and are given here so that the record shows them. Previous sentences were not otherwise changed. Each is marked "Correction (Iteration 2)" where it appears.

1. **Table 3, ECE row against temperature-scaled SGD.** The Iteration 1 row compared uncalibrated SAM with temperature-scaled SGD. Two rows were added beneath it, and the like-for-like comparison is in Table 20. Its ECE difference is +0.00054 with an interval that includes zero, so the calibrated ECE advantage claimed in Iteration 1 is not supported. The NLL difference of -0.01963 holds.
2. **Conclusions on criterion (a) and accuracy.** The conclusion on criterion (a) holds in direction, but its magnitude depends on the SGD learning rate and on the fit. The Iteration 1 accuracy null of +0.15 points is specific to SGD at learning rate 0.01, which was the lower edge of Iteration 1's grid. At the validation-selected learning rate of 0.001 the accuracy difference is +0.96 points (Table 13).
3. **Criterion (c), 5-epoch reversal.** The reversal at 5 epochs came from separately trained compressed schedules. Within one training run the test ECE gap at 5 epochs is -0.0133 (Table 23), which is negative, not positive.
4. **Table 6 label.** The label is corrected.
5. **Scope of the check, Artifact 1.** The check covers the primary paired contrasts and the two control checks. The ECE binning loop and the ACE, Brier and AURC values were not computed again.
6. **Table 9, Hessian ratio.** The text said the SGD top Hessian eigenvalue is two to three times the SAM value at every evaluated epoch. The ratio is 1.9 at epoch 40 and 3.35 at epoch 25, so the statement is corrected to the range given in Table 22.
7. **Table 9, loss increase within the radius.** The SAM loss increase inside the perturbation radius, which the per-epoch record also holds, is now reported at the logged epochs in Table 22.
8. **Wording, conclusions on the NLL advantage.** A sentence about the NLL advantage is restated as "the NLL advantage remains against temperature-scaled SGD, with Holm p = 0.0027".
9. **Reference year, Foret et al.** The reference record gave 2020, the arXiv year. The entry is corrected to ICLR 2021 in `references.bib`.

Table 20 is the like-for-like calibrated comparison that Correction 1 refers to.

Table 20. Experiment-1 calibrated comparison, epoch 40, five seeds: temperature-scaled SAM minus temperature-scaled SGD, computed again from the per-run metrics in `per_run_metrics.csv` (Artifact 1). Paired t-interval, df = 4.

| Metric | Per-seed differences (seeds 0-4) | Mean difference | 95% CI | p (two-sided) |
|---|---|---|---|---|
| ece | +0.00198, -0.00370, +0.00310, +0.00212, -0.00078 | +0.00054 | [-0.00291, +0.00399] | 0.6845 |
| nll | -0.01667, -0.02792, -0.01881, -0.01550, -0.01926 | -0.01963 | [-0.02569, -0.01357] | 0.0008 |
| accuracy | -0.00060, +0.00420, +0.00140, +0.00080, +0.00150 | +0.00146 | [-0.00071, +0.00363] | 0.1348 |

# What we have learned so far

This section covers the run as it is after Iteration 2. Previous sections keep the belief each iteration held at the time, and this section is the current reading of the evidence.

**Question.** Does sharpness-aware minimisation improve test accuracy and test calibration over standard SGD for a small CNN on Fashion-MNIST, and is any calibration gain more than a global change in confidence?

**Where the evidence lies.**

1. *NLL.* SAM has a lower test NLL than SGD in every design the run has trained, and the advantage persists under temperature scaling of the SGD baseline. Like-for-like (both conditions temperature-scaled, five seeds, 40 epochs), the NLL difference is -0.0196 in Artifact 1 (Table 20) and -0.0326 in Artifact 3 (Table 13, Holm p = 0.0018). At matched training loss it is -0.0082 (Table 17, Holm p = 0.0094). Artifact 2's three-seed trajectory gives -0.0164 at epoch 60, as a point estimate. The NLL advantage is therefore the most consistent result of the run, and its size depends on the SGD learning rate and on the training-length match.
2. *ECE.* The raw test ECE of SAM is lower than SGD's in every design (Tables 13, 17 and 2), by between 0.005 and 0.038 depending on the design. The advantage is a global confidence effect. Temperature scaling removes it: the like-for-like ECE differences are +0.00054 (Artifact 1), +0.00199 (Artifact 3) and -0.00013 (Artifact 4), each with an interval that includes zero. The hypothesis's ECE sub-claim is not supported in the form it was stated.
3. *Accuracy.* SAM's accuracy is higher in every design: +0.15 points at SGD learning rate 0.01 (Holm p = 0.135), +0.96 points at the validation-selected SGD learning rate of 0.001 (Holm p = 0.007), and +0.25 points at matched loss (Holm p = 0.055). The size of the accuracy effect is a function of the SGD baseline, and the run does not yet support a single accuracy number.
4. *Training length, criterion (c).* Along single trajectories the calibration gap grows with training. In Artifact 2 the validation ECE gap moves from near zero at epochs 1 to 5 to its largest value at epoch 25 and decreases after that. In Artifact 1's within-run checkpoints the test ECE gap grows from -0.013 at epoch 5 to -0.038 at epoch 40. The early-training prediction of a near-zero gap holds in one source and fails in the other.
5. *Mechanism.* In Artifact 2 SAM's top Hessian eigenvalue is 1.9 to 3.35 times smaller than SGD's across the logged epochs, and the SAM loss increase within the perturbation radius is between 0.016 and 0.066. Matched in training loss, SAM's validation ECE is lower by 0.010 to 0.014, which Artifact 2 reports at three seeds. These are relationships within the runs, not a test of the mechanism.

**What the run has changed or reduced.** The Iteration 1 SGD baseline at learning rate 0.01 was not the validation optimum; the lower edge of its grid was the selection. The Iteration 1 claim that accuracy shows no difference, and its calibrated ECE claim, do not hold under the like-for-like and re-tuned comparisons. The Iteration 1 reversal at 5 epochs came from comparing separately trained schedules. The fit gap between the optimisers changes sign between designs: SAM is less fitted at learning rate 0.01 (Artifact 1) and SGD is less fitted at learning rate 0.001 (Artifact 3), so a fit correction cannot be applied as one fixed correction.

**Hypothesis summary.** The NLL claim is kept, with a size that reflects the SGD baseline and the training-length match. The ECE sub-claim is not supported. Criterion (c) is supported in part. Criterion (a) holds in direction across designs, and its magnitude is not fixed. Accuracy is recorded as a positive difference whose size depends on the baseline. The run does not yet contain a result that is significant at 0.05 on an exact test, because five seeds set the smallest exact p-value at 0.0625. Every test in the run is parametric.

**What remains open.**

-  The related-work comparison with Tan et al. [4] and Morosini et al. [5] has not been run. Until it is, no positive claim about contribution can be made, and the two papers are used only through their reference entries and the summaries in the hypothesis record.
-  The matched-loss comparison has not been run with the validation-selected SGD learning rate of 0.001.
-  Five seeds are too few for an exact test at 0.05. More seeds are needed before any accuracy or calibration claim is reported as significant.
-  Criterion (c) has not been read from per-epoch test metrics on one trajectory with the validation-selected baseline.
-  The shuffled-label control results show that NLL and Brier respond to shuffled labels, so proper scoring rules do not isolate label information in this design.

# References

[1] P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR 2021. arXiv:2010.01412.

[2] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML 2017. arXiv:1706.04599.

[3] J. Nixon, M. W. Dusenberry, L. Zhang, G. Jerfel, D. Tran. Measuring Calibration in Deep Learning. CVPR Workshops 2019, pp. 38-41. arXiv:1904.01685.

[4] C. Tan, Y. Zhou, H. Ye, G. Dai, J. Liu, Z. Song, J. Zhang, Z. Zhao, Y. Hao, Y. Xu. Towards Understanding The Calibration Benefits of Sharpness-Aware Minimization. ICLR 2026. arXiv:2505.23866.

[5] A. Morosini, M. Gjika, T. Poggio, P. Beneventano. Too Sharp, Too Sure: When Calibration Follows Curvature. 2026. arXiv:2604.20614.

[6] H. Xiao, K. Rasul, R. Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017. arXiv:1708.07747.

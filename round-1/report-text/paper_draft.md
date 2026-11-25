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

**Check.** The artifact computed again the main paired differences and intervals from the saved test logits with separate code, and its control checks did not reject. Table 6 reproduces the independent computation, read from its check output. The artifact did not compute again the ECE binning loop or the ACE, Brier and AURC values for every run, and it did not retrain the model.

Table 6. Independent independent computation of the main contrasts from saved test logits, epoch 40. The SciPy and independent intervals agree to the reported digits.

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

Table 9. Per-epoch means over three seeds at selected epochs, read from `results/per_epoch.json`. Training loss is the mean over the epoch. Validation ECE is the mean of the three seeds. Top Hessian eigenvalue is the mean of the three seeds at evaluated epochs.

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

The SGD top Hessian eigenvalue is between about two and three times the SAM value at every evaluated epoch. SGD's validation ECE rises from 0.012 at epoch 5 to 0.043 at epoch 60, while its training loss falls to 0.005. SAM's validation ECE falls to a minimum of 0.0064 at epoch 15 and then rises to 0.025 at epoch 60, while its training loss stays near 0.03. Both optimisers become less calibrated as training continues past the point where their validation ECE is lowest, and the rise in SGD's validation ECE is larger. The per-epoch means do not show a gap at epoch 5, which matches the sign-mixed comparison in Table 7.

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

On the controlled comparison, SAM at a validation-selected learning rate and radius has a lower test ECE and a lower test NLL than SGD, with paired intervals excluding zero and a consistent sign across all five seeds [ARTIFACT:art_SL9gPfE09HlN]. Its test accuracy is the same within the seed variance. The NLL advantage remains a temperature-scaled SGD baseline with a significant corrected for multiplicity by Holm p-value. The ECE advantage over the same baseline is small and not significant after correction.

On the mechanism study, the ECE and NLL differences at epoch 60 keep their sign across three seeds, and the matched-loss comparison finds a lower top Hessian eigenvalue and a lower validation ECE for SAM [ARTIFACT:art_WOFinDmS2PoX]. Temperature scaling removes most of the test ECE and NLL gap at epoch 60, and what remains of the NLL gap is about one-fifth of the unscaled gap.

Against the hypothesis:

1. Criterion (a), lower test ECE and NLL than tuned SGD: supported by Artifact 1 with five seeds, and by Artifact 2 as summary statistics at three seeds.
2. Criterion (b), NLL advantage after temperature scaling: supported by Artifact 1 (Holm p = 0.0027). Artifact 2 shows a residual scaled NLL advantage of 0.016 NLL units on average, without a test.
3. Criterion (c), gap larger at full training than at short training: supported at 15 and 40 epochs in Artifact 1. Reversed at 5 epochs under compressed training, where SAM's ECE is higher. Artifact 2 shows no consistent sign at epoch 5.
4. Accuracy: no significant difference in either artifact, recorded as such.

Two caveats apply to every reading. SAM is less fitted than SGD at the end of both main schedules, so the calibration advantage may in part reflect a less-fitted model; the matched-loss comparison in Artifact 2 addresses this only at the training losses the runs reached. And the two artifacts use different learning rates, one tuned and one fixed, so their numbers are not comparable.

## Failed directions and dropped work

1. SAM at learning rate 0.03 with radius 0.2 failed to train (training loss 2.302, validation accuracy 0.1305) and was excluded from selection. This is a failed setting, not a finding about SAM's calibration.
2. The 60-epoch extension of Artifact 1 was not run. The artifact reports that SAM is under-fit at epoch 40 and that this extension would test it. The question stays open.
3. The early-stopping test of criterion (c) was replaced by compressed schedules, which are separate full training runs. The record keeps this change. The 40-epoch run's own epoch 5 and epoch 15 values are in Table 2 and differ from the compressed runs at the same epoch count.

## Designed but not run

1. Check of prior work (the third direction set for this iteration): designed to confirm whether any prior work runs SAM against temperature-scaled SGD on a small CNN, which baselines and radius ranges would be expected, and which calibration metric definitions to use. It was dropped at the artifact stage: the artifact stage returned no artifact for its design. This is not an experimental result, and it records no null or failed-direction finding. It remains open.

## What we have learned so far

On the evidence of this iteration, SAM lowers test ECE and NLL relative to tuned SGD for this small CNN on Fashion-MNIST, and its accuracy is not different beyond seed noise. The NLL advantage holds against temperature-scaled SGD. The ECE advantage over temperature-scaled SGD is small. The gap increases with training length from 5 to 40 epochs in the compressed-schedule comparison, but it is negative at 5 epochs, so the early-stopping prediction did not hold in that form. The main open questions are whether the calibration gap remains matching the training fit more closely, whether a longer schedule changes SAM's under-fit, and what the standard baselines are. Those questions are the subject of later iterations.

# References

[1] P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR 2021. arXiv:2010.01412.

[2] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML 2017. arXiv:1706.04599.

[3] J. Nixon, M. W. Dusenberry, L. Zhang, G. Jerfel, D. Tran. Measuring Calibration in Deep Learning. CVPR Workshops 2019, pp. 38-41. arXiv:1904.01685.

[4] C. Tan, Y. Zhou, H. Ye, G. Dai, J. Liu, Z. Song, J. Zhang, Z. Zhao, Y. Hao, Y. Xu. Towards Understanding The Calibration Benefits of Sharpness-Aware Minimization. ICLR 2026. arXiv:2505.23866.

[5] A. Morosini, M. Gjika, T. Poggio, P. Beneventano. Too Sharp, Too Sure: When Calibration Follows Curvature. 2026. arXiv:2604.20614.

[6] H. Xiao, K. Rasul, R. Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017. arXiv:1708.07747.

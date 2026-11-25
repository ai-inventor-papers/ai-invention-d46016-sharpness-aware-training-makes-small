# Caveats: does the SAM minus SGD calibration gap grow?

Gap convention throughout: SAM minus SGD. A negative ECE or NLL gap favours SAM.
No model was trained or re-run. Every number comes from the two dependency `full_method_out.json` files copied to `inputs/`.

## Scope and design limits

1. **Two trajectories, never pooled.**
   - Source 1: 5 paired seeds, 40-epoch headline set, learning rate 0.01 (lower edge of its grid).
   - Source 2: 3 paired seeds, 60-epoch trajectory, learning rate 0.03.
2. **Source 1 lengths are separate schedules.** The 5- and 15-epoch points are separate compressed cosine runs, not checkpoints of the 40-epoch run. The growth from 5 to 40 epochs therefore mixes training length with schedule shape. The within-run checkpoints (5 and 15 of the 40-epoch run) are reported as a secondary read in the table; they give the same sign pattern but are still mid-schedule at high learning rate.
3. **Source 1 has no validation metrics and no training loss in `full_method_out.json`.** Validation ECE, validation NLL and training loss are reported as MISSING for source 1. They were not recomputed from other workspaces.
4. **Power.** With n = 3 (source 2), paired t intervals have df = 2 (critical value 4.303) and are very wide. The percentile bootstrap over seed pairs is exact over all 27 resamples and takes few distinct values, so it is a coarse secondary check only. No p-values are used as evidence. Source 1 (n = 5, df = 4) has tighter intervals, but it is a different schedule design.
5. **Training-loss association is co-variation within one trajectory.** The per-seed slope of the gap on training loss is positive for both losses (more negative gap as loss falls, through epoch 25 or 30). It is not a mechanism test: SAM is under-fit relative to SGD at the same epoch, so a calibration difference can reflect fit as well as overfitting. The gap-versus-loss panels show that the relation reverses after the extremum.

## Correction: the p-values in the table are parametric and cannot reach significance at n = 3

The t-test p-values in `results/gap_table.*` assume normal paired differences. The independent re-derivation (`checks/rederive_headlines.py`, output in `checks/rederive_headlines_out.txt`) shows the limit:

- Source 2 has 3 seed pairs, so there are only 2^3 = 8 sign assignments. The smallest exact sign-flip permutation p-value is 0.25. The source 2 widening p-values (0.006 for NLL, 0.053 for ECE) and the epoch-60 ECE gap cannot be supported by any exact test on these data. Treat them as descriptive only, as the plan requires.
- The placebo check does not fail cleanly. Enumerating all 8 label swaps, the t-test rejects at 0.05 in 2 of 8, namely the all-same-sign assignments. Those two are the only assignments that reach the smallest exact p-value, so this is the design floor, not a false discovery.
- Source 1 (5 seeds, 32 sign assignments): the ECE gap at 40 epochs has a t-test p of 8.8e-6 but an exact sign-flip p of 0.0625. The Holm-adjusted p-values for source 1 (for example 0.016 for ECE at 5 epochs, and the 0.0027 and 0.079 contrasts against temperature-scaled SGD) are also parametric. The source 1 results are consistent in sign and direction across seeds, but the reported p-values overstate the evidence.

## What the numbers show

- **Accuracy is not detectably changed.** Source 1 at 40 epochs: +0.0015, CI [-0.0007, +0.0036]. Source 2 at 60 epochs: +0.0012, CI [-0.0033, +0.0057], p = 0.37.
- **Source 1 gap changes sign with length.** Test ECE gap is +0.0057 at 5 epochs (SAM worse, Holm p = 0.016), -0.0132 at 15 epochs, and -0.0378 at 40 epochs. Test NLL gap is +0.045, +0.0065, then -0.098. The early gap is not near zero. It is positive, meaning SAM is worse early in the separate short schedule. The within-run checkpoint at 5 epochs gives ECE -0.013 (p = 0.011), so the early sign depends on the schedule read.
- **Source 2 validation gap is near zero early and then reverses.**
  - Validation ECE gap: -0.004 at epoch 1, +0.004 at epoch 2, +0.002 at epoch 5 (intervals all include zero).
  - Validation ECE gap: -0.0125 at epoch 10 (p = 0.044), largest mean advantage -0.0326 at epoch 25, -0.0186 at epoch 60.
  - Validation NLL gap: +0.034 at epoch 1, near zero at 5, largest mean advantage -0.0953 at epoch 30, -0.0690 at epoch 60.
  - The gap widens and then narrows. The endpoint is not the extremum, so "the gap grows with training" is true only up to about epoch 25 to 30.
- **Widening test, source 2, per trajectory.**
  - Validation ECE, 60 minus 1: -0.0142, t interval [-0.0289, +0.0005], p = 0.053 (does not reach conventional significance).
  - Validation ECE, 60 minus 5: -0.0208, p = 0.115.
  - Validation NLL, 60 minus 1: -0.103 (parametric t p = 0.006; see the correction below).
  - Validation NLL, 60 minus 5: -0.068 (parametric t p = 0.046; see the correction below).
  - Source 1 widening (40 minus 5 and 40 minus 15, separate schedules) is significant for every metric, but it is confounded by schedule as stated above.
- **Test endpoints, source 2 at 60 epochs.** Raw SAM minus SGD: ECE -0.0171 (CI [-0.0207, -0.0135]), NLL -0.076 (CI [-0.092, -0.060]), Brier -0.010.
- **Temperature scaling removes most of the ECE advantage.**
  - SAM raw minus temperature-scaled SGD, source 2 at 60: ECE +0.018 (SAM is worse), NLL +0.003 (n.s.).
  - Both temperature-scaled (SAM-TS minus SGD-TS), source 2 at 60: ECE -0.0024 (CI includes zero), NLL -0.016 (CI [-0.030, -0.003], p = 0.035).
  - Source 1 at 40 epochs (SAM raw minus TS-SGD): NLL -0.021 (p = 0.0005), ECE -0.003 (p = 0.040 unadjusted, Holm p = 0.079).
  - The NLL advantage is the more robust calibration-related result. The ECE advantage is largely a global confidence shift.
- **A reported summary number does not reproduce.** The source 2 notes state that under temperature scaling the gap shrinks to 0.007 vs 0.009. The full-output values are SAM-TS ECE mean about 0.0074 and SGD-TS ECE mean about 0.0094, which matches that reading. The raw-SAM against TS-SGD comparison, which the plan names, gives +0.018 (SAM worse). The table reports both so a reader cannot take the shrunk figure as a SAM advantage over temperature-scaled SGD.

## Summary against the hypothesis

The hypothesis asks for a gap near zero early that widens with training. Source 2 (within one schedule) supports the widening for NLL and only weakly for ECE, and the widening stops at epoch 25 to 30. Source 1 supports widening, but its early read is positive and its design confounds length with schedule. Accuracy shows no detectable difference. The result is descriptive co-variation with n = 3 and n = 5, not a test of the overfitting mechanism.

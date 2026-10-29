# Native LandCover.ai local comparison

This centralized, noiseless simulator pilot uses 256 official training tiles, 64 validation tiles, three training seeds, 60 epochs, RGB inputs and five classes including background. All 1,024 validation crops are pooled. The common learning rate was chosen using only two affine seed 42 fits; the campaign contains 13 fits and 12 primary comparisons. No test images were evaluated.

| Control | Validation mIoU mean ± sample SD | Accuracy mean ± sample SD |
|---|---:|---:|
| Compact affine | 0.4590 ± 0.0042 | 0.8535 ± 0.0030 |
| Learned quantum | 0.4606 ± 0.0078 | 0.8580 ± 0.0090 |
| Frozen quantum | 0.4524 ± 0.0022 | 0.8544 ± 0.0102 |
| Zero bottleneck | 0.4605 ± 0.0167 | 0.8624 ± 0.0117 |

| Control | Total / trainable parameters | Core total / trainable | Profile step 1 / 2 (s) | Mean fit elapsed (s) |
|---|---:|---:|---:|---:|
| Compact affine | 8,155,843 / 8,155,843 | 1,568 / 1,568 | 0.103 / 0.096 | 446.6 |
| Learned quantum | 8,154,283 / 8,154,283 | 8 / 8 | 1.264 / 0.893 | 4758.2 |
| Frozen quantum | 8,154,283 / 8,154,275 | 8 / 0 | 1.068 / 0.855 | 4519.2 |
| Zero bottleneck | 8,154,283 / 5,988,512 | 8 / 0 | 0.079 / 0.080 | 423.1 |

Quantum minus reference, paired percentage points (mean ± sample SD):
- Compact affine: +0.157 ± 0.384; positive in 2/3 seeds.
- Frozen quantum: +0.818 ± 0.967; positive in 3/3 seeds.
- Zero bottleneck: +0.009 ± 1.381; positive in 2/3 seeds.

Seed 42 also calibrated the common rate through the affine candidates. Its contrast is separate from the mean and sample SD of follow-up seeds 77 and 123.

| Quantum minus reference | Calibration seed 42 (pp) | Follow-up seeds 77/123 mean ± sample SD (pp) |
|---|---:|---:|
| Compact affine | +0.504 | -0.017 ± 0.339 |
| Frozen quantum | +1.889 | +0.283 ± 0.388 |
| Zero bottleneck | +0.508 | -0.241 ± 1.854 |

The fixed practical screen did not pass. It requires at least +1 percentage point mean quantum improvement over both affine and frozen controls, positive in every seed for each, and nonnegative mean improvement over zero. This is a descriptive screening rule, not a significance test, an equivalence result or quantum advantage.

Selected checkpoints at epoch 60: 7/12. Selected checkpoints at or after epoch 55: 12/12. Positive final-interval validation changes: 7/12. The learning table retains epoch-zero to best changes and final interval behavior; a completed budget does not establish convergence. Late best epochs leave convergence unresolved.

Training quality: learning_observed. Fits with no epoch-zero-to-best gain: 0/12; all arms showed no learning: False. Late best epochs: 12/12; horizon unresolved: True. A failed screen is not a refutation of the paper. No observed learning across the arms is an inconclusive optimization result, and any late best epoch leaves the training horizon unresolved.

Curves show means and sample standard deviations over the same three training seeds at the common selected rate. Training loss retains every epoch; validation is recorded every five epochs. Validation was reused for rate and checkpoint selection. The cohort is not a scene-disjoint holdout. This independent implementation does not establish the paper’s full-test score, general quantum superiority, quantum hardware benefit or Anvil performance. No raw imagery or weights are exported.

IoU = TP/(TP+FP+FN), with zero for an empty union. Primary mIoU averages all five classes including background; foreground mIoU is secondary. Accuracy is total diagonal counts divided by total pixels. Reported trainable counts describe requires_grad flags, not functional influence. Profile costs use synthetic batches and are distinct from full fit elapsed costs.

| Control | Epoch-zero to best mIoU gain mean ± sample SD (pp) | Best epochs (42 / 77 / 123) |
|---|---:|---|
| Compact affine | +39.536 ± 5.494 | 60 / 55 / 60 |
| Learned quantum | +39.693 ± 5.110 | 60 / 55 / 60 |
| Frozen quantum | +38.875 ± 6.033 | 60 / 55 / 55 |
| Zero bottleneck | +39.684 ± 4.887 | 60 / 55 / 60 |

All twelve fits lacked an epoch-zero-to-best gain: False. This quality flag describes the observed optimization and does not change the frozen screening rule.

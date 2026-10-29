# Independent completed native LandCover numeric audit

All 13 completed fits, 13 checkpoint byte hashes, 169 validation confusion matrices, 793 full training-history points, 12 selected replays, 10 CSV tables per export and both 17-file exports passed independent metadata/numeric checks. No model, image or test-pixel data was loaded.

The common rate was selected only by the two seed 42 affine candidates, with lower rate on an exact tie: `0.001`.

| Arm | Validation mIoU mean | Sample SD | Accuracy mean |
|---|---:|---:|---:|
| compact_affine | 0.45900817 | 0.00420644 | 0.85350076 |
| quantum | 0.46057592 | 0.00784725 | 0.85798544 |
| frozen_quantum | 0.45239335 | 0.00224079 | 0.85444504 |
| zero_bottleneck | 0.46048724 | 0.01667332 | 0.86237649 |

| Quantum minus reference | Mean (pp) | Sample SD (pp) | Positive seeds | Seed 42 calibration (pp) | Seeds 77/123 mean (pp) |
|---|---:|---:|---:|---:|---:|
| compact_affine | +0.156775 | 0.384254 | 2/3 | +0.503777 | -0.016726 |
| frozen_quantum | +0.818257 | 0.966932 | 3/3 | +1.888780 | +0.282995 |
| zero_bottleneck | +0.008867 | 1.380740 | 2/3 | +0.508384 | -0.240891 |

The exact Fraction practical screen did not pass. This is a descriptive prioritization screen, not significance, equivalence or quantum advantage.

Primary learning-quality flags: `{"all_arms_no_learning": false, "horizons_unresolved": true, "initial_to_best_nonpositive_count": 0, "late_best_count": 12, "training_quality": "learning_observed"}`.

mIoU averages all five count-derived class IoUs; foreground mIoU excluding background is secondary. Accuracy is diagonal pixels divided by all pixels. Paired differences precede mean and sample SD (n-1); seed 42 calibration and seeds 77/123 follow-ups remain separate. Exact count fractions decide thresholds; float tolerance applies only to serialized reporting comparisons.

Checkpoint hashing establishes byte identity, not a new forward replay. The retained production collector independently restored/replayed the checkpoints. This oracle did not load weights, models, dataset images or test images. Parameter counts are checked across raw/profile/export records rather than reconstructed from models. Validation served rate/checkpoint selection and is not a geographic or independent-test claim.

Plan SHA256: `a3cae54c1f575fc46722513c8d274daf96ef4b3fc4635c4a3e051c7f6e1b30fc`.
Audit source SHA256: `42564517355697652f6eaf706098e760ccea38bcdabc14155f322b334b126d7e`.

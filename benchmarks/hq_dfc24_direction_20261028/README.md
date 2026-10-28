# HQ-style direction screen on DFC24

Architecture-transfer pilot, not a reproduction of the five-class LandCover.ai paper. Validation is primary; the previously exposed test split is descriptive.

All eight planned fits and checkpoints were audited, including both losing learning rates. Six selected saved test replays were checked for exact record/identity consistency and count-derived metrics; collection does not independently repeat inference.

128-pixel inputs; 128 optimization tiles per seed; 1,303 fit images used for normalization, excluding 230 validation images; all 205 test images (71 Optical/134 SAR). Three paired seeds, 60 epochs, batch size 8, Adam, Dice+CE and deterministic training-only D4 augmentation.

The MLP deliberately has more core parameters (5,216 versus 8); total parameters are 349,688 versus 344,480. Interfaces and initial non-core state, including BatchNorm buffers, are identical per seed. This is not exact parameter-count matching.

| Model | Selected LR | Validation IoU mean ± sample SD | Descriptive test IoU mean ± sample SD | Optical test | SAR test |
| --- | ---: | ---: | ---: | ---: | ---: |
| hq_quantum | 0.003 | 0.6023 ± 0.0493 | 0.6092 ± 0.0122 | 0.6160 | 0.5949 |
| hq_mlp | 0.003 | 0.5997 ± 0.0454 | 0.5872 ± 0.0078 | 0.5949 | 0.5700 |

Differences subtract paired seeds before aggregation. Units below are IoU percentage points; ± is sample SD, not a standard error or significance test.

| Quantum − MLP | Split | Mean ± sample SD | Positive seeds | Same selected LR |
| --- | --- | ---: | ---: | --- |
| iou | val | +0.264 ± 0.602 | 2/3 | True |
| optical_subset_iou | val | -0.432 ± 3.210 | 2/3 | True |
| sar_subset_iou | val | +0.118 ± 1.503 | 1/3 | True |
| iou | test | +2.199 ± 1.148 | 3/3 | True |
| optical_subset_iou | test | +2.103 ± 0.822 | 3/3 | True |
| sar_subset_iou | test | +2.496 ± 3.627 | 2/3 | True |

| Model | On − diagnostic test IoU | Mean ± sample SD (points) |
| --- | --- | ---: |
| hq_quantum | zero_core | +0.043 ± 0.074 |
| hq_quantum | random_core | +0.004 ± 0.008 |
| hq_quantum | zero_bottleneck | +0.033 ± 0.080 |
| hq_mlp | zero_core | -0.024 ± 0.179 |
| hq_mlp | random_core | -0.017 ± 0.156 |
| hq_mlp | zero_bottleneck | -0.028 ± 0.192 |

`zero_core` retains reconstruction bias; `zero_bottleneck` also removes that bias. `random_core` replaces only core coefficients. Drops measure dependence on an information path, not quantum superiority. Ordinary U-Net skips remain.

Angles are tanh-bounded radians without pi scaling. Each angle probe contains 384 values from the first eight unaugmented optimization images for its seed, measured at initialization and the selected checkpoint. Gradient/movement summaries establish observed optimization activity, not parameter importance or an accuracy mechanism. Curves retain all eight fits and every epoch training loss; incomplete convergence remains a limitation.

Three seeds and one classical control establish neither significance, equivalence, quantum advantage nor a general impossibility result. This local centralized screen makes no Anvil, hardware or throughput claim.

Figures: [all learning curves](plots/learning_curves.png), [paired pooled/sensor outcomes](plots/paired_iou.png), [knockouts](plots/knockouts.png), [angle/core diagnostics](plots/diagnostics.png). Each has a PDF export. [Reproduction and provenance](REPRODUCTION.md).

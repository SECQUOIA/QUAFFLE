# HQ training controls on DFC24

Validation is primary. The previously exposed test split remains descriptive; no fresh holdout or independent confirmation is claimed.

Five controls share the same depthwise U-Net, interfaces and initial non-core state, including BatchNorm buffers. The primary contrast fixes Adam learning rate 0.003 for every control and seed. Seed-42 observations at 0.001 are retained as auxiliary rate checks, without selecting a rate or changing the primary contrast. A common fixed rate isolates this training setting; it is not an independently optimized ranking of every control.

Twenty fit records comprise twelve fresh fits and eight byte-audited fits reused from the preceding HQ campaign. They are not twenty independent seeds. All twenty completed fits precede fifteen new saved descriptive test replays. Six older replays remain separately labeled historical evidence. The collector restores every checkpoint and checks saved replay identities and confusion-derived scores; it does not independently repeat inference.

Three paired seeds; 128-pixel images; base width 8; actual batch 8; 60 epochs and 960 optimizer batches per fit. Each seed optimizes on 128 tiles. Normalization uses 1,303 global-fit images excluding 230 validation images, so this is not pure 128-shot learning. All 205 test tiles comprise 71 Optical and 134 SAR.

| Control | Parameters / trainable | Core total / trainable | Bottleneck total / trainable | Validation IoU mean ± sample SD | Descriptive test IoU mean ± sample SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned quantum | 344,480 / 344,480 | 8 / 8 | 270,539 / 270,539 | 0.6023 ± 0.0493 | 0.6092 ± 0.0122 |
| Frozen quantum | 344,480 / 344,472 | 8 / 0 | 270,539 / 270,531 | 0.5995 ± 0.0423 | 0.6038 ± 0.0059 |
| Zero bottleneck | 344,480 / 73,941 | 8 / 0 | 270,539 / 0 | 0.6073 ± 0.0455 | 0.6022 ± 0.0052 |
| Compact affine | 346,040 / 346,040 | 1,568 / 1,568 | 272,099 / 272,099 | 0.5993 ± 0.0450 | 0.6052 ± 0.0099 |
| MLP | 349,688 / 349,688 | 5,216 / 5,216 | 275,747 / 275,747 | 0.5997 ± 0.0454 | 0.5872 ± 0.0078 |

Differences are computed within each paired seed before aggregation. Units are IoU percentage points; ± denotes sample SD, not a standard error or significance test.

| Learned quantum minus control | Split | Mean ± sample SD (points) | Positive seeds |
| --- | --- | ---: | ---: |
| Frozen quantum | val | +0.277 ± 0.887 | 2/3 |
| Frozen quantum | test | +0.543 ± 1.465 | 2/3 |
| Zero bottleneck | val | -0.504 ± 0.912 | 1/3 |
| Zero bottleneck | test | +0.703 ± 0.897 | 2/3 |
| Compact affine | val | +0.297 ± 0.684 | 2/3 |
| Compact affine | test | +0.402 ± 0.519 | 2/3 |
| MLP | val | +0.264 ± 0.602 | 2/3 |
| MLP | test | +2.199 ± 1.148 | 3/3 |

The frozen quantum control retains differentiable input dependence while its circuit coefficients stay at initialization. The zero-bottleneck control removes the deepest reconstructed map throughout training; it differs from knocking out a trained model afterward. Compact affine is a cheap classical transformation control, and the larger MLP remains a separate strong control. Ordinary encoder-to-decoder skips remain available.

Raw zero-bottleneck angle probes describe counterfactual encodings on an inactive path; no circuit uses those angles during that control's training or inference. Stored and trainable parameter counts also differ from the number of parameters that actively affect predictions.

Training-time contrasts can constrain proposed mechanisms, but three paired seeds do not establish quantum advantage, equivalence, significance or a general impossibility result. Seed changes vary the initialization, optimization subset and validation cohort together. Shared initial state and data schedules are paired across arms.

This is a noiseless local classical simulation and a binary DFC24 adaptation, not a reproduction of the motivating LandCover.ai result, a hardware experiment, an Anvil accuracy result or a throughput benchmark. Alan owns Anvil execution.

Tables: [summary](summary.csv), [individual seeds](per_seed.csv), [paired contrasts](paired.csv), [all primary/auxiliary fits](tuning.csv), [trainability diagnostics](trainability.csv). Curves retain every fit, including auxiliary rates. [Provenance and regeneration](REPRODUCTION.md).

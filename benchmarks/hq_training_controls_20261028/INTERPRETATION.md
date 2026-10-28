# What the HQ training controls tell us

The new controls weaken the claim that learning the quantum circuit improves accuracy. Learning the circuit coefficients gives a small, mixed validation gain over freezing them; the permanently zero bottleneck has the highest validation mean. Quantum retains the highest mean on the previously exposed, descriptive test split. This positive pattern does not establish a quantum-specific improvement.

All five arms use learning rate 0.003 and three paired seeds: 60 epochs, batch eight, 128-pixel inputs and 960 optimizer batches per fit. Initial non-core state, data and augmentation schedules are paired. The 20 fit records comprise 12 fresh fits and eight byte-audited reused fits, followed by 15 new primary test replays. Six older replays remain historical evidence; reused fits are not fresh replications.

| Model | Validation foreground IoU, mean ± sample SD | Descriptive test foreground IoU, mean ± sample SD |
| --- | ---: | ---: |
| Learned quantum | 0.6023 ± 0.0493 | 0.6092 ± 0.0122 |
| Frozen quantum | 0.5995 ± 0.0423 | 0.6038 ± 0.0059 |
| Zero throughout training | 0.6073 ± 0.0455 | 0.6022 ± 0.0052 |
| Compact affine | 0.5993 ± 0.0450 | 0.6052 ± 0.0099 |
| MLP | 0.5997 ± 0.0454 | 0.5872 ± 0.0078 |

Differences below are paired within seeds before aggregation, in IoU percentage points. Wins count positive differences; ± is sample SD, not a standard error.

| Learned quantum minus control | Validation difference | Validation wins | Descriptive test difference | Test wins |
| --- | ---: | ---: | ---: | ---: |
| Frozen quantum | +0.277 ± 0.887 | 2/3 | +0.543 ± 1.465 | 2/3 |
| Zero throughout training | -0.504 ± 0.912 | 1/3 | +0.703 ± 0.897 | 2/3 |
| Compact affine | +0.297 ± 0.684 | 2/3 | +0.402 ± 0.519 | 2/3 |
| MLP | +0.264 ± 0.602 | 2/3 | +2.199 ± 1.148 | 3/3 |

Freezing changes coefficient trainability while retaining input derivatives through the circuit. The small, sign-changing validation contrast does not demonstrate a reliable benefit from learning the eight coefficients at this budget. Frozen and zero cores stayed unchanged, with zero core-gradient norms throughout their four fits each. This verifies the interventions without establishing accuracy equivalence.

Training with the entire reconstructed bottleneck map zero wins against learned quantum on validation in two of three seeds. The deepest quantum path therefore is not needed to obtain competitive validation performance in this screen. In the [preceding direction screen](../hq_dfc24_direction_20261028/INTERPRETATION.md), removing that map after training changed quantum test IoU by at most 0.124 points. These distinct interventions neither identify a causal skip-connection mechanism nor prove absence of a training-time circuit effect.

The compact classical control narrows the descriptive test contrast from +2.199 points against MLP to +0.402 against affine. Its 48-to-32 affine map plus tanh has 1,568 coefficients and maximum rank 32. Whole models contain 344,480 quantum, 346,040 affine and 349,688 MLP parameters: quantum saves 0.45% and 1.49%, respectively. Core counts are 8, 1,568 and 5,216. This is not exact whole-model parameter parity or an exhaustive U-Net comparison.

The zero arm retains 344,480 registered parameters, including 270,539 frozen bottleneck parameters. The 73,941 with `requires_grad=True` include 32,768 first-upsampling weights multiplying an always-zero input, so this is not a functionally active count. The frozen-circuit arm has 344,472 trainable parameters; its eight fixed coefficients still participate in the computation.

All auxiliary seed-42 outcomes follow. Rate 0.003 wins in this seed for every arm; no primary rate is reselected. These observations do not establish optimal rates across seeds.

| Model | LR 0.001 validation IoU (best epoch) | LR 0.003 validation IoU (best epoch) | 0.003 minus 0.001, points |
| --- | ---: | ---: | ---: |
| Learned quantum | 0.560390 (24) | 0.571934 (58) | +1.154 |
| Frozen quantum | 0.566163 (46) | 0.568695 (44) | +0.253 |
| Zero throughout training | 0.564154 (24) | 0.571966 (24) | +0.781 |
| Compact affine | 0.561552 (22) | 0.566829 (44) | +0.528 |
| MLP | 0.564279 (60) | 0.567522 (54) | +0.324 |

The seed-77 quantum, zero, affine and MLP checkpoints occur at epoch 60; the other quantum checkpoints occur at epoch 58. Convergence limits remain unresolved. A common rate is not a separately optimized leaderboard. Seeds vary initialization, optimization subset and validation cohort together. Three seeds establish neither significance nor equivalence.

Our cleaned binary DFC24 task uses 128 optimization tiles per seed, 230 validation tiles and 205 test tiles (71 Optical, 134 SAR). Normalization uses 1,303 global-fit tiles excluding validation, so this is not pure 128-shot learning. The exposed test split supplies no independent confirmation; pooled and sensor tables remain together. Zero-arm angle probes are inactive counterfactual encodings.

The finite study is complete without adding fits after test inspection. Retain the positive descriptive quantum pattern while withholding an accuracy-superiority claim: learned-coefficient gains are inconsistent, the absent deepest path remains competitive, and a simpler classical map recovers much of the earlier gap. Full 16-qubit computation, tanh-bounded radians without pi and 32 X/Z outputs were preserved. This noiseless classical simulation is not LandCover.ai reproduction, hardware, full-size, federated-accuracy or Anvil evidence. Alan owns Anvil execution.

The [paired contrast figure](plots/contrasts.pdf) shows colored seed points and black markers for the mean paired difference; black error bars span one sample standard deviation above and below that mean. They are not confidence intervals. Read the [generated tables and plots](README.md), [all primary and auxiliary fits](tuning.csv), [pooled and sensor contrasts](paired.csv), [fixed protocol](../../docs/HQ_TRAINING_CONTROLS_PROTOCOL.md) and [regeneration instructions](REPRODUCTION.md) for the complete evidence.

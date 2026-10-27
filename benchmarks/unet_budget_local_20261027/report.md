# Ordinary U-Net parameter-budget comparison

Seeds: [42, 77, 123]. One synthetic dataset, generated with seed 42.

Target: 500,394 parameters. Selected U-Net: 510,631 (+2.05%). Width chosen by counts before training.

| Model | Parameters | Test IoU mean ± sample SD | On − zero IoU |
| --- | ---: | ---: | ---: |
| classical_bc7_g1 | 510,631 | 0.8783 ± 0.0144 | — |
| classical_bc8_g1 | 648,106 | 0.8887 ± 0.0055 | — |
| quantum_bc8_g1 | 500,394 | 0.8829 ± 0.0138 | +0.0005 |
| ablation_bc8_g1 | 500,522 | 0.8938 ± 0.0142 | +0.0009 |
| classical_bc8_g4 | 648,106 | 0.8950 ± 0.0219 | — |
| quantum_bc8_g4 | 500,394 | 0.8956 ± 0.0436 | -0.0000 |
| ablation_bc8_g4 | 500,522 | 0.8932 ± 0.0377 | -0.0003 |

Paired differences below are IoU percentage points (left minus right).

| Left | Right | Mean ± sample SD |
| --- | --- | ---: |
| classical_bc8_g1 | classical_bc7_g1 | +1.041 ± 1.987 |
| quantum_bc8_g1 | classical_bc7_g1 | +0.462 ± 2.585 |
| ablation_bc8_g1 | classical_bc7_g1 | +1.550 ± 2.858 |
| classical_bc8_g1 | classical_bc8_g4 | -0.625 ± 1.812 |
| quantum_bc8_g1 | quantum_bc8_g4 | -1.269 ± 2.981 |
| ablation_bc8_g1 | ablation_bc8_g4 | +0.062 ± 2.818 |
| quantum_bc8_g4 | ablation_bc8_g4 | +0.242 ± 0.717 |
| quantum_bc8_g1 | ablation_bc8_g1 | -1.089 ± 1.001 |

`g1`/`g4` indicate GroupNorm groups; `bc` is the base channel width.
The one-group comparisons control normalization across widths. Different-width models cannot share identical initial tensors.
All fits use Adam, Dice + cross-entropy loss, the same paired splits/shuffles, and validation-IoU checkpoint selection.
This tiny synthetic experiment does not establish real-data superiority, statistical equivalence, or quantum advantage.

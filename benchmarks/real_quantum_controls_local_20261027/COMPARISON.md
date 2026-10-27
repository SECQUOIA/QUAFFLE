# Quantum training-control comparison

Same dataset, three paired training/split seeds; descriptive differences, not significance or equivalence claims

| Model | Total parameters | Trainable parameters | Selected LR | Test IoU mean ± sample SD | Optical IoU | SAR IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unet_width7 | 510,631 | 510,631 | 0.001 | 0.5813 ± 0.0348 | 0.5946 | 0.5523 |
| unet_width8 | 648,106 | 648,106 | 0.003 | 0.5926 ± 0.0286 | 0.6041 | 0.5671 |
| mlp_global | 500,522 | 500,522 | 0.001 | 0.5791 ± 0.0150 | 0.5910 | 0.5529 |
| mlp_patch4x4 | 500,522 | 500,522 | 0.003 | 0.5957 ± 0.0169 | 0.6112 | 0.5624 |
| quantum_global | 500,394 | 500,394 | 0.001 | 0.5874 ± 0.0271 | 0.6016 | 0.5551 |
| quantum_patch4x4 | 500,394 | 500,394 | 0.003 | 0.5922 ± 0.0125 | 0.6095 | 0.5564 |
| quantum_frozen_global | 500,394 | 499,882 | 0.001 | 0.5834 ± 0.0238 | 0.5971 | 0.5519 |
| quantum_no_entanglement_global | 500,394 | 500,394 | 0.003 | 0.5906 ± 0.0191 | 0.6051 | 0.5589 |

| Paired contrast | Mean ± sample SD, IoU percentage points | Wins |
| --- | ---: | ---: |
| quantum_global − quantum_frozen_global | +0.394 ± 0.342 | 3/3 |
| quantum_global − quantum_no_entanglement_global | -0.323 ± 1.216 | 1/3 |
| quantum_no_entanglement_global − quantum_frozen_global | +0.717 ± 0.910 | 2/3 |
| quantum_frozen_global − mlp_global | +0.437 ± 0.947 | 1/3 |

Control choice used validation-only records. Every control received the original rate grid and budget.
Each model chose its learning rate independently from that grid; the comparisons include those selected-rate differences.
Frozen circuits retain their initial angles and input scales while the classical model trains. No-entanglement controls remove CNOT gates and train all parameters.
Initial parameters, cohorts, normalization and parameter counts were checked against the original quantum model for every seed.

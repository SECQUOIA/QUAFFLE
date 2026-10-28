# Local learning-rate and stability follow-up

Phase: long120; 18 factorial cells.
All validation fits completed before test replay. Test results are descriptive because the original test split was already consulted.
Normalization uses all global fit images, validation excluded; optimization uses 128 images. This is not pure few-shot learning.

| Model | Fixed LR | Validation IoU mean ± sample SD |
| --- | ---: | ---: |
| quantum_global | 0.001 | 0.6248 ± 0.0209 |
| quantum_global | 0.003 | 0.6150 ± 0.0165 |
| quantum_no_entanglement_global | 0.001 | 0.6245 ± 0.0206 |
| quantum_no_entanglement_global | 0.003 | 0.6280 ± 0.0252 |
| mlp_patch4x4 | 0.001 | 0.6230 ± 0.0295 |
| mlp_patch4x4 | 0.003 | 0.6172 ± 0.0210 |

| Model | Fixed LR | Descriptive test IoU mean ± sample SD | Optical IoU | SAR IoU |
| --- | ---: | ---: | ---: | ---: |
| quantum_global | 0.001 | 0.6015 ± 0.0215 | 0.6203 | 0.5597 |
| quantum_global | 0.003 | 0.5935 ± 0.0171 | 0.6132 | 0.5504 |
| quantum_no_entanglement_global | 0.001 | 0.6058 ± 0.0284 | 0.6224 | 0.5693 |
| quantum_no_entanglement_global | 0.003 | 0.6012 ± 0.0232 | 0.6155 | 0.5692 |
| mlp_patch4x4 | 0.001 | 0.5910 ± 0.0131 | 0.6021 | 0.5664 |
| mlp_patch4x4 | 0.003 | 0.5949 ± 0.0300 | 0.6089 | 0.5656 |

| Validation paired contrast | Fixed LR | IoU percentage points, mean ± sample SD | Wins |
| --- | ---: | ---: | ---: |
| quantum_global − quantum_no_entanglement_global | 0.001 | +0.037 ± 0.153 | 2/3 |
| quantum_global − quantum_no_entanglement_global | 0.003 | -1.296 ± 1.644 | 0/3 |
| quantum_global − mlp_patch4x4 | 0.001 | +0.181 ± 1.571 | 1/3 |
| quantum_global − mlp_patch4x4 | 0.003 | -0.225 ± 1.701 | 1/3 |
| quantum_no_entanglement_global − mlp_patch4x4 | 0.001 | +0.144 ± 1.441 | 1/3 |
| quantum_no_entanglement_global − mlp_patch4x4 | 0.003 | +1.071 ± 0.647 | 3/3 |

| Descriptive test paired contrast | Fixed LR | IoU percentage points, mean ± sample SD | Wins |
| --- | ---: | ---: | ---: |
| quantum_global − quantum_no_entanglement_global | 0.001 | -0.433 ± 1.091 | 1/3 |
| quantum_global − quantum_no_entanglement_global | 0.003 | -0.762 ± 1.198 | 1/3 |
| quantum_global − mlp_patch4x4 | 0.001 | +1.046 ± 0.899 | 3/3 |
| quantum_global − mlp_patch4x4 | 0.003 | -0.132 ± 1.408 | 1/3 |
| quantum_no_entanglement_global − mlp_patch4x4 | 0.001 | +1.479 ± 1.831 | 2/3 |
| quantum_no_entanglement_global − mlp_patch4x4 | 0.003 | +0.630 ± 2.096 | 2/3 |

Every cell, full learning curve, initialization identity and provenance are retained. Training times are not controlled performance benchmarks.

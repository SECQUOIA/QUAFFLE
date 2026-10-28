# Local learning-rate and stability follow-up

Phase: matched60; 12 factorial cells.
All validation fits completed before test replay. Test results are descriptive because the original test split was already consulted.
Normalization uses all global fit images, validation excluded; optimization uses 128 images. This is not pure few-shot learning.

| Model | Fixed LR | Validation IoU mean ± sample SD |
| --- | ---: | ---: |
| quantum_global | 0.001 | 0.6149 ± 0.0165 |
| quantum_global | 0.003 | 0.6067 ± 0.0113 |
| quantum_no_entanglement_global | 0.001 | 0.6107 ± 0.0086 |
| quantum_no_entanglement_global | 0.003 | 0.6147 ± 0.0209 |

| Model | Fixed LR | Descriptive test IoU mean ± sample SD | Optical IoU | SAR IoU |
| --- | ---: | ---: | ---: | ---: |
| quantum_global | 0.001 | 0.5874 ± 0.0271 | 0.6016 | 0.5551 |
| quantum_global | 0.003 | 0.5888 ± 0.0201 | 0.6089 | 0.5447 |
| quantum_no_entanglement_global | 0.001 | 0.5849 ± 0.0297 | 0.5954 | 0.5624 |
| quantum_no_entanglement_global | 0.003 | 0.5906 ± 0.0191 | 0.6051 | 0.5589 |

| Validation paired contrast | Fixed LR | IoU percentage points, mean ± sample SD | Wins |
| --- | ---: | ---: | ---: |
| quantum_global − quantum_no_entanglement_global | 0.001 | +0.420 ± 0.991 | 2/3 |
| quantum_global − quantum_no_entanglement_global | 0.003 | -0.795 ± 1.037 | 1/3 |

| Descriptive test paired contrast | Fixed LR | IoU percentage points, mean ± sample SD | Wins |
| --- | ---: | ---: | ---: |
| quantum_global − quantum_no_entanglement_global | 0.001 | +0.252 ± 1.062 | 2/3 |
| quantum_global − quantum_no_entanglement_global | 0.003 | -0.180 ± 1.184 | 2/3 |

Every cell, full learning curve, initialization identity and provenance are retained. Training times are not controlled performance benchmarks.

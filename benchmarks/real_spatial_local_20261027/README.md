# Real DFC24 spatial pilot

See [interpretation and limitations](INTERPRETATION.md) for the parameter
breakdown, seed variation, knockout evidence, qualitative figures and the
validation-only decision to run additional controls.

Paired seeds: [42, 77, 123]. 64-pixel inputs; at most 128 optimization tiles per seed; 60 epochs.
Normalization uses all global fit tiles with validation excluded; every model shares the same selected optimization and validation indices.
The first seed (42) selects one of the same learning-rate candidates per model using validation IoU. Replication uses the chosen rate. Test replay starts after every fit is complete.

| Model | Parameters | Selected LR | Test IoU mean ± sample SD | Optical IoU | SAR IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| unet_width7 | 510,631 | 0.001 | 0.5813 ± 0.0348 | 0.5946 | 0.5523 |
| unet_width8 | 648,106 | 0.003 | 0.5926 ± 0.0286 | 0.6041 | 0.5671 |
| mlp_global | 500,522 | 0.001 | 0.5791 ± 0.0150 | 0.5910 | 0.5529 |
| mlp_patch4x4 | 500,522 | 0.003 | 0.5957 ± 0.0169 | 0.6112 | 0.5624 |
| quantum_global | 500,394 | 0.001 | 0.5874 ± 0.0271 | 0.6016 | 0.5551 |
| quantum_patch4x4 | 500,394 | 0.003 | 0.5922 ± 0.0125 | 0.6095 | 0.5564 |

| Paired contrast | IoU percentage points, mean ± sample SD |
| --- | ---: |
| quantum_patch4x4 − quantum_global | +0.480 ± 1.459 |
| mlp_patch4x4 − mlp_global | +1.661 ± 0.246 |
| quantum_global − mlp_global | +0.832 ± 1.289 |
| quantum_patch4x4 − mlp_patch4x4 | -0.350 ± 0.595 |
| quantum_patch4x4 − unet_width7 | +1.088 ± 2.242 |
| quantum_patch4x4 − unet_width8 | -0.039 ± 2.135 |

Per-seed histories, validation selections, best epochs, gradient norms, parameter movement, output variation and zero/random knockout results are retained in the JSON records.
This is a centralized pilot with capped optimization and full-fit normalization, not a pure few-image learning experiment or an Anvil/federated result. Seed variation, incomplete convergence and small tuning budgets limit conclusions; this does not establish quantum computational advantage.

Validation selected a checkpoint in the final evaluation window for 5 of 18 runs; those curves warrant longer training before a final ranking.

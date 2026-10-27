# Real DFC24 spatial pilot

See the [complete original-plus-control comparison](COMPARISON.md) and
[interpretation](INTERPRETATION.md) for the paired training effects, selected
rates, parameter counts, audit evidence and limitations.

Paired seeds: [42, 77, 123]. 64-pixel inputs; at most 128 optimization tiles per seed; 60 epochs.
Normalization uses all global fit tiles with validation excluded; every model shares the same selected optimization and validation indices.
The first seed (42) selects one of the same learning-rate candidates per model using validation IoU. Replication uses the chosen rate. Test replay starts after every fit is complete.

| Model | Parameters | Selected LR | Test IoU mean ± sample SD | Optical IoU | SAR IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| quantum_frozen_global | 500,394 | 0.001 | 0.5834 ± 0.0238 | 0.5971 | 0.5519 |
| quantum_no_entanglement_global | 500,394 | 0.003 | 0.5906 ± 0.0191 | 0.6051 | 0.5589 |

| Paired contrast | IoU percentage points, mean ± sample SD |
| --- | ---: |

Per-seed histories, validation selections, best epochs, gradient norms, parameter movement, output variation and zero/random knockout results are retained in the JSON records.
This is a centralized pilot with capped optimization and full-fit normalization, not a pure few-image learning experiment or an Anvil/federated result. Seed variation, incomplete convergence and small tuning budgets limit conclusions; this does not establish quantum computational advantage.

Validation selected a checkpoint in the final evaluation window for 4 of 6 runs; those curves warrant longer training before a final ranking.

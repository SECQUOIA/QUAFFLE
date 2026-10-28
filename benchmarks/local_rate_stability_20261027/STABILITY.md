# Descriptive training stability summaries

Every loss is the recorded per-epoch optimization Dice plus cross-entropy loss. An initially high loss is not automatically instability. The maximum strictly after epoch 10 is a specified descriptive summary, not a diagnostic threshold or a criterion for selecting models.
Adjacent-epoch jumps use later minus earlier loss; zero with no epoch means no positive increase occurred. Ties select the earliest epoch. A late checkpoint was selected in the final evaluation window (last two epochs).

| Budget | Model | Seed | Fixed LR | Initial loss | Maximum loss (epoch) | Maximum after epoch 10 (epoch) | Largest positive adjacent jump (epoch) | Selected epoch; late |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 60 | quantum_global | 42 | 0.001 | 1.2531 | 1.2531 (1) | 0.6267 (19) | 0.1955 (19) | 60; yes |
| 60 | quantum_global | 42 | 0.003 | 1.1231 | 15.5828 (36) | 15.5828 (36) | 12.3548 (36) | 24; no |
| 60 | quantum_global | 77 | 0.001 | 1.1518 | 1.1518 (1) | 0.6111 (11) | 0.0958 (38) | 52; no |
| 60 | quantum_global | 77 | 0.003 | 1.0942 | 1.0942 (1) | 0.6433 (11) | 0.1647 (38) | 48; no |
| 60 | quantum_global | 123 | 0.001 | 1.4378 | 1.4378 (1) | 0.5595 (20) | 0.0611 (48) | 48; no |
| 60 | quantum_global | 123 | 0.003 | 1.2751 | 646.2063 (53) | 646.2063 (53) | 579.2358 (53) | 28; no |
| 60 | quantum_no_entanglement_global | 42 | 0.001 | 1.2527 | 1.2527 (1) | 0.5679 (19) | 0.1176 (19) | 50; no |
| 60 | quantum_no_entanglement_global | 42 | 0.003 | 1.1247 | 1.1247 (1) | 0.5277 (19) | 0.1026 (5) | 40; no |
| 60 | quantum_no_entanglement_global | 77 | 0.001 | 1.1529 | 1.1529 (1) | 0.6105 (11) | 0.0834 (36) | 56; no |
| 60 | quantum_no_entanglement_global | 77 | 0.003 | 1.0986 | 1.0986 (1) | 0.6711 (11) | 0.1159 (6) | 60; yes |
| 60 | quantum_no_entanglement_global | 123 | 0.001 | 1.4372 | 1.4372 (1) | 0.5454 (20) | 0.0631 (35) | 52; no |
| 60 | quantum_no_entanglement_global | 123 | 0.003 | 1.2527 | 1.2527 (1) | 1.1915 (39) | 0.4728 (39) | 28; no |
| 120 | mlp_patch4x4 | 42 | 0.001 | 1.2532 | 1.2532 (1) | 0.4996 (19) | 0.0653 (54) | 118; yes |
| 120 | mlp_patch4x4 | 42 | 0.003 | 1.1237 | 1.1237 (1) | 0.5705 (19) | 0.1304 (9) | 76; no |
| 120 | mlp_patch4x4 | 77 | 0.001 | 1.1517 | 1.1517 (1) | 0.5996 (11) | 0.0991 (38) | 120; yes |
| 120 | mlp_patch4x4 | 77 | 0.003 | 1.0937 | 1.0937 (1) | 0.6033 (11) | 0.1861 (3) | 70; no |
| 120 | mlp_patch4x4 | 123 | 0.001 | 1.4385 | 1.4385 (1) | 0.5586 (12) | 0.0523 (31) | 34; no |
| 120 | mlp_patch4x4 | 123 | 0.003 | 1.2752 | 1.8602 (50) | 1.8602 (50) | 1.3091 (50) | 38; no |
| 120 | quantum_global | 42 | 0.001 | 1.2531 | 1.2531 (1) | 0.6267 (19) | 0.1955 (19) | 108; no |
| 120 | quantum_global | 42 | 0.003 | 1.1231 | 15.5828 (36) | 15.5828 (36) | 12.3548 (36) | 24; no |
| 120 | quantum_global | 77 | 0.001 | 1.1518 | 1.1518 (1) | 0.6111 (11) | 0.0958 (38) | 84; no |
| 120 | quantum_global | 77 | 0.003 | 1.0942 | 1.0942 (1) | 0.6433 (11) | 0.1647 (38) | 120; yes |
| 120 | quantum_global | 123 | 0.001 | 1.4378 | 1.4378 (1) | 0.5595 (20) | 0.0611 (48) | 82; no |
| 120 | quantum_global | 123 | 0.003 | 1.2751 | 646.2063 (53) | 646.2063 (53) | 579.2358 (53) | 28; no |
| 120 | quantum_no_entanglement_global | 42 | 0.001 | 1.2527 | 1.2527 (1) | 0.5679 (19) | 0.1176 (19) | 112; no |
| 120 | quantum_no_entanglement_global | 42 | 0.003 | 1.1247 | 2.7613 (99) | 2.7613 (99) | 2.1586 (99) | 90; no |
| 120 | quantum_no_entanglement_global | 77 | 0.001 | 1.1529 | 1.1529 (1) | 0.6105 (11) | 0.1002 (114) | 112; no |
| 120 | quantum_no_entanglement_global | 77 | 0.003 | 1.0986 | 183.6088 (101) | 183.6088 (101) | 96.3045 (99) | 94; no |
| 120 | quantum_no_entanglement_global | 123 | 0.001 | 1.4372 | 1.4372 (1) | 0.5454 (20) | 0.0631 (35) | 108; no |
| 120 | quantum_no_entanglement_global | 123 | 0.003 | 1.2527 | 1.2527 (1) | 1.1915 (39) | 0.4728 (39) | 112; no |

These summaries describe observed trajectories; they do not establish significance, an optimization mechanism or a causal quantum benefit.

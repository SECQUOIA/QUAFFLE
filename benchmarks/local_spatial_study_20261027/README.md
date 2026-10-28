# Local spatial quantum study

Exploratory same-data duration/readout study; validation-led selection; descriptive test evidence, not untouched-holdout confirmation.

Seven spatially matched configurations received two-rate tuning on seed 42 and selected-rate replication on seeds 77/123 (28 fits).
The validation-selected quantum readout and strongest validation-selected classical control then received the same fresh 120-epoch protocol (8 fits).
All 36 checkpoints, including losing-rate candidates, were strictly restored and checked for finite weights, exact saved settings, initial parameter identity and parameter movement.

All runs use 64-pixel inputs, 128 optimization tiles, 230 validation tiles and 205 test tiles (71 Optical/134 SAR).
Normalization uses all 1,303 global fit tiles while excluding validation; this is not pure few-image learning.
Readout/gate comparisons at independently selected rates include tuning differences. The two common-rate readout rows use only one tuning seed and have no across-seed uncertainty estimate.

| Phase | Model | Total / trainable / core parameters | LR | Validation IoU mean ± SD | Test IoU mean ± SD | Optical IoU | SAR IoU | Late selections |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot60 | local_quantum_z | 503,294 / 503,294 / 16 | 0.003 | 0.6110 ± 0.0138 | 0.5875 ± 0.0222 | 0.5998 | 0.5596 | 0/3 |
| pilot60 | local_quantum_xz_zz | 503,294 / 503,294 / 16 | 0.003 | 0.6093 ± 0.0180 | 0.5782 ± 0.0189 | 0.5901 | 0.5531 | 0/3 |
| pilot60 | local_quantum_frozen | 503,294 / 503,278 / 16 | 0.001 | 0.6090 ± 0.0137 | 0.5823 ± 0.0285 | 0.5920 | 0.5622 | 0/3 |
| pilot60 | local_quantum_no_entanglement | 503,294 / 503,294 / 16 | 0.003 | 0.6123 ± 0.0187 | 0.5927 ± 0.0193 | 0.6103 | 0.5559 | 2/3 |
| pilot60 | local_mlp | 503,562 / 503,562 / 284 | 0.001 | 0.6051 ± 0.0178 | 0.5672 ± 0.0471 | 0.5747 | 0.5536 | 1/3 |
| pilot60 | local_conv | 503,338 / 503,338 / 60 | 0.003 | 0.6053 ± 0.0265 | 0.5908 ± 0.0230 | 0.6103 | 0.5499 | 0/3 |
| pilot60 | local_sincos | 503,438 / 503,438 / 160 | 0.001 | 0.6078 ± 0.0125 | 0.5586 ± 0.0436 | 0.5631 | 0.5495 | 1/3 |
| extend120 | local_quantum_z | 503,294 / 503,294 / 16 | 0.003 | 0.6110 ± 0.0138 | 0.5875 ± 0.0222 | 0.5998 | 0.5596 | 0/3 |
| extend120 | local_sincos | 503,438 / 503,438 / 160 | 0.001 | 0.6088 ± 0.0119 | 0.5641 ± 0.0417 | 0.5706 | 0.5492 | 0/3 |

| Phase | Paired contrast | Split | Mean ± sample SD (IoU percentage points) | Wins | Same selected rates |
| --- | --- | --- | ---: | ---: | --- |
| pilot60 | local_quantum_xz_zz − local_quantum_z | val | -0.166 ± 0.437 | 1/3 | True |
| pilot60 | local_quantum_xz_zz − local_quantum_z | test | -0.927 ± 0.425 | 0/3 | True |
| pilot60 | local_quantum_xz_zz − local_quantum_frozen | val | +0.032 ± 0.919 | 1/3 | False |
| pilot60 | local_quantum_xz_zz − local_quantum_frozen | test | -0.409 ± 1.607 | 1/3 | False |
| pilot60 | local_quantum_xz_zz − local_quantum_no_entanglement | val | -0.302 ± 1.021 | 1/3 | True |
| pilot60 | local_quantum_xz_zz − local_quantum_no_entanglement | test | -1.454 ± 0.534 | 0/3 | True |
| pilot60 | local_quantum_z − local_mlp | val | +0.590 ± 0.856 | 2/3 | False |
| pilot60 | local_quantum_z − local_mlp | test | +2.024 ± 4.351 | 2/3 | False |
| pilot60 | local_quantum_z − local_conv | val | +0.568 ± 2.259 | 2/3 | True |
| pilot60 | local_quantum_z − local_conv | test | -0.329 ± 0.703 | 1/3 | True |
| pilot60 | local_quantum_z − local_sincos | val | +0.318 ± 0.936 | 1/3 | False |
| pilot60 | local_quantum_z − local_sincos | test | +2.890 ± 2.205 | 3/3 | False |
| pilot60 | local_quantum_xz_zz − local_mlp | val | +0.424 ± 0.743 | 2/3 | False |
| pilot60 | local_quantum_xz_zz − local_mlp | test | +1.097 ± 4.090 | 1/3 | False |
| pilot60 | local_quantum_xz_zz − local_conv | val | +0.402 ± 2.176 | 1/3 | True |
| pilot60 | local_quantum_xz_zz − local_conv | test | -1.256 ± 0.556 | 0/3 | True |
| pilot60 | local_quantum_xz_zz − local_sincos | val | +0.151 ± 1.110 | 2/3 | False |
| pilot60 | local_quantum_xz_zz − local_sincos | test | +1.963 ± 2.478 | 2/3 | False |
| extend120 | local_quantum_z − local_sincos | val | +0.214 ± 0.803 | 1/3 | False |
| extend120 | local_quantum_z − local_sincos | test | +2.335 ± 2.180 | 2/3 | False |

| Fixed rate | Seed | Rich Z/X/ZZ minus Z-only validation IoU (percentage points) |
| ---: | ---: | ---: |
| 0.001 | 42 (n=1) | +0.004 |
| 0.003 | 42 (n=1) | +0.240 |

Learning rates were retuned for the 120-epoch extension. A changed selected rate makes the following a joint duration/tuning contrast, not an epoch-only effect.

| Model | Split | LR at 60 / 120 epochs | Same rate | 120 − 60 IoU points, mean ± sample SD |
| --- | --- | ---: | --- | ---: |
| local_quantum_z | val | 0.003 / 0.003 | True | +0.000 ± 0.000 |
| local_quantum_z | test | 0.003 / 0.003 | True | +0.000 ± 0.000 |
| local_sincos | val | 0.001 / 0.001 | True | +0.104 ± 0.139 |
| local_sincos | test | 0.001 / 0.001 | True | +0.556 ± 0.675 |

Of the eight long fits, 8 have an actually run 60-epoch prefix at the same model/seed/rate; 0 have no comparable short replication rate. Exact first-60-epoch histories match in 0 of the 8 comparable cases.
The [prefix inventory](duration_evidence.json) retains every comparable/unmeasured case and the measured maximum loss/validation-IoU differences.

| Phase | Model | On − zero IoU points, mean ± SD | On − random IoU points, mean ± SD |
| --- | --- | ---: | ---: |
| pilot60 | local_quantum_z | +0.036 ± 0.091 | +1.209 ± 2.140 |
| pilot60 | local_quantum_xz_zz | +0.244 ± 0.286 | +0.250 ± 0.243 |
| pilot60 | local_quantum_frozen | +0.329 ± 0.401 | +0.334 ± 0.465 |
| pilot60 | local_quantum_no_entanglement | +0.099 ± 0.120 | +0.257 ± 0.272 |
| pilot60 | local_mlp | +0.573 ± 1.007 | +0.225 ± 0.529 |
| pilot60 | local_conv | -0.048 ± 0.061 | -0.041 ± 0.060 |
| pilot60 | local_sincos | +0.746 ± 0.807 | +0.520 ± 0.467 |
| extend120 | local_quantum_z | +0.036 ± 0.091 | +1.209 ± 2.140 |
| extend120 | local_sincos | +0.541 ± 0.435 | +0.518 ± 0.341 |

Zero knockout removes the entire projected block output. Random knockout changes core weights while retaining trained projections and input dependence.
Z-only pads eight readout slots with zero; its equal stored projection count does not give those inactive columns capacity.
Separable ZZ readouts retain classical product features. Larger classical core capacities are reported rather than artificially reduced.
The 120-epoch fits reuse the same seeds/data and are a duration check, not independent replication. Test scores did not choose finalists or learning rates.
Fit times are retained for provenance, not treated as controlled throughput measurements. These are local classical-simulator results; Alan owns Anvil execution.

Plots: [paired outcomes](plots/paired_quantum_classical.png), [validation curves](plots/validation_curves.png), [input-angle saturation](plots/input_angle_saturation.png).

[Individual angle/scaling diagnostics](individual_angle_audit.json) retain each coefficient, selected-minus-initial movement and a Dice+CE derivative on the fixed first-eight optimization tiles. These are local probe derivatives, not accumulated training gradients or proof of parameter usefulness. Frozen gradients remain absent.

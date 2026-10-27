# What the real-data pilot establishes

A small classical MLP bottleneck reproduces the model-size reduction and has
the highest mean test IoU in this comparison. The results contain a narrower
positive global-quantum contrast, with substantial seed variation and very
weak dependence on the quantum output at inference. They do not establish
overall quantum superiority.

The [generated results](README.md) contain all six models and paired contrasts;
[per-seed scores](per_seed.csv) retain the individual outcomes. Differences
below are **IoU percentage points**, with sample standard deviation across
three paired seeds, rather than a confidence interval.

- Global quantum minus global MLP: **+0.832 ± 1.289 points**, wins in two of
  three test seeds. The individual gains are +0.219, −0.036 and +2.313 points;
  the mean is sensitive to seed 123.
- Patchwise MLP minus global MLP: **+1.661 ± 0.246 points**, wins in all three
  seeds. Retaining spatial information also benefits a classical bottleneck.
- Patchwise quantum minus global quantum: **+0.480 ± 1.459 points**, wins in
  two seeds. This spatial gain is less consistent than the MLP gain.
- Patchwise quantum minus patchwise MLP: **−0.350 ± 0.595 points**, wins in
  one seed. Patchwise MLP has the highest mean test IoU: **0.5957**, compared
  with **0.5922** for patchwise quantum and **0.5926** for the ordinary width-8
  U-Net. These small differences do not demonstrate statistical equivalence.
- Patchwise quantum minus the nearest-count width-7 U-Net: **+1.088 ± 2.242
  points**, wins in two seeds. Uniform narrowing changes the entire U-Net;
  the MLP comparison better isolates bottleneck processing.

## The parameter reduction is in the bottleneck

Counts below are unique parameter objects, including shared blocks only once.
The [count and initialization audit](provenance/parameter_breakdown.json)
and its [source](provenance/parameter_breakdown.py) retain the calculation.

| Architecture | Encoder | Bottleneck attention | Bottleneck residual processing | Decoder | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary U-Net, width 7 | 92,085 | 28,728 | 113,568 | 276,250 | 510,631 |
| Ordinary U-Net, width 8 | 115,288 | 32,832 | 148,224 | 351,762 | 648,106 |
| MLP bottleneck, width 8 | 115,288 | 32,832 | 640 | 351,762 | 500,522 |
| Quantum bottleneck, width 8 | 115,288 | 32,832 | 512 | 351,762 | 500,394 |

Global and patchwise variants have identical parameter counts. The quantum
core has 384 rotation angles and 128 input scales; it is 0.1023% of the hybrid
model's parameters. Classical attention remains in the bottleneck. The total
reduction from the ordinary width-8 U-Net is 22.79%; a classical MLP achieves
almost the same reduction. Width 7 is the nearest ordinary U-Net width and is
2.05% above the quantum total. At seed 42, all four Q/MLP variants have exactly
the same initial values for their 499,882 remaining classical parameters.

## Trainability does not establish useful circuit learning

The retained diagnostics show nonzero circuit gradients and selected-parameter
movement. However, global quantum's mean on-minus-zero test change is
**−0.003 ± 0.012 points**, and on-minus-random is **−0.006 ± 0.030 points**.
The corresponding patchwise changes are **+0.103 ± 0.143** and
**+0.131 ± 0.184 points**. [Individual knockout outcomes](knockout_per_seed.csv)
are retained. These interventions suggest weak inference dependence, especially
for global processing; they do not isolate the effect of learning the circuit.

The fixed validation-only rule selected **global** for additional training
controls: quantum beat its matching MLP in all three validation seeds, averaging
0.7865 points, above the 0.5-point threshold. Patchwise averaged 0.3807 points
and did not qualify. The [decision record](provenance/control_trigger.json)
was computed without test-score input. Frozen-at-initialization and
no-entanglement training controls follow this decision with the original
rate grid, seeds and training budget; their [completed results are reported
separately](../real_quantum_controls_local_20261027/INTERPRETATION.md).

## Limits and reproducibility

This is centralized training at 64-pixel resolution with 128 optimization tiles
per seed. Normalization uses all global fit tiles, excluding validation, so this
is not a pure few-image experiment. Each selected model is evaluated on all 205
test tiles: 71 optical and 134 SAR. Pooled IoU uses combined pixel counts.
Three seeds of one dataset measure training/split variation, not replication
across independent datasets.

Five of 18 selected checkpoints fall in the last evaluation window. The
[learning curves](learning_curves.png) also retain a temporary loss spike in
the seed-123 patchwise quantum run. Longer training and broader validation
tuning could change the ranking. The preceding 30-epoch validation-only screen
and its stopping reason are preserved in `validation_screen_30ep.json`.

[Global](qualitative/global/qualitative_median_worst_fp_fn.png) and
[patchwise](qualitative/patch4x4/qualitative_median_worst_fp_fn.png) overlays
include every seed, with median/worst examples selected by the same width-7
classical baseline rule. Their scatter plots include every test tile and both
sensors. Per-tile empty-union IoU is zero; this is not the pooled test metric.
All checkpoint and normalization identities were verified.

Separate sensor-subset views give each sensor its own median/worst selection:
[global Optical](qualitative/by_sensor/global/optical/qualitative_median_worst_fp_fn.png),
[global SAR](qualitative/by_sensor/global/sar/qualitative_median_worst_fp_fn.png),
[patchwise Optical](qualitative/by_sensor/patch4x4/optical/qualitative_median_worst_fp_fn.png),
and [patchwise SAR](qualitative/by_sensor/patch4x4/sar/qualitative_median_worst_fp_fn.png).
Each uses the same combined-trained checkpoints, with all 71 Optical or 134 SAR
test tiles in its scatter plot. Their per-tile scores exactly match the pooled
exports. Metadata retains both subset and original dataset indices.

The producing source is `71d1680c1c565e5880eec44783cf270b1ac918fb`; the original
collection and figures used `33b67bf2f17841f70bd5f3b5a082f9ad64a20689` with matching
executable hashes. Later control support changes those source files. Rebuild
this historical collection from the recorded retention revision; the collector
intentionally rejects a different executable snapshot. Archived audit helpers
contain the original absolute scratch paths. [Execution provenance](provenance/execution.json)
records commands, resource caps, raw artifacts and the one overlapping fit.
Fit durations are not controlled timing benchmarks. These are classical
simulator results; Anvil and quantum-hardware performance are not measured.
Alan owns the Anvil follow-up.

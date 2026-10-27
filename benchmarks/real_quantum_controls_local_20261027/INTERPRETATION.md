# What the additional training controls show

Training the global circuit's rotation angles and input scales produced a small
observed gain over keeping them at initialization: **+0.394 ± 0.342 IoU
percentage points**, positive in all three paired seeds. Both models selected
learning rate 0.001. This is a useful trainability signal within this model;
three seeds do not establish a general advantage.

Removing the entangling gates produced a higher mean test IoU under the same
validation tuning budget: **0.5906**, compared with **0.5874** for the original
global quantum model. The paired original-minus-no-entanglement difference is
**−0.323 ± 1.216 points**, with one win in three seeds. The no-entanglement model
selected rate 0.003 independently from the same two candidates, so that contrast
includes the selected-rate difference. These results do not establish a benefit
from entangling gates.

The highest mean in the complete comparison remains the **patchwise classical
MLP at 0.5957**, with almost the same total parameter count as the quantum model.
The [full comparison](COMPARISON.md) includes every original model, both controls,
selected rates, total/trainable parameter counts, per-sensor scores and paired
sample standard deviations. [Machine-readable comparisons](comparison.json)
retain every seed difference and cross-run audit result.

The original global model was also almost unchanged by post-hoc zero/random
quantum-output knockouts. Taken together, the results support further study of
training dynamics while providing little evidence for an important quantum
contribution at inference. How the trainable bottleneck changes optimization of
the classical model remains a hypothesis to investigate.

## What was held fixed

The global mode was selected by the recorded validation-only rule before these
control fits. Both controls used the original two-rate tuning grid, three seeds,
60 epochs, 128 optimization tiles, 64-pixel inputs, batch size 16 and validation
evaluation every two epochs. All models share the same seed-specific cohorts,
normalization and initial quantum-model parameter values. The paired audit
checked the full configuration except the intended control and selected rate,
as well as recorded runtime settings and original input identity.

The frozen control keeps all 384 angles and 128 input scales fixed, giving
**499,882 trainable parameters out of 500,394 total**. The classical backbone
still trains. The no-entanglement control removes every CNOT ring and trains
all 500,394 parameters. All eight checkpoints were strictly restored, including
control settings and gradient flags; all selected weights are finite. Six
validation-selected checkpoints were evaluated on all 205 test tiles after all
eight fits finished. Every tuning candidate and learning history is retained.

## Limits and reproduction

Four of six control checkpoints were selected in the last evaluation window.
Longer training and a broader validation budget could change the ranking.
The no-entanglement seed-123 curve also has a temporary loss spike and
validation drop. Optimization stability deserves attention in a larger follow-up.
Normalization uses all global fit images, including unused optimization tiles;
this is not pure few-image learning. These are three seeds on one dataset,
using ideal classical simulation, with no statistical-equivalence claim.

The [original pilot](../real_spatial_local_20261027/INTERPRETATION.md), its early
validation-only screen, parameter breakdown, knockout results and pooled/
sensor-subset figures are retained separately. This control run was produced by
`b57572896c821c423fcef1a42ba17de1056ca376` and collected with matching executable
hashes. [Verification provenance](provenance/verification.json) records commands,
source revisions, resource caps, raw artifacts and test evidence. Alan owns the
larger Anvil follow-up; these local results do not predict its ranking.

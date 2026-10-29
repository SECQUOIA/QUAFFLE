# Fixed-rule figures from the completed HQ controls

This protocol fixes the figure population and example rule before new validation
prediction. It produces descriptive figures from the existing 60-epoch study;
it adds no training, hyperparameter selection, test prediction or independent
confirmation. Historical producers, checkpoints and evidence remain unchanged.

## Population and learning curves

Use the complete audited HQ training-controls campaign. Select learned quantum,
frozen quantum, permanently zero bottleneck and compact affine, at the existing
fixed primary learning rate 0.003 and seeds 42, 77 and 123. Do not use the
auxiliary rate 0.001 to select models. The twelve selected histories comprise
nine fresh control fits and three reused learned-quantum fits. The full source
campaign has twenty fit records, twelve fresh and eight reused.

Each fit used 60 epochs, batch eight, 128 optimization tiles, 230 validation
tiles, 128-pixel inputs and base width eight. Normalization used all 1,303 global
fit images and excluded all 230 validation images. These are centralized,
noiseless simulations with the full 16-qubit circuit and ordinary U-Net skips.

Plot training Dice+CE loss and validation foreground IoU separately by seed,
using common colors for the four controls. Retain every recorded epoch and mark
the original selected best epoch. The seed-specific validation cohorts differ;
curves do not establish convergence or statistical superiority.

## Validation predictions and example selection

Use **seed 42 only** for the qualitative comparison. Load all 230 validation
tiles in their saved order, with the saved fit-only normalization, and restore
the four already selected native checkpoints at rate 0.003. No test dataset may
be constructed or evaluated. Reading existing test records and hashing input
files for the historical audit do not constitute new test prediction.

Predict compact affine first. For Optical and SAR separately, retain tiles with
at least 5% ground-truth flood pixels after the original mask preprocessing.
Sort eligible tiles by `(affine foreground IoU, validation index)` in ascending
order. Select the first tile as worst and index `(n - 1) // 2` as lower median.
Retain duplicate selections if a sensor has a single eligible example; label an
empty eligible set explicitly. Persist these selected identities and bindings
before predicting the learned-quantum, frozen-quantum and zero arms. Predictions
from those three arms never affect which examples are shown.
Selection is conditioned on affine errors and can emphasize weaknesses of that
reference. These examples describe individual cases; they cannot establish an
overall model ranking.

Evaluate all four models on exactly the same ordered validation cohort. Retain
per-image and sensor-pooled TP, FP, FN and TN, sample counts, foreground IoU and
prediction disagreements. Match original saved sensor/pooled validation counts
before exporting a verified result. Investigate any mismatch; do not silently
change the example rule or metric tolerance after seeing outcomes.

Foreground IoU is `TP / (TP + FP + FN)`, with zero for an empty denominator.
Pooled scores use summed integer counts, not mean per-image IoU. Prediction
disagreement measures differing labels, not accuracy improvement or causality.
The visual display maps normalized input channels to a labeled display range;
SAR display channels are not asserted to be natural-color imagery.

## Rendering and provenance

Keep inference and rendering separate. Preserve a local cache of predictions,
targets and normalized display inputs, with shapes, dtypes and SHA256 digests.
Load numeric archives without pickle. Bind the cache to historical source,
data, runtime, checkpoint, cohort, normalization and selection identities,
and retain the producing visualization source. Preserve incomplete or failed
attempts and reject unsafe output aliases or nonempty replacement outputs.

The renderer performs no model inference. It produces PDF and PNG learning
curves and error overlays with explicit input, ground truth and four model
columns. TP, FP and FN colors and example-selection captions must remain
readable. It exports machine-readable values and rendering provenance. Disable
variable PDF timestamps and verify whole-file byte identity across two actual
render invocations. A layout change does not require another prediction run.

Inspect the delivered figures visually and independently check their plotted
histories, selected identities and displayed counts against the raw records.
Raw image tiles and checkpoint weights are not committed. Published figures
contain the requested selected validation displays and overlays.

These examples were chosen from validation data already used for checkpoint
selection. They are descriptive illustrations of the tested models, not an
untouched holdout, a new quantum accuracy experiment, an Anvil measurement,
or evidence of quantum hardware or federated accuracy. The prospective full-fit
campaign and actual Anvil profiling remain Alan's responsibility.

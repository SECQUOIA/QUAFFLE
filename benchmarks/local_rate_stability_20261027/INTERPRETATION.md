# What the rate and duration follow-up shows

The global quantum model has a configuration-specific positive signal, while the
matched-rate controls provide little support for a consistent entanglement
benefit. At 120 epochs and learning rate 0.001, the entangled model exceeded the
patchwise MLP on the previously exposed test split by **+1.046 ± 0.899 IoU
percentage points**, positive in all three paired seeds. The primary validation
contrast was **+0.181 ± 1.571 points**, positive in only one seed. At rate 0.003,
the entangled model's mean validation and test scores were below the MLP's. The
low-rate test result is useful descriptive evidence to carry forward; its mixed
validation support leaves superiority unresolved. All ± values here are sample
standard deviations of paired seed differences, not standard errors.

The [120-epoch tables](long120/README.md) separate the gate comparison from the
classical comparison. Entangled minus separable validation IoU was
**+0.037 ± 0.153 points at 0.001**, and **−1.296 ± 1.644 points at 0.003**.
Descriptive test differences favored the separable model at both rates:
−0.433 ± 1.091 and −0.762 ± 1.198 points, respectively. These observations do not
establish a benefit from the CNOT gates.

Longer training mattered at the lower rate. Relative to its matching 60-epoch
fits, the entangled 0.001 model gained **+0.990 ± 0.471 validation points** and
**+1.411 ± 0.620 descriptive test points**, each positive in all three seeds.
The separable controls also improved. Validation checkpoint selection over 120
epochs includes the first 60 epochs, so the expanded selection window must be
considered when interpreting that improvement. The [duration comparisons](duration_comparison.json)
and [60-epoch tables](matched60/README.md) retain every paired difference.

The [prefix audit](prefix60_comparison.json) shows that all 12 matching quantum
cells had exactly the same validation IoU and sensor confusion counts through
epoch 60. None of the 12 complete histories was exactly identical: the largest
training-loss difference was 7.629 × 10⁻⁶, and loss was the only validation field
that differed. These recorded loss differences should not be described as
substantial classification drift.

The higher rate also produced finite loss excursions. The entangled seed-123
fit reached 646.21 at epoch 53; the separable seed-77 120-epoch fit reached 183.61
at epoch 101. The largest recorded MLP loss was 1.86. Their causes remain
undetermined. The [stability table](STABILITY.md) and full training-loss curves
retain the peaks, adjacent-epoch jumps and checkpoint selections. An initially
high loss is not automatically instability; the maximum after epoch 10 is a
specified descriptive summary, not a diagnostic threshold.

This was a local centralized study with three paired seeds, 64-pixel inputs,
128 optimization images, 230 validation images and all 205 test images
(71 Optical, 134 SAR). Normalization used all 1303 global-fit images while
excluding validation, which limits pure few-image learning claims. The 30 cells
comprise eight audited historical fits and 22 new fits; they are not 30
independent seeds. The test split had already informed research decisions and
remains descriptive. The MLP's bottleneck operates patchwise while the circuits
pool globally; encoder and attention context, decoder paths and skips remain.
That spatial-operation difference is part of the hybrid-model comparison.

The [study overview](README.md), both linked phase tables and the stability
records provide the complete results. The [protocol](../../docs/LOCAL_QUANTUM_FOLLOWUP.md),
[reproduction runner](../../scripts/run_local_rate_stability.py) and
[strict collector](../../scripts/collect_local_rate_stability.py) should be used
with the source and input identities in the two frozen plans. These local
results identify a low-rate configuration worth further confirmation; they do
not establish significance, equivalence, a quantum mechanism or performance on
Anvil or physical quantum hardware.

Exact collection commands and retained local-artifact locations are in the
[reproduction instructions](REPRODUCTION.md).

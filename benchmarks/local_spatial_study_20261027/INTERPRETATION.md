# What the spatial/readout study shows

The Z-only spatial circuit produced a positive descriptive comparison with the
validation-chosen sine/cosine control, but the complete control set does not show
a consistent quantum lead. The separable rich-readout circuit had the highest
mean validation and test IoU. The convolution's test mean also exceeded the
Z-only circuit's. These outcomes should accompany any favorable quantum result.

In the 60-epoch pilot, each model received the same two-rate tuning on seed 42, followed by replication
at its selected rate on seeds 77 and 123: 28 fits, with 21 selected results.
The table reports IoU means and sample standard deviations across the three
paired seeds. The rates were selected separately for each model.

| Model | Selected LR | Validation IoU, mean ± sample SD | Descriptive test IoU, mean ± sample SD |
| --- | ---: | ---: | ---: |
| Quantum Z | 0.003 | 0.6110 ± 0.0138 | 0.5875 ± 0.0222 |
| Quantum Z/X/ZZ | 0.003 | 0.6093 ± 0.0180 | 0.5782 ± 0.0189 |
| Frozen quantum Z/X/ZZ | 0.001 | 0.6090 ± 0.0137 | 0.5823 ± 0.0285 |
| Separable quantum Z/X/ZZ | 0.003 | 0.6123 ± 0.0187 | 0.5927 ± 0.0193 |
| Spatial MLP | 0.001 | 0.6051 ± 0.0178 | 0.5672 ± 0.0471 |
| Spatial convolution | 0.003 | 0.6053 ± 0.0265 | 0.5908 ± 0.0230 |
| Sine/cosine features | 0.001 | 0.6078 ± 0.0125 | 0.5586 ± 0.0436 |

Paired contrasts below subtract each seed before taking the mean and sample
standard deviation. Units are IoU percentage points; wins count strictly
positive seed differences. Neither table reports standard errors.

| Left minus right | Validation difference, mean ± sample SD | Validation wins | Descriptive test difference, mean ± sample SD | Test wins |
| --- | ---: | ---: | ---: | ---: |
| Quantum Z − quantum Z/X/ZZ | +0.166 ± 0.437 | 2/3 | +0.927 ± 0.425 | 3/3 |
| Quantum Z − spatial MLP | +0.590 ± 0.856 | 2/3 | +2.024 ± 4.351 | 2/3 |
| Quantum Z − spatial convolution | +0.568 ± 2.259 | 2/3 | −0.329 ± 0.703 | 1/3 |
| Quantum Z − sine/cosine features | +0.318 ± 0.936 | 1/3 | +2.890 ± 2.205 | 3/3 |
| Quantum Z/X/ZZ − frozen Z/X/ZZ | +0.032 ± 0.919 | 1/3 | −0.409 ± 1.607 | 1/3 |
| Quantum Z/X/ZZ − separable Z/X/ZZ | −0.302 ± 1.021 | 1/3 | −1.454 ± 0.534 | 0/3 |

The protocol chooses the quantum finalist between Z and Z/X/ZZ, and the
classical finalist among MLP, convolution and sine/cosine features, using mean
paired validation IoU. That rule selects **Quantum Z and sine/cosine features**.
Sine/cosine was selected by validation even though convolution's 60-epoch test
mean was higher than Quantum Z's. The frozen and separable variants remain
diagnostic controls under that rule; their scores remain in the full comparison.

The readout comparison uses the same selected rate, 0.003: adding X and ZZ
features did not improve the mean here. Z-only retains the same twelve-feature
projection interface by padding unused slots with zeros, so equal stored
parameter counts include inactive projection columns. The frozen and separable
controls both use the richer readout. They therefore do not isolate freezing or
removing CNOTs in the Z-only winner. Within the richer readout, the separable
comparison also shares rate 0.003 and favors the separable model on all three
test seeds. The frozen comparison uses different selected rates and includes
that tuning difference.

Inference knockouts also limit attribution. For Quantum Z, on minus zero-block
output test IoU was only +0.036 ± 0.091 points; on minus random-core IoU was
+1.209 ± 2.140 points. Zero mode removes the whole local block output, while
random mode changes the core within the learned projections. Its lead over
sine/cosine features does not by itself
establish strong reliance on the learned circuit at inference. These perturbation
results and the fitted-model comparisons do not identify a causal mechanism.

All models mix actual neighboring 2 by 2 encoder locations through the same
pre/post projection dimensions, with shared backbone/projection initialization
and unchanged encoder, attention, decoder and skip paths. The local classical
cores retain their reported capacities rather than being forced into a weaker
rank to match the circuit. This is a comparison of complete hybrid models.

The study uses 64-pixel inputs, 128 optimization images per seed, 230 validation
images and all 205 test images (71 Optical, 134 SAR). Normalization uses all
1303 global-fit images while excluding validation; this is not pure few-image
learning. The test split had already informed research decisions and remains
descriptive. Three paired seeds do not establish significance, equivalence,
quantum advantage, physical-device behavior or an Anvil ranking.

## Individual-angle diagnostics

The audit covers 15 selected quantum checkpoints: the 12 pilot checkpoints and
three Quantum Z extension checkpoints. It records each parameter's movement and
one evaluation-mode Dice-plus-cross-entropy derivative on the fixed first eight
optimization images. The absolute derivative threshold is 10⁻¹⁰. Across the 12
nonfrozen records, all four input scales exceed that threshold; the angle counts
are either 8/12 or 12/12. In all three frozen records, angles and scales remain
exactly at initialization, require no gradients and have no probe gradients.

The Z-only final Rot-phase (`omega`) derivative maxima range from 4.13 × 10⁻¹²
to 2.08 × 10⁻⁹; seed 77 crosses the threshold on all four such coordinates,
whereas seeds 42 and 123 cross on none. The richer entangled readout has maxima
from 1.43 × 10⁻⁶ to 1.77 × 10⁻⁵, with all four final phases above threshold.
Final Z-axis phases are unobservable to the ideal Z-only readout; the small
recorded residuals and threshold crossings are retained explicitly. These are movement
and single-probe derivative observations, not accumulated training-gradient
coverage, parameter-importance measurements or evidence of accuracy gains.
The extension repeats the selected Z checkpoints' diagnostic pattern.

## Completed 120-epoch duration check

All eight planned finalist fits and six selected test replays are complete.
Quantum Z selects learning rate 0.003 and sine/cosine selects 0.001 in both
phases. The 120-epoch results are:

| Model | Selected LR | Validation IoU, mean ± sample SD | Descriptive test IoU, mean ± sample SD |
| --- | ---: | ---: | ---: |
| Quantum Z | 0.003 | 0.6110 ± 0.0138 | 0.5875 ± 0.0222 |
| Sine/cosine features | 0.001 | 0.6088 ± 0.0119 | 0.5641 ± 0.0417 |

| Paired contrast | Validation difference, mean ± sample SD | Validation wins | Descriptive test difference, mean ± sample SD | Test wins |
| --- | ---: | ---: | ---: | ---: |
| Quantum Z − sine/cosine, 120 epochs | +0.214 ± 0.803 | 1/3 | +2.335 ± 2.180 | 2/3 |
| Quantum Z, 120 minus 60 epochs | 0.000 ± 0.000 | 0/3 (three ties) | 0.000 ± 0.000 | 0/3 (three ties) |
| Sine/cosine, 120 minus 60 epochs | +0.104 ± 0.139 | 2/3 | +0.556 ± 0.675 | 2/3 |

Longer training did not improve the validation-selected Quantum Z results in
any seed. Its best epochs remain 38, 50 and 26 for seeds 42, 77 and 123,
respectively, and each seed's selected validation and test IoU is exactly
unchanged. Sine/cosine improves modestly in two seeds, narrowing the mean
descriptive test gap. The remaining quantum-versus-sine/cosine comparison still
uses different selected rates, and its positive test mean has mixed primary
validation support.

All eight matching 60-epoch prefixes have exactly identical validation IoU and
sensor confusion counts. Their full histories are not exactly identical: the
largest training-loss difference is 3.576 × 10⁻⁶, and loss is the only validation
field that differs. These recorded loss differences are preserved rather than
treated as classification drift. The 120-epoch validation selection window
contains the first 60 epochs, and the same seeds and cohorts are reused; this
duration check is not independent replication. The complete study contains 36
fits and 27 selected results, rather than 36 independent seeds.

Read the [complete tables and plots](README.md),
[fixed-rule 60-epoch qualitative comparisons](qualitative60/README.md),
[reproduction instructions](REPRODUCTION.md) and
[full-size handoff for Alan](../../docs/LOCAL_SPATIAL_CONFIRMATION.md).

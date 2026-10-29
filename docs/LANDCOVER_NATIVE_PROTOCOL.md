# Native LandCover.ai local protocol

This experiment asks whether the quantum bottleneck helps on the dataset and
five-class task of [HQ-UNet, arXiv:2604.27206v1](https://arxiv.org/html/2604.27206v1).
It is an independent, bounded native-dataset comparison, not an exact reproduction
of the reported 0.8050 test mIoU. The paper does not give the author split,
normalization constants, loss, optimizer schedule, epoch count, seed distribution,
or enough code-level detail to establish that equivalence. No author contact or
Anvil execution is part of this local campaign.

## Architecture and fidelity choices

The separate `utilsTorch/hq_landcover.py` model uses RGB inputs, five logits,
128-pixel inputs, depthwise/pointwise double convolutions, BatchNorm/ReLU, and
64/128/256/512/1024 encoder widths. Following Figure 1, the fourth MaxPool and
DoubleConv run **before** the quantum interface; the prose also calls the
quantum block a replacement of the deepest convolution, an unresolved ambiguity.
Four transpose-convolution decoder blocks use ordinary matching-resolution skips.

Adaptive average pooling makes a 4x4 grid; a 1x1 projection and tanh supply three
angles per qubit in radians without multiplication by pi. All 16 qubits undergo
RX, RY, RZ encoding. For the native 8x8-to-4x4 map, deterministic ordinary
average pooling computes the same disjoint adaptive bins; independent output
and input-derivative checks cover this implementation choice. The loss uses
explicit log-softmax/gather/mean to preserve the same cross entropy while avoiding
the runtime's nondeterministic spatial-NLL CUDA kernel.
The circuit applies
horizontal then vertical filters with eight shared trainable
RY coefficients, and 32 Pauli-Z/X measurements. Figure 2 is interpreted as even
nearest-neighbor pairs followed by odd pairs (brickwork), with lower-index control
and higher-index CNOT target. The full statevector is simulated on CPU in complex128;
samplewise activation checkpointing preserves the input gradient. The CNN runs
on the selected device. A dense 32-to-1024x8x8 map reconstructs the decoder input.
Independent PennyLane definitions check outputs and angle/weight derivatives.

Historical binary factories/configurations and the entire historical simulator
file stay byte-for-byte intact. The native circuit lives in a separate module
and reuses unchanged gate/readout primitives. Noncommuting overlapping gates make
brickwork a real model choice, not an assumed equivalent optimization. Numerical
and historical source-hash guards both remain in force.

Four controls share every non-core initial tensor, classical interface, training
sample/crop/order/D4 schedule and optimizer budget:

1. `quantum`: eight learned circuit coefficients.
2. `frozen_quantum`: same initial circuit, fixed coefficients, input derivatives
   retained through the circuit and into the encoder.
3. `compact_affine`:48-to-32 affine map plus tanh,1,568 core parameters.
4. `zero_bottleneck`: reconstructed deep path always zero, bottleneck parameters
   frozen, ordinary skips and encoder computation retained.

These compare representations/training procedures on a classical simulator.
Stored, trainable, core and bottleneck counts are reported separately. The eight
quantum coefficients do not describe the whole model parameter count. A zero
control's `requires_grad` count does not certify functional participation.

## Data and exposure

Use the official
[LandCover.ai v1 archive](https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip),
MD5 `3268c89070e8734b4e91d531c0617e03`, independently documented by
[TorchGeo](https://docs.torchgeo.org/en/v0.8.1/_modules/torchgeo/datasets/landcoverai.html).
Retain its SHA256, original split-list and splitter hashes. Match the original
splitter's row-major 512 windows and tile indices, including index increments at
incomplete edges. Unlike its JPEG example, save lossless PNG to follow HQ-UNet.
Preserve integer labels: 0 background, 1 building, 2 woodland, 3 water, 4 road.

Prepare only official train and validation windows. Test **IDs** are read to
verify split separation; test image/mask windows are neither prepared, used for
statistics, trained on, nor predicted. Hashing archive bytes is not test evaluation.
All 41 parent orthophotos appear in each official 7470/1602/1602 tile split;
this experiment does not establish scene-disjoint geographic generalization.

The fixed local cohort uses 256 official training tiles and 64 validation tiles.
Stable SHA256 rankings with independent subset seeds 742 and 943 choose the cohorts
without model outputs or class-based resampling. Retain all selected IDs. Every
validation ground-truth class must be present; failure stops planning rather than
searching for a favorable replacement cohort. All three model seeds use this same
cohort, so their variation measures training randomness, not dataset uncertainty.

Each training epoch draws one deterministic random 128 crop per 512 tile and one
paired D4 transform, keyed by seed/epoch/tile. Every selected validation tile is
evaluated as all 16 nonoverlapping 128 crops: 1024 examples, with no random sampling
or augmentation. RGB uses fixed `pixel/127.5-1` scaling. No normalization constants
are estimated from validation or test. File digests, shapes, label counts and
actual observed training schedule digests are checked.

## Frozen training and selection

Use 60 epochs, batch 8 if the exact-width profile passes (batch 4 only as a documented
resource fallback **before** the plan is frozen), no mixed precision, and Adam
with default betas/epsilon, zero weight decay. The local loss is 0.5 unweighted
cross entropy plus 0.5 macro soft Dice across all five classes, an explicit
assumption because the paper does not specify its loss. No class reweighting,
pretraining, scheduler, early stopping, architecture search, or extra post-result
fits are included. At batch 8, each fit has 1,920 optimizer steps and 15,360 sampled
training patches; at batch 4 the same examples require 3,840 steps.

First fit only `compact_affine`, seed 42, at learning rates 0.001 and 0.003. Choose
the **common** rate by its highest validation mIoU; lower rate wins exact ties.
Persist that choice before fitting any quantum/frozen model. Run all four arms
at the chosen common rate for seeds 42, 77, 123, reusing the selected affine seed 42
fit with checkpoint/digest audit. The resulting campaign has 13 total fits,
12 primary rows, one auxiliary classical LR row. This modest common schedule is
not a claim that the quantum arm's hyperparameters were optimized. The LR choice
uses validation and favors a competent classical reference rather than selecting
a rate from a favorable quantum outcome.
Report seed 42 as the calibration seed and the two follow-up seeds separately;
the three-seed average includes calibration and is not wholly independent
confirmation of learning-rate selection.

Evaluate before training (epoch 0) and every 5 epochs through 60. Select the earliest
strict best validation mIoU checkpoint, including epoch 0. Retain history,
training-step and observed schedule hashes, real first-step gradient norms, core
movement, full checkpoint/config, elapsed cost and peak CUDA allocation. Retain
the training loss for every epoch, including epochs without validation. Restore
selected checkpoints strictly and replay validation, checking exact integer
confusion counts. Never use the test split to select or rescue a result.

## Metrics and interpretation

Pool a 5x5 integer confusion matrix over all validation pixels, rows=truth and
columns=prediction. For each class, IoU=TP/(TP+FP+FN). Primary mIoU is the
unweighted mean across **all five classes including background**, with a declared
zero for zero-union classes (the actual cohort contains all five true classes).
Overall accuracy is total correct pixels/total pixels. Report per-class IoU,
true/predicted class pixel counts and secondary foreground-only mIoU. Do not
replace the primary metric with foreground flood IoU or average per-crop scores.

Report each seed, paired quantum-minus-affine/frozen/zero differences in percentage
points, and paired mean/sample standard deviation. Report epoch 0-to-best learning
and the location of best epochs. A practical signal is mean quantum improvement
of at least 1 percentage point over both affine and frozen, with positive paired
differences on all three seeds; quantum should also match or beat zero on average.
This is a descriptive screening rule, not a significance test or general quantum
advantage. Mixed results and training still improving at the epoch cap stay visible.
A weak/short local fit cannot refute the paper; a positive small-cohort result
does not establish its 0.8050 full-test result or hardware/computational advantage.
Export explicit quality flags: the number of nonpositive epoch-zero-to-best
changes, whether all primary arms failed to improve, the number of best epochs
at or after 55, and whether any such late checkpoint leaves the horizon unresolved.
These flags accompany the practical comparison rather than turning a failed
screen into a claim that the paper or quantum learning was disproved.

## Local resource and execution contract

Before freezing the plan, profile one real forward/backward/Adam warmup step,
two timed steps and an evaluation batch for every control, exact native width
and actual batch size. Keep startup cost separate and count it once per fit.
Record output finiteness, real trainable/frozen movement, frozen input gradients,
parameter counts, time and peak allocation. Require paired common initialization.
Profile data are synthetic and do not select scientific outcomes. The plan binds
all executable source dependencies, this protocol, profile, data manifest and
runtime; source changes invalidate an unstarted plan rather than silently reusing it.

Heavy local commands run in an OS scope with 6 GiB resident memory, no swap, two
numerical threads and 12% of the shared GPU's allocator memory. CUDA deterministic
operations require `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Existing shared environments
are inspected and reused, not modified. A persisted 24-hour campaign deadline
includes downtime; require the conservative 2x profile estimate to fit before
launch. A lock prevents concurrent writers. Completed fits are reused only after
bindings/checkpoint/history audits; partial/failed work is sticky and never
silently retried or overwritten. Completed run invocation performs no new fitting
or inference. Failure or deadline exhaustion yields an incomplete-study report,
not new hyperparameters selected from preliminary scores.
The estimate includes every fit, validation pass, selected checkpoint replay,
and a further 30-minute allowance for data loading, checkpoint writes and
collection. It remains a short-profile estimate rather than a duration guarantee.
The original resource probe included startup in per-step timing; this was
corrected before plan freezing or any data training, with Astra's assessment
retained in the preflight package. Validation replay has its own finite deadline
and must be scheduled within the remaining campaign headroom.

Typical commands (absolute data/output paths selected by the operator):

```bash
python scripts/prepare_landcover_native.py --archive /path/landcover.ai.v1.zip --output-dir /path/dataset
python scripts/run_landcover_native.py profile --output /path/profile.json --device cuda
python scripts/run_landcover_native.py plan --dataset /path/dataset --profile /path/profile.json --output /path/campaign
python scripts/run_landcover_native.py run --dataset /path/dataset --output /path/campaign --device cuda
python scripts/collect_landcover_native.py replay --campaign /path/campaign --dataset /path/dataset --output /path/replay --device cuda
python scripts/collect_landcover_native.py export --campaign /path/campaign --replay /path/replay --output /path/exports
```

The PR retains source, original Astra reports, provenance, aggregate results,
validation learning curves and explicitly bounded conclusions. Raw images,
checkpoints and full local caches remain outside Git. Follow the dataset's
attribution/noncommercial/share-alike terms for any redistributed imagery;
numerical summaries do not require copying the dataset into the repository.

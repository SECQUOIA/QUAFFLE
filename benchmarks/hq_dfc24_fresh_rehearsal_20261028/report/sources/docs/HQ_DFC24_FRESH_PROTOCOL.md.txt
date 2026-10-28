# Frozen fresh HQ execution protocol

This file is the runtime-bound protocol for `run_hq_dfc24_fresh.py` and
`collect_hq_dfc24_fresh.py`. It implements the scientific design in
[the DFC24 decision protocol](HQ_DFC24_DECISION_PROTOCOL.md). Producer executable
bytes, this file, actual input identities, runtime, cohorts and a matching
resource profile are frozen before fitting. A later Git commit containing only
reports or reviews does not change those executable identities.

## Modes and models

Both explicit modes use four controls, seeds 42/77/123, learning-rate candidates
0.001/0.003, batch eight, 128-pixel inputs, base width eight, Adam, Dice plus
cross-entropy and paired training-only D4 augmentation. No public arbitrary
epoch, resolution, qubit, seed, rate or cap override is part of this protocol.

| Mode | Epochs | Optimization tiles | Evaluated validation tiles | Evaluation cadence | Selected replay |
| --- | ---: | ---: | ---: | ---: | --- |
| `full` | 60 | All 1,303 fit tiles | All 230 | Every two epochs and final | All 205 exposed test tiles |
| `rehearsal` | 2 | 16 fit tiles | 16 validation tiles | Every epoch | The same validation subset only |

Epoch-zero validation is included. In either mode, normalization uses the full
1,303-image fit partition, excluding all 230 validation images. Rehearsal's
optimization and evaluation subsets are deterministic and paired; their names
and indices, global fit/validation partition and normalization identities are
retained. This is not pure 16-shot learning.

The unchanged full 16-qubit HQ-style architecture uses tanh-bounded input angles
directly as radians, without a pi multiplier, shared horizontal/vertical circuit
filters with eight coefficients, 32 X/Z outputs, full dense reconstruction and
ordinary U-Net skips. Compare learned quantum, the identical initial circuit
with frozen coefficients and retained input gradients, a permanently zero
whole reconstructed bottleneck with unused parameters frozen, and a compact
48-to-32 affine map plus tanh. Non-core initial parameters and BatchNorm buffers
are identical within a seed. Registered/trainable counts are not functional
activity counts.

The learned-quantum producer is the direction fitter and its native checkpoint
restorer; the other three use the training-control fitter/restorer. Preserve
their established model/config/state names. Blindly calling the latter fitter
for learned quantum writes an incompatible format/config and is prohibited.
Every saved model is strictly reconstructed and checked for finite state,
BatchNorm buffers, parameter counts, trainability, control mode and frozen
identities. Historical capped fits are never reused.

## Selection and complete-fit boundary

Fit all four controls at both candidate rates on seed 42: eight fresh tuning
fits. Choose each arm's rate by maximum validation foreground IoU, breaking
exact ties toward the lower rate. Choose each checkpoint by the earliest strict
best validation foreground IoU, including epoch zero. Fit seeds 77 and 123 at
that arm's chosen rate: eight further fresh fits.

The complete grid is exactly 16 fresh fits, with 12 selected checkpoints. Keep
all candidates and failed attempts. Compare common initialization, cohorts,
normalization and actual sample/augmentation schedules within each seed.
Source/data/runtime/config/profile mismatch, incomplete histories, invalid
identities, altered checkpoint bytes or contradictory selection are failures.

Audit all 16 fits and strictly restored checkpoints before writing the completed
fit boundary. Bind the boundary to the plan, selected rates, raw fit records
and checkpoint audits. Only after this boundary may selected replay datasets
be constructed and predictions evaluated. Full mode uses all 205 exposed test
images, retaining 71 Optical/134 SAR identities and normalization. Rehearsal
must not construct or evaluate any test dataset: its 12 selected replays use
the held-out validation subset and are explicitly labeled validation.

Replay records bind the plan, native fit/history/checkpoint bytes, ordered
cohort and split. Before prediction inference, persist an immutable replay
intent with that binding. An intent without a committed matching replay blocks
resume, including after a process kill that could not write an exception marker.
Successful intents remain retained and are validated with their replay.
Existing completed replay bytes are validated and reused;
failed or partial evidence blocks silent retry. A completed resume performs
verification without fitting, inference or rewriting completed raw artifacts.
An incomplete grid has no scientific screen verdict.

## Metrics and interpretation

Flood foreground IoU is the selection and primary metric. Retain pooled and
Optical/SAR TP/FP/FN/TN, sample counts, foreground Dice, background IoU, binary
class-mean IoU, accuracy and saved AUC. Compute pooled metrics from summed sensor
counts, not the mean of sensor/image scores:

```text
foreground IoU = TP/(TP+FP+FN)
background IoU = TN/(TN+FP+FN)
binary mIoU = (foreground IoU+background IoU)/2
foreground Dice = 2TP/(2TP+FP+FN)
accuracy = (TP+TN)/(TP+FP+FN+TN)
```

Empty IoU/Dice denominators give zero, matching the existing evaluator. AUC
requires prediction scores and cannot be recovered from confusion counts.
Compute each within-seed contrast before its mean and sample SD. Fractions and
percentage points must be labeled; multiplying a fractional difference by 100
gives IoU percentage points. Report all candidates and seed-42 tuning optimism,
with follow-up seeds 77 and 123 separately visible.

The complete `full`-mode screen passes only if the mean paired validation gain
over **both** compact affine and frozen quantum is at least 0.01 IoU (one point),
every seed's difference against each is positive, and quantum's mean validation
IoU is at least the zero arm's mean. This is a chosen practical resource screen,
not a significance/equivalence test. Different per-arm selected rates compare
training procedures, not isolated coefficient trainability. Three seeds and
exposed DFC24 outcomes provide no independent-data confirmation, entanglement
benefit or general quantum advantage. Test scores cannot rescue a failed
validation screen. Never add cells or alter this rule after outcome inspection.

The `rehearsal` scientific status is always `not_applicable_rehearsal`; its short
fits and validation-only replays establish software mechanics rather than
accuracy or convergence. Do not merge its scores with the 60-epoch evidence.

## Feasibility, resources and retained attempts

Before running, profile all four actual controls at batch eight on the selected
mode's own training data: two Adam steps, validation, checkpoint I/O and strict
roundtrip. Retain finite outputs/loss/gradients/state, movement, frozen identity,
count/flag/mode checks, measured stage times and CPU/GPU memory with explicit
units and measurement boundaries. Profiling does not load test data. It is an
instrumented feasibility probe, not a controlled throughput comparison. Both
quantum variants retain full 16-qubit CPU computation with sample checkpointing;
the surrounding CNN can use CUDA.

The resource profile must match the run's exact mode, settings, producer source,
protocol, data/cohort and actual runtime. Record the actual operator-selected
CUDA memory fraction and two numerical threads; do not inherit the historical
runtime helper's hardcoded fraction. A different GPU/allocation/runtime requires
its own profile. A local profile supplies no Anvil throughput or policy evidence.

Require finite positive CLI wall-clock allowances. Persist the learning budget
and fixed deadline so restarting cannot silently create a new allowance;
downtime may consume it. Completed read-only verification may be separately
bounded without admitting new learning. Preserve timeout/failed/partial evidence
and restore any temporary signal handlers. Use an OS advisory lock for one live
writer; an old lockfile alone is not evidence of a live job. Reject unsafe
input/output aliases, unexpected nonempty outputs and mismatched resumes. Run
heavy commands under an OS memory bound with swap disabled and retain logs.

The collector audits saved records, compatible runtime and current input/source
identities, strictly restores all checkpoints and verifies selected replay
counts/bindings without fitting or new inference. Collection has its own source
identity and deterministic exports. Keep raw weights, imagery, histories and
failed attempts persistently; publish only the authorized evidence packet.

The software can be implemented, tested and rehearsed locally. Alan still owns
Anvil profiling and full-data execution. No SSH, cluster submission, contact,
new dataset download or quantum-hardware execution is part of this local work.

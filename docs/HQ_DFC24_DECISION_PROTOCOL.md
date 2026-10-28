# DFC24 decision study and Alan handoff

## Current status and purpose

This is a **prospective protocol for an exploratory scale-transfer screen** on
our cleaned binary DFC24 data. The fresh runner provides separate `full` and
`rehearsal` modes; a short rehearsal verifies execution rather than accuracy.
The full-data 60-epoch campaign and Anvil jobs remain unperformed. **Anvil launch
requires a profile on the actual allocation and frozen source, data, runtime
and resource identities.** Alan owns Anvil execution. Local rehearsal and
handoff preparation do not submit jobs or contact him.

The completed [local HQ controls](../benchmarks/hq_training_controls_20261028/INTERPRETATION.md)
give small, mixed learned-versus-frozen validation differences; the permanently
zero bottleneck has the highest validation mean. The quantum model retains the
highest descriptive test mean, with a much smaller contrast against compact
affine than against the larger MLP. These findings do not justify a quantum
accuracy-superiority claim or an open-ended architecture campaign.

The next question is whether the **learned-circuit training procedure**, given
the same two-rate tuning opportunity and full-fit training budget, produces a
practically useful validation improvement over frozen-circuit and compact
classical procedures. Independently selected learning rates can differ, so their
contrast cannot isolate circuit-coefficient trainability. Even coincident rates
restrict any conclusion to the architecture, cohorts, and budget tested. This
design measures neither entanglement benefit nor a general quantum advantage.

## Data-exposure prerequisite

Use conflict-policy `drop`, the bound cleaned manifests, and fit-only
normalization. The retained local archive contains 1,533 clean training and 205
clean testing pairs. For each existing seed, all training tiles already belong
to either the 1,303-image normalization-fit pool or the 230-image validation
pool. A 128-image optimization cap does not create an untouched reserve. All 205
test images (71 Optical, 134 SAR) have already been evaluated.

The [decision audit](../benchmarks/hq_dfc24_decision_20261028/README.md)
records the inspected archive and its limits. No verifiably untouched
scene/event partition was identified there. Manifest pixel hashes establish
identity and exact deduplication, not geographic independence. The retained TIFF
copies contain no checked geolocation/GPS tags or ASCII metadata. This makes no
claim about metadata in upstream DFC24 distributions. Numeric tile names,
collision suffixes, and `geo_cluster` feature clusters are not authoritative
scene/event identifiers. Known duplicate/leakage/conflicting-label exclusions
must not be restored as new test data.

A group holdout first requires an authoritative tile-to-scene/event/acquisition
map tied to the manifest. Repartitioning previously exposed images does not
erase that exposure. A recovered map can support a prospective grouped stress
test, with normalization fit on training groups only. Genuinely independent data
requires documented unexposed cohorts/outcomes and a separately frozen design;
it is not an automatic branch of this protocol.

## Frozen model and fitting design

Keep the existing full 16-qubit HQ-style bottleneck, base width eight, 128-pixel
inputs, actual batch size eight, ordinary encoder-to-decoder skips, depthwise
BatchNorm/ReLU convolutions, 32 X/Z outputs, and full dense reconstruction.
Encode tanh outputs directly as radians, **without multiplying by pi**. This
differs from other pi-scaled spatial implementations in the repository.

Four arms share the same initial non-core parameters and BatchNorm buffers:

| Arm | Intervention |
| --- | --- |
| Learned quantum | Train the eight circuit coefficients and classical network. |
| Frozen quantum | Keep the identical initial coefficients fixed; retain input derivatives through the circuit. |
| Permanent zero | Zero the complete reconstructed bottleneck during training and evaluation; freeze its unused parameters. |
| Compact affine | Replace the core with a 48-to-32 affine map and tanh, without an artificial narrow hidden layer. |

The earlier MLP remains historical evidence. Registered and trainable parameter
counts are not functional-activity counts: downstream trainable weights can
multiply the zero arm's permanently absent input. A whole-model efficiency claim
would also require an ordinary U-Net baseline and controlled cost measurements;
this four-arm screen alone does not establish one.

Optimize **all 1,303 fit tiles**, excluding the 230 validation tiles from both
optimization and normalization. Keep 60 epochs, Adam, Dice plus cross-entropy,
deterministic training-only D4 augmentation, validation every two epochs and at
the final epoch, and identical paired sample/augmentation schedules. This gives
163 optimizer batches per epoch and 9,780 per fit. Relative to the capped study,
both sample coverage and optimizer-step budget change; historical differences
cannot be attributed to sample size alone. This remains a base-eight model at
128 pixels, not full-capacity or native-resolution evidence, and 60 epochs do
not demonstrate convergence.

Give each arm the same two candidate learning rates, 0.001 and 0.003, on seed 42.
Select its rate solely by best validation foreground IoU; break equal-rate scores
in favor of the lower rate. Select checkpoints by the earliest strict maximum
validation foreground IoU, including epoch zero. Fit seeds 77 and 123 at each
arm's selected rate. The complete campaign is **16 fresh fits**:

- Eight seed-42 tuning candidates: four arms times two rates.
- Eight selected-rate follow-ups: four arms times two further seeds.
- Twelve selected primary checkpoints: four arms times three seeds.

No capped historical fit is eligible for reuse. Retain all tuning candidates,
histories, source/runtime/input/cohort/normalization identities, gradients,
checkpoint bytes, and unsuccessful attempts. Frozen parameters must remain
byte-identical to initialization. Audit all 16 fits and strictly restored
checkpoints before any selected test replay. Preserve bound replay outputs;
reject mismatched resumes and incomplete/failed attempts rather than silently
overwriting or retrying them. A failed or incomplete grid has **no screen verdict**.

## Metrics, screen and interpretation

Foreground flood IoU remains the selection and primary reporting metric. Report
foreground Dice, binary class-mean IoU, overall pixel accuracy, AUC where saved
probabilities permit it, and all pooled/Optical/SAR confusion counts and metrics.
Define pooled metrics from summed pixel counts, not the mean of image or sensor
scores. In fractions:

```text
flood IoU      = TP / (TP + FP + FN)
non-flood IoU  = TN / (TN + FP + FN)
binary mIoU   = (flood IoU + non-flood IoU) / 2
flood Dice     = 2*TP / (2*TP + FP + FN)
pixel accuracy = (TP + TN) / (TP + FP + FN + TN)
```

Use zero for an empty IoU/Dice denominator, matching the existing evaluator;
report the relevant counts. AUC cannot be recovered from confusion counts alone.
Binary mIoU is not the five-class LandCover.ai metric or a reproduction of that
paper. Publish both metric definitions rather than choosing the favorable one.

Compute paired quantum-minus-control differences within each seed, then their
mean and sample standard deviation. Also publish each of seeds 77 and 123
separately from seed 42: seed 42 participates in rate selection and the screen,
and all selected epochs were chosen on validation. These are adaptive validation
results on previously studied data, not independent-data confirmation.

The chosen practical screen passes only if all of the following hold:

- Mean paired validation gain is at least **0.01 IoU (one percentage point)**
  against both compact affine and frozen quantum.
- Every paired seed difference against each of those two controls is positive.
- Mean quantum validation IoU is at least the permanent-zero mean.

This threshold is a resource-prioritization rule, not a calibrated significance
or equivalence test. It was chosen after the earlier local study; do not apply it
retroactively as a preregistered test of those results. A pass motivates an
independent evaluation. A failure supplies no basis from this screen to
prioritize a larger accuracy campaign; it does not prove circuit effects absent.
Report every outcome, late checkpoint selections and failures. The exposed test
set is descriptive and cannot rescue a failed validation screen. Do not add
seeds, architectures, rates or epochs, or change the threshold after inspecting
outcomes. Changes required by feasibility must precede accuracy inspection and
create a separately identified protocol revision.

## Executable local audit

From the repository root, using the existing project Python environment:

```bash
python scripts/audit_hq_dfc24_decision.py \
  --data-root /path/to/retained/dfc24 \
  --output-dir results/hq_dfc24_decision_audit
```

Use a new empty output directory. This command audits the retained data and
saved local evidence and derives metrics from existing confusion counts. It
does not fit models, repeat inference, recover missing AUC, or launch Anvil.
The committed audit packet records the actual source and input identities.

## Fresh runner and local rehearsal

The [completed local rehearsal](../benchmarks/hq_dfc24_fresh_rehearsal_20261028/README.md)
executed all 16 fresh fits and 12 selected validation replays in 163.64 seconds.
Both local four-arm profiles passed, including the full-mode cohort configuration
with two Adam steps per arm. All selected checkpoints were epoch zero; this is
software execution evidence with no scientific accuracy verdict. The packet
retains deterministic regeneration, an independent numeric/identity audit and
guarded completed-resume evidence. Actual Anvil profiling and the full 60-epoch
campaign remain unperformed.

`scripts/run_hq_dfc24_fresh.py` implements the separately identified fresh
campaign under the runtime-bound [fresh execution protocol](HQ_DFC24_FRESH_PROTOCOL.md).
It dispatches learned quantum to the direction fitter and the other
three arms to the training-control fitter, preserving their native checkpoint
formats and strict restorers. The historical cap-128/reused-fit runner and its
frozen evidence remain unchanged. Neither old capped fits nor an incompatible
profile can be reused for a fresh full-data campaign.

Both modes execute the same 16-fit rate/seed selection sequence and audit every
checkpoint before 12 selected replays. Their scopes differ explicitly:

| Mode | Optimization | Evaluated validation | Epochs | Selected replay |
| --- | ---: | ---: | ---: | --- |
| `full` | All 1,303 fit tiles | All 230 validation tiles | 60 | All 205 exposed test tiles |
| `rehearsal` | 16 fit tiles | 16 held-out validation tiles | 2 | The same 16 validation tiles; no testing |

The rehearsal preserves normalization over all 1,303 global-fit tiles and
excludes the entire 230-tile validation partition. It is not pure 16-shot
learning, a convergence experiment, an independent confirmation or evidence for
quantum superiority. Its scientific screen status is `not_applicable_rehearsal`.
No test dataset is constructed or evaluated in rehearsal or profiling.

The CLI requires an explicit mode and finite wall-clock allowance. CUDA memory
fraction is an explicit operator choice, recorded in the runtime identity.
Profile all four actual arms before fitting: two Adam steps at batch eight,
validation, and checkpoint I/O/restoration, with measured stage times and memory.
These instrumented observations are feasibility probes, not controlled
throughput or accuracy comparisons. A profile from the workstation does not
establish Anvil feasibility or match a different allocation/runtime.

For local rehearsal, activate the existing project environment and run each
heavy command in the established OS memory limit with swap disabled and two
numerical-library threads. The following are local safety allowances, not
predicted runtimes or cluster settings:

```bash
python scripts/run_hq_dfc24_fresh.py profile \
  --mode rehearsal --data-root /path/to/retained/dfc24 \
  --work-dir results/hq_fresh_profile_rehearsal \
  --device cuda --gpu-memory-fraction 0.12 --budget-seconds 600

python scripts/run_hq_dfc24_fresh.py run \
  --mode rehearsal --data-root /path/to/retained/dfc24 \
  --profile results/hq_fresh_profile_rehearsal/resource_profile.json \
  --work-dir results/hq_fresh_rehearsal \
  --device cuda --gpu-memory-fraction 0.12 --budget-seconds 1800

python scripts/collect_hq_dfc24_fresh.py \
  --work-dir results/hq_fresh_rehearsal \
  --data-root /path/to/retained/dfc24 \
  --output-dir results/hq_fresh_rehearsal_report
```

Use new output directories. Resume a run by repeating its original `run`
arguments with `--resume`; it validates source, data, runtime, profile, cohorts
and completed artifacts before reusing them. Failed/partial evidence is
preserved and blocks a silent retry, including an immutable replay intent left
without its committed output after a process kill. An OS advisory lock prevents concurrent
writers. The saved budget/deadline bounds new learning work across resumes;
downtime may count against it. Completed verification does not authorize new
learning or overwriting replay files. The collector restores checkpoints and
audits saved metrics, without training or prediction inference, and exports a
deterministic report separately from the raw run.

## Alan's remaining launch gates

The implemented runner does not substitute for measurements on Anvil. Before
submission Alan must:

1. Review the fresh-runner and local-rehearsal evidence for the exact source
   revision to be used, including the 16-fit/12-selected flow, native quantum
   checkpoint dispatch, full-fit cohort guard, source/data/runtime bindings,
   deterministic selection, strict restoration, failure/resume behavior and
   complete-fit boundary before test replay.
2. Profile all four actual arms under his Anvil allocation, including the
   full 16-qubit CPU forward/backward path with per-sample checkpointing, CNN
   placement, data loading, validation and checkpoint I/O. Retain measured time,
   peak CPU/GPU memory, numerical behavior, source and runtime. A local random
   input profile is not a cluster throughput estimate. Fix a finite compute
   allowance before accepting the campaign.
3. Freeze the final input/protocol/source/runtime/resource identities before
   fitting. If infeasible, stop and revise before accuracy inspection. Do not
   silently substitute four qubits, reduce coverage, or change epochs.

Run `profile` and then `run` with `--mode full` on that allocation. Both require
Alan's chosen finite `--budget-seconds`; `run` requires the resulting matching
`--profile` file. Use the same data, device and CUDA memory fraction for both.
The collector command above also audits a full-mode run and computes the
predeclared screen only for its complete full-mode grid. Source-byte bindings
remain authoritative when Git HEAD later advances for report/review retention.

Alan supplies account, partition, walltime, interpreter and resource choices
through the established local cluster configuration. No scheduler policy,
allocation, GPU speedup or job duration is guessed here, and the workstation's
allocator cap is not prescribed as cluster policy. Full-data fits, independent
holdout confirmation, federated-accuracy and quantum-hardware evidence remain
unperformed. Broader issue [#42](https://github.com/SECQUOIA/QUAFFLE/issues/42)
remains open; this handoff does not close its production criteria.

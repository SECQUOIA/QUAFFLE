# HQ training controls — independent Astra initial source preflight, round 1

**Findings: Blocking 2; Nonblocking 2; Questions 0. Address the findings before starting the real-data grid.**

<!-- gh-review-pr:sha=d110b1e61d3023ecebde4c89579e86b6efaca7ca -->

Reviewed SECQUOIA/QUAFFLE local branch `research/hq-training-controls` at `d110b1e61d3023ecebde4c89579e86b6efaca7ca`, against base `d66014f16c8d3dc2c7bff4cc69091806563c4d51`, in `/tmp/quaffle-hq-training-controls-20261028`. This is the first review of this new campaign, not a repeat approval of the earlier direction study. The four committed changed files are the protocol, runner, control-model module and model tests. Concurrent untracked collector/test work is excluded from this verdict. No real-data results or final publication readiness are assessed here.

## HQTC-ASTRA-001 — Nonblocking: bind the legacy audit implementation
<!-- gh-review-pr:finding=hqtc-bind-legacy-audit-source -->

**Location:** `scripts/run_hq_training_controls.py:26`, `:356–357`.

`run` imports and relies on `scripts.collect_hq_dfc24_direction.audit_campaign` to accept the eight historical fits, but that module is absent from `SOURCE_FILES`. Consequently its implementation can change without changing the source identity checked by profiling, frozen-plan resume, or the before/after source checks. The existing ten historical producer hashes correctly protect the original fitting computation; they do not protect this newly executed acceptance gate.

**Remedy:** include the legacy collector in the new campaign's executable source inventory, retain that binding, and add a guard regression showing that a changed audit dependency is rejected before fitting/reuse. Preserve the original historical producer inventory separately.

## HQTC-ASTRA-002 — Blocking: require fresh and reused execution runtimes to agree
<!-- gh-review-pr:finding=hqtc-match-reused-runtime -->

**Location:** `scripts/run_hq_training_controls.py:357–377`.

The legacy collector validates package/interpreter identities against its current environment, but it does not require the fresh campaign's device/GPU runtime to equal the historical producer's runtime. The runner compares the new profile only with the current runtime, then accepts historical fits without comparing `old['plan']['runtime']`. For example, a newly generated CPU profile can satisfy the current checks while the reused baselines were trained with the CNN on CUDA. That would make the learned-versus-frozen/zero comparison change execution conditions as well as the intended training control, contrary to the exact-reuse protocol.

I independently checked the supplied profile and historical plan: their runtime dictionaries currently match exactly. This is a missing executable safeguard, not evidence that the submitted profile is already contaminated.

**Remedy:** compare the fresh execution runtime with the historical producer runtime before accepting the plan or starting any fresh fit. Include device, GPU identity, CUDA/runtime versions, thread settings and the declared allocator policy. Add a mismatch regression which proves fitting and test replay are not reached.

## HQTC-ASTRA-003 — Nonblocking: identify inactive zero-arm angle probes
<!-- gh-review-pr:finding=hqtc-label-inactive-angle-probes -->

**Location:** `scripts/run_hq_training_controls.py:175`, `:216`; inherited `angle_probe`/`angle_diagnostics` behavior.

The permanent-zero bottleneck correctly returns zeros without using its projections or circuit. However, the shared angle-probe pre-hook independently calls the inherited encoder and preprojection, then saves the result under the same angle-probe fields as active controls. For the zero arm these are hypothetical values from an unused path, and their changes can reflect the evolving upstream features even though the bottleneck is permanently absent. Without an explicit activity/scope label, later saturation tables can misrepresent these values as angles used by that model.

**Remedy:** either omit inactive-path angle measurements or retain them with explicit machine-readable inactive/counterfactual scope and corresponding report labels. Add a regression for the zero arm so it cannot be reported as an active angle path. Do not change the permanent-zero training semantics.

## HQTC-ASTRA-004 — Blocking: preserve replay evidence on resume
<!-- gh-review-pr:finding=hqtc-preserve-resumed-replays -->

**Location:** `scripts/run_hq_training_controls.py:420–437`.

A resumed campaign reuses completed fits, but unconditionally runs all primary test evaluations again and writes each existing `replays/<id>/seed<seed>.json` through `pilot.write_json`, which overwrites the file. A resume after partial replay therefore destroys earlier successful raw replay records; another interruption can also leave a mixture of replay attempts without an attempt identity. Source/runtime checks do not make discarding historical measured bytes acceptable. The protocol asks to retain raw evidence and keep new baseline replays distinct from historical records.

**Remedy:** validate and reuse compatible completed replay records, or refuse/retain an immutable prior attempt before creating an explicitly separate replay attempt. Bind reuse to the checkpoint, fit, plan and ordered cohort rather than merely trusting an existing filename. Preserve partial failures and add a regression proving that resume or failure cannot silently overwrite prior replay evidence.

## Checked behavior and limits

The model implementation otherwise follows the intended interventions. Frozen quantum retains the identical initial circuit weights and permits gradients through circuit inputs; permanent-zero freezes the entire unused bottleneck and cannot be reenabled by ordinary diagnostic cleanup. Encoder/decoder skips still train. The compact affine core is 48→32 plus tanh, with 1,568 parameters and no narrow hidden rank. Legacy quantum/MLP construction, parameter names and state semantics remain unchanged. The model tests meaningfully check these contracts, common initialization including BatchNorm buffers, frozen identity, live input gradients, zero-path exclusion and strict restoration of trainability.

The fixed-rate grid is explicit: five arms × three seeds at 0.003, plus five seed-42 auxiliary fits at 0.001, giving 20 cells with eight historical and 12 fresh fits. No new rate selection occurs. The runner's complete-grid pairing and strict checkpoint audits precede its 15 primary test replays. The historical six test records remain in their original root. The protocol correctly scopes the exposed test as descriptive, distinguishes 1,303 normalization images from 128 optimization images, and withholds entanglement/general quantum-advantage conclusions.

I inspected the retained **19 model tests passed in 5.65 seconds** and independently verified the supplied three-case batch-eight profile's source/protocol hashes against this candidate. The profile records finite actual-model training steps, strict restoration and unchanged frozen parameters; frozen/zero core gradients and movement remain zero, while the compact core updates. These observations establish the recorded mechanics/feasibility scope, not convergence or accuracy. No new heavy execution, model fitting, source change, Git mutation or GitHub write was performed during this review. The unfinished runner/collector tests and eventual full result packet require subsequent verification/review; this report is not formal human approval or merge authorization.

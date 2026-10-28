# Astra spatial preflight review 1

Reviewer: Astra (`gpt-6-astra`), independent read-only reviewer. Stage: initial LOCAL-SUBSTRATE source preflight, review round 1 of 3. Date: 2026-10-27.

Repository: SECQUOIA/QUAFFLE. Worktree: `/tmp/quaffle-local-spatial-kernel-20261027`. Branch: `research/local-spatial-kernel`. Inspected head: `cca85993d5d610ebe111998f593871bf6b6b614f`. Base: `dc5a3e7b140dfe73f27306f95b0e99e020b4b326`. The worktree was clean at the start and end of inspection. There is no PR yet; GitHub review, CI, and approval states do not apply to this local review.

This is the original reviewer report. Coordinator dispositions and later fixes belong in separate records. I used the assigned `gh-issue-to-human-review`, `gh-review-pr`, and shared delegation, resource, testing, checkout, and scientific-verification instructions. I made no source, Git, or GitHub changes, launched no accuracy fits, and used no GPU, Anvil, SSH, or additional reviewer.

Finding counts: **Blocking: 0; Nonblocking: 2; Question: 0.**

## Nonblocking — SPATIAL-ASTRA-001: emit the promised common-rate readout contrasts
<!-- gh-review-pr:finding=spatial-fixed-rate-readout-report -->

Location: `scripts/run_local_spatial_study.py:97–109`, especially the exclusive use of `summary['records']` at line 98. Protocol reference: the spatial/readout pilot in `/tmp/quaffle-local-rate-stability-20261027/docs/LOCAL_QUANTUM_FOLLOWUP.md` promises both fixed-seed readout contrasts and paired selected-rate contrasts.

`summarize()` currently produces only the three-seed, validation-selected-rate contrasts. It correctly discloses unequal selected learning rates, but never emits the seed-42 Z-versus-richer-readout comparison separately at each common candidate rate. Selecting a different rate for each readout can entangle the readout contrast with tuning, which is precisely why the protocol includes the fixed-seed comparison.

Remedy: derive validation IoU differences in percentage points between `local_quantum_xz_zz` and `local_quantum_z` at each of 0.001 and 0.003 from the existing `tuning_candidates.json`. Validate the seed/rate/model pairing and matched inputs, retain both results regardless of which rate is selected, and label them single-seed (`n=1`) tuned-checkpoint comparisons. Keep these distinct from the selected-rate, three-seed analysis. No additional fit or test-set replay is needed.

Suggested verification: a fixture where the readouts choose different best rates should still produce both common-rate contrasts with the expected signs and values; missing or duplicate pairs should fail clearly. This is a reporting gap, not a defect in the planned 28-fit schedule or its preserved raw evidence. Resolve before publishing results; adding it before freezing the producing source avoids a later provenance split.

## Nonblocking — SPATIAL-ASTRA-002: distinguish real angle saturation from hypothetical values
<!-- gh-review-pr:finding=spatial-classical-angle-diagnostic -->

Location: `utilsTorch/local_spatial.py:173–180` and `scripts/run_real_spatial_pilot.py:111–112`.

Every local block has `angle_diagnostics()`, so the forward hook records `input_angles` for all seven configurations. For `mlp` and `conv`, however, the method substitutes unit scales and computes `tanh(tokens)` even though `_evaluate_core()` sends their raw tokens straight into their learned core. These controls do not encode bounded angles. The resulting saturation values describe a hypothetical transformation and can mislead interpretation of the actual control mechanism.

Remedy: emit angle diagnostics only for `quantum` and `sincos`, or explicitly separate and name a hypothetical projected-token statistic for the other cores. A missing/inapplicable angle field is preferable to an apparent measurement of nonexistent angle saturation.

Suggested verification: exercise the output-probe hook for all four core kinds; quantum and sine/cosine should retain actual bounded-angle diagnostics, while MLP and convolution should not expose those values as actual `input_angles`. No training-computation change is required.

## Scientific and implementation assessment

I inspected all nine changed files and the unchanged model, data-loader, training, selection, knockout, and checkpoint paths they call. The extracted windows are stride-one, row-major 2×2 neighborhoods of each projected scalar map, with bottom/right replicate padding. The same core is shared across positions and projected maps; the twelve-feature reconstruction preserves that order. The MLP is full-input-rank, and the convolution implementation is algebraically a shared ordinary scalar 2×2 convolution. The larger classical cores are reported openly rather than artificially weakened.

The circuit and readout ordering agree with the stated four-qubit construction. Frozen mode freezes both angle and input-scale parameters while retaining gradients through the inputs/projections; the separable control removes the CNOTs and retains the same feature interface. Its ZZ features still provide classical products. Z-only padding preserves stored parameter shapes but has genuinely inactive projection columns; equal stored counts are not equal effective capacity. The existing residual affine modulation and encoder/decoder skips remain available in every arm. Neither nonzero aggregate gradients nor these controls establish usefulness of every angle or a uniquely quantum learning benefit.

The runner schedules seven models × two seed-42 rates plus two selected-rate replications per model: 28 fits. The duration extension schedules two finalists with the same policy: eight fresh fits. Finalists use mean validation IoU only, with the documented tie rules; under identical shared parameter counts, using total parameter counts for the classical tie is equivalent to comparing core counts. The duration extension is unconditional on which family wins. Data order uses explicit seeded loaders, test replay follows completion of the stage's fits, and the existing exposed test set is correctly described as exploratory. Fresh 120-epoch fits on the same seeds are a duration check, not independent replication.

## Evidence and limits

- Inspected the retained full-suite log: **328 passed, 21 warnings, 245.09 s**. This applies to the pre-final-delta source manifest, not a new full-suite execution at this head. The later CPU-RNG isolation/tag changes are explicitly recorded in `/tmp/quaffle-local-spatial-kernel-final-source.json`; the final local-kernel cohort passed **27 tests in 12.08 s**. The unchanged study-runner tests were covered by the full suite. I read the relevant tests, including independent PennyLane observable/gradient parity, spatial support, convolution equivalence, chunk/checkpoint gradient parity, shared initialization, frozen semantics, checkpoint restoration, selection, and pairing guards.
- My independent CPU probe verified all **183** final-manifest file hashes against this worktree and confirmed the imported local source path. For seeds **77 and 123**, all seven configurations had identical non-core state tensors. It also checked no-entanglement/zero-Rot readouts against analytical `cos(angle)`, `sin(angle)`, and cosine-product formulas, with independent input/scale gradients, and confirmed that exactly the unused Z-only projection columns have zero data gradient. All passed. Results: `/tmp/quaffle-local-followup-reviews/spatial-astra-preflight-probes-1.json`.
- That probe used the assigned Python 3.11 environment, CPU only, `PYTHONDONTWRITEBYTECODE=1`, the provided dependency path, two numerical-library threads, and `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0`. No source-tree writes occurred.
- I read the coordinator's frozen-source GPU profile at `/tmp/quaffle-local-spatial-profile-20261027/resource_profile.json`. All seven arms completed two finite-loss forward/backward/Adam steps at batch 16 and 64×64, with the same shared-initialization hash. Peak CUDA allocation was approximately **710.50–713.38 MiB**, below the stated 12% allocator budget. This supports the batch resource gate; it is not a controlled throughput comparison, proof of sustained training stability, or accuracy evidence. I did not independently rerun GPU profiling.
- No accuracy results existed for this study during review. Full-duration stability, comparative accuracy, all produced artifacts, and eventual PR/CI state require later evidence. This review provides no formal human approval or permission to merge.

Launch recommendation: **no Blocking finding prevents the bounded, prespecified accuracy study**. Address the two reporting items before publication, preferably before freezing its producing revision. Keep the planned learning rates, seeds, controls, 28-fit pilot, and eight-fit duration extension unchanged; do not alter the plan in response to test outcomes.

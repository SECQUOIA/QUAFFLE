# Astra source preflight — round 1

Reviewed base `5b16d3c40e2dbe852462aac438fcd04d5b96ab01` and pinned head `0dcc5e0e61daab7636d918355158b004b5f4e472` in `/tmp/quaffle-hq-handoff-figures-20261028`. The worktree was clean at the final inspection. This is the first review of the launch-package/figure task, using `gh-review-pr` in local, no-post mode; it does not reuse the preceding task’s convergence budget.

**Verdict: no blocking findings. The pinned source is ready for the authorized bounded validation-prediction and rendering stage. This is source-preflight readiness, not acceptance of unproduced scientific figures or Anvil execution.**

## Findings

- **Blocking:** none.
- **Nonblocking:** none.
- **Questions:** none requiring a source change or clarification before the bounded local stage.

## Inspected scope and evidence

I inspected all eight changed paths: the two protocols, launch generator, validation-cache collector, renderer, both test files, and the narrow `.gitignore` allowlist. I traced relevant inherited contracts through the historical controls/direction collectors and native restorers, training-data construction, validation evaluation, and cluster-configuration resolver. Historical producer/model behavior remains outside the changed implementation and is not newly validated by this review.

The launch generator emits three separately submitted jobs rather than submitting them. Its shell construction quotes command arguments, rejects header/control-character injection, binds source/protocol/config bytes, requires finite phase limits and cleanup headroom, and guards data/output aliases. Running requires an explicitly accepted passed full-profile digest and compatible full settings; native runtime/data/checkpoint guards remain authoritative. Resume is explicit and does not reset native deadlines. The documentation correctly leaves actual scheduler policy, allocation feasibility, profiling and submission to Alan, without presenting workstation timings as Anvil measurements.

The figure collector uses the fixed historical rate 0.003 and twelve histories (nine fresh and three reused), with no rate reselection. Qualitative inference uses seed 42’s ordered 230 validation tiles and original normalization over 1,303 non-validation images. The native checkpoint dispatcher handles learned quantum correctly. I checked batch-eight ordering, the actual channel-bearing mask shape and argmax conversion, evaluation-state preservation, and exact per-sensor confusion-count parity. Affine predictions determine worst/lower-median examples within each sensor’s flood-eligible set, with validation ordinal resolving ties; selection is persisted before other-arm inference. Partial attempts are retained and cannot be silently resumed.

The renderer imports no model or Torch code. It checks the complete cache inventory, bounded numeric archive headers/shapes/dtypes, binary masks, finite displays, source/checkpoint/cohort bindings and the affine phase boundary. It independently recomputes mask counts, disagreements and selection ranks. It retains all recorded curve epochs and original best-epoch markers. The final display mapping correctly maps normalized [-1,1] values to [0,1] with fixed clipping and no per-image stretching; its endpoint test covers negative features. Captions disclose affine-conditioned selection, reused histories, exposed validation data and the limits of these descriptive illustrations.

I inspected retained focused-test logs: **46 launcher tests passed in 4.09 seconds**, and **27 figure tests passed in 13.44 seconds** (one expected duplicate-ZIP-member warning from a rejection fixture). These are synthetic software checks, including shell dispatch stand-ins, native-shaped inference-helper fixtures, corruption/failure checks and repeat rendering; they are not actual DFC24 prediction evidence. I carried this evidence forward after matching the final source files. I did not rerun tests, infer predictions, fit models, contact a scheduler, modify sources or publish a GitHub review. The full suite was still pending when this report was written.

Pinned SHA256 identities:

- Launch generator: `bf79b1404c8181cca84c27161f35f75b574a040ed7fb48de0927d64b7a5105d4`
- Validation collector: `e4273e627101e604013ca9790cf0ad0db4d9834ad1c0e82e918ad40e1a4fd3e0`
- Renderer: `98f178884f689a9718d29a3e155ddb1b821e6bfe003cab5e5dc773339cb5163b`
- Launcher tests: `2c55db4ddb4b46c949f4489a76a349815920124fede2bdb77fed1f21699effa7`
- Figure tests: `317bae095042a9314ecfe7ab1423c0296b8aeea918ec1dd8c4126154666f2eab`

## Remaining results-stage gates

Run the authorized validation-only cache collection once with explicit resource bounds; require exact saved-count parity and unchanged parent/data identities. Render twice from that frozen cache and compare complete output bytes. Independently check actual history values, example ranks/identities and displayed counts, and visually inspect both delivered figure types at useful reading size. Retain full-suite and execution evidence, then pin the results for the planned second review. No test prediction, new training or scheduler submission is needed for these gates. No manuscript source is part of this preflight review.

A later report-retention commit must be distinguished from the source head inspected above.

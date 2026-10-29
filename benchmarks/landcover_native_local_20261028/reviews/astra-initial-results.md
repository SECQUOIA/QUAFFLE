# Astra initial empirical-results review — PR #67

**Verdict: no actionable findings; technically ready for the publisher’s final live checks and completion workflow.**

Reviewer: Astra, sole scientific reviewer. Stage: initial empirical-results review, round 1 of this PR’s three-round budget. This is a local, read-only review; no GitHub review, approval or merge was performed.

Inspected [SECQUOIA/QUAFFLE PR #67](https://github.com/SECQUOIA/QUAFFLE/pull/67) at head `01574d2f8f409f66a25abf194760308b4f238c7d`, against base `604cef0e164bf1ba2be301b82107ddbb0fb39a90`. This verdict covers that exact head, not a later commit retaining this report.

## Findings

- **Blocking:** none.
- **Nonblocking:** none.
- **Question:** none requiring resolution before publication.

## Scientific assessment

The headline is supported: this bounded native-dataset pilot demonstrates meaningful learning and a small descriptive learned-versus-frozen difference, but no convincing gain over the affine control. Independent count-based recomputation gives learned quantum **46.057592%** mean validation mIoU, affine **45.900817%**, frozen quantum **45.239335%**, and zero bottleneck **46.048724%**.

The paired quantum differences are **+0.156775 pp** versus affine (2/3 positive seeds), **+0.818257 pp** versus frozen (3/3), and **+0.008867 pp** versus zero (2/3). The frozen seed-77 difference is only **+0.008327 pp**. Separating calibration seed 42 leaves follow-up means of **−0.016726 pp** against affine and **+0.282995 pp** against frozen. These are descriptive observations, not evidence of statistical significance, equivalence or quantum advantage. The exact, preregistered practical screen fails; the documentation preserves that outcome without changing the criterion.

The comparison remains conditional on a common learning rate selected using affine seed 42, one fixed capped cohort, paired initialization/crop schedules and equal training work. It does not compare independently optimized procedures. The publication states these limitations and keeps the auxiliary rate fit separate from the twelve primary fits. All twelve primary checkpoints occur at epoch 55 or 60, with substantial improvement from initialization. Completed training therefore establishes learning, not convergence. Competitive zero-bottleneck performance is appropriately treated as motivation for a future intervention, not proof that the learned model ignores its bottleneck.

The cohort audit and publication correctly distinguish disjoint tile IDs from geographic independence: all 35 selected validation source parents also occur in the selected training cohort. Three model seeds measure optimization variation on that same cohort, not dataset uncertainty. Validation was reused for rate/checkpoint selection, and no test-score or paper-replication claim is made. The changed architecture, circuit schedule and dataset also prevent attributing a difference from DFC24 solely to the dataset. No further fitting or outcome-dependent extension is necessary to publish this bounded negative/mixed result.

## Evidence checked

- Independently reconciled all **133 changed paths** with GitHub’s live paginated pull-files inventory. The worktree was clean at the inspected head. Scope is the root README and results packet; no project executable source changes are introduced.
- Verified all **119 publication-map entries** against original local bytes, SHA256 values, sizes and committed Git blobs. Verified all **12 frozen source/protocol bindings** against current files and the base commit. Copied operational and audit sources remain inert and byte-preserved.
- Independently recomputed all **169 validation confusion-matrix mIoUs**, primary arm means/sample SDs and exact paired differences from integer counts using separate standard-library arithmetic. Checked all **793 training-history step records**, first-best checkpoint epochs, raw/embedded history agreement, and selected confusion-count agreement for all **12 retained replays**.
- Read the independent numeric oracle and exercised only its pure standard-library arithmetic/table functions against the published packet. Every cell of all **ten CSV tables**, the summary and exact screen matched. This is separate from my own headline/count recomputation; no production metric helper, model or inference was used.
- Checked export inventory hashes and retained replay-member hashes. The original replay inventory’s omitted `.replay.lock` is an explicitly excluded operational file, not an unexplained missing scientific artifact. Reconciled selected cohort IDs, class counts and the 35-parent overlap against the frozen plan.
- Inspected the actual PNG and the supplied rasterization of the actual PDF. Labels, legend, axes and displayed trajectories are readable and agree with the retained data and mean/sample-SD description. The plots do not imply convergence or a separated quantum advantage.
- Reviewed completion, replay, export, audit and staging records and their source contracts. The retained strict-replay and completed-path evidence supports the stated twelve replays, unchanged completed artifacts and two byte-identical seventeen-file exports. Prior **874-passed** source-test evidence is carried forward because the bound producer source is unchanged; it is not a new test run on this results-only head.

## Limits and remaining gates

This review did not rerun training, forward inference, the broad test suite, original TIFF-window decoding or checkpoint deserialization. Those execution claims rely on the retained, source-bound records and earlier checks; my new checks were static reads, hashes, numerical recomputation and visual inspection. Weights and imagery remain local, so the public packet supports numerical reconstruction and provenance inspection, not a standalone checkpoint replay without the documented external inputs.

The final live read still showed the exact inspected head, OPEN/draft, CLEAN, no submitted reviews and an empty status-check rollup. That empty rollup is not a CI pass. The publisher must refresh live gates before posting/completing the PR and distinguish an author-account COMMENT from formal human approval. No scientific or artifact correction is requested before that step.

<!-- gh-review-pr:sha=01574d2f8f409f66a25abf194760308b4f238c7d -->

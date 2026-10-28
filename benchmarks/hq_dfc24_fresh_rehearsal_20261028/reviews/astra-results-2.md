# Astra fresh-runner results review — round 2

**Verdict: ready for the coordinator's final verification and authorized PR/merge workflow. B1 is resolved. No remaining Blocking findings, Nonblocking findings, or unresolved Questions.** The actual local execution establishes the intended software rehearsal and instrumented profile mechanics. It does not establish an accuracy gain, convergence, Anvil feasibility, or independent scientific confirmation.

## Reviewed identities and scope

- Worktree: `/tmp/quaffle-hq-fresh-runner-20261028`, branch `research/hq-fresh-runner`.
- Base: `96f97c657312d80d19937a5119834f4c3a25dce0`.
- **Inspected results head:** `1af10fbad5c484011c77f6da79707b5b96f0c027`; clean at final inspection.
- **Actual producing/collecting/tested source:** `ef9119d9b168a4d1330802b86bac4ed66198fe66`. The runner, collector, frozen execution protocol and both test files are byte-identical at the inspected results head. The later commit retains results and updates status documentation.
- Runner SHA256: `1f5a244db37cd563cd0e0039de997dd140cee90527c4e50b585973dbf8ecbb81`.
- Corrected collector SHA256: `bd673749bfb37bd2488aa3b62940980d31db1c9411e6533f5690893ff21d6d4f`.
- Frozen execution protocol SHA256: `7149e9116e4d47661d497542d32d9b2c82ba1c8d9cd1dc5c4d1917e2224da8f4`.
- Public evidence: `benchmarks/hq_dfc24_fresh_rehearsal_20261028/`. I verified all 136 outer-manifest entries and exact inventory coverage, giving 137 packet files including the outer manifest. Its 99-file collector output is byte-identical to the actual retained generation. No checkpoint weights, images, PDFs or zip files are published in this packet.
- Private manuscript: current `main.tex` SHA256 `86ca98415fca48b301de77baa4a99c985ee7cf5d3dc57d3ebbdd46bffddfa3fe`. This and the other five changed manuscript artifacts remain separately retained, ignored/local files, not PR source.

This is review round 2 of the fresh executable-runner task. I reapplied the local/no-post `gh-review-pr` workflow, carrying forward the unchanged-source review from round 1, and inspected the correction, current documentation, actual generated evidence and local manuscript status changes. Root remains sole publisher. No helper, source/Git/remote/environment mutation, model fitting, prediction, profile, or cluster action was performed in this review. My additional execution consisted of read-only standard-library numerical and byte-identity checks; I did not rerun broad tests or physical experiments.

## Findings and original finding resolution

- **Blocking:** none remaining.
- **Nonblocking:** none requiring a change in this delivery.
- **Questions:** none unresolved for this scope.

### B1 resolved — inclusive scientific thresholds use exact audited counts
<!-- gh-review-pr:finding=hq-fresh-exact-gate-boundary -->

The corrected full-mode gate constructs `Fraction` foreground IoUs from the audited pooled TP/FP/FN counts. It evaluates mean gain against exactly `1/100`, per-seed positivity and quantum-versus-zero mean comparison rationally. Floating-point summary/SD reporting remains unchanged. This is the bounded correction requested in preflight; it does not introduce a tolerance that admits genuinely sub-threshold results.

I read the source/test delta and independently executed the actual corrected `identity`, `stats` and `analysis` function ASTs with standard-library globals. Exact `29/100 − 28/100` passes; a gain from `289999/1000000 − 28/100` fails; exact equality with the zero-arm mean passes; rehearsal remains `not_applicable_rehearsal`. The regression fixture now derives consistent saved sensor/pooled metrics from integer counts and also checks a seed tie failing despite an adequate mean. B1 is addressed, not merely documented.

## Actual execution evidence assessed

The recorded own-DFC24 run contains **16 entirely fresh fits and 12 selected validation replays**. All histories contain epochs 0, 1 and 2, and every fit records four optimizer steps. The selected-rate grid follows the frozen four-arm/two-rate seed-42 design and seeds 77/123 follow-ups. I verified the raw metrics/history/checkpoint hashes for every cell, all retained producer/source snapshots, and the validation-only replay scope. The normalization and cohort records preserve 1,303 global fit images and exclusion of all 230 global validation images while using 16 optimization and 16 evaluated validation tiles per seed.

**All 16 fit selections, including the 12 primary checkpoints, are epoch zero.** Each arm's seed-42 candidates tie at their initial best foreground IoU, so 0.001 is selected by the declared lower-rate rule. This is a legitimate execution of the selection policy, not a reason to add fits or reinterpret the rehearsal as an accuracy experiment. It exercises the initial-checkpoint path, complete orchestration and restoration. The packet clearly discloses this limitation and keeps the scientific screen inapplicable.

Both actual four-arm profiles passed. The rehearsal and full-mode profile driver durations are 58.388 and 76.942 seconds; the complete rehearsal driver duration is 163.642 seconds. Full mode here means a **profile using the 1,303/230 cohort configuration with two Adam steps per control**, not a 60-epoch campaign. I verified the eight profile-checkpoint hashes, bound source identities, two-step gradient observations and movement: learned quantum and affine cores have nonzero gradients/movement; frozen and zero cores remain unchanged. The recorded resources identify the actual local CUDA/CPU execution and allocator fraction. Instrumentation, warmup and CPU/cgroup accounting limits are disclosed; these timings are not controlled throughput or Anvil estimates.

The two collector executions ended with exit 0. I independently verified exact equality of every file across both 99-file exports, including the manifest itself, and checked the inner manifest's complete coverage. Saved CPU restoration audits and collector logs cover all 16 native checkpoints; collection does not repeat prediction inference. My own numerical oracle starts from raw saved integer confusion counts and uses rational formulas before checking reported values. **All 1,320 checks passed** across the 72 selected-observation rows, 120 metric-summary rows and 90 paired-contrast rows, including sample SD, signs, per-seed percentage-point differences and follow-up-only summaries. I also inspected the separately retained coordinator oracle source and result rather than substituting its assertion count for my own work.

The actual completed-resume probe installs tripwires on both fitters, `HQDirectionUNet.forward`, and test-dataset loading. It returns successfully while allowing strict checkpoint reconstruction. I read that probe and its result and independently rechecked the current hashes, sizes, modification times and inventory of **all 82 raw campaign files** against its preserved pre-resume inventory. They match. This supplies actual completed-resume evidence, beyond the synthetic source tests.

## Tests, provenance and source continuity

The final full repository suite at corrected source `ef9119d9b168a4d1330802b86bac4ed66198fe66` reports **691 passed, 21 warnings in 279.61 seconds**, exit 0. The outer execution driver reports 282.367 seconds; these are different timing boundaries, not conflicting results. I verified its log digest, exact relevant source hashes, zero-swap/6-GiB cgroup record and terminal result. I carry this evidence forward because those source bytes are unchanged at the inspected head. The earlier full suite and focused tests remain separately labeled and are not additional unique passing tests.

Execution bindings identify the actual commands, frozen source and retained inputs; the producer excludes the collector from its fitting dependencies, while collection records its own source identity. The packet retains the original preflight report byte-for-byte and labels the coordinator's B1 disposition separately. Historical benchmark packages and native model/fitter sources remain unchanged. No real-data fit is inferred from a synthetic test, and this review's source inspection is not presented as experimental execution.

## Separately retained manuscript

I verified all six current artifact hashes and before-copies, current PDF/build and render bindings, and the 30-member zip's uniqueness/CRC and exact member bytes against the local package. The `main.tex` diff contains only the two intended implementation-status replacements; historical scientific statements and tables are not revalidated by this work. The updated README distinguishes the completed validation-only rehearsal from the still-unperformed 60-epoch study and Anvil profiling.

The recorded builds produce IOP 39-page and arXiv 33-page PDFs. I directly viewed the bound IOP page 19 and arXiv page 15 renders carrying the revised status paragraph: text is readable and within margins, without observed clipping or overlap. I carry forward the recorded complete builds and the coordinator's four-page visual inspection; I did not rebuild or inspect every historical page. These local artifacts and the published hash summary are distinct deliverables.

## Readiness and remaining work

No further scientific fit, source fix, or full review round is justified solely to retain this report unchanged. Before final publication/merge, the coordinator should verify the retention-only delta and refreshed manifests, unchanged tested/producing source and private-artifact identities, and live GitHub head/CI/review/merge gates under the authorized workflow. This local assessment is not a posted formal approval and makes no hosted-CI claim.

The exact reviewed head is `1af10fbad5c484011c77f6da79707b5b96f0c027`; a later report-retention head should be identified separately. Actual Anvil profiling and resource acceptance, the full 60-epoch campaign and its exposed test replays, and genuinely independent confirmation remain pending. The current software rehearsal supplies no quantum-accuracy, hardware or federated-accuracy conclusion, and broader issue #42 remains open.

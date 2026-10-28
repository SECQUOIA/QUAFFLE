# Astra implementation review — round 2

**Verdict: ready for the coordinator's final verification and authorized repository closeout. No Blocking findings, Nonblocking findings, or unresolved Questions.** The implemented scope is a historical exposure/metric audit, a prospective protocol with explicit launch gates, and separately retained local manuscript corrections. It does not deliver or validate the future full-fit experiment.

## Reviewed identities and scope

- PR: https://github.com/SECQUOIA/QUAFFLE/pull/63, draft, `SECQUOIA/QUAFFLE`.
- Inspected head: `be1c9258c227b80325ecce122943c58c8c132a8a`.
- Base: `4231eb89b5913ef3595588397651a8b8a6cca8be`.
- Worktree: `/tmp/quaffle-hq-dfc24-decision-20261028`; clean at inspection and final readback.
- Public scope: 28 changed paths, comprising the audit script, its tests, protocol, root README addition, and 24 retained packet files. I independently matched all 28 paths and Git blob identities to the coordinator's retained GitHub inventory. Historical producer/collector/model packages are unchanged by this diff.
- Private manuscript source: `/home/bernalde/repos/QUAFFLE/papers/QUAFFLE_Journal_2026/main.tex`, SHA256 `3bb03511837d3a8770479d28df74133bffaac62cf7e8ab1b22ba73e7fcc7864b`. I checked the actual seven private artifact hashes against the retained verification record and read the source/companion-note/README changes against preserved originals. These ignored files are not part of the PR head.
- Audit script SHA256: `3a366038c1ea401614f03a620d7bd5e485927a1ef6bdae4b4491e951a7de4d40`; protocol SHA256: `2140c4fbd48529aae2ad7ed6ea7220917ea272b600ae37465991ef8e272f9c89`.

Applied `gh-review-pr` in local, no-post mode, including its testing/evidence and review-state guidance and the previously read shared scientific, ownership, and resource guidance. Root remains the sole publisher. No helpers, source edits, fits, inference, environment changes, cluster access, or GitHub writes were performed during this review.

## Findings inventory

- **Blocking:** none.
- **Nonblocking:** none requiring a change in this delivery.
- **Questions:** none unresolved for the bounded scope.

The absence of findings is not an approval of an unimplemented runner, a claim of Anvil feasibility, or independent confirmation of a quantum benefit.

## Assessment of the implementation

The audit is appropriately separate from the future campaign. Its exact grid checks concern the completed common-rate five-arm study: 20 records, eight historical reuses, 12 fits originally fresh in that campaign, and 15 saved primary replays. It never presents these as the proposed four-arm 16-fresh-fit campaign, and does not retrospectively apply the new practical gate.

The script checks the complete parent artifact inventory, retained source/protocol/profile bindings, fit/history and replay bindings, cohort pairing, validation schedules and earliest strict best checkpoints. It hashes current kept image/mask bytes against the bound manifest without decoding or loading model tensors. Output/input overlap and nonempty output are rejected. The tests exercise consequential failures, including altered raw bytes, source snapshot corruption despite a resealed outer manifest, missing/duplicated fit and replay grids, wrong replay bindings, numeric identity types, input/output aliasing, and disagreement between saved metrics and counts. The hand-computable confusion-count example and absent-class convention supply an independent metric basis; these are useful guards rather than just assertions that an implementation copies itself.

Metric definitions and units are consistent: sensor counts are summed before pooled metrics; seed metrics are aggregated using sample SD; paired contrasts are formed within seeds and multiplied by 100 for percentage points. Binary class-mean IoU remains distinct from flood IoU and the motivating five-class task. AUC is only range-checked, not reconstructed or advertised as independently verified from counts.

The exposure conclusion is appropriately bounded. Normalization and validation exhaust the 1,533 clean training tiles per seed; cap128 does not leave an untouched reserve, and the 205 test outcomes are exposed. The separate retained TIFF inspection describes local copies only. I checked aggregation of its 1,937 per-file records and canonical digest: one IFD each, no recorded checked geolocation/GPS tags or ASCII metadata. The observed tag inventory contains only ordinary image-layout/sample tags. This supports the limited local archive statement, not a claim about upstream DFC24 metadata or event independence.

The protocol resolves the scientific ambiguity: equal per-arm two-rate tuning compares **training procedures** and cannot isolate coefficient trainability when rates differ. The 16-fit/12-selected arithmetic, lower-rate and earliest-epoch ties, incomplete-grid rule, one-percentage-point threshold, positive-seed requirement, and no-rescue-by-exposed-test rule are explicit. It discloses seed-42 tuning optimism and validation checkpoint selection, requires follow-up seeds to be shown separately, and does not confuse three seeds with equivalence or independent-data confirmation. Full-fit sample coverage and increased optimizer work are distinguished.

Deferring the fresh runner is the right scope decision here. The protocol plainly documents the cap128/reuse dependency and incompatible naive quantum checkpoint-producer/restorer route. It gives no unsupported full-data training command. Future implementation, all-four-arm Anvil profiling, and frozen resource/input/source identities are explicit prerequisites. It does not guess scheduler policy, timing, or GPU speedup.

## Original plan findings

All five items are addressed by the actual deliverables, not only by the coordinator's disposition:

- `dfc24-tuned-procedure-estimand`: resolved by the procedure estimand and attribution limits.
- `dfc24-screen-interpretation`: resolved by deterministic selection, adaptive-validation disclosure, finite practical gate, and failure/no-expansion rules.
- `dfc24-full-fit-exposure`: resolved by the exposure audit and precise full-fit/budget/independence limits.
- `dfc24-manuscript-boundary`: resolved locally with separately identified artifacts and historical-evidence caveats.
- `dfc24-runner-readiness`: resolved through the explicitly accepted implementation/profiling gate, not through a claimed runner implementation.

I also verified that the original round-1 report is retained byte-for-byte.

## Checks actually performed and evidence carried forward

I independently verified both public packet hash manifests, their nonoverlap and complete coverage of all 24 files; all four implementation/documentation source hashes match the tested source revision. I checked the retained full-suite log digest. This supports carrying forward the recorded **609 passed, 21 warnings** full-suite result at `3249d63611e2b4a51fd07af2a85d391bdbaebc43`; subsequent changes at the reviewed head are retained evidence. The focused 31-test result is part of that suite, not additional unique coverage. I read the tests and did not repeat the full suite merely because this is a new review round.

My own standard-library numerical check read retained raw summary records and independently formed rational count metrics, then checked **1,470 values** across all 90 primary rows, 30 summaries and 120 paired contrasts. It passed, including sample SD, paired signs and seed differences. This is separate from the inspected retained 2,640-check oracle, which additionally covers all 60 fit-validation rows and byte-identical regeneration. I inspected that oracle's source without executing its fixed-output writer. I did not repeat the coordinator's two complete 3,476-file audit runs; their generated outputs and provenance were verified and their scope is described accurately.

For the local manuscript, I verified preserved original artifact hashes and the packaged `main.tex` bytes. I read the scientific corrections and new historical/prospective separation. The new control table and paired contrasts agree with retained evidence. Unsupported equivalence, skip-dominance, and automatic transfer of legacy noise/factorization claims are corrected. Historical tables/propositions and unrelated campaign claims remain separate; this review has not revalidated them.

I verified the retained rendered-page hashes and directly viewed the actual IOP page 18 and arXiv page 14 renders carrying the new table. The table, caption, and surrounding claims are readable, without observed clipping or overlap. I carry forward the recorded successful IOP 39-page and arXiv 33-page builds with their stated warnings; I did not rebuild or visually audit every page. Compilation is not validation of historical scientific claims.

## Readiness and remaining gates

This review covers exactly `be1c9258c227b80325ecce122943c58c8c132a8a` and the separately hashed private manuscript. A later commit retaining this report is not automatically the head I inspected: the coordinator should verify its retention-only delta, manifests, unchanged source/private identities, and fresh live GitHub state before publication or merge.

The retained GitHub snapshot has no executable hosted checks or Actions runs, no formal reviews, and no protected-branch rules. Local evidence must not be described as hosted CI success or formal independent peer approval. Root should refresh those live gates and the exact head under the already authorized workflow. No additional full scientific review round is necessary solely to retain this report unchanged; substantive changes require proportionate reinspection within the remaining budget.

Full-fit runner implementation, actual Anvil profiling and execution, and genuinely independent evaluation remain unperformed future work. Issue #42 should remain open. The public repository deliverable and the completed local manuscript package should be reported separately.

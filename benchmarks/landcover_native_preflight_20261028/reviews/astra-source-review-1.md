# Astra source review — PR #66, round 1

Reviewer: **Astra**, sole scientific/code reviewer for this assignment. Stage: **initial full changed-source and preflight review**, before scientific training. Reviewed [SECQUOIA/QUAFFLE PR #66](https://github.com/SECQUOIA/QUAFFLE/pull/66), base `bf8947ec90adc9857d1027c48bda98692c8a8013`, exact head `1ddbf61b8f36cda2fe7402dd79a71310ac98a22c`, in `/tmp/quaffle-landcover-native-20261028`.

**Verdict: no blocking findings. The pinned implementation and retained preflight evidence are technically ready for the bounded native campaign and implementation PR closeout, subject to the coordinator’s final live head/workflow checks. This is not acceptance of scientific training results or a formal human approval.**

## Findings

- **Blocking:** none.
- **Nonblocking:** none.
- **Questions:** none outstanding for this implementation/preflight deliverable.

## Scope and identity

I read all25 changed paths: native architecture and isolated circuit, preparation/data/metric modules, runner, collector, all four changed test files, protocol/README changes, and retained preflight metadata/logs/original assessments. I traced relevant inherited model, gate, pooling, state-hashing and checkpoint contracts. The earlier planning assessments are design history, not prior formal code-review rounds.

Live scoped `gh` reads confirmed the exact base/head, OPEN draft status, author/account `bernalde`,25 changed paths matching the local diff, no reviews, no inline/issue comments and no reported status checks. The worktree was clean at the inspected head. I applied `gh-review-pr` and relevant shared delegation, scientific and testing guidance in local/no-post mode. No review or other GitHub mutation was performed.

## Technical and scientific assessment

The standalone model addresses the known native-task differences: five classes, full64/128/256/512/1024 widths, explicit fourth convolutional down block, matching-resolution skips and full dense quantum reconstruction. The protocol explains the paper’s diagram/prose ambiguity and the chosen brickwork interpretation. The independent PennyLane tests explicitly define the gates and compare full16-qubit outputs and both angle/coefficient derivatives. They also show that historical sequential behavior is preserved and distinct from brickwork. I independently checked that the entire historical simulator file exactly matches base bytes; no old archived-source test was weakened.

The controls preserve paired non-core initialization and actual crop/augmentation/order schedules. Frozen quantum retains derivatives through circuit inputs; zero removes the reconstructed deep path while retaining ordinary skips. Counts distinguish stored/trainable/core parameters and do not equate flags with functional influence. Deterministic pooling and explicit CE have independent value/derivative checks rather than merely changed tolerances.

Preparation checks the fixed official archive identity and splitter semantics, including numbering of incomplete edge windows, without executing archive code. It requests only train/validation pixel windows and preserves original RGB/class values in lossless PNGs. Dataset access refuses test splits, checks file hashes even on cache hits, preserves labels0–4 and enumerates every validation crop deterministically. The source-window fixture deliberately corrupts a test-only label region while observing actual read requests, providing meaningful evidence for the exclusion boundary.

The runner freezes60epochs,256 training and64 validation tiles, three seeds and four arms. It retains all61 training-history entries per fit, separate periodic validation histories, actual optimizer-step counts and observed schedule hashes. Two affine seed42 fits choose the common rate before quantum/frozen training; the selected candidate is reused once, giving13fits and12primary rows. This is accurately described as a classical-calibrated common recipe, not optimized quantum hyperparameters. Calibration-seed and follow-up-seed contrasts remain separately visible. No test prediction or outcome-dependent extra fitting is part of the design.

The collector independently derives five-class metrics from pooled integer confusion matrices, audits the complete ordered grid and selected epochs, strictly reconstructs selected native checkpoints and verifies exact validation-count replay. Replay has a persisted deadline, lock, write-before-forward intent and sticky failures; completed replay/export paths avoid repeated inference. Export uses numeric records without model imports. The practical1percentage-point boundary uses count-derived Fraction arithmetic, with tests distinguishing an exact boundary from a just-below case that rounds identically in binary64. Learning and late-horizon diagnostics accompany, rather than replace, the explicitly descriptive screening rule.

## Evidence inspected and independently checked

Retained focused logs show74 native/historical model tests passed,28 data tests passed, and40 final runner/collector tests passed. Their fixtures cover meaningful failure paths, independent circuit/loss arithmetic, corrupt artifacts, split exclusion, checkpoint semantics, deadlines, orphan intents and deterministic model-free exports. I inspected these tests and carried forward the execution evidence rather than repeating expensive checks.

The final full repository log reports **874 passed,22 warnings in422.14seconds**. The coordinator confirmed terminal exit0 and unchanged source bytes throughout that run through the reviewed head. I directly inspected its terminal log; the full log is currently retained at `/home/bernalde/repos/QUAFFLE/results/landcover_native_20261028/full_tests_v2.log` for the coordinator’s retention-only follow-up.

My own lightweight read-only checks confirmed:

- All12 committed source-manifest entries match current files and the accepted profile’s source map.
- The cost formula independently recomputes to34,272.839407seconds; its2x margin is68,545.678814seconds, below the86,400-second limit. This remains a short-profile estimate, not a completion guarantee.
- The accepted actual batch-four profile records all four controls, one warmup plus two timed steps, paired initialization, expected gradient/core movement, and peak CUDA allocation683,477,504bytes. Its recorded runtime is PyTorch2.5.1/PennyLane0.44.1 on RTX A1000, with two threads and0.12 allocator fraction.
- The actual prepared manifest matches the committed digest,9,072 prepared tile identities, official7470/1602/1602 split counts, all41 shared parent orthophotos and every reported class-pixel aggregate. I did not independently decode the whole dataset or rerun preparation.

Primary source identities include runner `f00990836b025b47976041ef3f5504b102e7a6cdb574b2baac59a049f4270085`, collector `ea1f89beddb3179697955c87421d6f514b6648da9b72e22f940488f24d96c06c`, native model `c262aaeaac6d9b56e949fcf0a24c722eab2f85e33dc098eaa15b57a7e696a133`, and isolated circuit `e0d05a66a7633ff87516fdc862c997ee1a18ba1aa95e69d69b3dbaec7b9f02ae`.

## Boundaries and next gates

No scientific fit outcomes exist in this reviewed packet. A resource profile and software tests establish implementation feasibility, not meaningful native learning. After source acceptance, freeze the actual plan and run the complete bounded campaign; retain failures, independently audit all13fits, replay all12selected checkpoints and review the actual curves, counts and interpretation before publishing empirical conclusions. Replay must fit within remaining campaign headroom as the protocol requires.

The capped training cohort and official tile splits do not establish full-data performance or geographic generalization. Validation is used for rate/checkpoint selection. Neither a positive screen nor a failed screen establishes the paper’s reported test score, general quantum advantage or its impossibility. No Anvil result, manuscript rewrite or tracker closure is supplied by this PR.

I performed no training, prediction, heavy-test rerun, dataset/source/environment/Git mutation, or remote write. Only this original report was retained. A subsequent commit adding logs/review retention must be distinguished from the source head actually inspected above. The coordinator must refresh live head and workflow state before publishing the single author-account COMMENT review or merging.

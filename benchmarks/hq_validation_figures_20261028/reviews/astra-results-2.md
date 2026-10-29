# Astra actual-results review — round 2

Reviewed base `5b16d3c40e2dbe852462aac438fcd04d5b96ab01` and pinned results head `0d72d217fdba49fa66324dfaac41ab65a4d2958f` in `/tmp/quaffle-hq-handoff-figures-20261028`. The worktree was clean. This is round 2 of the launch-package/publication-figures task, in local, no-post mode; the coordinator remains the sole publisher.

**Verdict: no blocking findings. The requested local launch package and descriptive publication figures are ready for publication/merge, subject to the coordinator’s live GitHub head and workflow checks. Actual Anvil execution remains Alan’s separate responsibility.**

## Findings

- **Blocking:** none.
- **Nonblocking:** none.
- **Questions:** none outstanding for this deliverable.

## Source and execution continuity

I carried forward the round-1 review of all eight initial changed paths and inspected the subsequent renderer diff and final README/operator/decision-document changes. The renderer change adjusts figure dimensions, typography and spacing only. It leaves selection, predictions, normalization, numerical tables and scientific captions unchanged. The cache collector remains the reviewed source with SHA256 `e4273e627101e604013ca9790cf0ad0db4d9834ad1c0e82e918ad40e1a4fd3e0`; the final renderer is `d0f1639d1bbc46ce703f46e79e442dec4252ce5a5fef0c6a5d8085a238313c18`.

The retained execution evidence records one successful validation collection at source `0dcc5e0e61daab7636d918355158b004b5f4e472`, taking 175.127 seconds. I inspected its fitting/test-dataset tripwire wrapper and receipt, original-count parity evidence, cache provenance and source bindings. These establish the claimed validation-only scope along the inspected execution paths; they do not add an independent holdout or quantum-gain result. The layout correction at `93e7ba66898dbbab31d62ed21272b37faeb80287` consumed the same frozen cache, with no repeated model inference.

I inspected the final test log and execution receipt: **764 tests passed, 22 warnings, 296.73 seconds**, at `93e7ba66898dbbab31d62ed21272b37faeb80287`. I checked the retained log hash and matched relevant executable, test and figure-protocol hashes to the reviewed results head. The earlier 46 launcher and 27 figure focused-test passes remain applicable. These tests were run by the coordinator/workers; I did not rerun them.

## Actual artifacts and numerical evidence

I directly viewed both final PNGs and both retained rasterizations of the actual PDFs. Curve axes, seed labels, four-control legend and best-epoch markers are readable. The overlay rows, input/ground-truth/model columns, per-example IoUs and TP/FP/FN legend are separated and legible. The earlier inter-row label collision is absent. Fixed normalized-input displays remain aligned with their targets and overlays, and the affine-conditioned selection disclosure is visible.

My own lightweight read-only checks established:

- All **16 files** in the two final render directories are byte-identical, and the published `plots/` directory is an unchanged copy.
- The local cache inventory matches all **28 listed files**, and all **17 producer-dependency hashes** plus the figure protocol match the current source.
- All **732 curve rows**, including training losses, recorded validation IoUs and selected-epoch indicators, match the original historical records exactly.
- The published packet’s outer manifest matches all **87 covered files**; all **88 physical packet files**, including that manifest, are tracked.
- Prediction, both final rendering, and full-suite result receipts have matching log hashes and appropriate unchanged-source bindings. Both rendering receipts record unchanged cache bytes and mtimes.

I also inspected the independent audit implementation and its successful receipt. It derives counts from masks, uses Fraction-based eligibility/rank arithmetic, checks 920 per-image rows, 12 pooled/sensor rows, 1,380 disagreement rows, 732 historical curve rows and displayed counts, and reloads all 230 original targets plus the four displayed inputs without model evaluation. Historical raw hashes and mtimes remained unchanged. This is substantive independent numerical evidence, not merely a comparison with golden rendered files; I did not independently repeat its data reload.

The selected validation ordinals are Optical worst **17**, Optical lower median **3**, SAR worst **219**, and SAR lower median **44**, from 20 and 53 eligible tiles respectively. The displayed SAR worst case is missed by every control. These cases illustrate the disclosed affine-conditioned rule rather than establish an overall ranking. The documentation correctly distinguishes the twelve fixed-rate 60-epoch histories (nine fresh, three reused) from the separate two-epoch fresh-runner rehearsal.

## Launch and publication boundaries

The launch generator and its reviewed safeguards are unchanged. I inspected the retained demonstration manifest and its inert-script presentation: synthetic account/partition/data paths and tiny limits are explicitly unsuitable for production. The package supplies separate profile/run/collect jobs, source/configuration guards and manual acceptance of a passed full profile; local syntax/guard checks are not scheduler execution evidence. Alan must regenerate for his actual checkout/allocation and establish resource feasibility before submission.

Published cache metadata is explicitly identified as incomplete for rendering: raw numeric arrays and checkpoint weights remain local. The original complete cache supports inference-free reproduction. The outer packet manifest and inner renderer-output inventory have distinct, accurately documented scopes. No manuscript rewrite, new training, test prediction, scheduler contact, hardware claim or unrelated noise study is part of this result.

I performed no fitting, prediction, broad test rerun, source/Git/environment mutation or GitHub write. A later commit retaining this report must be identified as a retention-only head rather than the results head inspected above.

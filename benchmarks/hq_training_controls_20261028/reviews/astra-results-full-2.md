# Astra full results review — PR #62

**READY FOR HUMAN REVIEW. Blocking: 0. Nonblocking: 0. Questions: 0.**

Reviewed [SECQUOIA/QUAFFLE PR #62](https://github.com/SECQUOIA/QUAFFLE/pull/62) at exact head `ab59e994e712dff3b66bd662955889133e7167af`, against base `d66014f16c8d3dc2c7bff4cc69091806563c4d51`. This is completed review round 2 of this training-control campaign, following source review 1 and resolution verification 1. It is an automated, local, read-only review, not formal human approval or authorization to merge.

## Finding inventory

No new actionable findings. The four original findings remain addressed: `hqtc-bind-legacy-audit-source`, `hqtc-match-reused-runtime`, `hqtc-label-inactive-angle-probes`, and `hqtc-preserve-resumed-replays`. Their original reports and resolution evidence remain retained separately. No additional review or verification round was silently substituted for this full results review.

## Scope and implementation

I checked the complete 150-file changed inventory against the retained paginated GitHub pull-files list: seven source/protocol/test additions and 143 packet files. The source scope is `docs/HQ_TRAINING_CONTROLS_PROTOCOL.md`, both training-control runner/collector scripts, `utilsTorch/hq_training_controls.py`, and their three test modules. I reviewed the model and runner contracts, collector validation/export paths, relevant regression tests, and measured package contents rather than relying only on the interpretation summary. The checkout remained unchanged.

The implementation preserves the legacy quantum/MLP paths and common non-core initialization, including BatchNorm buffers. Frozen quantum freezes the eight circuit coefficients while retaining input derivatives. Permanent zero bypasses the complete reconstructed bottleneck map throughout training and evaluation and freezes its unused parameters; ordinary U-Net skips remain. The compact control is a 48-to-32 affine map followed by tanh, without an artificial narrow hidden layer. Stored and trainable parameter counts are correctly distinguished from functional influence: the zero model's requires-grad count includes downstream weights multiplying zeros and is not an active-parameter count.

The fixed protocol is implemented as 15 primary cells at learning rate 0.003 and five seed-42 auxiliary cells at 0.001, with no new rate selection. Eight historical fits are explicitly reused and 12 are fresh. The source/runtime/input guards, paired cohorts and actual D4 schedules, validation checkpoint selection, strict checkpoint reconstruction including BatchNorm and trainability, complete-grid replay boundary, and preserved replay bindings/failure evidence are consistent with this design. The collector audits recorded inference and strict checkpoint restoration; it does not claim to rerun inference independently.

## Checks and evidence

- Independently verified both packet manifests: 113 generated entries and 28 supplemental entries, with the two manifests themselves accounting for the 143-file packet. Compared all 114 original generated files against both persistent report generations and the committed packet: whole-file byte identity, including all PDFs, with no normalization.
- Verified current source hashes and the full-suite log hash against the retained binding/result records. Changes after producer `7754773ca2c36020df22a8971e07765b2357cbc8` are packet-only. The completed **578 passed, 21 warnings, 205.50 seconds, exit 0** suite therefore carries forward to this head. It includes the model, runner and collector regressions, real small synthetic control fits, frozen-input-gradient/restore checks, legacy output compatibility, provenance mismatch rejection, and replay no-overwrite/failure tests. I did not duplicate the expensive suite or launch any fit.
- Inspected and reran the retained standard-library numeric auditor against the persistent raw campaign and immutable initial report: **PASS, 21,736 assertions**, including 635 count-derived metric blocks and 2,448 CSV fields (2,112 numeric). This rechecks all 20 checkpoint byte bindings, complete raw inventories, eight legacy fits, 15 new primary replays, six separate historical replays, schedules, selected epochs, counts and paired summaries. The new reviewer output is retained as `astra-r2-numeric-reaudit.json` alongside this report. This execution uses no Torch/project imports, checkpoint deserialization or inference.
- Separately recomputed the four pooled quantum-minus-control primary validation/test contrasts directly from saved records. They agree with the report. Inspected all three actual PDF renders and verified their render/PDF hash bindings. Learning curves cover all 20 fits; paired plots cover pooled and sensor results; contrasts show individual seeds and mean/sample-SD bars. The explicit black-bar caption clarification in `INTERPRETATION.md` resolves the retained artifact worker's caption observation.

These checks complement the retained actual batch-eight profiles, producer/collector strict restoration, persistent data/checkpoint retention, and original source-review evidence. Hash identity establishes byte provenance, not an independent reproduction of model predictions. AUC cannot be recalculated from confusion counts alone; the numeric auditor accurately states this limit.

## Scientific assessment

The primary validation results do not establish a learned quantum advantage. Quantum-minus-frozen is **+0.277 ± 0.887** percentage points (2/3 seed wins); quantum-minus-permanent-zero is **−0.504 ± 0.912** (1/3); quantum-minus-compact-affine is **+0.297 ± 0.684** (2/3); quantum-minus-MLP is **+0.264 ± 0.602** (2/3). These are paired differences followed by mean and sample SD, not confidence intervals.

The exposed descriptive test pattern is more positive: the corresponding differences are **+0.543 ± 1.465**, **+0.703 ± 0.897**, **+0.402 ± 0.519**, and **+2.199 ± 1.148** percentage points. Quantum wins 2/3 seeds against each new control and 3/3 against MLP. The package preserves both results: zero has the highest mean validation IoU, while learned quantum has the highest mean descriptive test IoU. It appropriately withholds significance, equivalence, general quantum advantage and a causal skip-connection explanation.

The scope qualifications are material and accurately presented: three paired seeds, not 20 independent replications; seed variation includes initialization and optimization cohorts; normalization uses 1,303 global fit images and excludes 230 validation images, so this is not pure 128-shot training; the auxiliary rate check uses only seed 42; angle probes cover eight fixed unaugmented optimization images and are counterfactual for the zero arm; gradient records are epoch aggregates covering 960 optimizer batches, not raw batch histories. Four primary selections occur at epoch 60, precluding a convergence claim. Training-time permanent removal and the earlier inference-only knockout are different interventions. No additional grid or favorable rate was introduced after test inspection.

## Review limits and remaining workflow

My direct live GitHub read encountered a network restriction. The local file comparison uses the coordinator's retained fresh paginated inventory and PR context, which identify this exact head/base and no formal reviews or required approvals. Those records report no executable CI checks; disabled/empty CodeQL configuration is not a passing CI result. The coordinator must refresh live PR state and native gates before any publication or merge action. I performed no GitHub write.

This review clears the completed local training-control code and evidence packet for human review. It does not close issue #42, establish independent holdout or hardware confirmation, or satisfy the outstanding Anvil/federated criteria owned by Alan.

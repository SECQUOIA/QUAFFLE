# Astra assessment of the concrete 13-fit native protocol

Read-only planning review of the uncommitted protocol/runner over base `bf8947ec90adc9857d1027c48bda98692c8a8013` in `/tmp/quaffle-landcover-native-20261028`. Inspected protocol SHA256 `5d8c29866a6d43349b370985ea6b4eba22040000697fcf487d95edc4eff80242` and runner SHA256 `e5051675db6172f14cd43af4f5379842c63876708d38f3bfc4b86562be00be9b`. This supersedes my earlier proposed grid as design advice; it is not the later full implementation review, and no fit outcomes or resource-profile results were assessed here.

**Judgment:** the 13-fit design is a useful, neutral and affordable-in-principle native-dataset pilot. I support it subject to the pre-fit remedies below and a passing actual resource estimate. Three paired seeds at a common classical-calibrated rate are preferable to treating a single favorable seed as a result. No additional outcome-dependent fit expansion is needed in this campaign.

## Scientific adequacy

256 training parent tiles provide 15,360 dynamically sampled patches over60epochs and1,920 Adam steps at batch8. This is substantial optimization work rather than a two-epoch smoke test. It remains only about3.43% of the official training tile population; neither epoch count nor random crop diversity establishes convergence or full-data performance.64 validation parents cover16,777,216 pixels through1,024 deterministic crops, about4% of the official validation tile population. That is adequate for a local diagnostic if all five classes are represented, with class supports reported. Those pixels are correlated within tiles and scenes: they are not millions of independent samples. Three seeds measure optimization randomness on the same cohort, not spatial/data uncertainty.

Keep256/64 and60epochs unless the pre-outcome resource profile shows that even this complete design is infeasible. Do not automatically enlarge the cohort merely because classical controls are fast. Quantum/frozen training and repeated complete validation determine affordability. A larger independently preregistered population would improve representativeness later; it is not necessary to make this first pilot useful.

Affine seed42 at0.001/0.003 followed by one common selected rate across all arms is a defensible recipe calibration. The interpretation must be: **matched controls under a classical-reference-calibrated common training recipe**. It is neither each arm's optimized performance nor a faithful recovery of the authors' optimizer settings. A negative result can reflect quantum-specific rate sensitivity as well as limited bottleneck utility. This is a limitation to disclose, not a reason to add post-result quantum tuning.

The quantum-versus-frozen contrast is particularly useful because coefficients alone change trainability while initial circuit/input dependence remains paired. Affine has a larger/different core and zero removes the deep map; those contrasts are representation/control tests, not a clean estimate of quantum parameter efficiency. Show seed42 separately from follow-up seeds77/123 because seed42 validation helped choose the common rate. The all-three-seed screening statistic may remain primary, accompanied by visible follow-up-only differences.

## Pre-fit remedies

**P1 — retain every epoch's training diagnostics.** In the inspected `fit` implementation, history rows and `train_loss` are retained only inside the every-five-epochs validation branch. The intervening four losses are discarded. Save a row after every completed epoch, with validation present only at0,5,...,60. This preserves the promised learning history and supports honest undertraining/instability interpretation. Keep actual optimizer-step and schedule identities; do not impute unrecorded losses later.

**P2 — enforce the frozen campaign shape.** The protocol fixes256 training tiles,64 validation tiles and60epochs, but `make_plan` accepts arbitrary cohort counts and any positive multiple of5epochs under the same format/protocol. Before launch, reject values inconsistent with this campaign's frozen constants, or explicitly version and label an alternate protocol before outcomes. Batch4 is the one documented resource fallback and must be chosen/profiled before plan freeze. Do not let a short-run override inherit the13-fit scientific-screen interpretation. This request is narrowly about the promised preregistration boundary, not a demand for a general configuration framework.

These are concrete pretraining issues, not findings from a completed formal PR review. The subsequent source review should independently check checkpoint replay, complete-grid collection, finite/frozen-state checks, dataset binding and resume semantics; this narrow assessment does not certify those paths.

## Decision and undertraining interpretation

Precommit all13 fits and the12primary comparisons. The candidate affine fit not selected is retained as auxiliary, not counted twice or discarded. A failed/deadline-limited grid is incomplete and receives no scientific verdict. Fit count, checkpoint restore and exact confusion-matrix replay must pass before interpretation.

Keep the proposed practical signal: quantum's paired mean gain at least1percentage point over both affine and frozen, positive differences on every seed, and mean at least zero. Label it a spending/research screen, not a significance test. Compute boundary comparisons from confusion-derived values carefully; the earlier exact-threshold lesson applies.

Add separate, predeclared diagnostic flags rather than allowing a failed gain screen to stand for a broad negative conclusion:

- **No convincing learning:** report every arm's epoch0-to-best and epoch0-to-final mIoU change, training/validation losses and per-class predictions. If all primary arms gain less than0.05 mIoU from their own epoch0, classify the experiment as an inconclusive learning/optimization pilot. The0.05 cutoff is a declared diagnostic convention, not a universal definition of learning.
- **Horizon-limited:** flag selected checkpoints at epoch60 and positive recent validation trends; show best-versus-final behavior and the last three validation observations. Such a flag does not prove further training would help and must not trigger extra fits. It limits a negative interpretation.
- **Mixed native result:** show all paired signs and classwise effects without changing primary metric/cohort/rate. No test prediction may rescue a weak validation result.

A positive practical signal with competent learning motivates a separately authorized/frozen broader comparison. A failed signal with competent learning establishes only that this common recipe on this capped cohort did not deliver the predefined advantage. All-arms underlearning, instability or resource truncation is inconclusive about quantum potential. These categories preserve the user's question without biasing the experiment toward a positive result.

## Resource acceptance

The persisted12-hour limit and2x profile estimate are sensible operator limits, not runtime guarantees. Include every epoch0/periodic validation, selected-checkpoint replay and reasonable data/checkpoint/collector overhead in the estimate; the2x factor supplies margin but does not replace explicit timing boundaries. Keep6GiB/no-swap/two-thread limits and the shared-GPU allocator cap. A synthetic profile can establish shape, gradient and memory feasibility; it does not establish native learning. If the full13-fit forecast fails, revise the plan before scientific fitting and seek a clearly labeled bounded design, rather than silently completing one seed and presenting the original criterion.

I recommend proceeding to the full source preflight after P1/P2 and the interpretation wording are addressed, then freezing source/data/profile/plan before the first affine candidate fit. No actual experiment, source edit or GitHub write was performed for this assessment.

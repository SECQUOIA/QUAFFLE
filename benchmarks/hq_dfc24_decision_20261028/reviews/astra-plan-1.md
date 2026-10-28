# Astra plan review — round 1

**Verdict: revise the scientific estimand before implementation, then proceed with the bounded audit/protocol/local-manuscript deliverable. Do not add a fresh campaign runner in this task.** Inventory: one Blocking finding, three Nonblocking recommendations, and one Question resolved by the explicit implementation gate below. This is a plan review, not implementation acceptance or experimental execution evidence.

## Inspected identities and evidence

- Repository base, verified in `/tmp/quaffle-hq-dfc24-decision-20261028`: `4231eb89b5913ef3595588397651a8b8a6cca8be`.
- Plan: `/home/bernalde/repos/QUAFFLE/results/hq_dfc24_decision_20261028/plan-for-review.md`, SHA256 `5dd7d53121e6268482d91d745a25832cfdee4b65deec8b5f44243d396383d0dc`.
- Read the HQ direction and training-controls interpretations, `docs/HQ_TRAINING_CONTROLS_PROTOCOL.md`, retained paired contrasts, input-manifest identity description, and local resource profile. Inspected the controls runner, direction-runner restoration contract, training-data loader, and CPU circuit/checkpointing source. The retained evidence supports the quoted +0.277 ± 0.887 validation percentage-point learned-minus-frozen contrast and limited conclusions.
- Separately inspected relevant local manuscript passages in `/home/bernalde/repos/QUAFFLE/papers/QUAFFLE_Journal_2026/main.tex`, SHA256 `ec64daa75e085f0f6265b29fc5b1304a157010bf2660cce1a520a35dd55c723c`, the package README, and companion notes. This manuscript is untracked and absent from the pinned Git base. I inspected correction targets, not every historical result or proof.
- Applied `gh-review-pr` in local/no-post mode and shared conventions, delegation/resource, maintainability, and scientific-evidence guidance. No helpers, fits, inference, tests, environment changes, cluster access, or GitHub writes were performed.

## Blocking

### B1 — State the estimand of the independently tuned arms
<!-- gh-review-pr:finding=dfc24-tuned-procedure-estimand -->

**Location:** Evidence and question; finite design items 2–3.

The plan asks whether “learning the circuit” supplies a reproducible gain while each arm may select its own learning rate. If learned and frozen circuits select different rates, their contrast changes both trainability and optimization. It cannot identify the effect of circuit-coefficient trainability alone. The existing fixed-rate protocol explicitly distinguishes its intervention comparison from a separately optimized leaderboard.

**Remedy:** retain equal per-arm tuning opportunity at 0.001 and 0.003. Rewrite the question as whether the learned-circuit **training procedure**, under this equal tuning budget and full-fit regime, delivers a practical validation improvement over frozen-circuit and compact-classical procedures. State that differing selected rates prevent trainability-only causal attribution. Even coincident rates restrict the conclusion to this architecture, split construction, and budget. No additional fixed-rate grid is needed. “Competitive within this two-rate budget” is defensible; an unqualified “optimized classical baseline” is not.

## Nonblocking

### N1 — Make the screen a finite resource decision, with tuning optimism visible
<!-- gh-review-pr:finding=dfc24-screen-interpretation -->

**Location:** finite design items 3 and 5.

The arithmetic is coherent: eight seed-42 candidates plus eight selected-rate follow-ups gives 16 fresh fits and 12 selected checkpoints. A one-percentage-point threshold and all-positive paired differences are acceptable as a chosen practical screen, not a calibrated statistical test. Three seeds cannot establish equivalence or exclude useful effects when the gate fails. Seed 42 both selects the rate and enters the three-seed screen; all selected checkpoints also use validation for epoch selection. This remains an adaptive validation screen on an already studied dataset.

Publish all candidates, the three-seed gate, and the two follow-up seed differences separately; the latter are not independent-data confirmation. Freeze ties before fitting (lower rate; earliest strict best epoch including epoch zero, consistent with existing rules), missing/failed-fit handling, and units (`0.01` IoU = one percentage point). An incomplete campaign has no gate verdict. Failure means no basis to prioritize a larger accuracy campaign from this screen, not absence of circuit benefit. Passing motivates independent-data confirmation, not superiority. Do not add seeds or change the threshold after seeing outcomes.

### N2 — Keep full-fit scale, exposure, and independence separate
<!-- gh-review-pr:finding=dfc24-full-fit-exposure -->

**Location:** finite design items 1 and 3.

The current loader forms 1,303 fit and 230 validation tiles per seed and enforces normalization over exactly the fit partition. “All fit” means all 1,303 optimization tiles, not fit plus validation. At 60 epochs this changes both sample coverage and optimizer-step budget relative to cap128; a historical comparison cannot isolate sample-size effects. It remains base8 at 128 pixels, not full-capacity or native-resolution evidence.

Retain per-seed cohort identities and distinguish optimization, normalization, selection, and prior-inspection exposure. Exact-pixel deduplication does not establish geographic/event independence. If trustworthy metadata or unexposed cohorts are absent, say so and retain the exploratory design. Do not manufacture a fresh holdout by repartitioning exposed pixels. A newly found independent cohort requires a separately frozen design; it is not an automatic branch of this experiment.

### N3 — Correct the local manuscript without endorsing unrelated evidence
<!-- gh-review-pr:finding=dfc24-manuscript-boundary -->

**Location:** manuscript deliverable; local `main.tex` abstract and internal-validity discussion.

The inspected manuscript contains “statistically indistinguishable” and an inference from weak knockout effects to skip connections dominating inference. Replace unsupported equivalence and causal language with descriptions of observed comparisons. Add a separate full16q HQ-controls account with fit/sample counts, historical common-rate design, exposed-test status, and learned/frozen/zero/affine outcomes. Do not transfer legacy cosine-factorization/noise results to this different circuit or conflate the manuscript's pi-scaled spatial implementation with the tested tanh-radian, no-pi HQ architecture.

Preserve unrelated historical results/placeholders as separately sourced evidence, not results validated by this audit. Keep before-copies and build the edited manuscript if available. Since the source is untracked and outside the pinned base, report local manuscript edits/builds separately from the public PR; do not force-add the entire unpublished package.

## Question and resolved scope recommendation

### Q1 — Is an executable full-data campaign part of this delivery?
<!-- gh-review-pr:finding=dfc24-runner-readiness -->

The plan offers a new orchestration layer or an implementation gate. **Choose the implementation gate.** A frozen protocol, exposure/metric audit, and honest Alan handoff are sufficient bounded deliverables here. Root's subsequent clarification explicitly accepts this alternative. A ready-to-submit full-data command is not currently available.

This is a concrete compatibility gap: `scripts/run_hq_training_controls.py` restricts `--fit-cap` to 128, requires legacy reuse, and validates that cohort. Its fresh `fit_one` writes `hq-training-controls-v1`, but learned-quantum restoration dispatches to the direction runner, which expects `hq-direction-v1`. Blindly calling that fresh fitter for all four arms is unsafe. A future runner must deliberately dispatch the appropriate producer/restorer or establish and test a corrected format contract. This review does not validate that future implementation.

Do not build a second orchestration/collector framework for this task. State “full-data runner unimplemented; campaign launch gated,” give executable audit commands only for capabilities delivered, and list future requirements: exact 16-fit grid, per-arm selection, source/data/runtime bindings, strict restoration, failure/resume integrity, and complete-fit audit boundary before selected replay. Do not advertise unsupported flags as a runnable handoff.

Anvil feasibility must be measured with the **actual four arms**, including CPU full16q forward/backward and per-sample checkpointing, data loading, validation, and checkpoint I/O. Retain measured time and peak CPU/GPU memory under Alan's actual allocation before accepting a budget. The retained RTX A1000 profile is neither Anvil throughput nor scheduler policy. Do not guess partition, account, walltime, GPU speedup, or transfer a local allocator cap as cluster policy. If infeasible, stop and revise before outcome inspection; do not silently reduce epochs/coverage. No cluster access or contact occurs in this local task.

## Finite revised plan

1. Freeze a revised protocol resolving B1, specifying the 16-fit/12-selected design, deterministic selection, current-data exploratory status, and one-point resource screen. Keep four arms and equal per-arm rate tuning.
2. Produce a reproducible exposure/metric audit from existing manifests and saved confusion counts. Retain pooled and Optical/SAR counts, foreground IoU/Dice, binary macro IoU and accuracy; define aggregation and zero-denominator conventions. No inference reruns or historical-package mutation.
3. Deliver the Alan handoff with explicit **fresh-runner implementation** and **actual Anvil profiling** gates. No fictitious training command, guessed scheduler script, fits, submissions, contact, or downloads.
4. Make scoped local manuscript corrections and add the completed-controls account. Preserve original source and build/check actual changed artifacts where possible, reporting limitations. Keep ignored manuscript sources local.
5. Validate the audit with independently checkable count identities and relevant corruption/mismatch cases, then applicable repository checks. Do not invent runner coverage when no runner changes. Preserve environment and hosted-CI spending constraints.
6. Use remaining review capacity to inspect actual protocol/audit/local-manuscript changes at exact identities. Root handles authorized repository publication. Report local manuscript completion separately and full-data execution as pending; neither this source review nor a manuscript build supplies experimental validation.

The negative/mixed controls are a useful scientific result. A bounded scale check can determine whether this tuned procedure merits independent evaluation; it should not become a search for a favorable quantum result.

# Astra fresh-runner source preflight — round 1

**Verdict: one Blocking correctness finding in the full-mode scientific gate. No other Blocking findings, Nonblocking findings, or unresolved Questions.** Fix and verify B1 before accepting the complete runner/collector substrate. The finding does not affect rehearsal's `not_applicable_rehearsal` status, but completing the fix before the planned source freeze avoids unnecessarily splitting provenance.

This is the initial review of the new executable-runner task. It is not another review of PR63 and does not consume that earlier task's review budget.

## Inspected source and scope

- Worktree: `/tmp/quaffle-hq-fresh-runner-20261028`, branch `research/hq-fresh-runner`.
- Base: `96f97c657312d80d19937a5119834f4c3a25dce0`.
- Pinned reviewed head: `5645a523294c09bc39def58919a72a0e8cc604f6`; clean at final readback.
- Seven changed paths: root README, both decision/execution protocols, new runner and collector, and their two test files. I read those changes and relevant unchanged native fitter/restorer, model/control, data-loading, scoring, source-identity and runtime contracts.
- Runner SHA256: `1f5a244db37cd563cd0e0039de997dd140cee90527c4e50b585973dbf8ecbb81`.
- Collector SHA256: `8f71d0807ec594ff72fe23fafa6195cf83ef4b1997b1567ecfdd6197beb99b99`.
- Frozen execution protocol SHA256: `7149e9116e4d47661d497542d32d9b2c82ba1c8d9cd1dc5c4d1917e2224da8f4`.

I read the task's implementation contract and requirements, reapplied `gh-review-pr` in local/no-post mode, and used the applicable previously read shared scientific-evidence, testing, resource and delegation guidance. I reconciled the collector's final self-source guard after the source pin. Root is sole publisher. No source/Git/remote/environment edits, helpers, real-data fits, predictions, profiles, or cluster activity were performed by this review.

## Blocking

### B1 — Exact one-point gains can incorrectly fail the declared inclusive gate
<!-- gh-review-pr:finding=hq-fresh-exact-gate-boundary -->

**Location:** `scripts/collect_hq_dfc24_fresh.py:503`, with related comparisons at lines 504–509 in `analysis`.

**Evidence:** The protocol requires a mean paired validation gain **at least** 0.01 IoU against both frozen and compact controls. The implementation first subtracts floating-point IoUs, multiplies by 100, and compares the resulting mean directly with `1.`. For all three seeds, let quantum pooled foreground IoU be exactly `29/100` and each comparator's be exactly `28/100`. These are realizable integer-confusion-count ratios. All paired differences are positive, the true mean difference is exactly one percentage point, and quantum can also be above the zero arm. Nevertheless, the implemented differences become `0.9999999999999953` percentage points, so both `mean_gain_at_least_one_point` checks are false and the gate returns `fail`.

I reproduced this by extracting and executing the actual pinned `identity`, `stats` and `analysis` function ASTs with standard-library globals and a complete synthetic 16-fit/12-record analysis structure. No project module, model, fitting or prediction was invoked. Observed output included:

```text
mean: 0.9999999999999953
wins: 3
mean_gain_at_least_one_point: False
quantum_mean_at_least_zero: True
status: fail
```

This is a scientific-decision boundary error, not merely a displayed rounding issue. The current test covers a comfortable pass, a half-point failure, a zero-arm failure and a seed tie, but does not cover the inclusive one-point boundary.

**Concrete remedy:** compute the full-mode gate inequalities from the already audited pooled integer counts using exact rational foreground IoU, e.g. `Fraction(TP, TP+FP+FN)` with the documented zero-denominator convention. Compute paired rational differences, their mean against `Fraction(1,100)`, all-positive checks, and the quantum-versus-zero mean comparison on that basis. Keep floating-point summary/SD/CSV reporting unchanged. Do not solve this by rounding displayed scores or using an arbitrary tolerance that admits genuinely sub-threshold results.

Add regressions demonstrating that an exact one-point gain passes when all other conditions hold, a genuinely just-below-one-point gain fails, and exact equality versus the zero mean passes. Preserve all-positive seed and rehearsal-N/A behavior. Derive the synthetic expectations from integer counts, rather than setting contradictory saved metric fields merely to exercise a branch. This is a bounded collector/test correction; no model or producer change is needed.

## Assessment of the remaining source contracts

The learned-quantum route dispatches to `hq.fit_one`, preserving its native `hq-direction-v1` format; the other three routes preserve the training-control format. Strict restoration reconstructs control trainability, parameter counts, finite state/BatchNorm buffers, permanent-zero mode and frozen identities. Native-format and input-gradient tests exercise actual model mechanics, including the frozen circuit's continuing input derivatives. This addresses the prior incompatible-fitter concern without changing historical producers or model source.

The orchestrator implements the eight seed-42 candidates and eight selected-rate follow-ups, lower-rate exact ties, and inherited earliest strict best-epoch selection including zero. Pair checks cover non-core initialization, cohort/normalization identities and sample/augmentation schedules. Full mode uses the 1,303/230 split; rehearsal retains that normalization exclusion while evaluating deterministic 16/16 subsets. No test dataset is constructed by the rehearsal/profile branches. Selected replay is behind the complete 16-checkpoint audit boundary, and rehearsal does not calculate the scientific screen.

The fixed deadline, live OS writer lock, retained failed/partial fits, bound replay intent before inference, and completed-resume guards are proportionate to the scientific artifact contract. Tests check that an interrupted intent blocks retry, a completed resume performs no fit/prediction or raw rewrite, and expired learning allowance cannot be replenished for an incomplete campaign. The source/profile/runtime/cohort checks and explicit CUDA allocator fraction avoid presenting historical cap128 reuse or a workstation setting as a fresh full-data execution profile.

The collector does more than trust the runner's summary: it checks the exact grid and selection, raw byte inventories, data/cohort reconstruction, all native checkpoint restores, recorded movement and frozen parameters, histories/gradient coverage, boundaries/intents/replays, and independently derives metrics from integer counts. Its source identity is separate from producer dependencies. It permits byte-identical relocated data while retaining original producing paths, and records the restrictions on CPU restoration under matching software builds. The empty-attempt, replay, identity, source and checkpoint corruption tests exercise meaningful rejection paths. Deterministic export and inference-forbidden tests are relevant evidence, not real-data execution evidence.

## Verification evidence and limits

I inspected the runner integrity log reporting **51 passed in 10.45 seconds**. The earlier file named `runner_tests_final.log` contains an older 43-test run; the integrity log is the relevant later evidence. I also inspected the collector evidence manifest and logs: **30 passed in 62.94 seconds** before the isolated self-source guard, then **4 passed, 27 deselected in 17.62 seconds** covering final-source complete grids, deterministic export and self-source corruption. These overlap and must not be added into a count of unique tests. Their bound source hashes match the inspected files.

I did not repeat those focused suites or the coordinator-owned full suite. The full suite was still a coordinator task at review time; this report does not claim its final result. My additional executed check is the lightweight exact-boundary reproducer described in B1. No actual own-DFC24 execution result exists in this preflight assessment.

After B1 is corrected, verify the exact arithmetic regression and affected collector contracts, freeze the corrected source, and proceed with the already authorized actual four-arm profiles and complete validation-only rehearsal under the declared resource bounds. Preserve all real raw results and failed attempts. Subsequent results review must inspect those actual artifacts, collector regeneration, independent numerical/identity checks and completed-resume evidence. Full 60-epoch accuracy, test prediction replays, Anvil feasibility/execution and independent confirmation remain outside this local rehearsal and cannot be inferred from these source tests.

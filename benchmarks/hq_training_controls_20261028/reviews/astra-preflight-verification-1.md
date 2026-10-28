## 1. Overall assessment

All four findings in `astra-preflight-1.md` are **Addressed** at `7754773ca2c36020df22a8971e07765b2357cbc8`, against reviewed source `d110b1e61d3023ecebde4c89579e86b6efaca7ca` and base `d66014f16c8d3dc2c7bff4cc69091806563c4d51`. The addressing commit is `7754773` (“Preserve replay evidence and verify HQ control provenance”). Coverage is 4/4 original report items: two Blocking and two Nonblocking; none were declined or deferred.

The narrower verification is adequate to resolve the source-preflight concerns before the frozen-protocol fits. This is verification pass 1, not a new review round or approval of experimental results. Cumulative use is one completed review round and one verification pass. There is no remote PR or formal human approval at this boundary. The original report remains unchanged.

## 2. Comment-by-comment status

- **HQTC-ASTRA-001 / `hqtc-bind-legacy-audit-source` — Addressed.** `SOURCE_FILES` now includes `scripts/collect_hq_dfc24_direction.py`, so profiling, the frozen plan and subsequent source equality checks bind the executed historical acceptance gate. The new profile contains the collector hash, which I verified against current source. The source-inventory regression requires that dependency and uniqueness; source-change/resume tests verify rejection without changing completed outputs. The historical producer inventory remains separate.
- **HQTC-ASTRA-002 / `hqtc-match-reused-runtime` — Addressed.** Immediately after the historical audit, `run` requires exact equality between `hq.runtime(device)` and `old['plan']['runtime']`, before profile acceptance, plan creation or fresh fitting. That dictionary includes interpreter/packages, device/GPU, CUDA version, threads and allocator fraction. The regression gives the historical plan a different runtime while keeping the current profile valid, then checks rejection, no work directory and no fit events. I independently confirmed the new actual profile runtime equals the retained historical runtime.
- **HQTC-ASTRA-003 / `hqtc-label-inactive-angle-probes` — Addressed.** New fit diagnostics record `angle_probe_scope`: `inactive_counterfactual` for permanent zero and `active_fixed_probe` otherwise. Cached-fit validation rejects the wrong scope. The protocol explicitly explains that unused-path encodings are not prediction angles; collector diagnostics export the scope and the report labels zero-arm probes as counterfactual. Actual one-epoch new-control tests check the field. Permanent-zero and frozen model computation is unchanged.
- **HQTC-ASTRA-004 / `hqtc-preserve-resumed-replays` — Addressed.** Saved replay envelopes bind the frozen plan, origin/path, checkpoint/metrics/history digests and ordered test names. `validate_replay` checks that binding, the exact fit record, checkpoint/cohort identity and count-derived metrics. Existing replays are validated and reused without evaluation or rewriting; malformed evidence is rejected. New replay errors retain a bound failure record, and its presence blocks silent retry. Tests prohibit inference during completed resume, compare both bytes and modification times for all 15 replays, reject and preserve a tampered binding, and retain failure bytes while refusing retry. A partially written existing replay is likewise read and rejected rather than replaced.

## 3. Tests or CI evidence inspected

I inspected the test bodies relevant to the four remedies and reused the completed bounded local evidence: **19 model tests passed in 5.65 seconds**, **55 runner tests passed in 8.89 seconds**, and **22 collector tests passed in 79.98 seconds**. The runner cohort includes actual small-model fit/restore checks plus synthetic full-grid orchestration, runtime rejection, fixed-rate selection boundaries, replay preservation, tampering and failure handling. Synthetic tests establish those contracts, not scientific accuracy.

I verified every hash in `collector-verification.json` against the current files, and every source/protocol hash in `/tmp/quaffle-hq-training-controls-profile-v2-20261028/resource_profile.json`. The new actual-model profile passed all three new controls at batch eight with strict restoration and frozen-parameter identity. Those source bindings now include the historical collector. The original control-model module remains byte-identical, so its earlier behavioral evidence carries forward.

The separate full-suite run was still in progress when inspected: `full_suite_binding.json` pins `7754773` and `full_suite.log` contained progress but no final result. I do not count the expected 578 tests as a completed pass. No additional heavy test, GPU job or fitting was launched by this verifier; no hosted CI was inspected or claimed.

## 4. Remaining issues

No original finding remains partially addressed or unaddressed, and no Blocking request was declined. The already-running full suite still needs its actual terminal result recorded. Real-data fit, replay, collector-output, deterministic-regeneration and scientific-interpretation evidence does not yet exist for this campaign and remains outside this pre-fit verification. No GitHub threads or review verdicts exist to close out.

## 5. Whether another review round is justified

**No** additional pre-fit review round is justified for these fixes. The remedies are bounded, directly implement the requested contracts and have relevant regression evidence; no revised scientific design requires another judgment pass before fitting.

The separately planned **full results review, round 2**, remains necessary once the measured grid and artifact packet exist. That later review should cover the complete seven-file source/protocol change and results, including the new collector, all 20 cells, eight historical versus 12 fresh fits, all 15 bound primary replays, fixed-rate auxiliary interpretation, generated artifacts and retained provenance. It is not satisfied by this verification.

## 6. Recommended next action

Record the terminal outcome of the already-running current-head full suite. If it passes and the pinned source/input/profile/runtime bindings remain unchanged, proceed with the authorized finite frozen-protocol grid; the four source-preflight findings no longer prevent that work. Retain this assessment separately from the original report, then request the planned full results review after collection. No publication, merge, source modification, Git mutation or formal approval was performed here.

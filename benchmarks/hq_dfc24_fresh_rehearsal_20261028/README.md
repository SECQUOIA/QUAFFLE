# Local fresh HQ runner rehearsal

The complete four-arm software rehearsal passed on our cleaned DFC24 data: **16 fresh fits, 12 selected validation replays, zero test predictions**, completed in **163.64 seconds**. This verifies training, selection, native checkpoint restoration and saved-result audits.

Each fit used two epochs, 16 optimization tiles and 16 validation tiles per seed, with batch size 8, 128-pixel inputs and base width 8. Normalization uses all 1,303 global-fit images and excludes all 230 global validation images. The quantum controls retain full 16-qubit CPU computation; surrounding convolutions use CUDA. Controls are learned quantum, frozen quantum, permanently zero bottleneck and compact affine. Seeds 42, 77 and 123 and candidate rates 0.001/0.003 follow the [execution protocol](../../docs/HQ_DFC24_FRESH_PROTOCOL.md).

**All 12 selected checkpoints are epoch zero.** None of the short fits improved its initial validation foreground IoU. Both rates tied on that initial best score; every arm selected 0.001 by the declared tie rule. These initial-checkpoint results provide no new accuracy evidence for quantum. The scientific screen is `not_applicable_rehearsal`. Keep this packet separate from earlier 60-epoch capped-fit evidence.

Both four-arm resource profiles passed: rehearsal in 58.39 seconds and full mode in 76.94 seconds. The full-mode profile exercised the 1,303/230 cohort configuration with two Adam steps per control. Peak GPU allocations were about 144–146 MiB. Timings include instrumentation and unequal warmup, so they are feasibility observations. CPU RSS and cgroup memory peaks have different accounting boundaries; the first profile reached its 6 GiB cgroup cap including cache without a process failure.

Two collector invocations strictly restored all 16 native checkpoints on CPU, audited all 12 saved validation replays and produced identical complete output bytes. A standard-library oracle independently checked source/checkpoint identities, cohorts, selection, count-derived metrics, sample SD and paired contrasts. Guarded completed resume called no fitter or model prediction and preserved every raw file's bytes and modification time. The final full repository suite passed **691 tests** (21 retained warnings) in 279.61 seconds.

- [Audited records, metrics and source snapshots](report/README.md); [regeneration instructions](report/REPRODUCTION.md).
- [Rehearsal profile](profiles/rehearsal_resource_profile.json) and [full-mode profile](profiles/full_resource_profile.json).
- [Independent audit](verification/independent_audit.json) and [completed resume](verification/completed_resume.json).
- Original Astra reports and separate dispositions are retained under `reviews/`; command logs, bindings and exit codes are under `execution/`.

Heavy commands used a 6 GiB OS memory limit, no swap and two numerical threads. GPU phases capped the process allocator at 12%. Raw checkpoints and imagery remain in the persistent local results area and are required to regenerate checkpoint audits. No weights or images are published here. `report/artifact_sha256.json` covers the unchanged collector output; the outer `artifact_sha256.json` covers this packet except itself.

Alan owns actual Anvil profiling and the prespecified 60-epoch campaign using all 1,303 optimization tiles and 230 validation tiles. No full training campaign, cluster access or job submission, new dataset, test-prediction campaign, independent holdout, quantum hardware, or full-capacity/federated accuracy result was produced here. Issue #42 remains open for its wider criteria.

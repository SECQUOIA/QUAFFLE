# Generate the fresh HQ launch package

`misc/generate_hq_campaign_scripts.py` writes three separate SLURM jobs and an
immutable launch manifest. Generation and local validation submit no jobs and
do not contact Anvil. Alan supplies the scheduler account, partition, paths,
environment and resource limits through the existing cluster configuration.

This package launches the [fresh full-mode protocol](HQ_DFC24_FRESH_PROTOCOL.md):
four controls, 16 fresh fits and 12 selected replays, with all 1,303 optimization
tiles and 230 validation tiles at 60 epochs. It does not launch the old capped
study or the two-epoch rehearsal.

## Generate on the execution checkout

Alan should regenerate the package from the checkout he will execute on Anvil.
`--repo-root` identifies both the existing source tree and its execution path.
Data and interpreter paths can refer to the target machine; generation does not
require those paths to exist on this workstation.

Fill in ignored `config/cluster.local.json` using
[the existing template](../config/cluster.example.json), then supply explicit
phase budgets, walltimes and allocator fraction. These are operator limits,
not workstation-derived runtime estimates. Cluster flags override environment
variables, which override the selected JSON configuration and defaults.
SLURM reads `#SBATCH` directives as literal values, while `--export` can forward
the accepted-profile environment variable. See the
[official sbatch manual](https://slurm.schedmd.com/sbatch.html) for directive,
environment-export and walltime syntax.

After setting the variables below from the intended allocation:

```bash
python misc/generate_hq_campaign_scripts.py generate \
  --repo-root "$PWD" --cluster-config config/cluster.local.json \
  --data-root "$DFC24_ROOT" \
  --profile-dir "$HQ_CAMPAIGN_ROOT/profile" \
  --run-dir "$HQ_CAMPAIGN_ROOT/run" \
  --report-dir "$HQ_CAMPAIGN_ROOT/report" \
  --output-dir "$HQ_CAMPAIGN_ROOT/launch" \
  --gpu-memory-fraction "$HQ_GPU_MEMORY_FRACTION" \
  --profile-budget-seconds "$HQ_PROFILE_BUDGET_SECONDS" \
  --profile-walltime "$HQ_PROFILE_WALLTIME" \
  --run-budget-seconds "$HQ_RUN_BUDGET_SECONDS" \
  --run-walltime "$HQ_RUN_WALLTIME" \
  --collect-budget-seconds "$HQ_COLLECT_BUDGET_SECONDS" \
  --collect-walltime "$HQ_COLLECT_WALLTIME"
```

The output contains `01_profile.sbatch`, `02_run.sbatch`, `03_collect.sbatch`,
`launch_manifest.json` and `OPERATIONS.md`. Logs stay in the package's configured
log directories. Source, protocol and selected configuration bytes are bound;
regenerate the package after changing them. Do not edit generated commands to
change the experimental design.

## Profile, accept, run, collect

1. Submit only the profile job and inspect its actual exit status and complete
   four-arm `resource_profile.json` on that allocation.
2. Decide whether the full campaign fits the accepted compute budget and
   scheduler walltime. Stop and revise before fitting if it is infeasible.
3. Export `HQ_ACCEPTED_PROFILE_SHA256` with the exact inspected profile digest
   and submit the run job using the generated instructions. The job rejects
   missing acceptance, altered bytes, a rehearsal profile, or incompatible
   source/settings. Native guards then check the actual data, runtime and
   cohorts before learning.
4. After successful completion, submit collection separately. It requests no
   GPU and audits native checkpoints on CPU. Alan must choose a partition that
   permits this resource request; collection does not establish a new accuracy
   result or independently repeat inference.

There is no automatic submission chain. The native runner preserves failed or
partial fits and replays. Generate with `--resume` only to opt into its existing
resume contract, using the original budget, paths and profile. Incomplete-run
deadlines include downtime and queue time; deleting evidence is not a retry
mechanism. All commands have finite outer deadlines in addition to the native
profile/run budgets, with explicit cleanup headroom.

The package enforces one GPU and at least two allocated CPU threads. The circuit
retains full 16-qubit CPU computation, the CNN uses CUDA, and numerical thread
counts remain two. A process allocator fraction does not cap total host memory
or all GPU allocations. No resource setting here is a measured Anvil throughput
or policy recommendation.

The [decision protocol](HQ_DFC24_DECISION_PROTOCOL.md) retains Alan's remaining
launch gates and the interpretation limits of the already exposed test split.

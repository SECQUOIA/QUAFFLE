# Full HQ campaign handoff

Alan submits each job separately after checking the preceding result. This package submits nothing.
These are operator-selected resource limits, not Anvil runtime or allocation recommendations. The generator has not contacted or validated a cluster.

Immutable launch manifest SHA256: `d40f07f17d0f7e33b8e1367952cc4046e1683d31efeab5d0884166b1349d8ba7`. Source/protocol and selected cluster-config bytes are checked before every phase; changes require a new package.

1. Alan regenerates this package from his actual source checkout and cluster configuration; data and Python paths need not exist on the generator host. Put the exact checkout, dataset, package and configured environment at the recorded execution paths. Log folders are inside this package. Confirm scheduler account/partition, Python, modules, memory, walltimes and allocator fraction with local policy. The collection job uses the same configured partition without requesting a GPU; confirm that this is permitted or regenerate a separately configured collection package.
2. Submit only the profile job:

```bash
sbatch /home/bernalde/repos/QUAFFLE/results/hq_handoff_figures_20261028/launch_demo_v1/package/01_profile.sbatch
```

3. Inspect its exit status and full `resource_profile.json`: all four actual batch8 cases must pass. Decide whether the explicitly bounded full campaign is affordable. A laptop profile is not a cluster profile. Acceptance is manual, and binds the exact inspected bytes:

```bash
sha256sum -- /home/bernalde/repos/QUAFFLE/results/hq_handoff_figures_20261028/launch_demo_v1/profile/resource_profile.json
export HQ_ACCEPTED_PROFILE_SHA256='PASTE_THE_INSPECTED_64_CHARACTER_SHA256'
sbatch --export=ALL,HQ_ACCEPTED_PROFILE_SHA256 /home/bernalde/repos/QUAFFLE/results/hq_handoff_figures_20261028/launch_demo_v1/package/02_run.sbatch
```

4. After the run exits successfully and all 16 fresh fits and 12 saved test replays are complete, submit collection separately:

```bash
sbatch /home/bernalde/repos/QUAFFLE/results/hq_handoff_figures_20261028/launch_demo_v1/package/03_collect.sbatch
```

The native runner rechecks full settings, exact runtime/source/data/profile/cohorts and strict checkpoints. The native collector refuses incomplete grids. The 205 test tiles were already exposed; this is not a new holdout or a quantum-superiority test.

Resume opt-in for this package: `False`. No failed/partial fit or replay is silently retried. Native incomplete-run deadlines include downtime/queue time; completed resumes verify without new fitting or inference. Do not erase or edit preserved attempts to obtain a retry. A different resume choice requires a new package using the original run budget/profile/paths.

| Phase | Native/phase budget (seconds) | Outer TERM deadline (seconds) | SLURM walltime |
|---|---:|---:|---|
| profile | 60.0 | 90.0 | 00:03:00 |
| run | 60.0 | 90.0 | 00:03:00 |
| collect | 60.0 | 60.0 | 00:02:00 |

All commands have an outer TERM deadline with a 30-second KILL grace. Profile/run receive the native persisted learning budget and an outer deadline 30 seconds longer, allowing native failure preservation. Collection has an outer execution limit and performs CPU restoration/audits only. Walltimes allow at least 90 seconds beyond profile/run budgets and 60 beyond collection; these limits do not guarantee campaign completion.

CUDA allocator fraction: `0.12`. This is a process allocator cap, not total GPU/host memory accounting. Numerical threads are 2. Collection requests no GPU and hides CUDA. The four-arm model remains the production 16-qubit full bottleneck at image size 128 and base width 8; there are no geometry or epoch overrides.

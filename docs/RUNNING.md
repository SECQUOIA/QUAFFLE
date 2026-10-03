# Running the QUAFFLE experiment grid

All jobs run on a SLURM cluster, partition `ai`, account `YOUR_ACCOUNT` (set with `--account`), one GPU each. One GPU-hour is 1 SU.

## 1. Generate the scripts

`misc/generate_grid.py` writes one SLURM script per job for

- methods `classical` (FedU-Net), `ablation` (matched classical bottleneck, `abalation_studies.py`) and `quantum` (QUAFFLE)
- modes `centralized`, `fed-random` and `fed-dirichlet` (label-skew, `--alpha 0.5` by default)
- datasets `optical`, `sar` and `combined`
- the seeds in `config/seeds.py` (a `--seeds` subset is checked against that list)

Always look at the plan first.

```bash
cd /path/to/QUAFFLE
python misc/generate_grid.py --dry-run --budget <SU left>
```

Then write it, filtering any axis.

```bash
python misc/generate_grid.py --budget <SU left>                    # full grid, 270 jobs
python misc/generate_grid.py --datasets optical                    # optical only
python misc/generate_grid.py --methods quantum --modes fed-dirichlet --seeds 7 17
```

Other options

- `--q-ansatz {reupload,legacy}` (default reupload) and `--q-layers N` go to quantum and ablation jobs.
- `--quantum-knockout-eval` is passed to quantum jobs unless `--no-quantum-knockout-eval` is given.
- `--ckpt-every 5` is passed to federated jobs only if the target script source contains `ckpt-every`. The generator prints which mode it used.
- `--root` is the checkout the jobs `cd` into (default the checkout the generator lives in). Generate from the checkout you will run.
- `--out-dir` sets the output folder. The default is `misc/grid/<timestamp>/`. The generator refuses a non-empty folder and never writes into `misc/` itself.
- `--check-flags` runs each target script with `--help` once and fails if any emitted flag is unknown. It needs a user that can run the conda python. This takes a few minutes because each `--help` imports torch.

The output folder holds the job scripts (named `<method>_<mode>_<dataset>[_<ansatz>]_seed<N>.sh`), `submit_all.sh` and `manifest.csv` (script, job name, command, time limit, expected SU).

The two older generators are thin wrappers that share the same time limits and header.

- `misc/generate_bottleneck_ablation_scripts.py` is the grid restricted to `--methods ablation`.
- `misc/generate_qc_ablation_scripts.py` writes federated quantum jobs with `--quantum_channels 64 128 256`.

## 2. Time limits and cost

Runtimes measured on 2026-10-02 on one `ai` GPU, optical (200 training tiles), federated with 4 clients and 25 rounds of 4 local epochs.

| method | centralized | federated |
|---|---|---|
| classical | 0.10 h | 0.51 h |
| ablation | 0.12 h | 0.56 h |
| quantum | 0.40 h | 3.21 h |

Expected hours are scaled by training tiles (SAR 1119 is 5.6x optical, combined 1318 is 6.6x). Dirichlet is assumed equal to random. For quantum jobs the circuit overhead (quantum minus ablation) is also scaled by `quantum_channels/32` and `q_layers/2`. That part is an extrapolation. The time limit is `ceil(expected x 2)` hours with a floor of 1 hour (`--safety`, `--floor-hours`). The summary table shows expected SU and the worst case if every job hits its limit. With `--budget` the generator refuses to write when the expected total is over budget.

Full grid (270 jobs, 10 seeds) is about 1210 SU expected and 2580 SU worst case. Quantum federated on SAR and combined is about 18 h and 21 h per job and makes up two thirds of it.

The generator also reads `scontrol show reservation` and warns when a maintenance reservation on the partition starts before a job's limit would end. Such jobs stay PENDING until the reservation ends. Check `sinfo -T` before a large submission.

## 3. Submit

```bash
bash misc/grid/<timestamp>/submit_all.sh
```

It must be run as the submitting user. Logs go to `output_logs/<jobname>.o<jobid>` and `error_logs/<jobname>.e<jobid>` under the root. Submitted job ids are appended to `submitted.txt` in the grid folder.

To avoid a stampede on the shared data and the Excel tracker

- `--stagger-minutes N` adds `--begin=now+i*N minutes` to the i-th submission.
- `--max-concurrent K` chains the jobs into K lanes with `--dependency=afterany`, so at most K run at once and a failed job does not block its lane.

Both are baked into `submit_all.sh` at generation time.

## 4. Resume

Rerun the same script. The federated scripts take `--ckpt-every` and resume from the last checkpoint in their output directory, and regenerating the grid picks this up automatically. Resubmit only failed jobs with `sbatch misc/grid/<timestamp>/<script>.sh`, or regenerate a subset with `--seeds` and the axis filters into a new folder.

## 5. Results and analysis

Each run writes to `results/<run name>_seed<N>/` (curves, checkpoints, `summary.txt` with provenance, or `FAILED.txt` on failure) and adds a row to the Excel tracker. Then

```bash
python analyze_results.py
```

reads `results/` (or `$QUAFFLE_RESULTS_DIR`) and writes `results/analysis_summary.xlsx` and `results/plots/`. Missing seeds are listed per cell.

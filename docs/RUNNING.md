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
- `--q-backend pennylane` is passed to every quantum job, so the job script records which circuit simulator ran. `--q-backend torch` or `pennylane_pergroup` in the generator only makes sense for a cross-check grid, the time limits assume `pennylane`.
- The job header does not export `CUDA_LAUNCH_BLOCKING`. It forced a sync after every CUDA kernel and doubled the A100 step time of every model (see `docs/MODEL.md`).
- `--ckpt-every 5` is passed to federated jobs only if the target script source contains `ckpt-every`. The generator prints which mode it used.
- `--root` is the checkout the jobs `cd` into (default the checkout the generator lives in). Generate from the checkout you will run.
- `--out-dir` sets the output folder. The default is `misc/grid/<timestamp>/`. The generator refuses a non-empty folder and never writes into `misc/` itself.
- `--check-flags` runs each target script with `--help` once and fails if any emitted flag is unknown. It needs a user that can run the conda python. This takes a few minutes because each `--help` imports torch.

The output folder holds the job scripts (named `<method>_<mode>_<dataset>[_<ansatz>]_seed<N>.sh`), `submit_all.sh` and `manifest.csv` (script, job name, command, time limit, expected SU).

The two older generators are thin wrappers that share the same time limits and header.

- `misc/generate_bottleneck_ablation_scripts.py` is the grid restricted to `--methods ablation`.
- `misc/generate_qc_ablation_scripts.py` writes federated quantum jobs with `--quantum_channels 64 128 256`.

## 2. Time limits and cost

Runtimes measured on 2026-10-02 on one `ai` GPU, optical (200 training tiles), federated with 4 clients and 25 rounds of 4 local epochs. Classical and ablation were measured with `CUDA_LAUNCH_BLOCKING=1`, so they are conservative now that the flag is gone.

| method | centralized | federated |
|---|---|---|
| classical | 0.10 h | 0.51 h |
| ablation | 0.12 h | 0.56 h |
| quantum | 0.18 h (provisional) | 0.84 h (provisional) |

The quantum values are ablation x 1.5 and are provisional until replaced by measured hours from the smoke job (section 6). Under the 2x safety factor they still cover the measured runtimes of the earlier torch simulator grid (per-cell mean hours with blocking on: centralized optical 0.16, SAR 0.70, combined 0.83, federated optical 0.65, SAR 2.81, combined 3.38).

Expected hours are scaled by training tiles (SAR 1119 is 5.6x optical, combined 1318 is 6.6x). Dirichlet is assumed equal to random. For quantum jobs the circuit overhead (quantum minus ablation) is also scaled by `quantum_channels/32` and `q_layers/2`. That part is an extrapolation. The time limit is `ceil(expected x 2)` hours with a floor of 1 hour (`--safety`, `--floor-hours`). The summary table shows expected SU and the worst case if every job hits its limit. With `--budget` the generator refuses to write when the expected total is over budget.

Full grid (270 jobs, 10 seeds) is about 556 SU expected and 1290 SU worst case with the provisional quantum hours. Regenerate after updating `MEASURED_HOURS` in `misc/generate_grid.py`.

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

## 6. Smoke job and simulator comparison

Quantum runs made with older code used the torch statevector simulator (`q_backend='torch'` was the default then). Current runs use `q_backend='pennylane'` (one broadcast `default.qubit` QNode call per block forward, circuit on the CPU). Classical and ablation runs never touch the circuit and are unaffected. Keep runs of the two simulators in separate results trees, so that old checkpoints are not picked up by the resume logic.

1. Smoke job. `misc/smoke_pennylane_fast.sh` runs the benchmark on the A100 with and without `CUDA_LAUNCH_BLOCKING`, the GPU exactness test, and one centralized and one federated optical run (seed 17). It runs the code of `ROOT` (default: the directory `sbatch` is run from) but writes all results under `SMOKE_ROOT` (default `$ROOT/smoke_plfast`) by running from there with a `data` symlink, so nothing lands in `$ROOT/results`. Submit with `sbatch misc/smoke_pennylane_fast.sh`. Go criteria: the `pennylane` cpu full step is at most 1.6x the ablation step without blocking, the exactness test passes, both runs finish with the PennyLane simulator line in `summary.txt` and `"q_backend": "pennylane"` in `metrics.json`.
2. Put the smoke runs' sacct elapsed hours into `MEASURED_HOURS` in `misc/generate_grid.py`, then generate the quantum grid, for example `python misc/generate_grid.py --methods quantum --seeds 17 314 999` (27 jobs).
3. Compare the two simulators.

```bash
python tools/compare_backends.py <results tree A> <results tree B>
```

`tools/compare_backends.py` matches quantum runs on (mode, dataset, variant, seed), reads `results/metrics.json` (or parses `summary.txt` when it is missing) and prints, per (mode, dataset), the mean and std over the matched seeds of test AUC, IoU, Dice, loss and the knockout `zero` and `random` IoU for both roots, the difference of means and that difference in units of the seed std. `--per-run` lists every run, `--csv` writes the table. Run against itself the differences are exactly 0. The simulators are mathematically identical, but float32 summation order and GPU reductions differ, so judge agreement against the seed-to-seed spread, not equality.

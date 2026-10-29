# QUAFFLE: Quantum U-Net Assisted Federated Flood Learning and Estimation

QUAFFLE is a hybrid quantum-classical framework for flood segmentation from satellite imagery. It places a variational quantum circuit at the bottleneck of a U-Net and trains the model either centrally or with federated learning across several clients. The code is written in PyTorch, uses PennyLane for the quantum circuit and Flower for federated learning, and is evaluated on the SAR and optical flood data of the IEEE GRSS Data Fusion Contest 2024, Track 1.

QUAFFLE was a finalist in NASA's *Beyond the Algorithm Challenge: Novel Computing Architectures for Flood Analysis*.

## Overview

The repository contains training and evaluation code for three model families, each in a centralized and a federated setting.

- **Classical U-Net** (`UNetSegmentation`). The baseline, with no quantum component.
- **Hybrid QUAFFLE model** (`QVUNetSegmentation`). The same U-Net with a quantum bottleneck. 32 of the 256 bottleneck channels are split into 8 groups of 4. Each group is average pooled to 4 numbers, passed through a 4-qubit PennyLane circuit and broadcast back. Two circuit ansatzes are available: `reupload` (data re-uploading with bounded encoding, default) and `legacy` (the original single-layer circuit).
- **Parameter-matched classical ablation** (`ClassicalAblationUNetSegmentation`). The hybrid architecture with the circuit replaced by a small classical transform with about the same number of parameters, so that the effect of the circuit itself can be isolated.

What the code supports:

- Centralized training, and federated training with FedAvg over 4 clients (Flower simulation) with a random IID split or a label-skew Dirichlet split.
- Exact statevector simulation of the circuit with PennyLane `default.qubit` (backprop, one broadcast QNode call per block forward). A pure torch statevector simulator and the older per-group PennyLane path are kept as cross-checks.
- Quantum knockout evaluation. The test set is evaluated again with the quantum channels set to zero and with a freshly randomized circuit.
- A seeded, deduplicated data pipeline with fixed train, validation and test splits and per-run provenance.

Model details, circuit definitions, equivalence tests and parameter counts are in [docs/MODEL.md](docs/MODEL.md).

## Repository layout

```
centralized_demoTorchClassical.py   centralized classical U-Net
centralized_demoTorchQ.py           centralized hybrid model
federated_demoTorchClassical.py     federated classical U-Net (Flower, FedAvg)
federated_demoTorchQ.py             federated hybrid model
ablation_studies.py                 parameter-matched classical ablation, centralized and federated
utilsTorch/                         models (unetTorch.py) and data, training and evaluation helpers (utilsTorch.py)
flood_data.py                       shared data loader (splits, bands, normalization)
tools/                              shared training code, provenance, benchmarks, data tools, backend comparison
config/seeds.py                     the pre-declared seed list
config/cluster.example.json         template for SLURM cluster settings
splits/                             deduplicated split manifests and normalization statistics
misc/                               SLURM script generators and a smoke-test job template
analyze_results.py                  aggregation of results across runs and seeds
scripts/                            figure and table builders
tests/                              equivalence tests for the quantum block
results_tracker.py                  per-dataset Excel summary of runs
visualize_circuit.py                draws the quantum circuit
docs/                               DATA.md, MODEL.md, RUNNING.md
install_requirements.sh             pinned pip install, GPU or CPU
centralized_demoJAX.py, federated_demoJAX.py, utilsJAX/   legacy JAX code (see below)
```

## Installation

Python 3.11 is required. For GPU runs the scripts were used with CUDA 12. `install_requirements.sh` installs with `pip` into the active environment, so create and activate one first, for example with conda:

```bash
conda create -n quaffle python=3.11
conda activate quaffle
./install_requirements.sh gpu    # default
./install_requirements.sh cpu
```

The script installs pinned versions, among them torch 2.1.2 (cu121 wheels in GPU mode), torchvision 0.16.2, PennyLane 0.41.1, PennyLane-Lightning 0.41.1, flwr 1.19.0 with the simulation extra and pandas 3.0.2. It also installs the JAX stack (jax 0.4.28, flax 0.8.4) and TensorFlow 2.19.0, which are only needed for the legacy JAX code. seaborn is optional for `analyze_results.py`.

## Data

The DFC 2024 Track 1 data is not distributed with this repository. Obtain it from the contest organizers (https://www.grss-ieee.org/community/technical-committees/2024-ieee-grss-data-fusion-contest/) and arrange it under `data/` as follows.

```
data/
├── flood_optical/
│   ├── Training/{images,labels}/
│   └── Testing/{images,labels}/
├── flood_SAR/
│   ├── Training/{images,labels}/
│   └── Testing/{images,labels}/
└── flood_combined/          # optical and SAR tiles copied into one folder
    ├── Training/{images,labels}/
    └── Testing/{images,labels}/
```

`data` may also be a symlink to a shared copy. `flood_combined` is built from the other two folders. SAR files whose names clash with an optical file get a ` copy` suffix (`164.tif` becomes `164 copy.tif`), which is what `splits/flood_combined.json` expects. The exact steps are in [docs/DATA.md](docs/DATA.md#building-flood_combined).

- **Split manifests.** The provided SAR folders contain duplicate training tiles and test tiles that are pixel-identical to training tiles. `splits/flood_*.json` list the tiles that are kept, and the loader applies them, so nothing under `data/` is modified. After deduplication and excluding the 18 duplicate SAR training groups with conflicting masks (`--conflict-policy drop`), there are 235 training and 71 test tiles for optical, 1298 and 134 for SAR (1316 under `--conflict-policy first`), and 1533 and 205 for combined (1551 under `--conflict-policy first`). The manifests store per-file hashes and can be regenerated with `tools/dedup_splits.py`.
- **Bands.** Three input channels per tile. SAR uses VV, VH and the Copernicus DEM. Optical uses green, NIR and SWIR1.
- **Normalization.** Per-band 1st and 99th percentiles computed on the deduplicated training split only (`splits/norm_stats.json`, from `tools/compute_norm_stats.py`), with log1p applied to VV and VH, mapped to [-1, 1]. Tiles are resized to 128 x 128 (SAR is natively 512 x 512).
- **Validation.** Every run carves a 15 percent validation set from the training pool, stratified on the binarised flood label and determined by the seed only, before any federated partitioning.

The legacy loader behaviour is still available with `--legacy-data`. Details, counts and the band verification are in [docs/DATA.md](docs/DATA.md).

## Running

Run every command from the repository root. Results go to `results/<run name>_seed<N>/`.

For the completed local 16-qubit HQ controls, see the
[results and interpretation](benchmarks/hq_training_controls_20261028/INTERPRETATION.md).
The [DFC24 decision protocol and Alan handoff](docs/HQ_DFC24_DECISION_PROTOCOL.md)
define the next exploratory full-fit screen and its execution/profiling
requirements. Its [exposure and metric audit](benchmarks/hq_dfc24_decision_20261028/README.md)
uses existing data and saved counts; it launches no training or cluster jobs.
The [fresh execution protocol](docs/HQ_DFC24_FRESH_PROTOCOL.md) provides a separate
two-epoch rehearsal with 16 training and 16 validation tiles per seed, exercising
the complete fresh selection flow without test-set evaluation. Its purpose is
software verification; full-data accuracy and Anvil feasibility remain separate.
The [completed local rehearsal](benchmarks/hq_dfc24_fresh_rehearsal_20261028/README.md)
retains 16 fresh fits and 12 validation replays. All selected checkpoints were
epoch zero, so it supplies no new accuracy evidence for quantum.
The [publication figures](benchmarks/hq_validation_figures_20261028/README.md)
use the earlier 60-epoch control histories and fixed-rule validation examples.
The [fresh HQ SLURM launch guide](docs/HQ_ANVIL_LAUNCH.md) packages separate
profile, run and collect jobs for Alan's configuration and manual profile acceptance.

The separate [native LandCover.ai protocol](docs/LANDCOVER_NATIVE_PROTOCOL.md)
tests the five-class task of the HQ-UNet paper with its diagram's full-width
architecture and paired learned, frozen, compact classical and zero controls.
It provides checksum-verified data preparation, an exact-model local resource
profile, a frozen three-seed campaign and validation-only checkpoint collection.
This capped native-dataset comparison is distinct from the binary DFC24 study
and from an exact reproduction of the paper's reported test accuracy.
The [completed native comparison](benchmarks/landcover_native_local_20261028/README.md)
finds only +0.157 percentage points of mean quantum mIoU over affine, with
unresolved training horizons and no convincing gain over the classical control.

```bash
# Centralized
python centralized_demoTorchClassical.py --dataset optical --seed 17
python centralized_demoTorchQ.py --dataset optical --seed 17 --q-ansatz reupload --q-backend pennylane --quantum-knockout-eval
python ablation_studies.py --framework torch --mode centralized --dataset optical --seed 17

# Federated, random IID split
python federated_demoTorchClassical.py --partition random --dataset optical --seed 17
python federated_demoTorchQ.py --partition random --dataset optical --seed 17 --quantum-knockout-eval
python ablation_studies.py --framework torch --mode federated --partition random --dataset optical --seed 17

# Federated, label-skew Dirichlet split
python federated_demoTorchQ.py --partition dirichlet --alpha 0.5 --dataset sar --seed 17
python ablation_studies.py --framework torch --mode federated --partition dirichlet --alpha 0.5 --dataset sar --seed 17
```

Key flags (run any script with `--help` for the full list):

| Flag | Scripts | Meaning |
|---|---|---|
| `--dataset {optical,sar,combined}` | all | Dataset under `data/` |
| `--seed N` | all | Seed for data splits, model initialization and training |
| `--partition {random,dirichlet}`, `--alpha A` | federated | Client split. `dirichlet` skews the flood versus non-flood mix per client with concentration `A` |
| `--min-client-samples N` | federated | Minimum training samples per client under the Dirichlet split (default 16) |
| `--q-ansatz {reupload,legacy}`, `--q-layers N` | hybrid, ablation | Circuit ansatz and number of re-uploading layers (default reupload, 2 layers) |
| `--q-backend {pennylane,pennylane_pergroup,torch}` | hybrid | Circuit simulator. All three compute the same circuit. `pennylane` is the default |
| `--quantum-knockout-eval` | hybrid | Also evaluate the test set with the quantum block set to `zero` and `random` |
| `--quantum_channels {32,64,128,256}` | federated hybrid | Number of bottleneck channels routed through circuits (default 32) |
| `--ckpt-every N`, `--no-resume` | federated | Checkpoint interval in rounds, and disabling resume |
| `--max-samples N`, `--epochs N`, `--rounds N`, `--local-epochs N` | see `--help` | Debug overrides for quick runs |

The default schedule is 100 epochs (full passes over the training set) for centralized runs and 25 rounds of 4 local epochs over 4 clients for federated runs, with batch size 8 and Adam at learning rate 1e-4.

### SLURM

`misc/generate_grid.py` writes one SLURM script per (method, mode, dataset, seed), with time limits derived from measured runtimes, plus `submit_all.sh` and `manifest.csv`. Cluster settings are not hard-coded. Copy `config/cluster.example.json` to `config/cluster.local.json` (ignored by git) and fill in your account, partition, Python interpreter, conda setup and modules, or pass them as `QUAFFLE_*` environment variables or command-line flags.

```bash
cp config/cluster.example.json config/cluster.local.json   # then edit
python misc/generate_grid.py --dry-run
python misc/generate_grid.py --methods quantum --datasets optical --seeds 17
```

Filters, time limits, staggered submission, resuming and the smoke-test job are described in [docs/RUNNING.md](docs/RUNNING.md).

## Reproducibility

- **Seeds.** `config/seeds.py` holds the single pre-declared seed list (7, 17, 31, 42, 77, 123, 256, 314, 512, 999). The SLURM generators and the analysis scripts read it. Missing runs are reported with the missing seed ids rather than hidden.
- **Seeding.** python, numpy, torch and CUDA are seeded once per run, every DataLoader has its own seeded generator, and federated clients reseed their shuffle from (seed, client id, round).
- **Provenance.** Every run records the git SHA, dirty flag, branch, full command line, seed, host, SLURM job id, library versions and the circuit simulator that was used. This is printed at start and written to `results/summary.txt` and `results/metrics.json` inside the run directory.
- **Resume.** Centralized runs resume from `checkpoint.pt` in their output directory. Federated runs save a checkpoint every `--ckpt-every` rounds and resume from it when rerun with the same arguments. Checkpoints from an incompatible run format or a different circuit simulator are not resumed.
- **Failures.** A federated run in which any round fails writes `FAILED.txt` instead of a summary and exits non-zero.

## Analysis

```bash
python analyze_results.py
```

reads the runs under `results/` (or `$QUAFFLE_RESULTS_DIR`), parses `summary.txt` of each run, aggregates per method, mode and dataset across seeds with paired t-tests and Wilcoxon signed-rank tests, and writes `analysis_summary.xlsx` and plots to `$QUAFFLE_ANALYSIS_OUT_DIR` (default: the results folder).

By default only hybrid runs with `q_backend='pennylane'` are kept. Runs that record provenance but no `q_backend` count as `torch`, and older `summary.txt` files without a provenance block count as `pennylane_pergroup`, so both are left out by default. `--include-backend` takes one or more of `pennylane`, `pennylane_pergroup` and `torch`, or `all`, for example `python analyze_results.py --include-backend pennylane torch`. With more than one simulator the non-PennyLane methods get a ` [qtorch]` or ` [qpergroup]` label suffix.

`tools/compare_backends.py ROOT_A ROOT_B` compares the hybrid runs of two results trees (for example the same grid run with two circuit simulators) per mode, dataset, variant and seed. Three scripts build figures and tables. Each takes `--out-dir`, and without it they write to:

- `scripts/build_forest_plot.py`: paired IoU forest plot (PNG and JSON) in `$QUAFFLE_FOREST_OUT_DIR`, else `$QUAFFLE_FIGURES_DIR/forest`, else `figures/forest/`.
- `scripts/build_table2_data.py`: summary table data of the pooled optical and SAR setting (`headline_pooled_table2_data.json`) in `$QUAFFLE_ANALYSIS_OUT_DIR`, else `results/`.
- `scripts/build_exemplar_figure.py`: exemplar prediction figure and `meta.json` in `$QUAFFLE_FIGURES_DIR/exemplars`, else `figures/exemplars/`.

## Tests

```bash
python tests/test_quantum_backend.py
python tests/test_pennylane_fast.py
```

They run with plain Python. Each script prints `PASS` or `FAIL` per check and a final count, and exits non-zero if a check fails. pytest is optional and is not installed by `install_requirements.sh`. The tests check that the PennyLane, per-group PennyLane and torch simulators give the same outputs and gradients, that state dicts load across simulators, and that the legacy circuit matches its closed form. `tests/test_pennylane_fast.py --device cuda` repeats the comparison on a GPU.

## Legacy code

- `centralized_demoJAX.py`, `federated_demoJAX.py`, `utilsJAX/` and the `--framework jax` mode of `ablation_studies.py` are an earlier JAX implementation. They are kept for reference, are not maintained, and do not include the data, seeding, provenance and failure-handling changes of the PyTorch code. None of the current experiments use them.
- `utilsTorch/unetTorch.py` also contains an experimental photonic backend (`quantum_backend='ptlayer'`) built on ORCA Computing's `ptseries` SDK. It is only active when `ptseries` is installed, which `install_requirements.sh` does not do. It is not tested here and was not used in the reported experiments.

## Citation

Paper in preparation. Until then, please cite this repository.

## Related work

The quantum bottleneck design is inspired by hybrid quantum diffusion models.

```bibtex
@article{defalco2024quantum,
  title={Towards Efficient Quantum Hybrid Diffusion Models},
  author={De Falco, Francesca and Ceschini, Andrea and Sebastianelli, Alessandro and Le Saux, Bertrand and Panella, Massimo},
  journal={KI - K{\"u}nstliche Intelligenz},
  volume={38},
  number={4},
  pages={311--326},
  year={2024},
  publisher={Springer},
  doi={10.1007/s13218-024-00858-5}
}
```

Previous work of the group on federated learning:

```bibtex
@article{dutta2025federated,
  title={Federated Learning in Chemical Engineering: A Tutorial on a Framework for Privacy-Preserving Collaboration across Distributed Data Sources},
  author={Dutta, Siddhant and de Freitas, Iago Leal and Xavier, Pedro Maciel and de Farias, Claudio Miceli and Bernal Neira, David E.},
  journal={Industrial \& Engineering Chemistry Research},
  volume={64},
  number={15},
  pages={7767--7783},
  year={2025},
  publisher={ACS Publications},
  doi={10.1021/acs.iecr.4c03805}
}

@article{dutta2024federated,
  title={Federated learning with quantum computing and fully homomorphic encryption: A novel computing paradigm shift in privacy-preserving ML},
  author={Dutta, Siddhant and Karanth, P. P. and Xavier, P. M. and de Freitas, I. L. and Innan, Nouhaila and Yahia, Sadok Ben and Shafique, Muhammad and Bernal Neira, David E.},
  journal={arXiv preprint arXiv:2409.11430},
  year={2024}
}
```

## Acknowledgements

- NASA ESTO for hosting the *Beyond the Algorithm Challenge: Novel Computing Architectures for Flood Analysis* (https://www.nasa-beyond-challenge.org/) and recognizing QUAFFLE as a finalist.
- The SECQUOIA research group at Purdue University for support and collaboration.

## Contact

For questions or collaboration, contact the SECQUOIA group at Purdue University (https://secquoia.github.io/).

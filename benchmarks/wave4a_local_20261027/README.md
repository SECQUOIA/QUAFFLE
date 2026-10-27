# Wave 4A local verification — 2026-10-27

This is a bounded synthetic-data exploration of the controls added in PRs #50–#52.
It measures ordinary end-to-end training, without freezing encoder or decoder weights.
It does not establish real DFC24 performance, statistical superiority, or hardware quantum advantage.

## Method and reproduction

The producing implementation revision is `a2664cadd20765a22250449a6280ffbaa195b3dd`.
The dataset is generated once from seed 42. Model initialization, validation splitting,
and client training use the paired seeds 42, 77, and 123. These are three training/split
seeds on one synthetic dataset realization, not three independent datasets.

All runs use base width 8, 32×32 images, batch size 4, learning rate 0.005,
Dice + cross-entropy loss, and no time embeddings. Centralized training runs for four
epochs on a pool of 16 training tiles before validation holdout and six test tiles per sensor. The federated panel uses both
sensors, four modality-silo clients, full participation, three FedAvg rounds, and two
local epochs per round. Checkpoints are selected by held-out validation IoU; test
IoU is evaluated after selection. Quantum models use the default two-layer,
four-qubit re-uploading block, PennyLane on CPU, and shared bottleneck parameters.

Panel 5 has 14 configurations: two classical references (`none`/`affine` decoder gates),
and quantum/parameter-matched MLP/full-rank-permitted MLP controls crossed with
32/64 processed channels and both gate settings. At this scale, 64 is the full
bottleneck. The two MLP hidden widths are three and four respectively. Panel 6 has
12 configurations: classical, MLP, and quantum FedAvg crossed with none/FP16/INT8/top-10%
update compression. The compression panel uses partial bottlenecks and ungated
skips, keeping the architecture comparison separate.

```bash
QUAFFLE_INPROCESS_FED=1 python scripts/run_local_mini_sweep.py \
  --panels 5 6 --seeds 42 77 123 --device cuda \
  --work-dir /tmp/quaffle-wave4-reproduction --keep-work-dir \
  --out-json summary.json --out-md report.md
python benchmarks/wave4a_local_20261027/reproduce_comparisons.py
```

The last command regenerates the paired comparisons from the archived `summary.json`
beside the script. To analyze a fresh reproduction, replace that archived input with
the new summary in a disposable copy. Differences are computed seed by seed before
reporting their mean and sample standard deviation. The retained values use IoU units;
multiply by 100 for percentage points. No significance claim or tuning decision is
based on this small test set.

Execution used an NVIDIA RTX A1000, two Torch intra/inter-op threads, a 6 GiB host
memory limit with swap disabled, and a 12% GPU allocator cap (about 0.96 GiB).
See the provenance JSON files for Python/package versions. This reused local environment
differs from the older versions pinned by `install_requirements.sh`; it is not evidence
of cluster-environment parity. Rasterio 1.4.3 and support packages were added in an
isolated dependency overlay, without modifying the shared environment.

The first invocation completed all 42 architecture fits, then stopped before any
federated fit because Ray was unavailable. All architecture metrics/settings were
recovered directly from their written metrics files; their per-run timings were not
retained and are null. The 36 compression fits use the existing in-process Flower
`Server.fit` executor with sequential client proxies. This exercises actual client
training and aggregation strategies, but does not validate Ray execution or wire
transport. Compression upload figures model raw tensor payloads and exclude RPC framing.
Shared workstation timings should not be used for speed comparisons.

## Evidence

- `summary.json`: all 78 per-seed records, including validation/test outcomes,
  both zero/random knockouts, parameter budgets, and modeled payload accounting.
- `report.md`: aggregate means and sample standard deviations.
- `paired_comparisons.json`: paired architecture and compression differences.
- `provenance.json` and `compression_provenance.json`: producing revision,
  runner hash, runtime versions, resource limits, and executor details.

## Validation

At the producing revision, the full local test suite passed: 214 tests. The initial seven missing-Rasterio
failures were resolved through the isolated dependency overlay and rechecked.

## Results and implications

| Centralized configuration | Parameters | Test IoU (mean ± sample SD) | On − zero IoU |
| --- | ---: | ---: | ---: |
| Classical U-Net (gate=none) | 648,106 | 0.8950 ± 0.0219 | — |
| Quantum (qc=32, gate=none) | 537,386 | 0.8824 ± 0.0172 | +0.0000 |
| MLP param_matched (qc=32, gate=none) | 537,378 | 0.8736 ± 0.0262 | -0.0003 |
| MLP full_rank (qc=32, gate=none) | 537,450 | 0.8890 ± 0.0304 | -0.0003 |
| Quantum (qc=64, gate=none) | 500,394 | 0.8956 ± 0.0436 | -0.0000 |
| MLP param_matched (qc=64, gate=none) | 500,378 | 0.8873 ± 0.0352 | -0.0031 |
| MLP full_rank (qc=64, gate=none) | 500,522 | 0.8932 ± 0.0377 | -0.0003 |
| Quantum (qc=64, gate=affine) | 518,074 | 0.8545 ± 0.0568 | -0.0134 |

The full ungated quantum model's mean IoU is close to the classical reference with
22.8% fewer parameters (500,394 versus 648,106). The full ungated wider MLP reaches
0.8932 with 500,522 parameters. The quantum-minus-wider-MLP mean difference there is
only about +0.24 IoU percentage points. On the partial bottleneck, widening the MLP
changes the quantum-minus-control gap from about +0.88 to −0.66 points. These tiny
synthetic comparisons support an architectural compression hypothesis; they do not
establish a quantum-specific benefit or statistical equivalence.

Full ungated quantum knockout remains approximately zero. Full quantum plus affine
decoder gates reduces mean IoU to 0.8545, and its mean on-minus-zero delta is −0.0134:
zeroing the quantum output improves IoU by about 1.34 percentage points on average.
Making the output consequential did not make that contribution beneficial in this
experiment. The full-rank-permitted MLP label removes a hidden-width rank ceiling;
it does not guarantee full Jacobian rank at every trained input.

| Federated model | None | FP16 | INT8 | Top-10% |
| --- | ---: | ---: | ---: | ---: |
| classical | 0.8732 | 0.8727 | 0.8591 | 0.4548 |
| ablation | 0.8737 | 0.8740 | 0.8741 | 0.6818 |
| quantum | 0.8674 | 0.8673 | 0.8665 | 0.6861 |

FP16 is close to uncompressed IoU in all three models while modeling a 50% raw-payload
reduction. INT8 has mixed effects, including a roughly 1.4-point classical IoU decrease.
Top-10% substantially reduces IoU under this three-round training schedule. The full
report includes seed dispersion and both knockout interventions. These observations
do not establish long-horizon compression behavior or actual network-byte savings.

A useful next Anvil deliverable is a small paired real-DFC24 pilot of ungated partial
and full bottlenecks with both MLP controls, alongside the ordinary U-Net reference.
Keep affine gating as a diagnostic rather than assuming it improves performance.
Use FP16 as the first compression pilot. Resolve the strict production manifest and
fit-only normalization gate in issue #45 before interpreting production results.
No Anvil jobs were submitted as part of this local experiment.

# Ordinary U-Net at the quantum parameter budget

This follow-up to PR #53 tests whether a uniformly narrower ordinary U-Net
achieves the same small-model tradeoff as the full quantum bottleneck. All 21
fits completed: seven configurations across paired seeds 42, 77, and 123.
The results are a descriptive synthetic-data comparison, with no claim of
real-data superiority, statistical equivalence, or quantum advantage.

## Parameter matching and normalization

The width-8 full quantum bottleneck has **500,394** trainable parameters.
Before training, ordinary U-Nets with base widths 4–8 were counted, preserving
the standard `(1, 2, 4, 8)` stage multipliers, spatial bottleneck convolutions,
and ungated encoder–decoder skips. Width 7 was closest at **510,631** parameters,
**2.05% above** the target. This is an approximate budget match, not an exact one;
the ordinary baseline receives the slightly larger budget. No width was selected
using validation or test performance. All candidate counts are in `summary.json`.

Seven channels cannot divide into four GroupNorm groups. The primary comparison
therefore uses **one normalization group throughout**: width-7 ordinary U-Net,
width-8 ordinary U-Net, full quantum, and the wider MLP replacement in the hybrid
architecture. The last three also run with four groups as a bridge to the
previous experiment. The MLP hidden width is four (`full_rank` mode); that
removes the width-three rank ceiling without guaranteeing full Jacobian rank
at every input. It has **500,522** parameters in either normalization arm.

| Primary comparison, one GroupNorm group | Parameters | Test IoU mean ± sample SD |
| --- | ---: | ---: |
| Slim ordinary U-Net, base width 7 | 510,631 | 0.8783 ± 0.0144 |
| Ordinary U-Net, base width 8 | 648,106 | 0.8887 ± 0.0055 |
| Full quantum bottleneck, base width 8 | 500,394 | 0.8829 ± 0.0138 |
| Full wider-MLP bottleneck, base width 8 | 500,522 | 0.8938 ± 0.0142 |

Quantum minus slim U-Net averaged **+0.462 IoU percentage points**, with paired
sample SD **2.585 points**. The individual differences were **+3.391, −0.504,
and −1.502 points** for seeds 42, 77, and 123. Quantum was ahead on one seed
and behind on two. The wider MLP averaged **1.089 points above quantum** in
this same-normalization comparison. The larger ordinary U-Net averaged
1.041 points above the slim ordinary U-Net.

These runs show that a comparably sized ordinary U-Net is competitive. They
do not establish that the quantum model is better at a fixed budget. The wider
classical replacement also remains a strong control, so any proposed benefit
from changing the bottleneck architecture must be separated from the quantum
mapping itself. Small sample size, one synthetic dataset realization, four
epochs, and a common fixed optimizer setting limit inference.

The four-group bridge reproduces the previous sweep's three quantum test
metric dictionaries exactly (mean IoU 0.8956). The four-group ordinary and
wider-MLP means are 0.8950 and 0.8932. Changing quantum normalization from four
groups to one changes mean IoU by −1.269 points, which is larger than its
same-normalization mean lead over the slim baseline. The report contains all
paired comparisons; cross-normalization headline differences are confounded.
Mean on-minus-zero quantum knockout effects remain small: about +0.0005 IoU
with one group and −0.00004 with four groups.

## Protocol and evidence

The producing revision is `6d9f2808b3d015b509df2d59dac1a317c3dee2ae`.
The experiment reuses the repository's model wrappers, seeded construction,
dataset loading, fit-only normalization, validation split, seeded loaders,
`train_one_epoch`, `evaluate_model`, checkpoint-selection rule, and knockout
evaluation. Its standalone loop is identical across all seven configurations.
All model parameters are optimized with Adam. It evaluates validation at epoch
zero and after every epoch, retains the first best validation-IoU checkpoint,
then evaluates that checkpoint on test data. Test metrics never select widths
or checkpoints. Per-fit validation histories and selected epochs are retained.

Common settings: base image size 32×32, batch size 4, learning rate 0.005,
four epochs, Dice + cross-entropy loss, no time embeddings, no augmentation,
no decoder gating, and the ordinary weight-standardized spatial convolutions.
Quantum uses the default shared, two-layer, four-qubit re-uploading circuit
with 64 processed channels and PennyLane on CPU. The wider MLP uses the same
full hybrid-bottleneck architecture with a classical mapping instead.

One synthetic dataset is generated with seed 42: 16 optical training-pool
tiles before the 15% stratified validation holdout, and six optical test tiles.
The paired seeds affect initialization, validation splitting and train shuffling;
they are not independent dataset replications. Each seed has identical split
indices and input names across all seven models. Different-width models cannot
share identical initial tensors. Quantum/MLP at the same width and normalization
retain the repository's paired initialization behavior.

All **184** generated input files match the earlier sweep's hashes, and every
hash was checked against the original local file. The **21** retained records
match their per-fit original metrics, including all **24** knockout evaluations.
The selected validation epochs were checked against their full histories.

- `summary.json`: all settings, candidate parameter counts, selected width,
  per-fit splits/names, validation histories, test and knockout metrics,
  aggregate statistics, and eight seed-paired comparisons.
- `report.md`: generated descriptive table and paired differences.
- `input_hashes.json`: every generated input file's SHA256 hash.
- `provenance.json`: producing source revision and hashes, Python/package
  versions, GPU identity, and Torch thread/allocation settings.

The NVIDIA RTX A1000 run used two Torch intra/inter-op threads, a 12% GPU
allocation cap, and an external 6 GiB host-memory limit with swap disabled.
The reused Python environment and isolated Rasterio dependency overlay match
the previous local sweep, but differ from older versions pinned by the install
script. Runtime equality with Anvil is unverified. Timings include loading,
construction, training, and evaluation on a shared workstation and support no
speed claim. No Anvil jobs or external services were modified.

## Reproduction

Run in the project's Python environment under an appropriate host-memory cap:

```bash
python scripts/run_parameter_budget_comparison.py \
  --work-dir /tmp/quaffle-budget-reproduction --seeds 42 77 123 --device cuda
```

The output directory must be new or empty. Each completed fit is saved
immediately, including a selected checkpoint and a cumulative partial summary;
later failure does not delete earlier evidence. Source code and production
training defaults are unchanged by this standalone experiment.

To regenerate the archived aggregate report from the retained records:

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.run_parameter_budget_comparison import summarize, format_report
p = Path('benchmarks/unet_budget_local_20261027')
s = json.loads((p / 'summary.json').read_text())
assert summarize(s['records']) == s['analysis']
assert format_report(s).encode() == (p / 'report.md').read_bytes()
PY
```

# QUAFFLE model notes

This file describes the hybrid bottleneck in `utilsTorch/unetTorch.py`, the
construction options in `utilsTorch/utilsTorch.py`, and the parameter counts
of every model variant.

## Bottleneck layout

The QVUNet bottleneck (`QVertex`) has two `QResnetBlock`s with attention in
between. Each block splits the 256 bottleneck channels into 32 quantum channels
and 224 classical channels. The 32 quantum channels form 8 groups of 4. Each
group is spatially average pooled to 4 numbers, mapped by a 4 qubit circuit to
4 `<Z>` values, and broadcast back over (H, W). The quantum output is therefore
spatially constant within a group. The classical ablation keeps this exact
pool, transform, broadcast pattern and only swaps the per-group transform.

## Circuit options (`q_ansatz`)

### `legacy` (original model)

Per group: `RX(x_j)` on wire j, then `RY(theta_j)` on wire j, then a CNOT ring
0->1->2->3->0, then `<Z_j>`. The inputs `x_j` are the raw spatial means with no
scaling. There are 4 parameters per group and 32 in total.

Closed form. Before the CNOT ring the state is a product state with single
qubit `<Z_j> = c_j`, where `c_j = cos(x_j) cos(theta_j)`. A CNOT ring is a
permutation of computational basis states, so each output is a parity of
input bits. Checked numerically against the PennyLane circuit (20 random
draws, error below 1e-9):

| output | equals |
| --- | --- |
| `<Z_0>` | `c_1 c_2 c_3` |
| `<Z_1>` | `c_0 c_1` |
| `<Z_2>` | `c_0 c_1 c_2` |
| `<Z_3>` | `c_0 c_1 c_2 c_3` |

So the legacy block computes fixed products of cosines and can be evaluated
classically in a few multiplications. It is kept only to reproduce the
original results.

### `reupload` (default for new runs)

`q_layers` layers (default 2). Each layer applies:

1. Bounded encoding `RY(pi * tanh(s_lj * x_j))` on wire j, where `s_lj` is a learnable input scale initialized to 1. Angles stay inside (-pi, pi).
2. A general rotation `Rot(w_lj0, w_lj1, w_lj2)` on wire j, initialized uniformly in [-pi, pi].
3. A CNOT ring 0->1->2->3->0.

Measurement is `<Z_j>` on every wire. The inputs are encoded again after each
entangling layer, so the outputs are no longer products of single qubit
functions of the inputs. Parameters per group are `16 * q_layers` (12 rotation
angles and 4 input scales per layer). The circuit runs on `default.qubit` with
the torch interface, `diff_method="backprop"` and parameter broadcasting over
the batch (one QNode call per group per forward pass).

Measured CPU cost (8 groups, batch 4, forward and backward of the quantum
block only): legacy 186 ms, reupload with 2 layers 110 ms. The reupload circuit
is faster because it is batched while legacy loops over batch items.

### Sharing between the two bottleneck blocks (`q_share_blocks`)

`q_share_blocks=True` (default, original behaviour) passes the same
`QuantumBlock` to `qresblock1` and `qresblock2`, so both use one set of circuit
parameters. `q_share_blocks=False` builds a second block with its own
parameters. The classical ablation follows the same setting.

## Knockout switch

`set_quantum_mode(mode, knockout_seed=0)` exists on `QuantumBlock`,
`ClassicalBottleneckBlock`, `QVUNet`, `QVUNetSegmentation` and
`ClassicalAblationUNetSegmentation`.

| mode | effect on the bottleneck output |
| --- | --- |
| `on` | trained circuit (default) |
| `zero` | all quantum channels replaced with zeros |
| `random` | same circuit with fresh random parameters from the init distribution |

The random parameters come from a local generator seeded with
`knockout_seed + 1000003` (unshared blocks use `knockout_seed` and
`knockout_seed + 1`). The trained parameters, the state_dict and the global
RNG are not modified. Switching back to `on` restores the exact original
outputs. Call it after `model.eval()` and before evaluation.

## Seeding

`QuantumBlock` no longer calls `torch.manual_seed`. Previously it reseeded the
global RNG with 42 three times and drew circuit parameters from a generator
seeded with `42 + group_idx`. As a result the quantum parameters, every module
built after the block (the whole QVUNet) and later DataLoader shuffles were
identical for every `--seed`.

Now every model wrapper accepts `seed=None`.

* `seed=None` uses the global torch RNG as seeded by the caller. Construction advances the global RNG but never resets it.
* `seed=<int>` builds the whole model under `torch.random.fork_rng` seeded with that value. The result depends only on the seed and the caller's global RNG state is left untouched. `QuantumBlock(seed=...)` uses a local `torch.Generator` directly.

The JAX models (`utilsJAX/unetJAX.py`) take their key from `config.seed` and
already standardize weights without mutation, so they need no change.

## Weight standardization

`WeightStandardizedConv` used to write the standardized kernel back into
`conv.weight.data` on every forward pass, including evaluation. That changed
the stored parameters outside the optimizer and put standardized weights into
FedAvg payloads. It now standardizes into a local tensor and calls `F.conv2d`.
Gradients flow through the standardization. This affects the classical and
hybrid models in the same way, so results from before and after this change
are not directly comparable.

## How training scripts pass the options

Config keys read by `QVUNetSegmentation` and `ClassicalAblationUNetSegmentation`
(`UNetSegmentation` ignores them):

| key | default | values |
| --- | --- | --- |
| `q_ansatz` | `'reupload'` | `'legacy'`, `'reupload'` |
| `q_layers` | `2` | int >= 1 (reupload only) |
| `q_share_blocks` | `True` | bool |

Suggested CLI wiring:

```python
config['q_ansatz'] = args.q_ansatz                # --q-ansatz {legacy,reupload}
config['q_layers'] = args.q_layers                # --q-layers 2
model = QVUNetSegmentation(config, seed=config['seed'])
if args.quantum_knockout_eval != 'on':            # --quantum-knockout-eval {on,zero,random}
    model.eval()
    model.set_quantum_mode(args.quantum_knockout_eval, knockout_seed=config['seed'])
```

To reproduce the original architecture pass `q_ansatz='legacy'`. Note that
the seeding and weight standardization fixes still apply, so legacy runs are
not bit identical to the original ones. Code that does not pass `seed` keeps
working and uses the global RNG. In federated code every client and the
server should build the template model with the same `seed` so the initial
global weights agree.

## Parameter counts

Full configuration: `base_channels=32`, `dim_mults=(1, 2, 4, 8)`,
`resnet_block_groups=4`, `quantum_channels=32`, PennyLane backend. Counted with
`sum(p.numel() for p in model.parameters())`.

| model | q_ansatz | q_layers | shared | params | ablation | diff | rel. diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classical FedU-Net (`UNetSegmentation`) | n/a | n/a | n/a | 9,327,970 | n/a | n/a | n/a |
| QUAFFLE | legacy | n/a | yes | 8,758,146 | 8,758,274 | +128 | +0.0015% |
| QUAFFLE | reupload | 2 | yes | 8,758,370 | 8,758,362 | -8 | -0.0001% |
| QUAFFLE | reupload | 3 | yes | 8,758,498 | 8,758,506 | +8 | +0.0001% |
| QUAFFLE | reupload | 2 | no | 8,758,626 | 8,758,610 | -16 | -0.0002% |

Ablation transform per group: `Linear(4, 4)` (20 params) for legacy, and
`Linear(4, h)-Tanh-Linear(h, 4)-Tanh` (9h + 4 params) for reupload with
`h = round((16 * q_layers - 4) / 9)`, which gives h = 3 for 2 layers and h = 5
for 3 layers. The final Tanh bounds the output to [-1, 1] like `<Z>`.

The parameter reduction relative to the classical FedU-Net (about 570k) comes
from the 32 quantum channels removing classical convolutions in both bottleneck
blocks. The circuit itself holds at most a few hundred parameters, so the
reduction is a property of the channel split and is shared with the ablation.

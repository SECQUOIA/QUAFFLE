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
the batch and the 8 groups (one QNode call per block forward pass, see
`q_backend` below).

## Circuit simulator (`q_backend`)

`QuantumBlock(q_backend=..., q_sim_device=...)`, the config keys `q_backend`
and `q_sim_device`, and the training flags `--q-backend` and `--q-sim-device`
choose how the circuits are evaluated. The math is the same for all three
simulators. Every reported QUAFFLE number comes from `pennylane`.

| `q_backend` | how |
| --- | --- |
| `'pennylane'` (default) | one PennyLane `default.qubit` QNode call per block forward, torch interface, `diff_method="backprop"`, broadcast over (batch x groups) including the trainable weights and input scales |
| `'pennylane_pergroup'` | the previous PennyLane path, one QNode call per 4 qubit group (legacy also loops over batch items). Kept as a test reference and for `visualize_circuit.py` |
| `'torch'` | exact statevector simulation written in torch, all groups and the whole batch in one pass, on the device of the input. Kept as an independent cross-check |

| `q_sim_device` | where the `pennylane` circuit runs |
| --- | --- |
| `'cpu'` (default) | the pooled inputs [B x 8, 4] and the stacked parameters are copied to the CPU, the QNode runs there, the [B, 32] output is copied back to the model device |
| `'model'` | the QNode runs on the device of the input (the GPU in a job) |

### How the single call works

`QuantumBlock` pools the 32 quantum channels to [B, 8, 4] and flattens them to
[B x 8, 4]. The per-group parameters are stacked to [8, L, 4, 3] (rotations)
and [8, L, 4] (input scales), expanded over the batch and flattened to
[B x 8, L, 4, 3] and [B x 8, L, 4]. Stacking keeps the autograd link to the
per-group `nn.Parameter`s, so parameter names, shapes, the state_dict and the
parameter count are unchanged. One QNode then applies the native gate list
(`RY`, `Rot`, `CNOT` for reupload, `RX`, `RY`, `CNOT` for legacy) with every
gate angle broadcast along the B x 8 axis. The QNodes are built once per
(ansatz, layers) in a module level cache, which also makes the block
picklable.

The previous path paid a nearly fixed Python cost per QNode call (tape
construction, transform program, `qml.math` dispatch, one small einsum per
gate) and made 8 calls per block forward for reupload and 64 for legacy at
batch 8. The shared block is called twice per model forward, so a training
step made 16 or 128 calls. The single call removes about 87 percent
(reupload L=2) to 98 percent (legacy) of that work.

### Why the circuit runs on the CPU

The statevector has 16 amplitudes per row. On a GPU every one of the
thousands of tiny gate operations becomes its own CUDA kernel launch, so the
same code was about 3x slower on an A100 than on one CPU core (319 ms against
109 ms for the per-group reupload block). With `q_sim_device='cpu'` the data
moved per block call is a few kB and the GPU synchronizes four times per
training step (two block calls in the forward pass, two in the backward pass)
instead of after every gate. `q_sim_device='model'` exists so the smoke job
can measure the GPU placement directly.

### Why `CUDA_LAUNCH_BLOCKING` was removed

Every job script used to export `CUDA_LAUNCH_BLOCKING=1`, inherited from the
first torch job scripts. It is a debugging switch for synchronous CUDA error
reporting and forces a host sync after every kernel launch. Nothing in the
code needs it, and it does not change results of deterministic kernels
(`seed_everything` keeps `cudnn.deterministic=True`, which is what controls
reproducibility). On the A100 it doubled the full training step of every model
(benchmark on one A100, batch 8, image 128):

| model | `CUDA_LAUNCH_BLOCKING=1` | blocking off |
| --- | --- | --- |
| quantum, `pennylane_pergroup` | 814.6 ms | 514.0 ms |
| quantum, `torch` | 164.7 ms | 78.4 ms |
| classical ablation | 153.6 ms | 76.9 ms |

`misc/generate_grid.py` no longer writes it into the job header.

### Equivalence

`tests/test_quantum_backend.py` and `tests/test_pennylane_fast.py` (both run
with pytest or plain `python`) compare the three simulators on random inputs
and random parameters, including input scales away from 1. The forward output
of `pennylane` is bitwise identical to `pennylane_pergroup`. Parameter
gradients differ only by float summation order (the gradient of a shared
weight is now summed over the batch inside one autograd graph), at most
6.0e-07 in float32 and 1.8e-15 in float64 for the CPU prototype.

Maximum absolute differences, `pennylane_pergroup` against `torch`, 8 groups,
batch 5:

| case | output | grad wrt input | grad wrt rotation angles | grad wrt input scales |
| --- | --- | --- | --- | --- |
| legacy, float64 | 0 | 1.7e-16 | 5.7e-14 | n/a |
| reupload L=2, float64 | 0 | 2.0e-15 | 3.6e-14 | 2.0e-14 |
| reupload L=3, float64 | 0 | 3.3e-15 | 3.9e-14 | 4.6e-14 |
| legacy, float32 | 1.8e-7 | 6.0e-8 | 3.1e-5 | n/a |
| reupload L=2, float32 | 2.4e-7 | 1.4e-6 | 2.3e-5 | 2.7e-5 |
| reupload L=3, float32 | 3.3e-7 | 1.7e-6 | 3.8e-5 | 3.2e-5 |
| full model, reupload L=2, shared | 7.5e-7 (logits) | n/a | 4.6e-11 | all params 6.3e-8 |
| full model, reupload L=2, unshared | 5.4e-7 (logits) | n/a | 1.1e-10 | all params 6.0e-8 |
| full model, legacy | 5.4e-7 (logits) | n/a | 1.2e-11 | all params 4.1e-8 |
| knockout `random`, legacy and reupload | 2.4e-7 | n/a | n/a | n/a |

The float64 runs agree to rounding error. In float32 the block output agrees to
float32 resolution and the float32 gradients differ only by summation order
(relative difference at most 1.3e-6 of the gradient scale, gradients of the
rotation angles reach about 200 in this test). The full model test loads one
state_dict into models with different simulators (`base_channels=8`, input
2x3x16x16) and compares the logits, the loss and every parameter gradient after
one backward pass. A state_dict written by a `torch` model loads into a
`pennylane` model with `strict=True`, so archived checkpoints stay usable.

The legacy circuit also matches the closed form products in the table above
to 3.3e-16.

### Speed

`tools/bench_quantum.py` times the quantum block alone (forward and backward,
8 groups, 8x8 spatial input, batch 4, 8 and 16) for every simulator, and a
full training step (forward, cross entropy, backward, Adam) of the production
configuration (`base_channels=32`, image 128, batch 8) for every simulator
and for the parameter matched classical ablation. It prints the ratio of each
quantum step to the ablation step. Options: `--device`, `--quick` (reupload
L=2 only), `--skip-block`, `--skip-model`, `--threads`, `--json`. On a
checkout without `pennylane_pergroup` or `q_sim_device` it skips those
variants with a message.

CPU, cluster login node, one thread, PennyLane 0.41.1, medians of 15 repetitions
after 3 warmups, block input [B, 32, 8, 8] float32 (login node load about 14
on 32 cores, so a few percent noise):

| quantum block fwd+bwd, B=8 | QNode calls | per-group path | single call | speedup |
| --- | --- | --- | --- | --- |
| reupload L=2 | 8 then 1 | 108.99 ms | 15.61 ms | 7.0x |
| reupload L=3 | 8 then 1 | 154.32 ms | 22.11 ms | 7.0x |
| legacy | 64 then 1 | 364.37 ms | 7.56 ms | 48x |

The cost of one call is nearly flat in the number of broadcast rows (reupload
L=2: 13.6 ms for 8 rows, 15.1 ms for 64, 15.7 ms for 128), and 2 or 4 threads
give the same time as 1. Both QResnetBlock calls of one training step cost
30.98 ms (reupload L=2), 44.16 ms (L=3) and 14.99 ms (legacy) at B=8.

Alternatives measured on the same node (reupload L=2, one call over 64 rows,
forward and backward): `default.qubit` with backprop 15.05 ms,
`lightning.qubit` with adjoint 178.53 ms, `default.qubit` with adjoint
2099.44 ms. Parameter-shift cannot differentiate broadcast trainable
parameters in PennyLane 0.41.1. `lightning.gpu` is not installed. So the
default stays `default.qubit` with backprop.

GPU numbers for the single call are pending the smoke job
(`misc/smoke_pennylane_fast.sh`, see `docs/RUNNING.md`). The estimate for an
A100 without `CUDA_LAUNCH_BLOCKING` is about 110 ms per full step against
76.9 ms for the ablation (about 1.4x), against 514.0 ms for the per-group path
(6.7x). That estimate assumes the CPU of an `ai` node is about as fast as the
login node and is not a measurement.

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
| `q_backend` | `'pennylane'` | `'pennylane'`, `'pennylane_pergroup'`, `'torch'` (simulator only, same math) |
| `q_sim_device` | `'cpu'` | `'cpu'`, `'model'` (where the `pennylane` circuit runs) |
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
`resnet_block_groups=4`, `quantum_channels=32`, any `q_backend`. Counted with
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

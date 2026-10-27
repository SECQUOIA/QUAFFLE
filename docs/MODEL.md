# QUAFFLE model notes

This file describes the hybrid bottleneck in `utilsTorch/unetTorch.py`, the
construction options in `utilsTorch/utilsTorch.py`, and the parameter counts
of every model variant.

## Bottleneck layout

The QVUNet bottleneck (`QVertex`) has two `QResnetBlock`s (`AblationResnetBlock`
is an alias used by the parameter-matched ablation) with attention in
between. At the default settings (`base_channels=32`, `quantum_channels=32`,
`n_qubits=4`), each block splits the 256 bottleneck channels into 32 quantum
channels and 224 classical channels, and the 32 quantum channels form 8 groups
of 4 (in general, `quantum_channels` channels form `G = quantum_channels // n_qubits`
groups of `n_qubits` channels for `n_qubits in {2, 4, 8, 12, 16}`). Under the
default `q_spatial_mode='global'`, each group is spatially average pooled to
`n_qubits` numbers (4 numbers by default), mapped by an `n_qubits`-qubit circuit
to `n_qubits` `<Z>` values, and broadcast back over `(H, W)`. The quantum output `y_q` is
therefore spatially constant within a group. The classical ablation keeps this
exact pool, transform, broadcast pattern and only swaps the per-group transform.

### Skip modulation (`skip_modulation`)

Because `y_q` has shape `(B, qc, 1, 1)` broadcast over `(H, W)`, adding `y_q`
directly (`x_skip[:, :qc] + y_q`) shifts each quantum skip channel by a pure
spatial DC offset that is removed by spatial centering or downstream spatial
`GroupNorm`. `QResnetBlock` (and `AblationResnetBlock`) therefore defaults to
multiplicative/affine channel modulation (`skip_modulation='affine'`):

* `'affine'` (default): `x_skip[:, :qc] * (1.0 + y_q)`, scaling spatial AC
  variation in each quantum skip channel while collapsing cleanly to the identity
  skip (`1.0 + 0 = 1.0`) when `quantum_mode='zero'`.
* `'additive'`: `x_skip[:, :qc] + y_q`, retained for backward-compatible
  evaluation or reproduction of legacy additive checkpoints (`--skip-modulation additive`).

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
classically in a few multiplications. It is kept to reproduce results of
the original model.

### `reupload` (default for new runs)

`q_layers` layers (default 2). Each layer applies:

1. Bounded encoding `RY(pi * tanh(s_lj * x_j))` on wire j, where `s_lj` is a learnable input scale initialized to 1. Angles stay inside (-pi, pi).
2. A general rotation `Rot(w_lj0, w_lj1, w_lj2)` on wire j, initialized uniformly in [-pi, pi].
3. A CNOT ring 0->1->2->3->0.

Measurement is `<Z_j>` on every wire. The inputs are encoded again after each
entangling layer, so the outputs are no longer products of single qubit
functions of the inputs. Parameters per group are `4 * n_qubits * q_layers`
(`16 * q_layers` at default `n_qubits=4`: `3 * n_qubits` rotation angles and
`n_qubits` input scales per layer). The circuit runs on `default.qubit` with
the torch interface, `diff_method="backprop"` and parameter broadcasting over
the batch and the `num_groups = quantum_channels // n_qubits` groups (one QNode
call per block forward pass, see `q_backend` below).

### Configurable per-group qubit width (`--n-qubits {2, 4, 8, 12, 16}`, `#39`)

For `q_ansatz in ('reupload', 'legacy')`, `--n-qubits {2, 4, 8, 12, 16}` (default `4`)
controls how many channels are entangled within each group:
- `num_groups = quantum_channels // n_qubits`
- `effective_qc = num_groups * n_qubits`
- For `q_ansatz='reupload'`, each group has `4 * q_layers * n_qubits` trainable
  parameters, so at fixed divisible `quantum_channels` (e.g., `QC = 32` across
  `n_qubits in {4, 8, 16}`, or `QC = 48` across `{2, 4, 8, 12, 16}`), the total
  unique quantum parameter count `4 * q_layers * effective_qc` (`256` at `QC=32, L=2`)
  is **invariant** to `n_qubits`, isolating within-group entanglement width
  ($2^4 = 16 \to 2^8 = 256 \to 2^{16} = 65{,}536$ Hilbert space per group) from
  parameter count.

### `hqunet` (2D spatial patch lattice, `#21`)

`--q-ansatz hqunet --hqunet-grid {2,4}` replaces global spatial pooling with a
2D spatial patch lattice of $G \times G$ qubits ($n_{\text{qubits}} = 4$ for
`hqunet_grid=2`, $n_{\text{qubits}} = 16$ for `hqunet_grid=4`):

1. A $1\times 1$ pre-projection compresses bottleneck channels into `hqunet_groups` groups, and each spatial feature map is adaptively pooled onto a $G \times G$ spatial grid (`1` qubit per spatial patch).
2. Each layer applies bounded angle encoding $\operatorname{RY}(\pi \tanh(s_{l,q} x_q))$, general single-qubit rotations $\operatorname{Rot}(\phi_{l,q}, \theta_{l,q}, \omega_{l,q})$, and 2D nearest-neighbor horizontal + vertical $\operatorname{CZ}$ entanglement across the $G \times G$ spatial grid.
3. Single-qubit $\langle Z_q \rangle$ expectations are reshaped back to $(G, G)$, projected via a $1\times 1$ post-projection, and bilinearly upsampled to $(H, W)$ to preserve 2D spatial boundary structure.

### Ansatz expressivity & classical simulability progression

| Ansatz / Bottleneck | Qubits per circuit | Entanglement & Non-commutativity | Classical Simulability & Analytical Form | Scientific Role in QUAFFLE |
| --- | --- | --- | --- | --- |
| `legacy` | $n_q \in \{2,4,8,12,16\}$ ($\lfloor c_q/n_q\rfloor$ groups; default $n_q=4$) | Single $\operatorname{RX}(x_j)\operatorname{RY}(\theta_j)$ layer before computational-basis $\operatorname{CNOT}$ ring | **Analytically trivial**: $\operatorname{CNOT}$ acts on a product state right before $Z$-basis measurement, reducing exact $\langle Z_k \rangle$ to closed-form monomials of $c_j = \cos(x_j)\cos(\theta_j)$ ($n_q$ multiplications) | Baseline showing that shallow single-layer computational-basis entanglement collapses to a classical product unit |
| `reupload` ($L \ge 2$) | $n_q \in \{2,4,8,12,16\}$ ($\lfloor c_q/n_q\rfloor$ groups; default $n_q=4$) | Interleaved $\operatorname{RY}(\pi\tanh(s_{l,j}x_j)) + \operatorname{Rot}(\phi,\theta,\omega) + \operatorname{CNOT}$ rings across $L$ layers | **Non-factorizable trigonometric Fourier series**: intermediate $\operatorname{Rot}$ gates between $\operatorname{CNOT}$ rings mix computational-basis amplitudes so outputs cannot be factored into single-qubit cosines; statevector dimension scales as $2^{n_q} \in \{4, 16, 256, 4096, 65536\}$ per group while total quantum parameter count stays invariant at $4 L c_q$ | Controlled test of multi-layer data re-uploading inductive bias and entanglement horizon scaling vs. parameter-matched classical bottleneck |
| `hqunet` ($G \in \{2,4\}$) | $G^2 \in \{4, 16\}$ | 2D nearest-neighbor horizontal + vertical $\operatorname{CZ}$ lattice across $G \times G$ spatial patches | $G=2$ ($2^4=16$ statevector) or $G=4$ ($2^{16}=65{,}536$ statevector) coupling adjacent spatial patches rather than globally pooled channels | Tests spatially-local quantum entanglement for flood boundary preservation |
| `ClassicalAblation` (`ablation_studies.py`) | $0$ (classical) | Parameter-matched per-group `Linear`/`MLP`, `conv1x1`, `depthwise3x3`, `lowrank`, or `hqunet_spatial` | Purely classical bottleneck with **100% identical paired-seed U-Net encoder/decoder initialization** (`#5`) | Isolates whether any segmentation gain comes from the PQC mapping vs. the bottleneck parameter budget / channel split |

## Circuit simulator (`q_backend`)

`QuantumBlock(q_backend=..., q_sim_device=...)`, the config keys `q_backend`
and `q_sim_device`, and the training flags `--q-backend` and `--q-sim-device`
choose how the circuits are evaluated. The math is the same for all three
simulators. `pennylane` is the default. When $N_{\text{tokens}} \cdot G \cdot 2^{n_q}$
exceeds `_MAX_SIM_STATES` ($2^{18}$ complex128 amplitudes), all three backends
automatically chunk across tokens/groups with non-reentrant activation
checkpointing.

| `q_backend` | how |
| --- | --- |
| `'pennylane'` (default) | broadcast PennyLane `default.qubit` QNode evaluation (`torch` interface, `diff_method="backprop"`) over `(tokens x groups)` per chunk, including trainable weights and input scales |
| `'pennylane_pergroup'` | per-group PennyLane QNode evaluation over each $n_q$-qubit group (`legacy` loops over tokens per chunk). Kept as a test reference and for `visualize_circuit.py` |
| `'torch'` | exact batched statevector simulation written in PyTorch over `(tokens x groups)` per chunk on the input device. Kept as an independent cross-check and fast wide-qubit simulator |

| `q_sim_device` | where the `pennylane` circuit runs |
| --- | --- |
| `'cpu'` (default) | the pooled inputs `[N x G, n_q]` and the stacked parameters are copied to the CPU, the QNode runs there, and the `[N, G * n_q]` output is copied back to the model device |
| `'model'` | the QNode runs on the device of the input (the GPU in a job) |

### How the batched call works

At default $c_q = 32, n_q = 4$ ($G = 8$ groups), `QuantumBlock` pools the 32
quantum channels to `[B, 8, 4]` (or `[B, G, n_q]` in general) and flattens them
to `[B x G, n_q]`. The per-group parameters are stacked to `[G, L, n_q, 3]`
(rotations) and `[G, L, n_q]` (input scales), expanded over the token batch and
flattened to `[B x G, L, n_q, 3]` and `[B x G, L, n_q]`. Stacking keeps the
autograd link to the per-group `nn.Parameter`s, so parameter names, shapes, the
`state_dict` and the parameter count are unchanged. The broadcast QNode applies
the native gate list (`RY`, `Rot`, `CNOT` for `reupload`, `RX`, `RY`, `CNOT` for
`legacy`) with every gate angle broadcast along the `B x G` axis. The QNodes
are built once per `(ansatz, layers, n_qubits)` in a module-level cache, which
also makes the block picklable.

The previous path paid a nearly fixed Python cost per QNode call (tape
construction, transform program, `qml.math` dispatch, one small einsum per
gate) and made 8 calls per block forward for reupload and 64 for legacy at
batch 8. The shared block is called twice per model forward, so a training
step made 16 or 128 calls. The single call removes about 87 percent
(reupload L=2) to 98 percent (legacy) of that work.

### Why the circuit runs on the CPU

At the default `n_qubits=4` width, the statevector has 16 complex amplitudes
per row ($2^{\text{n\_qubits}}$ amplitudes in general, up to 65,536 at
`n_qubits=16`). On a GPU every one of the thousands of tiny gate operations
becomes its own CUDA kernel launch, so at small qubit widths (`n_qubits=4`) the
same code was about 3x slower on an A100 than on one CPU core (319 ms against
109 ms for the per-group reupload block). With `q_sim_device='cpu'` the data
moved per block call is a few kB and the GPU synchronizes four times per
training step (two block calls in the forward pass, two in the backward pass)
instead of after every gate. `q_sim_device='model'` exists so the smoke job
and wider-circuit runs can evaluate or run GPU placement directly.

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

GPU numbers for the single call are not included here. The smoke job
(`misc/smoke_pennylane_fast.sh`, see `docs/RUNNING.md`) measures them. The
estimate for an A100 without `CUDA_LAUNCH_BLOCKING` is about 110 ms per full
step against 76.9 ms for the ablation (about 1.4x), against 514.0 ms for the
per-group path (6.7x). That estimate assumes the CPU of a GPU node is about as
fast as the login node and is not a measurement.

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

## Multimodal input stems & non-IID federated partitioning (`#14`)

In `flood_combined`, 12-band optical GeoTIFFs (default 3-band selection `[GREEN, NIR, SWIR1]`)
and 6-band SAR GeoTIFFs (default 3-band selection `[VV, VH, COP_DEM]`) have distinct physical
units and noise characteristics. `--combined-input-mode` controls how `flood_data.py`
and `DownUnet` (`QVUNet` / `UNet`) represent and process these modalities:

| `combined_input_mode` | Loader output | Input stem (`DownUnet`) |
| --- | --- | --- |
| `'shared3'` (default) | `(3, H, W)` per sample | Single shared 3-channel `init_conv = Conv2d(3, init_dim, 7, padding=3)` |
| `'concat6'` | `(6, H, W)` (`[Optical(3), SAR(3)]` with inactive half zero-padded) | 6-channel `init_conv = Conv2d(6, init_dim, 7, padding=3)` |
| `'dual_stem'` | `(4, H, W)` (`[C0, C1, C2, modality_id]`) | Separate 3-channel optical `init_conv` and SAR `init_conv_sar` stems routed per sample before the shared encoder (`init_conv_sar` is initialized under an isolated `fork_rng` so downstream paired-seed backbone weights remain 100% identical across Quantum, Ablation, and Classical models) |

Federated partitioning (`--partition`) supports four realistic data-silo regimes:

* `'random'` (default): IID uniform random shards.
* `'dirichlet'`: Label-skew Dirichlet ($\operatorname{Dir}(\alpha)$) over binarized flood dominance with `min_client_samples` guard.
* `'modality'`: Cross-sensor non-IID silos grouping optical tiles and SAR tiles onto dedicated pure sensor-specific clients (constrained by each modality's capacity under `min_client_samples`).
* `'geo_cluster'`: Geolocation / topographic feature-cluster non-IID silos using deterministic seeded $k$-means++ clustering on canonical 7-D per-tile feature vectors (`3` channel means + `3` channel standard deviations + `1` flood coverage fraction, invariant across `combined_input_mode`) with `min_client_samples` guard.

## Federated heterogeneity & circular PQC angle aggregation (`#15`)

* **FedProx (`--fl-strategy fedprox --fedprox-mu <float>`)**: Adds the proximal penalty
  $\frac{\mu}{2}\sum_p \|w_p - w_p^{(t)}\|_2^2$ relative to the round's initial global
  parameters $w^{(t)}$ during local client updates (`QuantumFlowerClient`,
  `ClassicalFlowerClient`, `_TorchClassicalFlowerClient`) to bound client drift on
  strongly non-IID splits.
* **Circular PQC angle aggregation (`--q-angle-agg {linear,circular}`)**: Because PQC
  rotation gates ($\operatorname{RX}, \operatorname{RY}, \operatorname{RZ}, \operatorname{Rot}$)
  are $2\pi$-periodic on $S^1 = [-\pi, \pi]$, naive Euclidean averaging across the branch
  cut ($\theta_1 = \pi - \epsilon$, $\theta_2 = -\pi + \epsilon$) averages to $0$ (the
  antipodal gate) instead of $\pm\pi$. Setting `--q-angle-agg circular` aggregates
  periodic rotation tensors (`quantum_params`, `hqunet_weights`) in `QuantumFedAvg.aggregate_fit`
  via the weighted circular mean
  $$\bar{\theta} = \operatorname{atan2}\!\left(\sum_k w_k \sin\theta_k,\; \sum_k w_k \cos\theta_k\right) \in [-\pi, \pi],$$
  while keeping unbounded Euclidean parameters (`input_scales`, `pre_q_conv`,
  `post_q_proj`, and classical backbone layers) on standard weighted averaging.
* **Partial client participation (`--num-clients K --clients-per-round C` or `--fraction-fit f`)**:
  Supports $K > 4$ client silos with $C < K$ sampled per round.

## How training scripts pass the options

Config keys read by `QVUNetSegmentation` and `ClassicalAblationUNetSegmentation`
(`UNetSegmentation` also reads `combined_input_mode`, `backbone_conv_mode`, and `use_time_emb`):

| key | default | values |
| --- | --- | --- |
| `q_backend` | `'pennylane'` | `'pennylane'`, `'pennylane_pergroup'`, `'torch'` (simulator only, same math) |
| `q_sim_device` | `'cpu'` | `'cpu'`, `'model'` (where the `pennylane` circuit runs) |
| `q_ansatz` | `'reupload'` | `'legacy'`, `'reupload'`, `'hqunet'` |
| `q_layers` | `2` | int >= 1 (`reupload` and `hqunet`) |
| `hqunet_grid` | `2` | `2`, `4` (`hqunet` 2D spatial patch grid size) |
| `q_spatial_mode` | `'global'` | `'global'`, `'pixel'`, `'patch2x2'`, `'patch4x4'`, `'se_gate'` |
| `ablation_mode` | `'auto'` | `'auto'`, `'linear'`, `'mlp'`, `'conv1x1'`, `'depthwise3x3'`, `'lowrank'`, `'hqunet_spatial'` |
| `ablation_mlp_mode` | `'param_matched'` | `'param_matched'` ($h = \operatorname{round}((4 L n_q - n_q)/(2 n_q + 1))$), `'full_rank'` ($h = n_q$, removes the hidden-width rank ceiling to permit full $n_q \times n_q$ input–output Jacobian rank; `#44`) |
| `ablation_mlp_hidden` | `None` | `None` or int >= 1 (explicit MLP hidden dimension override; `#44`) |
| `q_share_blocks` | `True` | bool |
| `skip_modulation` | `'affine'` | `'affine'`, `'additive'` (`QResnetBlock` / `AblationResnetBlock` skip modulation) |
| `combined_input_mode` | `'shared3'` | `'shared3'`, `'concat6'`, `'dual_stem'` (`#14`) |
| `fl_strategy` | `'fedavg'` | `'fedavg'`, `'fedprox'` (`#15`, federated only) |
| `fedprox_mu` | `0.0` | float >= 0.0 (`#15`, defaults to `0.01` when `fl_strategy='fedprox'`) |
| `q_angle_agg` | `'linear'` | `'linear'`, `'circular'` (`#15`, federated quantum only) |
| `select_metric` | `'iou'` | `'iou'`, `'dice'`, `'auc'`, `'loss'` (validation metric for best-checkpoint restoration) |

The training scripts set these keys from `--q-ansatz`, `--q-layers`,
`--q-backend`, `--q-sim-device`, `--skip-modulation`, and `--select-metric`
(`tools/training_common.py`, `apply_common_args`) and build the model with
`build_model`. The model reads the `q_*` and `skip_modulation` keys from the
config and `build_model` passes `seed`:

```python
config = tc.apply_common_args(config, args)
model = tc.build_model(QVUNetSegmentation, config)   # QVUNetSegmentation(config, seed=config['seed'])
```

`--quantum-knockout-eval` is an on/off flag. When it is set, the test set is
evaluated once more in mode `zero` and once in mode `random`
(`tc.knockout_eval`), the model is switched back to `on` and both results are
written to `summary.txt` and `metrics.json`.

To reproduce the original architecture pass `q_ansatz='legacy'`. Note that
the seeding and weight standardization fixes still apply, so legacy runs are
not bit identical to runs of the original code. Code that does not pass `seed` keeps
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 
from typing import Optional, Tuple

# Optional: import ORCA PTLayer if available
try:
    from ptseries.models.pt_layer import PTLayer
    from ptseries.tbi import create_tbi
    HAS_PT = True
except ImportError:
    HAS_PT = False

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

# --- Utility Functions ---
def l2norm(t, axis=1, eps=1e-12):
    denom = torch.clamp(torch.norm(t, p=2, dim=axis, keepdim=True), min=eps)
    return t / denom

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
    
    def forward(self, time):
        assert len(time.shape) == 1
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=time.dtype, device=time.device) * -emb)
        emb = time.to(time.dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# --- Core Blocks ---
class WeightStandardizedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        # Standardize into a local tensor. The stored parameter is never
        # overwritten, so eval passes and FedAvg payloads see the raw weights
        # and gradients flow through the standardization.
        kernel = self.conv.weight
        mean = kernel.mean(dim=(1,2,3), keepdim=True)
        var = kernel.var(dim=(1,2,3), keepdim=True)
        kernel = (kernel - mean) / torch.sqrt(var + 1e-5)
        return F.conv2d(x, kernel, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)

class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim, groups=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv0 = WeightStandardizedConv(in_dim, out_dim)
        self.norm0 = nn.GroupNorm(groups, out_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_dim * 2),
            nn.SiLU(),
        )
        self.conv1 = WeightStandardizedConv(out_dim, out_dim)
        self.norm1 = nn.GroupNorm(groups, out_dim)
        self.skip = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, time_emb):
        B, C, H, W = x.shape
        
        h = self.conv0(x)
        h = self.norm0(h)
        
        # Add timestep embedding
        time_emb_processed = self.time_mlp(time_emb)  # [B, 2*out_dim]
        time_emb_processed = time_emb_processed.view(B, -1, 1, 1)  # [B, 2*out_dim, 1, 1]
        scale, shift = time_emb_processed.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.silu(h)
        
        x_skip = self.skip(x)
        return x_skip + h

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

# --- Attention Blocks ---
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = scale
        self.inner_dim = dim_head * heads
        
        self.to_qkv = nn.Conv2d(dim, self.inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.inner_dim, dim, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        qkv = self.to_qkv(x)  # [B, inner_dim*3, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # Each [B, inner_dim, H, W]
        
        # Reshape to [B, heads, H*W, dim_head]
        q = q.view(B, self.heads, self.dim_head, H*W).transpose(2, 3)
        k = k.view(B, self.heads, self.dim_head, H*W).transpose(2, 3)
        v = v.view(B, self.heads, self.dim_head, H*W).transpose(2, 3)
        
        # Normalize
        q = l2norm(q, axis=-1)
        k = l2norm(k, axis=-1)
        
        # Attention
        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(sim, dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(2, 3).contiguous().view(B, self.inner_dim, H, W)
        
        out = self.to_out(out)
        return out

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        
        self.to_qkv = nn.Conv2d(dim, self.inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.inner_dim, dim, 1)
        self.out_norm = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        qkv = self.to_qkv(x)  # [B, inner_dim*3, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # Each [B, inner_dim, H, W]
        
        # Reshape to [B, heads, H*W, dim_head]
        q = q.view(B, self.heads, self.dim_head, H*W).transpose(2, 3)
        k = k.view(B, self.heads, self.dim_head, H*W).transpose(2, 3)
        v = v.view(B, self.heads, self.dim_head, H*W).transpose(2, 3)
        
        # Softmax normalization
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)
        
        q = q / math.sqrt(self.dim_head)
        v = v / (H * W)
        
        # Linear attention
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhde,bhnd->bhne', context, q)
        out = out.transpose(2, 3).contiguous().view(B, self.inner_dim, H, W)
        
        out = self.to_out(out)
        out = self.out_norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out

class AttnBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, use_linear_attention=True):
        super().__init__()
        self.use_linear_attention = use_linear_attention
        self.norm = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        
        if use_linear_attention:
            self.attn = LinearAttention(dim, heads, dim_head)
        else:
            self.attn = Attention(dim, heads, dim_head)
        
    def forward(self, x):
        normed_x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.attn(normed_x)
        return out + x

# --- Quantum Components ---
N_QUBITS = 4
Q_ANSATZ_CHOICES = ('legacy', 'reupload')
QUANTUM_MODES = ('on', 'zero', 'random')
# Added to knockout_seed so the 'random' knockout never reuses the exact stream
# that initialized the trained parameters (e.g. run seed == knockout seed).
KNOCKOUT_SEED_OFFSET = 1_000_003


def create_quantum_block(backend='ptlayer', quantum_channels=8, ptlayer_config=None,
                         seed=None, q_ansatz='reupload', q_layers=2, q_backend='torch'):
    """
    Create a QuantumBlock that processes channels in groups of 4.

    Args:
        backend: 'ptlayer' or 'pennylane'
        quantum_channels: Number of quantum channels (will be rounded down to multiple of 4)
        ptlayer_config: Configuration dict for PTLayer (if backend='ptlayer')
        seed: Optional int. If given, quantum parameters are drawn from a local
            torch.Generator seeded with it. If None, the global torch RNG is used.
        q_ansatz: 'legacy' (original circuit) or 'reupload'
        q_layers: Number of re-uploading layers (only used by 'reupload')
        q_backend: circuit simulator for backend='pennylane'. 'torch' (exact
            batched statevector simulation in torch) or 'pennylane' (QNodes on
            default.qubit). Both give the same outputs and gradients.

    Returns:
        QuantumBlock instance
    """
    return QuantumBlock(backend=backend, quantum_channels=quantum_channels,
                        ptlayer_config=ptlayer_config, seed=seed,
                        q_ansatz=q_ansatz, q_layers=q_layers, q_backend=q_backend)


def pennylane_default_circuit(n_qubits=4):
    """Legacy PennyLane circuit used in the original model (q_ansatz='legacy').

    RX(x_j) encoding, one RY(theta_j) per qubit, a CNOT ring, then <Z_j>.
    The state before the CNOT ring is a product state, so every output is a
    closed-form product of cos(x_j)cos(theta_j) terms (see docs/MODEL.md).
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def circuit(x, params=None):
        # x: [B, 4, H, W] -> process spatial average for each channel
        if len(x.shape) == 4:  # [B, 4, H, W]
            x_processed = x.mean(dim=(2, 3))  # [B, 4]
        elif len(x.shape) == 3:  # [B, 4, 1] or similar
            x_processed = x.squeeze(-1)  # [B, 4]
        else:
            x_processed = x
        
        # Process first batch item
        x_item = x_processed[0]  # [4]
        
        if params is None:
            params = torch.zeros(n_qubits, dtype=x_item.dtype, device=x_item.device)
            
        # Data encoding - each channel to one qubit
        for i in range(n_qubits):
            qml.RX(x_item[i], wires=i)
        
        # Parameterized gates
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
            
        # Entangling gates
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])  # Ring connectivity
            
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit


def pennylane_reupload_circuit(n_qubits=4, n_layers=2):
    """Data re-uploading circuit (q_ansatz='reupload').

    For each layer l = 0..n_layers-1:
      1. Bounded encoding  RY(pi * tanh(s[l, j] * x_j)) on every wire j,
         with a learnable input scale s[l, j] per layer and qubit.
      2. General rotation  Rot(w[l, j, 0], w[l, j, 1], w[l, j, 2]) on every wire.
      3. Entangling CNOT ring 0->1->2->3->0.
    Measurement: <Z_j> on every wire.

    Inputs are re-encoded after each entangling layer, so the outputs are not
    a product of single-qubit functions of the inputs. x has shape [B, n_qubits]
    and is handled with PennyLane parameter broadcasting (one QNode call per
    group for the whole batch). Uses backprop so gradients flow through torch.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights, scales):
        # x: [B, n_qubits], weights: [L, n_qubits, 3], scales: [L, n_qubits]
        for l in range(n_layers):
            enc = math.pi * torch.tanh(x * scales[l])  # [B, n_qubits], in (-pi, pi)
            for j in range(n_qubits):
                qml.RY(enc[:, j], wires=j)
            for j in range(n_qubits):
                qml.Rot(weights[l, j, 0], weights[l, j, 1], weights[l, j, 2], wires=j)
            for j in range(n_qubits):
                qml.CNOT(wires=[j, (j + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    return circuit


# --- Exact batched statevector simulator (q_backend='torch') ---
#
# Same math as PennyLane default.qubit for the two 4 qubit circuits above,
# written as plain torch ops so that the batch AND all groups are simulated in
# one pass on the model's device. Wire 0 is the most significant bit of the
# basis index, as in PennyLane. The simulation runs in complex128 because
# default.qubit returns float64 expectation values for these circuits. The
# result is cast to float32 by QuantumBlock exactly like the PennyLane path.
Q_BACKEND_CHOICES = ('torch', 'pennylane')
_SIM_CACHE = {}


def _cnot_ring_tables(n_qubits, device):
    """Permutation and <Z> sign tables for a 4 qubit CNOT ring, cached per device.

    perm: index tensor with psi_after = psi_before[..., perm] for the gate
        sequence CNOT(0,1), CNOT(1,2), ..., CNOT(n-1, 0).
    zsign: [2**n, n] float64 matrix with zsign[b, j] = +1 if bit j of b is 0
        else -1, so <Z_j> = sum_b |psi_b|^2 zsign[b, j].
    """
    key = (n_qubits, str(device))
    if key not in _SIM_CACHE:
        dim = 2 ** n_qubits
        perm = [0] * dim
        for b in range(dim):
            bits = [(b >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
            for c in range(n_qubits):
                t = (c + 1) % n_qubits
                if bits[c]:
                    bits[t] ^= 1
            out = 0
            for k in range(n_qubits):
                out = (out << 1) | bits[k]
            perm[out] = b
        zsign = torch.tensor([[1.0 - 2.0 * ((b >> (n_qubits - 1 - j)) & 1) for j in range(n_qubits)]
                              for b in range(dim)], dtype=torch.float64)
        _SIM_CACHE[key] = (torch.tensor(perm, dtype=torch.long, device=device), zsign.to(device))
    return _SIM_CACHE[key]


def _ry_mat(theta):
    """RY(theta) as a complex [..., 2, 2] tensor."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    m = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    return torch.complex(m, torch.zeros_like(m))


def _rx_mat(theta):
    """RX(theta) as a complex [..., 2, 2] tensor."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)
    cc = torch.complex(c, z)
    ms = torch.complex(z, -s)
    return torch.stack([torch.stack([cc, ms], -1), torch.stack([ms, cc], -1)], -2)


def _rot_mat(phi, theta, omega):
    """Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi) as [..., 2, 2] complex."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    c = torch.complex(c, torch.zeros_like(c))
    s = torch.complex(s, torch.zeros_like(s))
    p = (phi + omega).to(torch.float64) / 2
    m = (phi - omega).to(torch.float64) / 2
    cp, sp, cm, sm = torch.cos(p), torch.sin(p), torch.cos(m), torch.sin(m)
    e_mp = torch.complex(cp, -sp)   # exp(-i (phi+omega)/2)
    e_pp = torch.complex(cp, sp)    # exp(+i (phi+omega)/2)
    e_pm = torch.complex(cm, sm)    # exp(+i (phi-omega)/2)
    e_mm = torch.complex(cm, -sm)   # exp(-i (phi-omega)/2)
    row0 = torch.stack([e_mp * c, -e_pm * s], -1)
    row1 = torch.stack([e_mm * s, e_pp * c], -1)
    return torch.stack([row0, row1], -2)


def _simulate_layers(layer_gates, n_qubits=N_QUBITS):
    """Run the circuit |0..0> -> [single qubit gates on every wire, CNOT ring] * L
    and return <Z_j> for every wire.

    layer_gates: list of L complex tensors shaped [..., n_qubits, 2, 2]; entry
        [..., j] is the product of all single qubit gates applied to wire j in
        that layer (in circuit order, later gates on the left).
    Returns a float64 tensor [..., n_qubits].
    """
    assert n_qubits == 4, "the batched simulator is written for 4 qubits"
    u0 = layer_gates[0]
    perm, zsign = _cnot_ring_tables(n_qubits, u0.device)
    batch = u0.shape[:-3]
    dim = 2 ** n_qubits
    psi = None
    for u in layer_gates:
        a, b, c, d = u[..., 0, :, :], u[..., 1, :, :], u[..., 2, :, :], u[..., 3, :, :]
        if psi is None:
            # Input is |0000>, so the first layer gives the product state of the
            # first columns of the four single qubit gates.
            psi = torch.einsum('...a,...c,...e,...g->...aceg',
                               a[..., :, 0], b[..., :, 0], c[..., :, 0], d[..., :, 0])
        else:
            psi = psi.reshape(*batch, 2, 2, 2, 2)
            psi = torch.einsum('...ab,...bdfh->...adfh', a, psi)
            psi = torch.einsum('...cd,...adfh->...acfh', b, psi)
            psi = torch.einsum('...ef,...acfh->...aceh', c, psi)
            psi = torch.einsum('...gh,...aceh->...aceg', d, psi)
        psi = psi.reshape(*batch, dim)[..., perm]
    probs = psi.real ** 2 + psi.imag ** 2
    return probs @ zsign


def torch_reupload_expvals(x, weights, scales):
    """Exact batched equivalent of pennylane_reupload_circuit.

    x: [B, G, 4] pooled inputs, weights: [G, L, 4, 3], scales: [G, L, 4].
    Returns float64 <Z_j> shaped [B, G, 4].
    """
    L = weights.shape[1]
    gates = []
    for l in range(L):
        enc = math.pi * torch.tanh(x * scales[:, l].unsqueeze(0))        # [B, G, 4]
        ry = _ry_mat(enc.to(torch.float64))                               # [B, G, 4, 2, 2]
        w = weights[:, l].to(torch.float64)                               # [G, 4, 3]
        rot = _rot_mat(w[..., 0], w[..., 1], w[..., 2]).unsqueeze(0)      # [1, G, 4, 2, 2]
        gates.append(rot @ ry)
    return _simulate_layers(gates)


def torch_legacy_expvals(x, theta):
    """Exact batched equivalent of pennylane_default_circuit (RX(x) RY(theta), ring).

    x: [B, G, 4] pooled inputs, theta: [G, 4]. Returns float64 [B, G, 4].
    """
    rx = _rx_mat(x.to(torch.float64))                                     # [B, G, 4, 2, 2]
    ry = _ry_mat(theta.to(torch.float64)).unsqueeze(0)                    # [1, G, 4, 2, 2]
    return _simulate_layers([ry @ rx])


def _randn(shape, generator=None):
    return torch.randn(*shape, generator=generator)


def _uniform(shape, low, high, generator=None):
    return low + (high - low) * torch.rand(*shape, generator=generator)


class QuantumBlock(nn.Module):
    """
    Quantum processing block that handles channels in groups of 4.
    
    This implementation follows the JAX approach where channels are partitioned
    into groups of 4 and each group is processed by a separate quantum circuit.
    No padding is used - only complete groups of 4 are processed quantum mechanically.
    Each group is spatially average pooled to 4 numbers, mapped to 4 <Z> values,
    and broadcast back over (H, W).
    
    Args:
        backend: 'ptlayer' or 'pennylane'
        quantum_channels: Total number of quantum channels (will process quantum_channels//4 groups)
        ptlayer_config: Configuration for PTLayer backend
        seed: Optional int. If given, quantum parameters come from a local
            torch.Generator (PTLayer modules are built under a forked RNG). The
            global torch RNG is never reseeded. If None, the global RNG is used
            as seeded by the caller.
        q_ansatz: 'legacy' reproduces the original circuit (4 params per group).
            'reupload' is the data re-uploading circuit (16 * q_layers params
            per group). PennyLane backend only.
        q_layers: Number of re-uploading layers for q_ansatz='reupload'.
        q_backend: how the backend='pennylane' circuits are simulated.
            'torch' (default) runs an exact statevector simulation of all groups
            and the whole batch in one pass of torch ops on the input's device.
            'pennylane' evaluates one default.qubit QNode per group (the
            original path, kept for verification). Parameters, state_dict and
            outputs are the same for both. Switch at runtime with set_q_backend.

    Knockout: set_quantum_mode('on' | 'zero' | 'random') switches the forward
    pass between the trained circuit, an all-zero output, or the same circuit
    with freshly drawn random parameters (fixed knockout seed). The trained
    parameters and state_dict are not modified.
    """
    def __init__(self, backend='ptlayer', quantum_channels=8, ptlayer_config=None,
                 seed=None, q_ansatz='reupload', q_layers=2, q_backend='torch'):
        super().__init__()

        if q_ansatz not in Q_ANSATZ_CHOICES:
            raise ValueError(f"Unknown q_ansatz {q_ansatz!r}, expected one of {Q_ANSATZ_CHOICES}")
        if q_ansatz == 'reupload' and int(q_layers) < 1:
            raise ValueError("q_layers must be >= 1")
        if q_backend not in Q_BACKEND_CHOICES:
            raise ValueError(f"Unknown q_backend {q_backend!r}, expected one of {Q_BACKEND_CHOICES}")

        self.backend = backend
        self.quantum_channels = quantum_channels
        self.q_ansatz = q_ansatz
        self.q_layers = int(q_layers) if q_ansatz == 'reupload' else 1
        self.seed = seed
        self.quantum_mode = 'on'
        self._random_params = None
        self._random_stacked = None
        self.q_backend = q_backend
        
        # Calculate number of 4-channel groups
        self.num_groups = quantum_channels // 4
        self.remaining_channels = quantum_channels % 4

        # Local generator: never touches the global RNG.
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
        
        if backend == 'ptlayer':
            if not HAS_PT:
                raise ImportError("PTLayer not available. Please install ORCA PTLayer or use 'pennylane' backend.")
            
            # Set default PTLayer configuration
            if ptlayer_config is None:
                ptlayer_config = {
                    'n_loops': 2,
                    'n_samples': 200,
                    'observable': 'mean'
                }
            
            try:
                # PTLayer and nn.Linear draw from the global RNG internally. When a
                # seed is given, build them under a forked RNG so the caller's
                # global state is restored afterwards.
                with torch.random.fork_rng(devices=[], enabled=seed is not None):
                    if seed is not None:
                        torch.manual_seed(int(seed))
                    # Create TBI object
                    self.tbi = create_tbi(n_loops=ptlayer_config.get('n_loops', 2))
                    
                    # Define input state for 4 modes
                    self.input_state = (1, 0, 1, 0) 
                    self.m_modes = 4
                    
                    # Calculate required number of parameters
                    self.n_params = self.tbi.calculate_n_beam_splitters(self.m_modes)
                    
                    # Create PTLayer instances for each group
                    self.ptlayers = nn.ModuleList()
                    self.param_projections = nn.ModuleList()
                        
                    for group_idx in range(self.num_groups):
                        ptlayer = PTLayer(
                                input_state=self.input_state,
                                in_features=self.n_params,
                                tbi=self.tbi,
                                observable=ptlayer_config.get('observable', 'mean'),
                                n_samples=ptlayer_config.get('n_samples', 200)
                        )
                        self.ptlayers.append(ptlayer)
                        # Map 4 channels to PTLayer parameters
                        self.param_projections.append(nn.Linear(4, self.n_params))
                
                print(f"PTLayer initialized: {self.num_groups} groups, {self.m_modes} modes each, {self.n_params} parameters per group")
                
            except Exception as e:
                print(f"Warning: Could not initialize PTLayer: {e}")
                self.ptlayers = None
                self.param_projections = None
                self.tbi = None
                
        elif backend == 'pennylane':
            if not HAS_PENNYLANE and q_backend == 'pennylane':
                raise ImportError("PennyLane not available. Install PennyLane or use q_backend='torch'.")
            
            # QNodes for each 4-channel group (built lazily, only used by
            # q_backend='pennylane')
            self.pennylane_circuits = []
            self.quantum_params = nn.ParameterList()
            # Learnable per-layer input scales (reupload only)
            self.input_scales = nn.ParameterList()
            
            for group_idx in range(self.num_groups):
                if q_ansatz == 'legacy':
                    # Same distribution as the original model (N(0, 0.1^2)), but
                    # drawn from the seeded local generator or the global RNG.
                    params = nn.Parameter(_randn((N_QUBITS,), generator) * 0.1)
                    self.quantum_params.append(params)
                else:
                    weights = nn.Parameter(_uniform((self.q_layers, N_QUBITS, 3), -math.pi, math.pi, generator))
                    self.quantum_params.append(weights)
                    self.input_scales.append(nn.Parameter(torch.ones(self.q_layers, N_QUBITS)))
            if q_backend == 'pennylane':
                self._build_pennylane_circuits()
                
        else:
            raise ValueError(f'Unknown quantum backend: {backend}')

    @property
    def n_params_per_group(self):
        if self.backend != 'pennylane':
            return None
        if self.q_ansatz == 'legacy':
            return N_QUBITS
        return self.q_layers * N_QUBITS * 4  # 3 rotation angles + 1 input scale per qubit per layer

    def _build_pennylane_circuits(self):
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane not available, q_backend='pennylane' needs it.")
        if self.pennylane_circuits:
            return
        for _ in range(self.num_groups):
            if self.q_ansatz == 'legacy':
                self.pennylane_circuits.append(pennylane_default_circuit(n_qubits=N_QUBITS))
            else:
                self.pennylane_circuits.append(
                    pennylane_reupload_circuit(n_qubits=N_QUBITS, n_layers=self.q_layers))

    def set_q_backend(self, q_backend):
        """Switch the circuit simulator ('torch' or 'pennylane') in place.
        Parameters are untouched."""
        if q_backend not in Q_BACKEND_CHOICES:
            raise ValueError(f"Unknown q_backend {q_backend!r}, expected one of {Q_BACKEND_CHOICES}")
        if self.backend == 'pennylane' and q_backend == 'pennylane':
            self._build_pennylane_circuits()
        self.q_backend = q_backend
        return self

    # --- Knockout ---
    def set_quantum_mode(self, mode='on', knockout_seed=0):
        """Select 'on' (trained circuit), 'zero' (output replaced by zeros), or
        'random' (circuit evaluated with random parameters drawn from a local
        generator seeded by knockout_seed + KNOCKOUT_SEED_OFFSET, from the same distribution used at
        initialization). Does not modify trained parameters or the global RNG."""
        if mode not in QUANTUM_MODES:
            raise ValueError(f"Unknown quantum mode {mode!r}, expected one of {QUANTUM_MODES}")
        self.quantum_mode = mode
        self._random_params = None
        self._random_stacked = None
        if mode == 'random' and self.backend == 'pennylane':
            g = torch.Generator()
            g.manual_seed(int(knockout_seed) + KNOCKOUT_SEED_OFFSET)
            rand = []
            for i in range(self.num_groups):
                if self.q_ansatz == 'legacy':
                    rand.append((_randn((N_QUBITS,), g) * 0.1, None))
                else:
                    rand.append((_uniform((self.q_layers, N_QUBITS, 3), -math.pi, math.pi, g),
                                 torch.ones(self.q_layers, N_QUBITS)))
            self._random_params = rand
        elif mode == 'random':
            raise NotImplementedError("'random' knockout is implemented for the pennylane backend only")
        return self

    def _group_params(self, i, device):
        if self.quantum_mode == 'random':
            w, s = self._random_params[i]
            w = w.to(device)
            s = s.to(device) if s is not None else None
            return w, s
        w = self.quantum_params[i]
        s = self.input_scales[i] if self.q_ansatz == 'reupload' else None
        return w, s

    def _stacked_params(self, device):
        """All group parameters stacked along a leading group axis:
        weights [G, L, 4, 3] (legacy: [G, 4]) and scales [G, L, 4] (legacy: None).
        Stacking keeps the autograd link to the per-group nn.Parameters."""
        if self.quantum_mode == 'random':
            if self._random_stacked is None or self._random_stacked[0].device != torch.device(device):
                w = torch.stack([p[0] for p in self._random_params]).to(device)
                s = None
                if self.q_ansatz == 'reupload':
                    s = torch.stack([p[1] for p in self._random_params]).to(device)
                self._random_stacked = (w, s)
            return self._random_stacked
        w = torch.stack(list(self.quantum_params))
        s = torch.stack(list(self.input_scales)) if self.q_ansatz == 'reupload' else None
        return w, s

    def _forward_torch_sim(self, x):
        """q_backend='torch': every group and batch item in one batched simulation."""
        B, _, H, W = x.shape
        G = self.num_groups
        if G == 0:
            return torch.zeros(B, 0, H, W, dtype=x.dtype, device=x.device)
        # Same per-channel spatial mean as the PennyLane path
        pooled = x[:, :G * N_QUBITS].mean(dim=(2, 3)).reshape(B, G, N_QUBITS)
        w, s = self._stacked_params(x.device)
        if self.q_ansatz == 'legacy':
            y = torch_legacy_expvals(pooled, w)
        else:
            y = torch_reupload_expvals(pooled, w, s)
        y = y.reshape(B, G * N_QUBITS).to(torch.float32)
        return y.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
    
    def forward(self, x, params=None):
        B, C, H, W = x.shape

        if self.quantum_mode == 'zero':
            return torch.zeros(B, self.num_groups * 4, H, W, dtype=torch.float32, device=x.device)
        
        if self.backend == 'ptlayer':
            if self.ptlayers is None:
                raise ValueError('PTLayer instances not initialized for PTLayer backend')
            
            outputs = []
            
            # Process each 4-channel group
            for i in range(self.num_groups):
                # Extract 4 channels for this group
                start_idx = i * 4
                end_idx = start_idx + 4
                x_group = x[:, start_idx:end_idx]  # [B, 4, H, W]
                
                # Pool spatial dimensions to get per-channel features
                x_pooled = F.adaptive_avg_pool2d(x_group, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 4]
            
            # Project to PTLayer parameter space
                theta = self.param_projections[i](x_pooled)  # [B, n_params]
                
                # Apply PTLayer
                y_group = self.ptlayers[i](theta)  # [B, 4]
                
                # Reshape output to spatial dimensions
                y_group = y_group.unsqueeze(-1).unsqueeze(-1)  # [B, 4, 1, 1]
                y_group = y_group.expand(-1, -1, H, W)  # [B, 4, H, W]
                
                outputs.append(y_group)
            
            # Concatenate all group outputs
            if outputs:
                y = torch.cat(outputs, dim=1)  # [B, num_groups*4, H, W]
            else:
                y = torch.zeros(B, 0, H, W, dtype=x.dtype, device=x.device)
            
            return y.float()
            
        elif self.backend == 'pennylane' and self.q_backend == 'torch':
            return self._forward_torch_sim(x)

        elif self.backend == 'pennylane':
            if not self.pennylane_circuits:
                raise ValueError('PennyLane circuits not initialized for PennyLane backend')
            
            outputs = []
            
            # Process each 4-channel group
            for i in range(self.num_groups):
                # Extract 4 channels for this group
                start_idx = i * 4
                end_idx = start_idx + 4
                x_group = x[:, start_idx:end_idx]  # [B, 4, H, W]
                w, s = self._group_params(i, x.device)

                if self.q_ansatz == 'legacy':
                    # Process each batch item for this group (original behaviour)
                    group_outputs = []
                    for b in range(B):
                        x_item = x_group[b:b+1]  # [1, 4, H, W]
                        
                        y_item = self.pennylane_circuits[i](x_item, w)
                        y_tensor = torch.stack(y_item).unsqueeze(0)  # [1, 4]
                        group_outputs.append(y_tensor)
                    
                    # Stack batch outputs for this group
                    y_group = torch.cat(group_outputs, dim=0)  # [B, 4]
                else:
                    x_pooled = x_group.mean(dim=(2, 3))  # [B, 4]
                    y_list = self.pennylane_circuits[i](x_pooled, w, s)
                    y_group = torch.stack([torch.as_tensor(v).reshape(-1) for v in y_list], dim=1)
                    y_group = y_group.reshape(B, N_QUBITS)  # [B, 4]
                
                # Reshape to spatial dimensions
                y_group = y_group.to(torch.float32).unsqueeze(-1).unsqueeze(-1)  # [B, 4, 1, 1]
                y_group = y_group.expand(-1, -1, H, W)  # [B, 4, H, W]
                
                outputs.append(y_group)
            
            # Concatenate all group outputs
            if outputs:
                y = torch.cat(outputs, dim=1)  # [B, num_groups*4, H, W]
            else:
                y = torch.zeros(B, 0, H, W, dtype=x.dtype, device=x.device)
            
            return y.float()
        else:
            raise ValueError(f'Unknown quantum backend: {self.backend}')

class QResnetBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_emb_dim: int,
        quantum_block: QuantumBlock,
        quantum_channels: int = 8,
        groups: int = 8,
    ):
        super().__init__()
        self.quantum_channels = quantum_channels
        self.quantum_block = quantum_block
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Calculate effective quantum channels (must be multiple of 4)
        self.effective_qc = (quantum_channels // 4) * 4
        self.classical_channels = out_dim - self.effective_qc

        # Classical path for remaining channels
        if self.classical_channels > 0:
            # The classical path should handle the remaining channels after quantum processing
            # We'll use the actual number of classical channels as input dimension
            classical_in_dim = in_dim - self.effective_qc
            self.conv0 = WeightStandardizedConv(classical_in_dim, self.classical_channels)
            self.norm0 = nn.GroupNorm(min(groups, self.classical_channels), self.classical_channels)
            
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, 2 * self.classical_channels),
                nn.SiLU(),
            )
            
            self.conv1 = WeightStandardizedConv(self.classical_channels, self.classical_channels)
            self.norm1 = nn.GroupNorm(min(groups, self.classical_channels), self.classical_channels)
        else:
            self.conv0 = None
            self.norm0 = None
            self.time_mlp = None
            self.conv1 = None
            self.norm1 = None

        # Skip connection
        self.skip = (
            nn.Conv2d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x, time_emb, params=None):
        B, C, H, W = x.shape

        # Split channels: quantum channels (multiples of 4) and classical channels
        # Only process the first quantum_channels if input has more channels
        if C > self.effective_qc:
            x_q = x[:, :self.effective_qc] if self.effective_qc > 0 else None
            x_c = x[:, self.effective_qc:] if self.effective_qc < C else None
        else:
            # If input has fewer channels than quantum_channels, use all input for quantum
            x_q = x if self.effective_qc > 0 else None
            x_c = None

        outputs = []

        # Process quantum channels
        if x_q is not None and self.effective_qc > 0:
            y_q = self.quantum_block(x_q, params)  # [B, effective_qc, H, W]
            
            # Ensure spatial dimensions match
            if y_q.shape[2:] != (H, W):
                y_q = F.interpolate(y_q, size=(H, W), mode='nearest')

            outputs.append(y_q)

        # Process classical channels
        if self.classical_channels > 0 and x_c is not None:
            # Ensure we don't process more channels than the classical path is designed for
            if x_c.shape[1] > self.classical_channels:
                x_c = x_c[:, :self.classical_channels]
            elif x_c.shape[1] < self.classical_channels:
                # Pad with zeros if we have fewer channels
                padding = self.classical_channels - x_c.shape[1]
                x_c = torch.cat([x_c, torch.zeros(B, padding, H, W, device=x.device, dtype=x.dtype)], dim=1)
            
            # First convolution
            h_c = self.conv0(x_c)
            h_c = self.norm0(h_c)

            # Add time embedding
            t = self.time_mlp(time_emb).view(B, -1, 1, 1)
            scale, shift = torch.chunk(t, 2, dim=1)
            h_c = h_c * (1 + scale) + shift
            h_c = F.silu(h_c)

            # Second convolution
            h_c = self.conv1(h_c)
            h_c = self.norm1(h_c)
            h_c = F.silu(h_c)
            
            outputs.append(h_c)

        # Combine quantum and classical outputs
        if outputs:
            h = torch.cat(outputs, dim=1)  # [B, out_dim, H, W]
        else:
            h = torch.zeros(B, self.out_dim, H, W, device=x.device, dtype=x.dtype)

        # Residual connection
        x_skip = self.skip(x)
        return x_skip + h


# --- U-Net Components ---
class DownUnet(nn.Module):
    def __init__(self, dim, init_dim=None, dim_mults=(1, 2, 4, 8), resnet_block_groups=8):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        
        # Initial convolution
        self.init_conv = nn.Conv2d(3, self.init_dim, 7, padding=3)  # Assuming 3 input channels
        
        # Time embedding
        self.time_emb = SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4)
        )
        
        # Downsampling blocks
        self.downs = nn.ModuleList()
        dims = [self.init_dim] + [dim * m for m in dim_mults]
        
        for i in range(len(dim_mults)):
            dim_in = dims[i]
            dim_out = dims[i + 1]
            
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, dim * 4, resnet_block_groups),
                ResnetBlock(dim_in, dim_in, dim * 4, resnet_block_groups),
                AttnBlock(dim_in),
                Downsample(dim_in, dim_out) if i < len(dim_mults) - 1 else nn.Identity()
            ]))
        
        # Final conv
        mid_dim = dim * dim_mults[-1]
        self.final_conv = nn.Conv2d(dims[-2], mid_dim, 3, padding=1)
    
    def forward(self, x, time):
        # Initial conv
        h = self.init_conv(x)
        hs = [h]
        
        # Time embedding
        time_emb = self.time_emb(time)
        time_emb = self.time_mlp(time_emb)
        
        # Downsampling
        for resblock1, resblock2, attn, downsample in self.downs:
            h = resblock1(h, time_emb)
            hs.append(h)
            h = resblock2(h, time_emb)
            h = attn(h)
            hs.append(h)
            h = downsample(h)
        
        h = self.final_conv(h)
        return h, hs, time_emb

class UpUnet(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), resnet_block_groups=8):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.out_dim = out_dim or 3  # Assuming 3 output channels
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        
        # Upsampling blocks
        self.ups = nn.ModuleList()
        dims = [self.init_dim] + [dim * m for m in dim_mults]
        
        for i in reversed(range(len(dim_mults))):
            dim_in = dims[i + 1]
            dim_out = dims[i]
            
            # For skip connections, we need to handle concatenated dimensions
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_in + dim_out, dim_in, dim * 4, resnet_block_groups),  # After first skip
                ResnetBlock(dim_in + dim_out, dim_in, dim * 4, resnet_block_groups),  # After second skip
                AttnBlock(dim_in),
                Upsample(dim_in, dim_out) if i > 0 else nn.Identity()
            ]))
        
        # Final processing
        self.final_conv1 = nn.Conv2d(dims[1], self.init_dim, 3, padding=1)
        self.final_resblock = ResnetBlock(self.init_dim * 2, self.dim, self.dim * 4, resnet_block_groups)  # After final skip
        self.final_conv2 = nn.Conv2d(self.dim, self.out_dim, 1)
    
    def forward(self, x, h, hs, time_emb):
        num_resolutions = len(self.dim_mults)
        dims = [self.init_dim] + [self.dim * m for m in self.dim_mults]
        
        # Upsampling - following JAX logic exactly
        for i, (resblock1, resblock2, attn, upsample) in enumerate(self.ups):
            resolution_idx = num_resolutions - 1 - i
            dim_in = dims[resolution_idx + 1]
            dim_out = dims[resolution_idx] if resolution_idx > 0 else self.init_dim
            
            # First skip connection
            skip1 = hs.pop()
            h = torch.cat([h, skip1], dim=1)
            h = resblock1(h, time_emb)
            
            # Second skip connection  
            skip2 = hs.pop()
            h = torch.cat([h, skip2], dim=1)
            h = resblock2(h, time_emb)
            
            h = attn(h)
            h = upsample(h)
        
        # Final processing
        h = self.final_conv1(h)
        
        # Final skip connection
        final_skip = hs.pop()
        h = torch.cat([h, final_skip], dim=1)
        h = self.final_resblock(h, time_emb)
        h = self.final_conv2(h)
        
        return h
    
def matched_mlp_hidden(params_per_group):
    """Hidden width h for Linear(4,h)-Tanh-Linear(h,4)-Tanh (9h + 4 params) closest
    to params_per_group."""
    return max(1, int(round((params_per_group - 4) / 9.0)))


class ClassicalBottleneckBlock(nn.Module):
    """Drop-in classical replacement for QuantumBlock for parameter-matched ablation.

    Mirrors QuantumBlock's exact computational pattern so the ablation is a fair
    classical analogue, not a more-expressive one:
      1. Spatial average pool each 4-channel group: [B, 4, H, W] -> [B, 4]
      2. Per-group classical transform (one small module per group,
         analogous to the quantum circuit processing one group at a time)
      3. Broadcast back to spatial: [B, 4] -> [B, 4, H, W]

    Because the output is spatially constant (same value per (H, W) position
    within a group), the ablation has the same spatial-expressivity restriction
    as the quantum block. The only difference is the per-group transform itself.

    kind='linear': Linear(4, 4) per group (20 params/group). This is the
        original ablation and pairs with q_ansatz='legacy'.
    kind='mlp': Linear(4, h) - Tanh - Linear(h, 4) - Tanh per group (9h + 4
        params/group), bounded to [-1, 1] like <Z>. Pairs with
        q_ansatz='reupload' with h = matched_mlp_hidden(16 * q_layers).

    Interface matches QuantumBlock: forward(x, params=None) with x shaped
    [B, quantum_channels, H, W], plus set_quantum_mode for the same knockout.
    """
    def __init__(self, quantum_channels=8, kind='linear', hidden=None):
        super().__init__()
        if kind not in ('linear', 'mlp'):
            raise ValueError(f"Unknown classical bottleneck kind {kind!r}")
        self.quantum_channels = quantum_channels
        self.num_groups = quantum_channels // 4
        self.remaining_channels = quantum_channels % 4
        self.effective_qc = self.num_groups * 4
        self.kind = kind
        self.hidden = hidden if hidden is not None else 3
        self.quantum_mode = 'on'
        self._random_groups = None
        self.per_group = nn.ModuleList([self._make_group() for _ in range(self.num_groups)])

    def _make_group(self):
        if self.kind == 'linear':
            return nn.Linear(4, 4)
        return nn.Sequential(nn.Linear(4, self.hidden), nn.Tanh(),
                             nn.Linear(self.hidden, 4), nn.Tanh())

    def set_quantum_mode(self, mode='on', knockout_seed=0):
        """Same knockout switch as QuantumBlock: 'on', 'zero', or 'random'
        (freshly initialized per-group modules from a forked RNG)."""
        if mode not in QUANTUM_MODES:
            raise ValueError(f"Unknown quantum mode {mode!r}, expected one of {QUANTUM_MODES}")
        self.quantum_mode = mode
        self._random_groups = None
        if mode == 'random':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(knockout_seed) + KNOCKOUT_SEED_OFFSET)
                # Plain list so the random copy is not registered in state_dict
                self._random_groups = [self._make_group() for _ in range(self.num_groups)]
        return self

    def forward(self, x, params=None):
        if self.effective_qc == 0:
            return x
        B, _, H, W = x.shape
        if self.quantum_mode == 'zero':
            return torch.zeros(B, self.effective_qc, H, W, dtype=x.dtype, device=x.device)
        groups = self.per_group
        if self.quantum_mode == 'random':
            groups = [g.to(x.device) for g in self._random_groups]
        x_eff = x[:, :self.effective_qc]
        # Spatial pool per group
        pooled = F.adaptive_avg_pool2d(x_eff, 1).squeeze(-1).squeeze(-1)  # [B, effective_qc]
        outputs = []
        for i, layer in enumerate(groups):
            g = pooled[:, i * 4 : (i + 1) * 4]       # [B, 4]
            y = layer(g)                              # [B, 4]
            y = y.unsqueeze(-1).unsqueeze(-1)         # [B, 4, 1, 1]
            y = y.expand(-1, -1, H, W)                # [B, 4, H, W]
            outputs.append(y)
        return torch.cat(outputs, dim=1)


class QVertex(nn.Module):
    def __init__(self, dim, quantum_block: QuantumBlock, quantum_channels=8,
                 dim_mults=(1, 2, 4, 8), resnet_block_groups=8, quantum_block2=None):
        super().__init__()
        self.dim = dim
        self.quantum_channels = quantum_channels
        self.resnet_block_groups = resnet_block_groups
        # quantum_block2=None: qresblock1 and qresblock2 share one block (same
        # circuit parameters, original behaviour). Pass a second block to give
        # qresblock2 its own parameters.
        self.shared_quantum_block = quantum_block2 is None
        quantum_block2 = quantum_block if quantum_block2 is None else quantum_block2
        
        mid_dim = dim * dim_mults[-1]
        time_emb_dim = dim * 4  # Match the main model's time embedding dimension
        self.qresblock1 = QResnetBlock(mid_dim, mid_dim, time_emb_dim, quantum_block, quantum_channels, resnet_block_groups)
        self.attn = AttnBlock(mid_dim, use_linear_attention=False)
        self.qresblock2 = QResnetBlock(mid_dim, mid_dim, time_emb_dim, quantum_block2, quantum_channels, resnet_block_groups)
    
    def forward(self, h, time_emb, params=None):
        h = self.qresblock1(h, time_emb, params)
        h = self.attn(h)
        h = self.qresblock2(h, time_emb, params)
        return h, time_emb

class QVUNet(nn.Module):
    def __init__(self, dim, quantum_block: QuantumBlock, init_dim=None, out_dim=None, 
                 dim_mults=(1, 2, 4, 8), resnet_block_groups=8, quantum_channels=8,
                 quantum_block2=None):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.out_dim = out_dim or 2  # Binary segmentation
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        self.quantum_channels = quantum_channels
        
        self.down = DownUnet(dim, init_dim, dim_mults, resnet_block_groups)
        self.vertex = QVertex(dim, quantum_block, quantum_channels, dim_mults, resnet_block_groups,
                              quantum_block2=quantum_block2)
        self.up = UpUnet(dim, init_dim, out_dim, dim_mults, resnet_block_groups)

    def set_quantum_mode(self, mode='on', knockout_seed=0):
        """Apply the knockout switch to every bottleneck block (quantum or
        classical ablation). Unshared blocks get knockout_seed and knockout_seed + 1."""
        blocks = []
        for m in (self.vertex.qresblock1.quantum_block, self.vertex.qresblock2.quantum_block):
            if hasattr(m, 'set_quantum_mode') and not any(m is b for b in blocks):
                blocks.append(m)
        for k, m in enumerate(blocks):
            m.set_quantum_mode(mode, knockout_seed=knockout_seed + k)
        return self
    
    def forward(self, x, time, params=None):
        h, hs, time_emb = self.down(x, time)
        h, time_emb = self.vertex(h, time_emb, params)
        h = self.up(x, h, hs, time_emb)
        return h 

# --- Classical U-Net Components ---

class Vertex(nn.Module):
    """Classical vertex (bottleneck) of the U-Net."""
    def __init__(self, dim, dim_mults=(1, 2, 4, 8), resnet_block_groups=8):
        super().__init__()
        self.dim = dim
        self.resnet_block_groups = resnet_block_groups
        
        mid_dim = dim * dim_mults[-1]
        time_emb_dim = dim * 4  # Match the main model's time embedding dimension
        
        self.resblock1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim, resnet_block_groups)
        self.attn = AttnBlock(mid_dim, use_linear_attention=False)
        self.resblock2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim, resnet_block_groups)
    
    def forward(self, h, time_emb):
        h = self.resblock1(h, time_emb)
        h = self.attn(h)
        h = self.resblock2(h, time_emb)
        return h, time_emb

class UNet(nn.Module):
    """Classical U-Net architecture."""
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), resnet_block_groups=8):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.out_dim = out_dim or 3  # Default to 3 output channels
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        
        self.down = DownUnet(dim, init_dim, dim_mults, resnet_block_groups)
        self.vertex = Vertex(dim, dim_mults, resnet_block_groups)
        self.up = UpUnet(dim, init_dim, out_dim, dim_mults, resnet_block_groups)
    
    def forward(self, x, time):
        h, hs, time_emb = self.down(x, time)
        h, time_emb = self.vertex(h, time_emb)
        h = self.up(x, h, hs, time_emb)
        return h 
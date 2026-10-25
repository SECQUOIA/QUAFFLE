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
BACKBONE_CONV_CHOICES = ('ws_conv', 'standard', 'depthwise_sep')
DEFAULT_BACKBONE_CONV_MODE = 'ws_conv'
_BACKBONE_CONV_ALIASES = {
    'ws_conv': 'ws_conv',
    'default': 'ws_conv',
    'standard': 'standard',
    'conv2d': 'standard',
    'depthwise_sep': 'depthwise_sep',
    'depthwise': 'depthwise_sep',
    'dw_sep': 'depthwise_sep',
}


def normalize_backbone_conv_mode(mode):
    """Normalize backbone_conv_mode ('ws_conv' default, 'standard', or 'depthwise_sep')."""
    if mode is None:
        return DEFAULT_BACKBONE_CONV_MODE
    key = str(mode).strip().lower()
    if key not in _BACKBONE_CONV_ALIASES:
        raise ValueError(
            f"Unknown backbone_conv_mode {mode!r}, expected one of {BACKBONE_CONV_CHOICES}"
        )
    return _BACKBONE_CONV_ALIASES[key]


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


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise convolution followed by 1x1 WeightStandardizedConv (#22)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = WeightStandardizedConv(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def _compatible_group_count(channels: int, max_groups: int = 4) -> int:
    """Return the largest divisor of `channels` that is <= `max_groups` (at least 1)."""
    for g in range(min(int(max_groups), int(channels)), 0, -1):
        if int(channels) % g == 0:
            return g
    return 1


def _make_backbone_conv(in_channels, out_channels, backbone_conv_mode=DEFAULT_BACKBONE_CONV_MODE):
    mode = normalize_backbone_conv_mode(backbone_conv_mode)
    if mode == 'depthwise_sep':
        return DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    if mode == 'standard':
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    return WeightStandardizedConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        time_emb_dim,
        groups=8,
        use_time_emb: bool = True,
        backbone_conv_mode: str = DEFAULT_BACKBONE_CONV_MODE,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_time_emb = bool(use_time_emb)
        self.backbone_conv_mode = normalize_backbone_conv_mode(backbone_conv_mode)
        self.conv0 = _make_backbone_conv(in_dim, out_dim, self.backbone_conv_mode)
        self.norm0 = nn.GroupNorm(groups, out_dim)
        if self.use_time_emb:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_dim * 2),
                nn.SiLU(),
            )
        else:
            self.time_mlp = None
        self.conv1 = _make_backbone_conv(out_dim, out_dim, self.backbone_conv_mode)
        self.norm1 = nn.GroupNorm(groups, out_dim)
        self.skip = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, time_emb=None):
        B, C, H, W = x.shape
        
        h = self.conv0(x)
        h = self.norm0(h)
        
        # Add timestep embedding when enabled
        if self.time_mlp is not None and time_emb is not None:
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
Q_ANSATZ_CHOICES = ('legacy', 'reupload', 'hqunet')
Q_SPATIAL_MODES = ('global', 'pixel', 'patch2x2', 'patch4x4', 'se_gate')
DEFAULT_Q_SPATIAL_MODE = 'global'
HQUNET_GRID_CHOICES = (2, 4)
DEFAULT_HQUNET_GRID = 2
ABLATION_MODE_CHOICES = ('auto', 'linear', 'mlp', 'conv1x1', 'depthwise3x3', 'lowrank', 'hqunet_spatial')
DEFAULT_ABLATION_MODE = 'auto'
QUANTUM_MODES = ('on', 'zero', 'random')
# Added to knockout_seed so the 'random' knockout never reuses the exact stream
# that initialized the trained parameters (e.g. run seed == knockout seed).
KNOCKOUT_SEED_OFFSET = 1_000_003
QUANTUM_INIT_SEED_OFFSET = 500_009


def normalize_q_spatial_mode(mode):
    """Normalize q_spatial_mode ('global' default, or 'pixel', 'patch2x2', 'patch4x4', 'se_gate')."""
    if mode is None:
        return DEFAULT_Q_SPATIAL_MODE
    key = str(mode).strip().lower()
    if key not in Q_SPATIAL_MODES:
        raise ValueError(
            f"Unknown q_spatial_mode {mode!r}, expected one of {Q_SPATIAL_MODES}"
        )
    return key


def normalize_hqunet_grid(grid):
    """Normalize HQ-UNet spatial grid dimension G in {2, 4}."""
    if grid is None:
        return DEFAULT_HQUNET_GRID
    g = int(grid)
    if g not in HQUNET_GRID_CHOICES:
        raise ValueError(
            f"Unknown hqunet_grid {grid!r}, expected one of {HQUNET_GRID_CHOICES}"
        )
    return g


def normalize_ablation_mode(mode):
    """Normalize classical ablation bottleneck mode."""
    if mode is None:
        return DEFAULT_ABLATION_MODE
    key = str(mode).strip().lower()
    if key not in ABLATION_MODE_CHOICES:
        raise ValueError(
            f"Unknown ablation_mode {mode!r}, expected one of {ABLATION_MODE_CHOICES}"
        )
    return key


def hqunet_grid_edges(grid_size: int = DEFAULT_HQUNET_GRID):
    """Return (row_edges, col_edges) on a G x G 2D qubit grid.
    Qubit index q = r * G + c for r, c in 0..G-1.
    """
    G = normalize_hqunet_grid(grid_size)
    row_edges = [
        (r * G + c, r * G + (c + 1))
        for r in range(G)
        for c in range(G - 1)
    ]
    col_edges = [
        (r * G + c, (r + 1) * G + c)
        for r in range(G - 1)
        for c in range(G)
    ]
    return row_edges, col_edges


def create_quantum_block(backend='ptlayer', quantum_channels=8, ptlayer_config=None,
                         seed=None, q_ansatz='reupload', q_layers=2, q_backend='pennylane',
                         q_sim_device='cpu', q_spatial_mode=DEFAULT_Q_SPATIAL_MODE,
                         hqunet_grid=DEFAULT_HQUNET_GRID):
    """
    Create a QuantumBlock that processes channels in groups of 4 (or on a GxG grid for hqunet).
    """
    return QuantumBlock(backend=backend, quantum_channels=quantum_channels,
                        ptlayer_config=ptlayer_config, seed=seed,
                        q_ansatz=q_ansatz, q_layers=q_layers, q_backend=q_backend,
                        q_sim_device=q_sim_device, q_spatial_mode=q_spatial_mode,
                        hqunet_grid=hqunet_grid)


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


# Broadcast QNodes for q_backend='pennylane', one per (ansatz, layers, qubits).
# Kept at module level (not on the module instance) so QuantumBlock stays
# picklable and every block shares the same QNode.
_FLAT_QNODES = {}


def pennylane_flat_qnode(q_ansatz, n_layers=1, n_qubits=None):
    """One default.qubit QNode that evaluates all groups and batch items at once.

    Inputs are broadcast along a leading axis K = B * G:
      reupload: x [K, 4], weights [K, L, 4, 3], scales [K, L, 4]
      legacy:   x [K, 4], theta [K, 4]
    The gate sequence is identical to pennylane_reupload_circuit and
    pennylane_default_circuit (native RY / Rot / RX / CNOT gates), with
    diff_method='backprop' and the torch interface.
    """
    if n_qubits is None:
        n_qubits = N_QUBITS
    key = (q_ansatz, int(n_layers), int(n_qubits))
    if key in _FLAT_QNODES:
        return _FLAT_QNODES[key]
    if not HAS_PENNYLANE:
        raise ImportError("PennyLane not available, q_backend='pennylane' needs it.")
    dev = qml.device("default.qubit", wires=n_qubits)
    if q_ansatz == 'legacy':
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, theta):
            for j in range(n_qubits):
                qml.RX(x[:, j], wires=j)
            for j in range(n_qubits):
                qml.RY(theta[:, j], wires=j)
            for j in range(n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
    elif q_ansatz == 'reupload':
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, weights, scales):
            for l in range(int(n_layers)):
                enc = math.pi * torch.tanh(x * scales[:, l])  # [K, n_qubits]
                for j in range(n_qubits):
                    qml.RY(enc[:, j], wires=j)
                for j in range(n_qubits):
                    qml.Rot(weights[:, l, j, 0], weights[:, l, j, 1], weights[:, l, j, 2], wires=j)
                for j in range(n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
    else:
        raise ValueError(f"Unknown q_ansatz {q_ansatz!r}")
    _FLAT_QNODES[key] = circuit
    return circuit


# --- Exact batched statevector simulator (q_backend='torch') ---
#
# Same math as PennyLane default.qubit for the two 4 qubit circuits above,
# written as plain torch ops so that the batch AND all groups are simulated in
# one pass on the model's device. Wire 0 is the most significant bit of the
# basis index, as in PennyLane. The simulation runs in complex128 because
# default.qubit returns float64 expectation values for these circuits. The
# result is cast to float32 by QuantumBlock exactly like the PennyLane path.
#
# q_backend values: 'pennylane' (default, one broadcast default.qubit QNode
# call per block forward), 'pennylane_pergroup' (one QNode call per group,
# legacy: per group and batch item; test reference), 'torch' (this simulator,
# cross-check reference).
Q_BACKEND_CHOICES = ('pennylane', 'pennylane_pergroup', 'torch')
Q_SIM_DEVICE_CHOICES = ('cpu', 'model')
_PENNYLANE_Q_BACKENDS = ('pennylane', 'pennylane_pergroup')
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


def _rz_mat(theta):
    """RZ(theta) as a complex [..., 2, 2] tensor."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)
    e_m = torch.complex(c, -s)
    e_p = torch.complex(c, s)
    zc = torch.complex(z, z)
    return torch.stack([torch.stack([e_m, zc], -1), torch.stack([zc, e_p], -1)], -2)


def _hqunet_filter_matrix(theta1, theta2, phi1, phi2):
    """Build 4x4 real unitary matrix for 2-qubit filter
    F(theta, phi) = (RY(phi1) x RY(phi2)) @ CNOT(0->1) @ (RY(theta1) x RY(theta2)).
    """
    ct1, st1 = torch.cos(theta1 / 2), torch.sin(theta1 / 2)
    ct2, st2 = torch.cos(theta2 / 2), torch.sin(theta2 / 2)
    cp1, sp1 = torch.cos(phi1 / 2), torch.sin(phi1 / 2)
    cp2, sp2 = torch.cos(phi2 / 2), torch.sin(phi2 / 2)

    ry_t1 = torch.stack([torch.stack([ct1, -st1], -1), torch.stack([st1, ct1], -1)], -2)
    ry_t2 = torch.stack([torch.stack([ct2, -st2], -1), torch.stack([st2, ct2], -1)], -2)
    ry_p1 = torch.stack([torch.stack([cp1, -sp1], -1), torch.stack([sp1, cp1], -1)], -2)
    ry_p2 = torch.stack([torch.stack([cp2, -sp2], -1), torch.stack([sp2, cp2], -1)], -2)

    a = torch.kron(ry_t1, ry_t2)
    b = torch.kron(ry_p1, ry_p2)
    # CNOT(0->1) swaps basis states |10> (idx 2) and |11> (idx 3)
    a_cnot = torch.stack([a[0], a[1], a[3], a[2]], dim=0)
    return b @ a_cnot


def _dual_basis_expvals_from_state(state, n_qubits):
    """Compute [<Z_0>, ..., <Z_{Nq-1}>, <X_0>, ..., <X_{Nq-1}>] from a batched
    statevector `state` of shape [B, 2**n_qubits] (or [B, 2, 2, ..., 2]).
    Returns float64 tensor of shape [B, 2 * n_qubits].
    """
    B = state.shape[0]
    exp_z = []
    exp_x = []
    for q in range(n_qubits):
        s_q = state.reshape(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
        s0 = s_q[:, :, 0, :]
        s1 = s_q[:, :, 1, :]
        p0 = s0.real ** 2 + s0.imag ** 2
        p1 = s1.real ** 2 + s1.imag ** 2
        exp_z.append((p0 - p1).sum(dim=(1, 2)))
        exp_x.append((2.0 * (s0.real * s1.real + s0.imag * s1.imag)).sum(dim=(1, 2)))
    return torch.cat(
        [torch.stack(exp_z, dim=1), torch.stack(exp_x, dim=1)],
        dim=1,
    )


def torch_hqunet_expvals(angles, weights, grid_size=DEFAULT_HQUNET_GRID):
    """Exact statevector simulation of the HQ-UNet 2D spatial quanvolutional ansatz (#18).

    Args:
        angles: [B, N_q, 3] tensor of 3-axis (RX, RY, RZ) encoding angles per qubit.
        weights: [L, 2, 4] tensor of shared horizontal filter F_R (weights[l, 0]) and
            shared vertical filter F_C (weights[l, 1]) per layer (also accepts untied
            [L, 2 * G * (G - 1), 4]).
        grid_size: G in {2, 4} where N_q = G * G.

    Returns:
        float64 tensor of shape [B, 2 * N_q] containing (<Z_0..Nq-1>, <X_0..Nq-1>).
    """
    G = normalize_hqunet_grid(grid_size)
    n_qubits = G * G
    B = angles.shape[0]
    L = weights.shape[0]
    row_edges, col_edges = hqunet_grid_edges(G)
    all_edges = row_edges + col_edges
    shared_filters = weights.shape[1] == 2

    angles64 = angles.to(torch.float64)
    weights64 = weights.to(torch.float64)

    state = torch.zeros((B, 1 << n_qubits), dtype=torch.complex128, device=angles.device)
    state[:, 0] = 1.0 + 0.0j

    def _apply_two_qubit_gate(st, q1, q2, uf_c):
        a_len = 1 << q1
        m_len = 1 << (q2 - q1 - 1)
        c_len = 1 << (n_qubits - 1 - q2)
        s_pair = st.reshape(B, a_len, 2, m_len, 2, c_len)
        s00 = s_pair[:, :, 0, :, 0, :]
        s01 = s_pair[:, :, 0, :, 1, :]
        s10 = s_pair[:, :, 1, :, 0, :]
        s11 = s_pair[:, :, 1, :, 1, :]
        vec = torch.stack([s00, s01, s10, s11], dim=-1)  # [B, A, M, C, 4]
        vec_out = torch.matmul(vec, uf_c.t())
        row0 = torch.stack([vec_out[..., 0], vec_out[..., 1]], dim=3)  # [B, A, M, 2, C]
        row1 = torch.stack([vec_out[..., 2], vec_out[..., 3]], dim=3)
        return torch.stack([row0, row1], dim=2).reshape(B, 1 << n_qubits)

    for l in range(L):
        # 1. 3-axis (RX, RY, RZ) encoding on every qubit q = 0..N_q-1
        u_enc = (
            _rz_mat(angles64[:, :, 2])
            @ _ry_mat(angles64[:, :, 1])
            @ _rx_mat(angles64[:, :, 0])
        )  # [B, N_q, 2, 2]
        for q in range(n_qubits):
            s_q = state.reshape(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
            s0 = s_q[:, :, 0, :]
            s1 = s_q[:, :, 1, :]
            g = u_enc[:, q]  # [B, 2, 2]
            g00 = g[:, 0, 0].view(B, 1, 1)
            g01 = g[:, 0, 1].view(B, 1, 1)
            g10 = g[:, 1, 0].view(B, 1, 1)
            g11 = g[:, 1, 1].view(B, 1, 1)
            out0 = g00 * s0 + g01 * s1
            out1 = g10 * s0 + g11 * s1
            state = torch.stack([out0, out1], dim=2).reshape(B, 1 << n_qubits)

        # 2. 2D row and column quanvolutional filters F_R (shared across row_edges)
        #    and F_C (shared across col_edges)
        if shared_filters:
            w_r = weights64[l, 0]
            uf_r = _hqunet_filter_matrix(w_r[0], w_r[1], w_r[2], w_r[3])
            uf_r_c = torch.complex(uf_r, torch.zeros_like(uf_r))
            for q1, q2 in row_edges:
                state = _apply_two_qubit_gate(state, q1, q2, uf_r_c)
            w_c = weights64[l, 1]
            uf_c = _hqunet_filter_matrix(w_c[0], w_c[1], w_c[2], w_c[3])
            uf_c_c = torch.complex(uf_c, torch.zeros_like(uf_c))
            for q1, q2 in col_edges:
                state = _apply_two_qubit_gate(state, q1, q2, uf_c_c)
        else:
            for e, (q1, q2) in enumerate(all_edges):
                w_e = weights64[l, e]
                uf = _hqunet_filter_matrix(w_e[0], w_e[1], w_e[2], w_e[3])
                uf_c = torch.complex(uf, torch.zeros_like(uf))
                state = _apply_two_qubit_gate(state, q1, q2, uf_c)

    return _dual_basis_expvals_from_state(state, n_qubits)


_HQUNET_STATE_QNODES = {}


def pennylane_hqunet_state_qnode(n_layers=2, grid_size=DEFAULT_HQUNET_GRID):
    """PennyLane default.qubit QNode returning statevector for HQ-UNet (G=2 or G=4)."""
    G = normalize_hqunet_grid(grid_size)
    n_qubits = G * G
    key = (int(n_layers), G)
    if key in _HQUNET_STATE_QNODES:
        return _HQUNET_STATE_QNODES[key]
    if not HAS_PENNYLANE:
        raise ImportError("PennyLane not available, q_backend='pennylane' needs it.")
    dev = qml.device("default.qubit", wires=n_qubits)
    row_edges, col_edges = hqunet_grid_edges(G)
    all_edges = row_edges + col_edges

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(angles, weights):
        # angles: [B, N_q, 3], weights: [L, 2, 4] (shared F_R, F_C) or [L, n_edges, 4]
        shared = weights.shape[1] == 2
        for l in range(int(n_layers)):
            for q in range(n_qubits):
                qml.RX(angles[:, q, 0], wires=q)
                qml.RY(angles[:, q, 1], wires=q)
                qml.RZ(angles[:, q, 2], wires=q)
            if shared:
                for q1, q2 in row_edges:
                    qml.RY(weights[l, 0, 0], wires=q1)
                    qml.RY(weights[l, 0, 1], wires=q2)
                    qml.CNOT(wires=[q1, q2])
                    qml.RY(weights[l, 0, 2], wires=q1)
                    qml.RY(weights[l, 0, 3], wires=q2)
                for q1, q2 in col_edges:
                    qml.RY(weights[l, 1, 0], wires=q1)
                    qml.RY(weights[l, 1, 1], wires=q2)
                    qml.CNOT(wires=[q1, q2])
                    qml.RY(weights[l, 1, 2], wires=q1)
                    qml.RY(weights[l, 1, 3], wires=q2)
            else:
                for e, (q1, q2) in enumerate(all_edges):
                    qml.RY(weights[l, e, 0], wires=q1)
                    qml.RY(weights[l, e, 1], wires=q2)
                    qml.CNOT(wires=[q1, q2])
                    qml.RY(weights[l, e, 2], wires=q1)
                    qml.RY(weights[l, e, 3], wires=q2)
        return qml.state()

    _HQUNET_STATE_QNODES[key] = circuit
    return circuit


def _randn(shape, generator=None):
    return torch.randn(*shape, generator=generator)


def _uniform(shape, low, high, generator=None):
    return low + (high - low) * torch.rand(*shape, generator=generator)


class QuantumBlock(nn.Module):
    """
    Quantum processing block that handles channels in groups of 4 (for 'reupload'
    and 'legacy') or on a 2D GxG spatial grid (for 'hqunet').
    """
    def __init__(self, backend='pennylane', quantum_channels=8, ptlayer_config=None,
                 seed=None, q_ansatz='reupload', q_layers=2, q_backend='pennylane',
                 q_sim_device='cpu', q_spatial_mode=DEFAULT_Q_SPATIAL_MODE,
                 hqunet_grid=DEFAULT_HQUNET_GRID):
        super().__init__()

        if q_ansatz not in Q_ANSATZ_CHOICES:
            raise ValueError(f"Unknown q_ansatz {q_ansatz!r}, expected one of {Q_ANSATZ_CHOICES}")
        if q_ansatz in ('reupload', 'hqunet') and int(q_layers) < 1:
            raise ValueError("q_layers must be >= 1")
        if q_backend not in Q_BACKEND_CHOICES:
            raise ValueError(f"Unknown q_backend {q_backend!r}, expected one of {Q_BACKEND_CHOICES}")
        if q_sim_device not in Q_SIM_DEVICE_CHOICES:
            raise ValueError(f"Unknown q_sim_device {q_sim_device!r}, expected one of {Q_SIM_DEVICE_CHOICES}")

        self.backend = backend
        self.quantum_channels = quantum_channels
        self.q_ansatz = q_ansatz
        self.q_layers = int(q_layers) if q_ansatz in ('reupload', 'hqunet') else 1
        self.q_spatial_mode = normalize_q_spatial_mode(q_spatial_mode)
        self.hqunet_grid = normalize_hqunet_grid(hqunet_grid)
        self.seed = seed
        self.quantum_mode = 'on'
        self._random_params = None
        self._random_stacked = None
        self._random_hqunet_weights = None
        self.q_backend = q_backend
        self.q_sim_device = q_sim_device
        
        # Calculate number of 4-channel groups
        self.num_groups = quantum_channels // 4
        self.remaining_channels = quantum_channels % 4
        self.effective_qc = self.num_groups * 4
        self.pre_q_conv = None
        self.post_q_proj = None
        self.hqunet_weights = None

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
            if not HAS_PENNYLANE and q_backend in _PENNYLANE_Q_BACKENDS:
                raise ImportError("PennyLane not available. Install PennyLane or use q_backend='torch'.")
            
            self.pennylane_circuits = []
            self.quantum_params = nn.ParameterList()
            self.input_scales = nn.ParameterList()

            if q_ansatz == 'hqunet':
                G = self.hqunet_grid
                n_q = G * G
                if self.effective_qc > 0:
                    with torch.random.fork_rng(devices=[], enabled=seed is not None):
                        if seed is not None:
                            torch.default_generator.manual_seed(int(seed) + QUANTUM_INIT_SEED_OFFSET)
                        self.pre_q_conv = nn.Conv2d(self.effective_qc, 3, kernel_size=1)
                        self.post_q_proj = nn.Linear(2 * n_q, self.effective_qc * n_q)
                    # Shared horizontal filter F_R (index 0) and vertical filter F_C (index 1)
                    # per layer: 2 filters x 4 RY angles = 8 parameters per layer (arXiv:2604.27206)
                    self.hqunet_weights = nn.Parameter(
                        _uniform((self.q_layers, 2, 4), -math.pi, math.pi, generator)
                    )
            else:
                for group_idx in range(self.num_groups):
                    if q_ansatz == 'legacy':
                        params = nn.Parameter(_randn((N_QUBITS,), generator) * 0.1)
                        self.quantum_params.append(params)
                    else:
                        weights = nn.Parameter(_uniform((self.q_layers, N_QUBITS, 3), -math.pi, math.pi, generator))
                        self.quantum_params.append(weights)
                        self.input_scales.append(nn.Parameter(torch.ones(self.q_layers, N_QUBITS)))
                if q_backend == 'pennylane_pergroup':
                    self._build_pennylane_circuits()
                
        else:
            raise ValueError(f'Unknown quantum backend: {backend}')

    @property
    def n_params_per_group(self):
        if self.backend != 'pennylane':
            return None
        if self.q_ansatz == 'legacy':
            return N_QUBITS
        if self.q_ansatz == 'hqunet':
            return 8 * self.q_layers
        return self.q_layers * N_QUBITS * 4  # 3 rotation angles + 1 input scale per qubit per layer

    def _build_pennylane_circuits(self):
        if self.q_ansatz == 'hqunet':
            return
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane not available, q_backend='pennylane_pergroup' needs it.")
        if self.pennylane_circuits:
            return
        for _ in range(self.num_groups):
            if self.q_ansatz == 'legacy':
                self.pennylane_circuits.append(pennylane_default_circuit(n_qubits=N_QUBITS))
            else:
                self.pennylane_circuits.append(
                    pennylane_reupload_circuit(n_qubits=N_QUBITS, n_layers=self.q_layers))

    def set_q_backend(self, q_backend):
        """Switch the circuit simulator ('pennylane', 'pennylane_pergroup' or
        'torch') in place. Parameters are untouched."""
        if q_backend not in Q_BACKEND_CHOICES:
            raise ValueError(f"Unknown q_backend {q_backend!r}, expected one of {Q_BACKEND_CHOICES}")
        if self.backend == 'pennylane':
            if q_backend == 'pennylane_pergroup':
                self._build_pennylane_circuits()
            else:
                # Drop the per-group QNodes so the block stays picklable
                self.pennylane_circuits = []
                if q_backend == 'pennylane' and not HAS_PENNYLANE:
                    raise ImportError("PennyLane not available, q_backend='pennylane' needs it.")
        self.q_backend = q_backend
        return self

    def set_q_sim_device(self, q_sim_device):
        """'cpu' or 'model': where the q_backend='pennylane' circuit runs."""
        if q_sim_device not in Q_SIM_DEVICE_CHOICES:
            raise ValueError(f"Unknown q_sim_device {q_sim_device!r}, expected one of {Q_SIM_DEVICE_CHOICES}")
        self.q_sim_device = q_sim_device
        return self

    def simulator_info(self):
        """Plain dict describing how the circuit is simulated (for summaries,
        metrics.json and provenance)."""
        if self.backend != 'pennylane':
            return {'q_backend': None, 'family': self.backend}
        if self.q_backend == 'torch':
            return {'q_backend': 'torch', 'device': 'exact torch statevector simulator',
                    'diff_method': 'torch autograd', 'sim_device': 'model',
                    'q_spatial_mode': self.q_spatial_mode}
        return {'q_backend': self.q_backend, 'device': 'default.qubit', 'interface': 'torch',
                'diff_method': 'backprop',
                'broadcast': ('batch x groups, one call per block forward'
                              if self.q_backend == 'pennylane' else 'per group (legacy: per item)'),
                'sim_device': self.q_sim_device if self.q_backend == 'pennylane' else 'model',
                'q_spatial_mode': self.q_spatial_mode,
                'pennylane_version': qml.__version__ if HAS_PENNYLANE else None}

    # --- Knockout ---
    def set_quantum_mode(self, mode='on', knockout_seed=0, seed=None):
        """Select 'on' (trained circuit), 'zero' (output replaced by zeros), or
        'random' (circuit evaluated with random parameters drawn from a local
        generator seeded by knockout_seed + KNOCKOUT_SEED_OFFSET, from the same distribution used at
        initialization). Does not modify trained parameters or the global RNG."""
        if seed is not None:
            knockout_seed = seed
        if mode not in QUANTUM_MODES:
            raise ValueError(f"Unknown quantum mode {mode!r}, expected one of {QUANTUM_MODES}")
        self.quantum_mode = mode
        self._random_params = None
        self._random_stacked = None
        self._random_hqunet_weights = None
        if mode == 'random' and self.backend == 'pennylane':
            g = torch.Generator()
            g.manual_seed(int(knockout_seed) + KNOCKOUT_SEED_OFFSET)
            if self.q_ansatz == 'hqunet':
                self._random_hqunet_weights = _uniform(
                    (self.q_layers, 2, 4), -math.pi, math.pi, g
                )
            else:
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

    def _eval_tokens_torch(self, tokens):
        """Evaluate 4-qubit groups on `tokens` of shape [N, G * 4] using exact torch statevector."""
        N = tokens.shape[0]
        G = self.num_groups
        pooled = tokens.reshape(N, G, N_QUBITS)
        w, s = self._stacked_params(tokens.device)
        if self.q_ansatz == 'legacy':
            y = torch_legacy_expvals(pooled, w)
        else:
            y = torch_reupload_expvals(pooled, w, s)
        return y.reshape(N, G * N_QUBITS).to(torch.float32)

    def _eval_tokens_pennylane_flat(self, tokens):
        """Evaluate 4-qubit groups on `tokens` of shape [N, G * 4] using broadcast PennyLane QNode."""
        N = tokens.shape[0]
        G = self.num_groups
        sim = torch.device('cpu') if self.q_sim_device == 'cpu' else tokens.device
        pooled = tokens.reshape(N * G, N_QUBITS).to(sim)
        w, s = self._stacked_params(sim)
        w = w.to(sim)
        circuit = pennylane_flat_qnode(self.q_ansatz, self.q_layers)
        if self.q_ansatz == 'legacy':
            theta = w.unsqueeze(0).expand(N, G, N_QUBITS).reshape(N * G, N_QUBITS)
            y = circuit(pooled, theta)
        else:
            L = w.shape[1]
            s = s.to(sim)
            wf = w.unsqueeze(0).expand(N, G, L, N_QUBITS, 3).reshape(N * G, L, N_QUBITS, 3)
            sf = s.unsqueeze(0).expand(N, G, L, N_QUBITS).reshape(N * G, L, N_QUBITS)
            y = circuit(pooled, wf, sf)
        y = torch.stack([torch.as_tensor(v).reshape(-1) for v in y], dim=-1)  # [N*G, 4]
        return y.reshape(N, G * N_QUBITS).to(device=tokens.device, dtype=torch.float32)

    def _eval_tokens_pennylane_pergroup(self, tokens):
        """Evaluate 4-qubit groups on `tokens` of shape [N, G * 4] using per-group QNodes."""
        if not self.pennylane_circuits:
            raise ValueError('PennyLane circuits not initialized for PennyLane backend')
        N = tokens.shape[0]
        G = self.num_groups
        outputs = []
        for i in range(G):
            x_g = tokens[:, i * 4 : (i + 1) * 4]  # [N, 4]
            w, s = self._group_params(i, tokens.device)
            if self.q_ansatz == 'legacy':
                group_outputs = []
                for n in range(N):
                    x_item = x_g[n:n + 1].unsqueeze(-1).unsqueeze(-1)  # [1, 4, 1, 1]
                    y_item = self.pennylane_circuits[i](x_item, w)
                    y_tensor = torch.stack(y_item).unsqueeze(0)  # [1, 4]
                    group_outputs.append(y_tensor)
                y_group = torch.cat(group_outputs, dim=0)  # [N, 4]
            else:
                y_list = self.pennylane_circuits[i](x_g, w, s)
                y_group = torch.stack([torch.as_tensor(v).reshape(-1) for v in y_list], dim=1)
                y_group = y_group.reshape(N, N_QUBITS)
            outputs.append(y_group.to(torch.float32))
        return torch.cat(outputs, dim=1) if outputs else torch.zeros(N, 0, dtype=torch.float32, device=tokens.device)

    def _eval_4qubit_tokens(self, tokens):
        if self.q_backend == 'pennylane':
            return self._eval_tokens_pennylane_flat(tokens)
        if self.q_backend == 'torch':
            return self._eval_tokens_torch(tokens)
        return self._eval_tokens_pennylane_pergroup(tokens)

    def _forward_spatial_4qubit(self, x):
        """Apply `q_spatial_mode` ('global', 'pixel', 'patch2x2', 'patch4x4', 'se_gate')
        to 4-qubit group circuits."""
        B, _, H, W = x.shape
        G = self.num_groups
        if G == 0:
            return torch.zeros(B, 0, H, W, dtype=x.dtype, device=x.device)
        x_eff = x[:, :G * N_QUBITS]
        mode = self.q_spatial_mode
        if mode == 'global':
            tokens = x_eff.mean(dim=(2, 3))  # [B, G*4]
            y_tokens = self._eval_4qubit_tokens(tokens)  # [B, G*4]
            return y_tokens.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        if mode == 'se_gate':
            tokens = x_eff.mean(dim=(2, 3))  # [B, G*4]
            y_tokens = self._eval_4qubit_tokens(tokens)  # [B, G*4]
            gate = torch.sigmoid(y_tokens.unsqueeze(-1).unsqueeze(-1))
            return x_eff.to(torch.float32) * gate
        if mode == 'pixel':
            tokens = x_eff.permute(0, 2, 3, 1).reshape(B * H * W, G * N_QUBITS)
            y_tokens = self._eval_4qubit_tokens(tokens)
            return y_tokens.reshape(B, H, W, G * N_QUBITS).permute(0, 3, 1, 2).contiguous()
        if mode in ('patch2x2', 'patch4x4'):
            P = 2 if mode == 'patch2x2' else 4
            x_patch = F.adaptive_avg_pool2d(x_eff, (P, P))
            tokens = x_patch.permute(0, 2, 3, 1).reshape(B * P * P, G * N_QUBITS)
            y_tokens = self._eval_4qubit_tokens(tokens)
            y_patch = y_tokens.reshape(B, P, P, G * N_QUBITS).permute(0, 3, 1, 2).contiguous()
            if (P, P) != (H, W):
                return F.interpolate(y_patch, size=(H, W), mode='bilinear', align_corners=False)
            return y_patch
        raise ValueError(f"Unknown q_spatial_mode {mode!r}")

    def _forward_torch_sim(self, x):
        """q_backend='torch': every group and batch item in one batched simulation."""
        if self.q_ansatz == 'hqunet':
            return self._forward_hqunet(x)
        return self._forward_spatial_4qubit(x)

    def _forward_pennylane_flat(self, x):
        """q_backend='pennylane': one broadcast QNode call over (batch x groups)."""
        if self.q_ansatz == 'hqunet':
            return self._forward_hqunet(x)
        return self._forward_spatial_4qubit(x)

    def _eval_hqunet_expvals(self, angles):
        """Evaluate HQ-UNet circuit on `angles` of shape [B, N_q, 3].
        Returns float32 tensor of shape [B, 2 * N_q].
        """
        G = self.hqunet_grid
        n_q = G * G
        w = (
            self._random_hqunet_weights.to(angles.device)
            if self.quantum_mode == 'random'
            else self.hqunet_weights
        )
        if self.q_backend in _PENNYLANE_Q_BACKENDS:
            sim = (
                torch.device('cpu')
                if (self.q_backend == 'pennylane' and self.q_sim_device == 'cpu')
                else angles.device
            )
            circuit = pennylane_hqunet_state_qnode(self.q_layers, G)
            state = circuit(angles.to(sim), w.to(sim))
            if state.dim() == 1:
                state = state.unsqueeze(0)
            expvals = _dual_basis_expvals_from_state(state, n_q)
            return expvals.to(device=angles.device, dtype=torch.float32)
        expvals = torch_hqunet_expvals(angles, w, G)
        return expvals.to(device=angles.device, dtype=torch.float32)

    def _forward_hqunet(self, x):
        """Forward pass for HQ-UNet 2D spatial quanvolutional bottleneck (#18)."""
        B, _, H, W = x.shape
        if self.effective_qc == 0:
            return torch.zeros(B, 0, H, W, dtype=x.dtype, device=x.device)
        G = self.hqunet_grid
        n_q = G * G
        if self.quantum_mode == 'zero':
            # Zero the 2*G^2 quantum expectation vector before the Post-Q Linear projection (#18)
            z = torch.zeros(B, 2 * n_q, dtype=torch.float32, device=x.device)
        else:
            x_eff = x[:, :self.effective_qc]
            x_grid = F.adaptive_avg_pool2d(x_eff, (G, G))  # [B, C_q, G, G]
            angles = math.pi * torch.tanh(self.pre_q_conv(x_grid))  # [B, 3, G, G]
            angles = angles.permute(0, 2, 3, 1).reshape(B, n_q, 3)
            z = self._eval_hqunet_expvals(angles)  # [B, 2 * n_q]
        y_grid = self.post_q_proj(z).reshape(B, self.effective_qc, G, G)
        if (G, G) != (H, W):
            return F.interpolate(y_grid, size=(H, W), mode='bilinear', align_corners=False)
        return y_grid
    
    def forward(self, x, params=None):
        B, C, H, W = x.shape

        if self.q_ansatz == 'hqunet':
            return self._forward_hqunet(x)

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
            
        elif self.backend == 'pennylane':
            return self._forward_spatial_4qubit(x)
        else:
            raise ValueError(f'Unknown quantum backend: {self.backend}')

SKIP_MODULATION_CHOICES = ('affine', 'additive')
DEFAULT_SKIP_MODULATION = 'affine'
_SKIP_MODULATION_ALIASES = {
    'affine': 'affine',
    'multiplicative': 'affine',
    'se_scale': 'affine',
    'additive': 'additive',
    'legacy': 'additive',
}


def normalize_skip_modulation(mode):
    """Normalize skip_modulation mode ('affine' default, 'additive' legacy)."""
    if mode is None:
        return DEFAULT_SKIP_MODULATION
    key = str(mode).strip().lower()
    if key not in _SKIP_MODULATION_ALIASES:
        raise ValueError(
            f"Unknown skip_modulation {mode!r}, expected one of {SKIP_MODULATION_CHOICES}"
        )
    return _SKIP_MODULATION_ALIASES[key]


class QResnetBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_emb_dim: int,
        quantum_block: Optional[QuantumBlock] = None,
        quantum_channels: int = 8,
        groups: int = 8,
        skip_modulation: str = DEFAULT_SKIP_MODULATION,
        q_residual_mode: Optional[str] = None,
        q_block: Optional[QuantumBlock] = None,
        use_time_emb: bool = True,
        backbone_conv_mode: str = DEFAULT_BACKBONE_CONV_MODE,
    ):
        super().__init__()
        if quantum_block is None:
            quantum_block = q_block
        if quantum_block is None:
            raise ValueError("QResnetBlock requires quantum_block (or q_block)")
        self.quantum_channels = quantum_channels
        self.quantum_block = quantum_block
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_time_emb = bool(use_time_emb)
        self.backbone_conv_mode = normalize_backbone_conv_mode(backbone_conv_mode)
        eff_mode = q_residual_mode if q_residual_mode is not None else skip_modulation
        self.skip_modulation = normalize_skip_modulation(eff_mode)
        self.q_residual_mode = self.skip_modulation
        
        # Calculate effective quantum channels (must be multiple of 4)
        self.effective_qc = (quantum_channels // 4) * 4
        self.classical_channels = out_dim - self.effective_qc

        # Classical path for remaining channels
        if self.classical_channels > 0:
            classical_in_dim = in_dim - self.effective_qc
            cls_groups = _compatible_group_count(self.classical_channels, groups)
            self.conv0 = _make_backbone_conv(classical_in_dim, self.classical_channels, self.backbone_conv_mode)
            self.norm0 = nn.GroupNorm(cls_groups, self.classical_channels)
            
            if self.use_time_emb:
                self.time_mlp = nn.Sequential(
                    nn.Linear(time_emb_dim, 2 * self.classical_channels),
                    nn.SiLU(),
                )
            else:
                self.time_mlp = None
            
            self.conv1 = _make_backbone_conv(self.classical_channels, self.classical_channels, self.backbone_conv_mode)
            self.norm1 = nn.GroupNorm(cls_groups, self.classical_channels)
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

    @property
    def q_block(self):
        return self.quantum_block

    def forward(self, x, time_emb=None, params=None):
        B, C, H, W = x.shape

        # Residual connection
        x_skip = self.skip(x)

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
        qc_actual = 0

        # Process quantum channels
        if x_q is not None and self.effective_qc > 0:
            y_q = self.quantum_block(x_q, params)  # [B, effective_qc, H, W]
            
            # Ensure spatial dimensions match
            if y_q.shape[2:] != (H, W):
                y_q = F.interpolate(y_q, size=(H, W), mode='nearest')

            qc_actual = y_q.shape[1]
            if self.skip_modulation == 'additive':
                outputs.append(x_skip[:, :qc_actual] + y_q)
            else:
                # Multiplicative/affine spatial modulation: x_skip[:, :qc] * (1 + y_q)
                outputs.append(x_skip[:, :qc_actual] * (1.0 + y_q))

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

            # Add time embedding when enabled
            if self.time_mlp is not None and time_emb is not None:
                t = self.time_mlp(time_emb).view(B, -1, 1, 1)
                scale, shift = torch.chunk(t, 2, dim=1)
                h_c = h_c * (1 + scale) + shift
            h_c = F.silu(h_c)

            # Second convolution
            h_c = self.conv1(h_c)
            h_c = self.norm1(h_c)
            h_c = F.silu(h_c)
            
            outputs.append(x_skip[:, qc_actual:qc_actual + h_c.shape[1]] + h_c)

        # Combine quantum and classical outputs
        if outputs:
            out = torch.cat(outputs, dim=1)  # [B, out_dim, H, W]
            if out.shape[1] < x_skip.shape[1]:
                out = torch.cat([out, x_skip[:, out.shape[1]:]], dim=1)
            return out
        return x_skip


class AblationResnetBlock(QResnetBlock):
    """Bottleneck residual block used by the parameter-matched classical ablation.

    Applies the same skip_modulation ('affine': x_skip[:, :qc] * (1 + y_q),
    or 'additive': x_skip[:, :qc] + y_q) as QResnetBlock when paired with a
    ClassicalBottleneckBlock.
    """

    def forward(self, x, time_emb=None, params=None):
        return super().forward(x, time_emb, params)


# --- U-Net Components ---
class DownUnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8,
        use_time_emb: bool = True,
        backbone_conv_mode: str = DEFAULT_BACKBONE_CONV_MODE,
    ):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        self.use_time_emb = bool(use_time_emb)
        self.backbone_conv_mode = normalize_backbone_conv_mode(backbone_conv_mode)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(3, self.init_dim, 7, padding=3)  # Assuming 3 input channels
        
        # Time embedding
        if self.use_time_emb:
            self.time_emb = SinusoidalPosEmb(dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim * 4)
            )
        else:
            self.time_emb = None
            self.time_mlp = None
        
        # Downsampling blocks
        self.downs = nn.ModuleList()
        dims = [self.init_dim] + [dim * m for m in dim_mults]
        
        for i in range(len(dim_mults)):
            dim_in = dims[i]
            dim_out = dims[i + 1]
            
            self.downs.append(nn.ModuleList([
                ResnetBlock(
                    dim_in, dim_in, dim * 4, resnet_block_groups,
                    use_time_emb=self.use_time_emb,
                    backbone_conv_mode=self.backbone_conv_mode,
                ),
                ResnetBlock(
                    dim_in, dim_in, dim * 4, resnet_block_groups,
                    use_time_emb=self.use_time_emb,
                    backbone_conv_mode=self.backbone_conv_mode,
                ),
                AttnBlock(dim_in),
                Downsample(dim_in, dim_out) if i < len(dim_mults) - 1 else nn.Identity()
            ]))
        
        # Final conv
        mid_dim = dim * dim_mults[-1]
        self.final_conv = nn.Conv2d(dims[-2], mid_dim, 3, padding=1)
    
    def forward(self, x, time=None):
        # Initial conv
        h = self.init_conv(x)
        hs = [h]
        
        # Time embedding
        if self.use_time_emb and self.time_emb is not None and self.time_mlp is not None:
            if time is None:
                time = torch.zeros((x.shape[0],), dtype=torch.float32, device=x.device)
            time_emb = self.time_emb(time)
            time_emb = self.time_mlp(time_emb)
        else:
            time_emb = None
        
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
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8,
        use_time_emb: bool = True,
        backbone_conv_mode: str = DEFAULT_BACKBONE_CONV_MODE,
    ):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.out_dim = out_dim or 3  # Assuming 3 output channels
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        self.use_time_emb = bool(use_time_emb)
        self.backbone_conv_mode = normalize_backbone_conv_mode(backbone_conv_mode)
        
        # Upsampling blocks
        self.ups = nn.ModuleList()
        dims = [self.init_dim] + [dim * m for m in dim_mults]
        
        for i in reversed(range(len(dim_mults))):
            dim_in = dims[i + 1]
            dim_out = dims[i]
            
            # For skip connections, we need to handle concatenated dimensions
            self.ups.append(nn.ModuleList([
                ResnetBlock(
                    dim_in + dim_out, dim_in, dim * 4, resnet_block_groups,
                    use_time_emb=self.use_time_emb,
                    backbone_conv_mode=self.backbone_conv_mode,
                ),
                ResnetBlock(
                    dim_in + dim_out, dim_in, dim * 4, resnet_block_groups,
                    use_time_emb=self.use_time_emb,
                    backbone_conv_mode=self.backbone_conv_mode,
                ),
                AttnBlock(dim_in),
                Upsample(dim_in, dim_out) if i > 0 else nn.Identity()
            ]))
        
        # Final processing
        self.final_conv1 = nn.Conv2d(dims[1], self.init_dim, 3, padding=1)
        self.final_resblock = ResnetBlock(
            self.init_dim * 2, self.dim, self.dim * 4, resnet_block_groups,
            use_time_emb=self.use_time_emb,
            backbone_conv_mode=self.backbone_conv_mode,
        )
        self.final_conv2 = nn.Conv2d(self.dim, self.out_dim, 1)
    
    def forward(self, x, h, hs, time_emb=None):
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


def matched_lowrank_dim(effective_qc, target_total_params):
    """Rank r for Conv2d(qc, r, 1) + Tanh + Conv2d(r, qc, 1) + Tanh
    ((2 * qc + 1) * r + qc params) closest to target_total_params."""
    if effective_qc <= 0:
        return 1
    denom = max(1, 2 * effective_qc + 1)
    return max(1, int(round((target_total_params - effective_qc) / float(denom))))


class ClassicalBottleneckBlock(nn.Module):
    """Drop-in classical replacement for QuantumBlock for parameter-matched ablation.

    Supports:
      - kind in ('linear', 'mlp') with spatial_mode in ('global', 'pixel', 'patch2x2', 'patch4x4', 'se_gate') (#13)
      - kind in ('conv1x1', 'depthwise3x3', 'lowrank') spatial classical bottleneck baselines (#13)
      - kind == 'hqunet_spatial' parameter-matched 2D spatial grid control for q_ansatz='hqunet' (#18)
    """
    def __init__(
        self,
        quantum_channels=8,
        kind='linear',
        hidden=None,
        seed=None,
        q_ansatz=None,
        q_layers=None,
        spatial_mode=DEFAULT_Q_SPATIAL_MODE,
        ablation_mode=None,
        hqunet_grid=DEFAULT_HQUNET_GRID,
        lowrank_rank=None,
    ):
        super().__init__()
        eff_ablation = normalize_ablation_mode(ablation_mode) if ablation_mode is not None else None
        if q_ansatz is not None:
            if q_ansatz not in Q_ANSATZ_CHOICES:
                raise ValueError(f"Unknown q_ansatz {q_ansatz!r}, expected one of {Q_ANSATZ_CHOICES}")
            if eff_ablation in (None, 'auto'):
                if q_ansatz == 'hqunet':
                    kind = 'hqunet_spatial'
                elif q_ansatz == 'reupload':
                    kind = 'mlp'
                else:
                    kind = 'linear'
            else:
                kind = eff_ablation
            if hidden is None:
                hidden = matched_mlp_hidden(16 * int(q_layers or 2))
        elif eff_ablation not in (None, 'auto'):
            kind = eff_ablation

        if kind not in ('linear', 'mlp', 'conv1x1', 'depthwise3x3', 'lowrank', 'hqunet_spatial'):
            raise ValueError(f"Unknown classical bottleneck kind {kind!r}")
        self.quantum_channels = quantum_channels
        self.num_groups = quantum_channels // 4
        self.remaining_channels = quantum_channels % 4
        self.effective_qc = self.num_groups * 4
        self.kind = kind
        self.q_ansatz = q_ansatz
        self.q_layers = int(q_layers or 2)
        self.spatial_mode = normalize_q_spatial_mode(spatial_mode)
        self.hqunet_grid = normalize_hqunet_grid(hqunet_grid)
        self.hidden = hidden if hidden is not None else 3
        target_per_group = 4 if q_ansatz == 'legacy' else (16 * self.q_layers)
        self.lowrank_rank = (
            int(lowrank_rank)
            if lowrank_rank is not None
            else matched_lowrank_dim(self.effective_qc, self.num_groups * target_per_group)
        )
        self.quantum_mode = 'on'
        self._random_groups = None
        self._random_lowrank = None
        self._random_hqunet_weights = None
        self.per_group = None
        self.lowrank_net = None
        self.pre_q_conv = None
        self.post_q_proj = None
        self.hqunet_classical_weights = None

        with torch.random.fork_rng(devices=[], enabled=seed is not None):
            if seed is not None:
                torch.default_generator.manual_seed(int(seed) + QUANTUM_INIT_SEED_OFFSET)
            if self.kind == 'hqunet_spatial':
                G = self.hqunet_grid
                n_q = G * G
                if self.effective_qc > 0:
                    self.pre_q_conv = nn.Conv2d(self.effective_qc, 3, kernel_size=1)
                    self.post_q_proj = nn.Linear(2 * n_q, self.effective_qc * n_q)
                    gen = None
                    if seed is not None:
                        gen = torch.Generator()
                        gen.manual_seed(int(seed) + QUANTUM_INIT_SEED_OFFSET)
                    # Match shared horizontal/vertical filter count: (q_layers, 2, 4) = 8 * L params
                    self.hqunet_classical_weights = nn.Parameter(
                        _uniform((self.q_layers, 2, 4), -1.0, 1.0, gen)
                    )
            elif self.kind == 'lowrank':
                if self.effective_qc > 0:
                    self.lowrank_net = self._make_lowrank()
            else:
                self.per_group = nn.ModuleList([self._make_group() for _ in range(self.num_groups)])

    def _make_lowrank(self):
        r = self.lowrank_rank
        return nn.Sequential(
            nn.Conv2d(self.effective_qc, r, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(r, self.effective_qc, kernel_size=1),
            nn.Tanh(),
        )

    def _make_group(self):
        if self.kind == 'linear':
            return nn.Linear(4, 4)
        if self.kind == 'mlp':
            return nn.Sequential(
                nn.Linear(4, self.hidden), nn.Tanh(),
                nn.Linear(self.hidden, 4), nn.Tanh(),
            )
        if self.kind == 'conv1x1':
            if self.q_ansatz == 'legacy':
                return nn.Conv2d(4, 4, kernel_size=1)
            return nn.Sequential(
                nn.Conv2d(4, self.hidden, kernel_size=1), nn.Tanh(),
                nn.Conv2d(self.hidden, 4, kernel_size=1), nn.Tanh(),
            )
        if self.kind == 'depthwise3x3':
            return nn.Sequential(
                nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
                nn.Tanh(),
                nn.Conv2d(4, 4, kernel_size=1),
                nn.Tanh(),
            )
        raise ValueError(f"Unsupported per-group kind {self.kind!r}")

    def set_quantum_mode(self, mode='on', knockout_seed=0, seed=None):
        """Same knockout switch as QuantumBlock: 'on', 'zero', or 'random'."""
        if seed is not None:
            knockout_seed = seed
        if mode not in QUANTUM_MODES:
            raise ValueError(f"Unknown quantum mode {mode!r}, expected one of {QUANTUM_MODES}")
        self.quantum_mode = mode
        self._random_groups = None
        self._random_lowrank = None
        self._random_hqunet_weights = None
        if mode == 'random':
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(int(knockout_seed) + KNOCKOUT_SEED_OFFSET)
                if self.kind == 'hqunet_spatial':
                    g = torch.Generator()
                    g.manual_seed(int(knockout_seed) + KNOCKOUT_SEED_OFFSET)
                    self._random_hqunet_weights = _uniform(
                        (self.q_layers, 2, 4), -1.0, 1.0, g
                    )
                elif self.kind == 'lowrank':
                    if self.effective_qc > 0:
                        # Wrap in a plain list so nn.Module.__setattr__ does not register
                        # a temporary submodule during random knockout evaluation.
                        self._random_lowrank = [self._make_lowrank()]
                else:
                    self._random_groups = [self._make_group() for _ in range(self.num_groups)]
        return self

    def _eval_hqunet_classical(self, angles, weights):
        """Classical 2D spatial grid mixing counterpart to HQ-UNet's shared F_R / F_C circuit (#18).
        Maps [B, N_q, 3] -> [B, 2 * N_q] using the exact same parameter count (8 * L).
        """
        B, n_q, _ = angles.shape
        G = self.hqunet_grid
        row_edges, col_edges = hqunet_grid_edges(G)
        all_edges = row_edges + col_edges
        shared = weights.shape[1] == 2
        h0 = torch.zeros((B, n_q), dtype=angles.dtype, device=angles.device)
        h1 = torch.zeros((B, n_q), dtype=angles.dtype, device=angles.device)
        f1, f2, f3 = angles[:, :, 0], angles[:, :, 1], angles[:, :, 2]

        for l in range(self.q_layers):
            h0 = torch.tanh(h0 + torch.sin(f1) + 0.5 * torch.cos(f3))
            h1 = torch.tanh(h1 + torch.sin(f2) - 0.5 * torch.sin(f3))
            cols0 = list(torch.unbind(h0, dim=1))
            cols1 = list(torch.unbind(h1, dim=1))
            if shared:
                w_r = weights[l, 0]
                for q1, q2 in row_edges:
                    u1 = torch.tanh(w_r[0] * cols0[q1] + w_r[1] * cols0[q2])
                    u2 = torch.tanh(w_r[2] * cols1[q1] + w_r[3] * cols1[q2])
                    cols0[q1] = u1
                    cols1[q2] = u2
                w_c = weights[l, 1]
                for q1, q2 in col_edges:
                    u1 = torch.tanh(w_c[0] * cols0[q1] + w_c[1] * cols0[q2])
                    u2 = torch.tanh(w_c[2] * cols1[q1] + w_c[3] * cols1[q2])
                    cols0[q1] = u1
                    cols1[q2] = u2
            else:
                for e, (q1, q2) in enumerate(all_edges):
                    w = weights[l, e]
                    u1 = torch.tanh(w[0] * cols0[q1] + w[1] * cols0[q2])
                    u2 = torch.tanh(w[2] * cols1[q1] + w[3] * cols1[q2])
                    cols0[q1] = u1
                    cols1[q2] = u2
            h0 = torch.stack(cols0, dim=1)
            h1 = torch.stack(cols1, dim=1)
        return torch.cat([h0, h1], dim=1)

    def _eval_tokens_groups(self, tokens, groups):
        """Apply per-group Linear/MLP modules to `tokens` of shape [N, effective_qc]."""
        outputs = []
        for i, layer in enumerate(groups):
            g = tokens[:, i * 4 : (i + 1) * 4]
            outputs.append(layer(g))
        return torch.cat(outputs, dim=1)

    def forward(self, x, params=None):
        if self.effective_qc == 0:
            return x
        B, _, H, W = x.shape
        x_eff = x[:, :self.effective_qc]

        if self.kind == 'hqunet_spatial':
            G = self.hqunet_grid
            n_q = G * G
            if self.quantum_mode == 'zero':
                z = torch.zeros(B, 2 * n_q, dtype=x.dtype, device=x.device)
            else:
                w = (
                    self._random_hqunet_weights.to(x.device)
                    if self.quantum_mode == 'random'
                    else self.hqunet_classical_weights
                )
                x_grid = F.adaptive_avg_pool2d(x_eff, (G, G))
                angles = math.pi * torch.tanh(self.pre_q_conv(x_grid)).permute(0, 2, 3, 1).reshape(B, n_q, 3)
                z = self._eval_hqunet_classical(angles, w)
            y_grid = self.post_q_proj(z).reshape(B, self.effective_qc, G, G)
            if (G, G) != (H, W):
                return F.interpolate(y_grid, size=(H, W), mode='bilinear', align_corners=False)
            return y_grid

        if self.quantum_mode == 'zero':
            return torch.zeros(B, self.effective_qc, H, W, dtype=x.dtype, device=x.device)

        if self.kind == 'lowrank':
            net = self._random_lowrank[0].to(x.device) if self.quantum_mode == 'random' else self.lowrank_net
            return net(x_eff)

        groups = self.per_group
        if self.quantum_mode == 'random':
            groups = [g.to(x.device) for g in self._random_groups]

        if self.kind in ('conv1x1', 'depthwise3x3'):
            outputs = []
            for i, layer in enumerate(groups):
                g_map = x_eff[:, i * 4 : (i + 1) * 4]
                outputs.append(layer(g_map))
            return torch.cat(outputs, dim=1)

        # kind in ('linear', 'mlp') with spatial_mode support (#13)
        mode = self.spatial_mode
        if mode == 'global':
            pooled = F.adaptive_avg_pool2d(x_eff, 1).squeeze(-1).squeeze(-1)  # [B, effective_qc]
            y_tokens = self._eval_tokens_groups(pooled, groups)
            return y_tokens.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        if mode == 'se_gate':
            pooled = F.adaptive_avg_pool2d(x_eff, 1).squeeze(-1).squeeze(-1)
            y_tokens = self._eval_tokens_groups(pooled, groups)
            gate = torch.sigmoid(y_tokens.unsqueeze(-1).unsqueeze(-1))
            return x_eff * gate
        if mode == 'pixel':
            tokens = x_eff.permute(0, 2, 3, 1).reshape(B * H * W, self.effective_qc)
            y_tokens = self._eval_tokens_groups(tokens, groups)
            return y_tokens.reshape(B, H, W, self.effective_qc).permute(0, 3, 1, 2).contiguous()
        if mode in ('patch2x2', 'patch4x4'):
            P = 2 if mode == 'patch2x2' else 4
            x_patch = F.adaptive_avg_pool2d(x_eff, (P, P))
            tokens = x_patch.permute(0, 2, 3, 1).reshape(B * P * P, self.effective_qc)
            y_tokens = self._eval_tokens_groups(tokens, groups)
            y_patch = y_tokens.reshape(B, P, P, self.effective_qc).permute(0, 3, 1, 2).contiguous()
            if (P, P) != (H, W):
                return F.interpolate(y_patch, size=(H, W), mode='bilinear', align_corners=False)
            return y_patch
        raise ValueError(f"Unknown spatial_mode {mode!r}")


class QVertex(nn.Module):
    def __init__(
        self,
        dim,
        quantum_block: QuantumBlock,
        quantum_channels=8,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8,
        quantum_block2=None,
        skip_modulation=DEFAULT_SKIP_MODULATION,
        use_time_emb: bool = True,
        backbone_conv_mode: str = DEFAULT_BACKBONE_CONV_MODE,
    ):
        super().__init__()
        self.dim = dim
        self.quantum_channels = quantum_channels
        self.resnet_block_groups = resnet_block_groups
        self.skip_modulation = normalize_skip_modulation(skip_modulation)
        self.use_time_emb = bool(use_time_emb)
        self.backbone_conv_mode = normalize_backbone_conv_mode(backbone_conv_mode)
        self.shared_quantum_block = quantum_block2 is None
        quantum_block2 = quantum_block if quantum_block2 is None else quantum_block2
        
        mid_dim = dim * dim_mults[-1]
        time_emb_dim = dim * 4  # Match the main model's time embedding dimension
        block_cls1 = AblationResnetBlock if isinstance(quantum_block, ClassicalBottleneckBlock) else QResnetBlock
        block_cls2 = AblationResnetBlock if isinstance(quantum_block2, ClassicalBottleneckBlock) else QResnetBlock
        self.qresblock1 = block_cls1(
            mid_dim, mid_dim, time_emb_dim, quantum_block, quantum_channels,
            resnet_block_groups, skip_modulation=self.skip_modulation,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )
        self.attn = AttnBlock(mid_dim, use_linear_attention=False)
        self.qresblock2 = block_cls2(
            mid_dim, mid_dim, time_emb_dim, quantum_block2, quantum_channels,
            resnet_block_groups, skip_modulation=self.skip_modulation,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )
    
    @property
    def q_block(self):
        return self.qresblock1.quantum_block

    def forward(self, h, time_emb=None, params=None):
        h = self.qresblock1(h, time_emb, params)
        h = self.attn(h)
        h = self.qresblock2(h, time_emb, params)
        return h, time_emb

class QVUNet(nn.Module):
    def __init__(
        self,
        dim,
        quantum_block: QuantumBlock,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8,
        quantum_channels=8,
        quantum_block2=None,
        skip_modulation=DEFAULT_SKIP_MODULATION,
        use_time_emb: bool = True,
        backbone_conv_mode: str = DEFAULT_BACKBONE_CONV_MODE,
    ):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.out_dim = out_dim or 2  # Binary segmentation
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        self.quantum_channels = quantum_channels
        self.skip_modulation = normalize_skip_modulation(skip_modulation)
        self.use_time_emb = bool(use_time_emb)
        self.backbone_conv_mode = normalize_backbone_conv_mode(backbone_conv_mode)
        
        self.down = DownUnet(
            dim, init_dim, dim_mults, resnet_block_groups,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )
        self.vertex = QVertex(
            dim, quantum_block, quantum_channels, dim_mults, resnet_block_groups,
            quantum_block2=quantum_block2, skip_modulation=self.skip_modulation,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )
        self.up = UpUnet(
            dim, init_dim, out_dim, dim_mults, resnet_block_groups,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )

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
    
    def forward(self, x, time=None, params=None):
        h, hs, time_emb = self.down(x, time)
        h, time_emb = self.vertex(h, time_emb, params)
        h = self.up(x, h, hs, time_emb)
        return h 

# --- Classical U-Net Components ---

class Vertex(nn.Module):
    """Classical vertex (bottleneck) of the U-Net."""
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8,
        use_time_emb: bool = True,
        backbone_conv_mode: str = DEFAULT_BACKBONE_CONV_MODE,
    ):
        super().__init__()
        self.dim = dim
        self.resnet_block_groups = resnet_block_groups
        self.use_time_emb = bool(use_time_emb)
        self.backbone_conv_mode = normalize_backbone_conv_mode(backbone_conv_mode)
        
        mid_dim = dim * dim_mults[-1]
        time_emb_dim = dim * 4  # Match the main model's time embedding dimension
        
        self.resblock1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim, resnet_block_groups,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )
        self.attn = AttnBlock(mid_dim, use_linear_attention=False)
        self.resblock2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim, resnet_block_groups,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )
    
    def forward(self, h, time_emb=None):
        h = self.resblock1(h, time_emb)
        h = self.attn(h)
        h = self.resblock2(h, time_emb)
        return h, time_emb

class UNet(nn.Module):
    """Classical U-Net architecture."""
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8,
        use_time_emb: bool = True,
        backbone_conv_mode: str = DEFAULT_BACKBONE_CONV_MODE,
    ):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.out_dim = out_dim or 3  # Default to 3 output channels
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        self.use_time_emb = bool(use_time_emb)
        self.backbone_conv_mode = normalize_backbone_conv_mode(backbone_conv_mode)
        
        self.down = DownUnet(
            dim, init_dim, dim_mults, resnet_block_groups,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )
        self.vertex = Vertex(
            dim, dim_mults, resnet_block_groups,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )
        self.up = UpUnet(
            dim, init_dim, out_dim, dim_mults, resnet_block_groups,
            use_time_emb=self.use_time_emb, backbone_conv_mode=self.backbone_conv_mode,
        )
    
    def forward(self, x, time=None):
        h, hs, time_emb = self.down(x, time)
        h, time_emb = self.vertex(h, time_emb)
        h = self.up(x, h, hs, time_emb)
        return h
 
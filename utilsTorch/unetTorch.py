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
                         seed=None, q_ansatz='reupload', q_layers=2):
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

    Returns:
        QuantumBlock instance
    """
    return QuantumBlock(backend=backend, quantum_channels=quantum_channels,
                        ptlayer_config=ptlayer_config, seed=seed,
                        q_ansatz=q_ansatz, q_layers=q_layers)


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

    Knockout: set_quantum_mode('on' | 'zero' | 'random') switches the forward
    pass between the trained circuit, an all-zero output, or the same circuit
    with freshly drawn random parameters (fixed knockout seed). The trained
    parameters and state_dict are not modified.
    """
    def __init__(self, backend='ptlayer', quantum_channels=8, ptlayer_config=None,
                 seed=None, q_ansatz='reupload', q_layers=2):
        super().__init__()

        if q_ansatz not in Q_ANSATZ_CHOICES:
            raise ValueError(f"Unknown q_ansatz {q_ansatz!r}, expected one of {Q_ANSATZ_CHOICES}")
        if q_ansatz == 'reupload' and int(q_layers) < 1:
            raise ValueError("q_layers must be >= 1")

        self.backend = backend
        self.quantum_channels = quantum_channels
        self.q_ansatz = q_ansatz
        self.q_layers = int(q_layers) if q_ansatz == 'reupload' else 1
        self.seed = seed
        self.quantum_mode = 'on'
        self._random_params = None
        
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
            if not HAS_PENNYLANE:
                raise ImportError("PennyLane not available. Please install PennyLane or use 'pennylane' backend.")
            
            # Create circuits for each 4-channel group
            self.pennylane_circuits = []
            self.quantum_params = nn.ParameterList()
            # Learnable per-layer input scales (reupload only)
            self.input_scales = nn.ParameterList()
            
            for group_idx in range(self.num_groups):
                if q_ansatz == 'legacy':
                    circuit = pennylane_default_circuit(n_qubits=N_QUBITS)
                    # Same distribution as the original model (N(0, 0.1^2)), but
                    # drawn from the seeded local generator or the global RNG.
                    params = nn.Parameter(_randn((N_QUBITS,), generator) * 0.1)
                    self.quantum_params.append(params)
                else:
                    circuit = pennylane_reupload_circuit(n_qubits=N_QUBITS, n_layers=self.q_layers)
                    weights = nn.Parameter(_uniform((self.q_layers, N_QUBITS, 3), -math.pi, math.pi, generator))
                    self.quantum_params.append(weights)
                    self.input_scales.append(nn.Parameter(torch.ones(self.q_layers, N_QUBITS)))
                self.pennylane_circuits.append(circuit)
                
        else:
            raise ValueError(f'Unknown quantum backend: {backend}')

    @property
    def n_params_per_group(self):
        if self.backend != 'pennylane':
            return None
        if self.q_ansatz == 'legacy':
            return N_QUBITS
        return self.q_layers * N_QUBITS * 4  # 3 rotation angles + 1 input scale per qubit per layer

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
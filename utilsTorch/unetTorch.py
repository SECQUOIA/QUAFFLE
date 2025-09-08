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
        kernel = self.conv.weight
        mean = kernel.mean(dim=(1,2,3), keepdim=True)
        var = kernel.var(dim=(1,2,3), keepdim=True)
        kernel = (kernel - mean) / torch.sqrt(var + 1e-5)
        self.conv.weight.data = kernel
        x = self.conv(x)
        return x

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
def create_quantum_block(backend='ptlayer', quantum_channels=8, ptlayer_config=None):
    """
    Create a QuantumBlock that processes channels in groups of 4.
    
    Args:
        backend: 'ptlayer' or 'pennylane'
        quantum_channels: Number of quantum channels (will be rounded down to multiple of 4)
        ptlayer_config: Configuration dict for PTLayer (if backend='ptlayer')
    
    Returns:
        QuantumBlock instance
    """
    return QuantumBlock(backend=backend, quantum_channels=quantum_channels, ptlayer_config=ptlayer_config)

def pennylane_default_circuit(n_qubits=4):
    """Create PennyLane circuit for processing 4-channel groups."""
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

class QuantumBlock(nn.Module):
    """
    Quantum processing block that handles channels in groups of 4.
    
    This implementation follows the JAX approach where channels are partitioned
    into groups of 4 and each group is processed by a separate quantum circuit.
    No padding is used - only complete groups of 4 are processed quantum mechanically.
    
    Args:
        backend: 'ptlayer' or 'pennylane'
        quantum_channels: Total number of quantum channels (will process quantum_channels//4 groups)
        ptlayer_config: Configuration for PTLayer backend
    """
    def __init__(self, backend='ptlayer', quantum_channels=8, ptlayer_config=None):
        super().__init__()
        
        # Set deterministic seed for quantum block initialization
        torch.manual_seed(42)
        
        self.backend = backend
        self.quantum_channels = quantum_channels
        
        # Calculate number of 4-channel groups
        self.num_groups = quantum_channels // 4
        self.remaining_channels = quantum_channels % 4
        
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
                
                # Set seed for deterministic PTLayer initialization
                torch.manual_seed(42)
                    
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
            
            # Set seed for reproducible random initialization
            torch.manual_seed(42) 
            
            for group_idx in range(self.num_groups):
                circuit = pennylane_default_circuit(n_qubits=4)
                self.pennylane_circuits.append(circuit)
                # Initialize parameters with small random values for better quantum behavior
                # Use deterministic initialization based on group index for reproducibility
                param_generator = torch.Generator()
                param_generator.manual_seed(42 + group_idx)  # Deterministic seed per group
                params = nn.Parameter(torch.randn(4, generator=param_generator) * 0.1)
                self.quantum_params.append(params)
                
        else:
            raise ValueError(f'Unknown quantum backend: {backend}')
    
    
    def forward(self, x, params=None):
        B, C, H, W = x.shape
        
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
                
                # Process each batch item for this group
                group_outputs = []
                for b in range(B):
                    x_item = x_group[b:b+1]  # [1, 4, H, W]
                    
                    y_item = self.pennylane_circuits[i](x_item, self.quantum_params[i])
                    y_tensor = torch.stack(y_item).unsqueeze(0)  # [1, 4]
                    group_outputs.append(y_tensor)
                
                # Stack batch outputs for this group
                y_group = torch.cat(group_outputs, dim=0)  # [B, 4]
                
                # Reshape to spatial dimensions
                y_group = y_group.unsqueeze(-1).unsqueeze(-1)  # [B, 4, 1, 1]
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
    
class QVertex(nn.Module):
    def __init__(self, dim, quantum_block: QuantumBlock, quantum_channels=8, 
                 dim_mults=(1, 2, 4, 8), resnet_block_groups=8):
        super().__init__()
        self.dim = dim
        self.quantum_channels = quantum_channels
        self.resnet_block_groups = resnet_block_groups
        
        mid_dim = dim * dim_mults[-1]
        time_emb_dim = dim * 4  # Match the main model's time embedding dimension
        self.qresblock1 = QResnetBlock(mid_dim, mid_dim, time_emb_dim, quantum_block, quantum_channels, resnet_block_groups)
        self.attn = AttnBlock(mid_dim, use_linear_attention=False)
        self.qresblock2 = QResnetBlock(mid_dim, mid_dim, time_emb_dim, quantum_block, quantum_channels, resnet_block_groups)
    
    def forward(self, h, time_emb, params=None):
        h = self.qresblock1(h, time_emb, params)
        h = self.attn(h)
        h = self.qresblock2(h, time_emb, params)
        return h, time_emb

class QVUNet(nn.Module):
    def __init__(self, dim, quantum_block: QuantumBlock, init_dim=None, out_dim=None, 
                 dim_mults=(1, 2, 4, 8), resnet_block_groups=8, quantum_channels=8):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.out_dim = out_dim or 2  # Binary segmentation
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        self.quantum_channels = quantum_channels
        
        self.down = DownUnet(dim, init_dim, dim_mults, resnet_block_groups)
        self.vertex = QVertex(dim, quantum_block, quantum_channels, dim_mults, resnet_block_groups)
        self.up = UpUnet(dim, init_dim, out_dim, dim_mults, resnet_block_groups)
    
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
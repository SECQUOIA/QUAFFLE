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
def create_quantum_block(backend='ptlayer', quantum_channels=2, ptlayer_config=None):
    """
    Helper function to create a QuantumBlock with internal PTLayer/PennyLane handling.
    
    Args:
        backend: 'ptlayer' or 'pennylane'
        quantum_channels: Number of quantum channels (also determines modes)
        ptlayer_config: Configuration dict for PTLayer (if backend='ptlayer')
                       Default: {'n_loops': 2, 'n_samples': 200, 'observable': 'mean'}
    
    Returns:
        QuantumBlock instance
    """
    return QuantumBlock(backend=backend, quantum_channels=quantum_channels, ptlayer_config=ptlayer_config)

def pennylane_default_circuit(quantum_channels, n_qubits=None):
    n_qubits = 8  # Fixed to 8 qubits regardless of quantum_channels
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def circuit(x, params=None):
        # x: [B, C, H, W] -> process each batch item separately
        if len(x.shape) == 4:  # [B, C, H, W]
            x_processed = x.mean(dim=(2, 3))  # [B, C]
        elif len(x.shape) == 3:  # [B, C, 1] or similar
            x_processed = x.squeeze(-1)  # [B, C]
        else:
            x_processed = x
        
        # Process first batch item
        x_item = x_processed[0]  # [C]
        
        if params is None:
            params = torch.zeros(n_qubits, dtype=x_item.dtype, device=x_item.device)
            print("Quantum parameters are not being used")
            
        # Ensure we have the right number of qubits
        x_padded = torch.zeros(n_qubits, dtype=x_item.dtype, device=x_item.device)
        x_padded[:min(len(x_item), n_qubits)] = x_item[:min(len(x_item), n_qubits)]
        
        for i in range(n_qubits):
            qml.RX(x_padded[i], wires=i)
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit

class QuantumBlock(nn.Module):
    def __init__(self, backend='ptlayer', quantum_channels=8, ptlayer_config=None):
        super().__init__()
        self.backend = backend
        self.quantum_channels = quantum_channels
        
        if backend == 'ptlayer':
            if not HAS_PT:
                raise ImportError("PTLayer not available. Please install ORCA PTLayer or use 'pennylane' backend.")
            
            # Set default PTLayer configuration
            ptlayer_config = {
                'n_loops': 2,
                'n_samples': 200,
                'observable': 'mean'
            }
            
            try:
                # Create TBI object
                self.tbi = create_tbi(n_loops=ptlayer_config.get('n_loops', 2))
                
                # Define input state based on quantum channels
                # Use alternating pattern for photonic states
                self.input_state = tuple([1 if i % 2 == 0 else 0 for i in range(quantum_channels)])
                self.m_modes = len(self.input_state)
                
                # Calculate required number of parameters
                self.n_params = self.tbi.calculate_n_beam_splitters(self.m_modes)
                
                # Create PTLayer with correct API
                self.ptlayer = PTLayer(
                    input_state=self.input_state,
                    in_features=self.n_params,
                    tbi=self.tbi,
                    observable=ptlayer_config.get('observable', 'mean'),
                    n_samples=ptlayer_config.get('n_samples', 200)
                )
                
                # Create parameter projection layer to map quantum_channels to n_params
                self.param_projection = nn.Linear(quantum_channels, self.n_params)
                
                print(f"PTLayer initialized: {self.m_modes} modes, {self.n_params} parameters, input_state={self.input_state}")
                
            except Exception as e:
                print(f"Warning: Could not initialize PTLayer: {e}")
                self.ptlayer = None
                self.tbi = None
                
        elif backend == 'pennylane':
            if not HAS_PENNYLANE:
                raise ImportError("PennyLane not available. Please install PennyLane or use 'pennylane' backend.")
            self.pennylane_circuit = pennylane_default_circuit(quantum_channels)
            self.ptlayer = None
            self.tbi = None
            
            n_qubits = 8  # Fixed architecture
            self.quantum_params = nn.Parameter(
                torch.randn(n_qubits) * 0.1  # Small random initialization
            )
        else:
            raise ValueError(f'Unknown quantum backend: {backend}')
    
    
    def forward(self, x, params=None):
        B, C, H, W = x.shape
        
        if self.backend == 'ptlayer':
            if self.ptlayer is None:
                raise ValueError('PTLayer instance not initialized for PTLayer backend')
            
            # Convert spatial features to parameter vectors
            # Pool spatial dimensions to get per-channel features
            x_pooled = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
            
            # Project to PTLayer parameter space
            theta = self.param_projection(x_pooled)  # [B, n_params]
            
            # Apply PTLayer
            y = self.ptlayer(theta)  # [B, m_modes]
            
            # Reshape output to spatial dimensions
            y = y.unsqueeze(-1).unsqueeze(-1)  # [B, m_modes, 1, 1]
            y = y.expand(-1, -1, H, W)  # [B, m_modes, H, W]
            
            # Pad or truncate to match quantum_channels
            if y.shape[1] < self.quantum_channels:
                padding = torch.zeros(B, self.quantum_channels - y.shape[1], H, W, 
                                    dtype=y.dtype, device=y.device)
                y = torch.cat([y, padding], dim=1)
            elif y.shape[1] > self.quantum_channels:
                y = y[:, :self.quantum_channels]
            
            return y.float()
            
        elif self.backend == 'pennylane':
            if self.pennylane_circuit is None:
                raise ValueError('PennyLane circuit not initialized for PennyLane backend')
            # Process each batch item
            outputs = []
            for i in range(B):
                x_item = x[i:i+1]  # [1, C, H, W]
                
                y_item = self.pennylane_circuit(x_item, self.quantum_params)
                    
                y_tensor = torch.stack(y_item).unsqueeze(0)  # [1, n_qubits]
                outputs.append(y_tensor)
            
            # Stack outputs and reshape to match input spatial dimensions
            y = torch.cat(outputs, dim=0)  # [B, n_qubits] where n_qubits=8 (fixed)
            
            # Reshape to [B, C, H, W] by broadcasting
            y = y.unsqueeze(-1).unsqueeze(-1)  # [B, 8, 1, 1]
            y = y.expand(-1, -1, H, W)  # [B, 8, H, W]
            
            # Pad or truncate 8-qubit output to match quantum_channels
            if y.shape[1] < self.quantum_channels:  # quantum_channels > 8: pad with zeros
                padding = torch.zeros(B, self.quantum_channels - y.shape[1], H, W, 
                                    dtype=y.dtype, device=y.device)
                y = torch.cat([y, padding], dim=1)
            elif y.shape[1] > self.quantum_channels:  # quantum_channels < 8: truncate
                y = y[:, :self.quantum_channels]
            
            return y.float()  # Ensure output is float
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
        self.qc = quantum_channels
        self.oc = out_dim - quantum_channels   # classical channels that stay conv-based
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.quantum_block = quantum_block

        # Classical path (reduced-width)
        self.conv0 = WeightStandardizedConv(in_dim - self.qc, self.oc)
        self.norm0 = nn.GroupNorm(groups, self.oc)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 2 * self.oc),
            nn.SiLU(),
        )

        self.conv1 = WeightStandardizedConv(self.oc, self.oc)
        self.norm1 = nn.GroupNorm(groups, self.oc)

        # Skip path must produce the full out_dim (oc + qc)
        self.skip = (
            nn.Conv2d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x, time_emb, params=None):
        B, C, H, W = x.shape

        # --- Split channels: first qc → quantum, rest → classical
        qc = min(self.qc, C)
        x_q = x[:, :qc]                          # quantum slice
        x_c = x[:, qc:] if C > qc else None      # classical slice

        # --- Quantum branch
        y_q = self.quantum_block(x_q, params)    # [B, qc', H, W]
        if y_q.shape[2:] != (H, W):
            y_q = F.interpolate(y_q, size=(H, W), mode='nearest')

        # Ensure quantum output has exactly qc channels
        if y_q.shape[1] < qc:
            pad = torch.zeros(B, qc - y_q.shape[1], H, W, device=y_q.device, dtype=y_q.dtype)
            y_q = torch.cat([y_q, pad], dim=1)
        elif y_q.shape[1] > qc:
            y_q = y_q[:, :qc]

        # --- Classical branch (reduced conv width)
        if x_c is None or x_c.shape[1] == 0:
            # no classical slice left; just use zeros for conv path
            h_c_in = torch.zeros(B, 0, H, W, device=x.device, dtype=x.dtype)
        else:
            h_c_in = x_c

        # conv0 path
        h_c = self.conv0(h_c_in)
        h_c = self.norm0(h_c)

        # time embedding
        t = self.time_mlp(time_emb).view(B, -1, 1, 1)
        scale, shift = torch.chunk(t, 2, dim=1)
        h_c = h_c * (1 + scale) + shift
        h_c = F.silu(h_c)

        # conv1 path
        h_c = self.conv1(h_c)
        h_c = self.norm1(h_c)
        h_c = F.silu(h_c)

        # --- Recombine quantum + classical
        h = torch.cat([y_q, h_c], dim=1)  # [B, qc + oc == out_dim, H, W]

        # --- Residual
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

class QVUNetWithInternalQuantum(nn.Module):
    """
    QVUNet with internal quantum backend handling.
    This class automatically creates and manages the quantum backend internally.
    """
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), 
                 resnet_block_groups=8, quantum_channels=8, quantum_backend='ptlayer', 
                 ptlayer_config=None):
        super().__init__()
        self.dim = dim
        self.init_dim = init_dim or dim
        self.out_dim = out_dim or 2  # Binary segmentation
        self.dim_mults = dim_mults
        self.resnet_block_groups = resnet_block_groups
        self.quantum_channels = quantum_channels
        self.quantum_backend = quantum_backend
        
        # Create quantum block internally
        self.quantum_block = create_quantum_block(
            backend=quantum_backend,
            quantum_channels=quantum_channels,
            ptlayer_config=ptlayer_config
        )
        
        # Create the main QVUNet with the quantum block
        self.qvunet = QVUNet(
            dim=dim,
            quantum_block=self.quantum_block,
            init_dim=init_dim,
            out_dim=out_dim,
            dim_mults=dim_mults,
            resnet_block_groups=resnet_block_groups,
            quantum_channels=quantum_channels
        )
    
    def forward(self, x, time, params=None):
        return self.qvunet(x, time, params)
    
    @classmethod
    def create_with_ptlayer(cls, dim, ptlayer_config=None, **kwargs):
        """
        Convenience method to create QVUNet with PTLayer backend.
        
        Args:
            dim: Base dimension
            ptlayer_config: PTLayer configuration dict
                           Default: {'n_loops': 2, 'n_samples': 200, 'observable': 'mean'}
            **kwargs: Other QVUNet parameters
        """
        return cls(dim=dim, quantum_backend='ptlayer', ptlayer_config=ptlayer_config, **kwargs)
    
    @classmethod
    def create_with_pennylane(cls, dim, **kwargs):
        """
        Convenience method to create QVUNet with PennyLane backend.
        
        Args:
            dim: Base dimension
            **kwargs: Other QVUNet parameters
        """
        return cls(dim=dim, quantum_backend='pennylane', **kwargs) 

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
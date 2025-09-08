
import flax.linen as nn
from typing import Any, Callable, Sequence, Tuple, List, Optional,Union
from einops import rearrange
import jax.numpy as jnp
import jax
import math
import pennylane as qml
from flax.linen.linear import (canonicalize_padding, _conv_dimension_numbers)
import numpy as np 



def l2norm(t, axis=1, eps=1e-12):
    """Performs L2 normalization of inputs over specified axis.

    Args:
      t: jnp.ndarray of any shape
      axis: the dimension to reduce, default -1
      eps: small value to avoid division by zero. Default 1e-12
    Returns:
      normalized array of same shape as t


    """
    denom = jnp.clip(jnp.linalg.norm(t, ord=2, axis=axis, keepdims=True), eps)
    out = t/denom
    return (out)


class SinusoidalPosEmb(nn.Module):
    """Build sinusoidal embeddings 

    Attributes:
      dim: dimension of the embeddings to generate
      dtype: data type of the generated embeddings
    """
    dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, time):
        """
        Args:
          time: jnp.ndarray of shape [batch].
        Returns:
          out: embedding vectors with shape `[batch, dim]`
        """
        assert len(time.shape) == 1.
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=self.dtype) * -emb)
        emb = time.astype(self.dtype)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

class Downsample(nn.Module):

  dim :Optional[int] = None
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self,x):
    B, H, W, C = x.shape
    dim = self.dim if self.dim is not None else C 
    x = nn.Conv(dim, kernel_size = (4,4), strides= (2,2), padding = 1, dtype=self.dtype)(x)
    assert x.shape == (B, H // 2, W // 2, dim)
    return(x)
  
class Downsample1(nn.Module):

  dim :Optional[int] = None
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self,x):
    B, H, W, C = x.shape
    dim = self.dim if self.dim is not None else C 
    x = nn.Conv(dim, kernel_size = (7,7), strides= (2,2), padding = 1, dtype=self.dtype)(x)
   
    return(x)
  
class Upsample(nn.Module):

  dim: Optional[int] = None
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self,x):
    B, H, W, C = x.shape
    dim = self.dim if self.dim is not None else C 
    if H==2:
        x= jax.image.resize(x, (B, H*3 +1, W*3 +1, C), 'nearest')
    else:
        x = jax.image.resize(x, (B, H * 2, W * 2, C), 'nearest')
    
    x = nn.Conv(dim, kernel_size=(3,3), padding=1,dtype=self.dtype)(x)
    if H!=2:
        assert x.shape == (B, H * 2, W * 2, dim) 
    return(x)


class WeightStandardizedConv(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """ 
    features: int
    kernel_size: Sequence[int] = 3
    strides: Union[None, int, Sequence[int]] = 1
    padding: Any = 1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32


    @nn.compact
    def __call__(self, x):
        """
        Applies a weight standardized convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        x = x.astype(self.dtype)
        
        conv = nn.Conv(
            features=self.features, 
            kernel_size=self.kernel_size, 
            strides = self.strides,
            padding=self.padding, 
            dtype=self.dtype, 
            param_dtype = self.param_dtype,
            parent=None)
        
        kernel_init = lambda  rng, x: conv.init(rng,x)['params']['kernel']
        bias_init = lambda  rng, x: conv.init(rng,x)['params']['bias']
        
        # standardize kernel
        kernel = self.param('kernel', kernel_init, x)
        eps = 1e-5 if self.dtype == jnp.float32 else 1e-3
        # reduce over dim_out
        redux = tuple(range(kernel.ndim - 1))
        mean = jnp.mean(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        var = jnp.var(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        standardized_kernel = (kernel - mean)/jnp.sqrt(var + eps)

        bias = self.param('bias',bias_init, x)

        return(conv.apply({'params': {'kernel': standardized_kernel, 'bias': bias}},x))
    

class ResnetBlock(nn.Module):
    """Convolutional residual block."""
    dim: int = None
    groups: Optional[int] = 10
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time_emb):
        """
        Args:
          x: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        Returns:
          x: jnp.ndarray of shape [B, H, W, C]
        """

        B, _, _, C = x.shape
        assert time_emb.shape[0] == B and len(time_emb.shape) == 2

        h = WeightStandardizedConv(
            features=self.dim, kernel_size=(3, 3), padding=1, name='conv_0')(x)
        h =nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name='norm_0')(h)

        # add in timestep embedding
        time_emb = nn.Dense(features=2 * self.dim,dtype=self.dtype,
                           name='time_mlp.dense_0')(nn.swish(time_emb))
        time_emb = time_emb[:,  jnp.newaxis, jnp.newaxis, :]  # [B, H, W, C]
        scale, shift = jnp.split(time_emb, 2, axis=-1)
        h = h * (1 + scale) + shift

        h = nn.swish(h)

        h = WeightStandardizedConv(
            features=self.dim, kernel_size=(3, 3), padding=1, name='conv_1')(h)
        h = nn.swish(nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name='norm_1')(h))

        if C != self.dim:
            x = nn.Conv(
              features=self.dim, 
              kernel_size= (1,1),
              dtype=self.dtype,
              name='res_conv_0')(x)

        assert x.shape == h.shape

        return x + h
    

def embedding(x, wires):
    wires = list(wires)
    qml.templates.AngleEmbedding(x, wires=wires)
    

def get_ansatz(ansatz):
    if ansatz == 'HQConv_ansatz':
        return HQConv_ansatz,  28
    if ansatz == 'FQConv_ansatz':
        return FQConv_ansatz, 24
    if ansatz == 'basic_ansatz':
        return Basic_ansatz, 4
    
def create_circuit(n_qubits,layers,ansatz):
    
    device = qml.device("default.qubit", wires=n_qubits)
    ansatz, params_per_layer = get_ansatz(ansatz)

    @qml.qnode(device, interface='jax')
    def circuit(x, theta):
        embedding(x, wires=range(n_qubits))
        for i in range(layers):
            ansatz(theta[i * params_per_layer: (i + 1) * params_per_layer], wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    return jax.jit(circuit)#

def block_A(theta, wires):
    for i,j in zip(range(3,-1,-1), range(0,len(theta),2)):
         qml.CRZ(2*np.pi*theta[j], wires=[wires[(i+1)%4], wires[i]])
         qml.CRX(2*np.pi*theta[j+1], wires=[wires[(i+1)%4], wires[i]])

def block_B(theta, wires):
    qml.CRZ(2*np.pi*theta[0], wires=[wires[0], wires[4]])
    qml.CRX(2*np.pi*theta[1], wires=[wires[0], wires[4]])

def block_C(theta, wires):
    for i,j in zip(range(3,-1,-1), range(4)):
        qml.CRZ(2*np.pi*theta[j], wires=[wires[i+4], wires[i]])

def block_D(theta, wires):
    for i,j in zip(range(3,-1,-1), range(4)):
        qml.CRX(2*np.pi*theta[j], wires=[wires[i+4], wires[i]])
    
def HQConv_ansatz(theta, wires):
    for i,j in zip(range(len(wires), -1, -4), range(len(wires)%4)):
        block_A(theta[j*8:(j+1)*8], wires[i-4:i])
    theta1= theta[(len(wires)%4)*8:]
    for i,j in zip(range(len(wires)-1, 3, -4), range(len(wires))):
        block_B(theta1[j*2:(j+1)*2], wires[i-7:i+1])
        
def get_next_four_indices(i, max_val=11): 
    return [(i + j) % (max_val + 1) for j in range(1, 5)]
    
def FQConv_ansatz(theta,wires):
    for i,j in zip(range(len(wires)-1, -1, -4), range(len(wires))):
        wires_controlled= list(wires[i-3:i+1])
        indices= get_next_four_indices(i, len(wires)-1)
        wires_control= list([wires[idx] for idx in indices])
        wires1= wires_controlled + wires_control
        theta1= theta[8*j:8*(j+1)]
        block_C(theta1[:4], wires1)
        block_D(theta1[4:8], wires1)

def Basic_ansatz(theta, wires):
    N = len(wires)
    for i in range(N):
        qml.RX(theta[i], wires=i)
    for i in range(N-1):
        qml.CNOT(wires=[i, i+1])
    qml.CNOT(wires=[N-1, 0])

def apply_qnn(x,circuit, param):
        x = jax.numpy.reshape(x, (-1, 4))
        fn = jax.vmap(lambda z: circuit(z, param))
        h = fn(x)
        h = jnp.asarray(h)
        h = h.T
        h = jax.numpy.reshape(h, (-1, 2, 2, 1))
        return h

def create_circuit_4_qubits(layers, ansatz):
    device = qml.device("default.qubit", wires=4)
    ansatz_fn, params_per_layer = get_ansatz(ansatz)

    @qml.qnode(device, interface='jax')
    def circuit(x, theta):
        # x shape: (4,)
        embedding(x, wires=range(4))
        for i in range(layers):
            ansatz_fn(theta[i * params_per_layer: (i + 1) * params_per_layer], wires=range(4))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]

    return jax.jit(circuit)

def apply_qnn4_pooled(x, circuit, param):
    """
    Pooled 4-channel encoding:
      - x: [B, H, W, 4]
      - pool over spatial dims -> [B, 4]
      - apply circuit batched -> [B, 4]
      - broadcast back to [B, H, W, 4]
    """
    B, H, W, C = x.shape
    assert C == 4
    x_pooled = jnp.mean(x, axis=(1, 2))  # [B, 4]
    fn = jax.vmap(lambda z: circuit(z, param))
    y = fn(x_pooled)  # [B, 4]
    y = jnp.asarray(y)
    y = jnp.reshape(y, (B, 1, 1, 4))
    y = jnp.tile(y, (1, H, W, 1))
    return y



def quantum_circuit_creation_4ch(layers, varform, num_groups:int):
    """
    Create a list of 4-qubit circuits, one per 4-channel group.
    """
    circuits = []
    for _ in range(num_groups):
        circuits.append(create_circuit_4_qubits(layers, varform))
    return circuits


class QResnetBlock(nn.Module):
    """Quantum Convolutional residual block using 4-channel pooled groups."""
    dim: int = None
    groups: Optional[int] = 10
    dtype: Any = jnp.float32
    quantum_channels: int = 12  # total quantum channels (will use multiples of 4)
    name_ansatz: str = 'basic_ansatz'
    num_layer : int = 1

    def init_params(self, rng:2, num_qparams:16):
        return jax.random.uniform(rng,(num_qparams,))

    @nn.compact
    def __call__(self, x, time_emb):
        B, H, W, C = x.shape
        assert time_emb.shape[0] == B and len(time_emb.shape) == 2

        # Determine effective quantum channels as multiple of 4
        effective_qc = (self.quantum_channels // 4) * 4
        effective_qc = int(min(effective_qc, self.dim))
        num_groups = effective_qc // 4

        # Prepare parameters and circuits for groups
        # For basic_ansatz with 4 wires: params per layer = 4
        params_per_layer = 4
        num_qparams = params_per_layer * self.num_layer
        params = []
        circuits = quantum_circuit_creation_4ch(self.num_layer, self.name_ansatz, num_groups)
        for i in range(num_groups):
            param_name = f"param4_{i}"
            p = self.param(param_name, self.init_params, num_qparams)
            params.append(p)

        # Split channels
        x_q = x[:, :, :, :effective_qc] if effective_qc > 0 else None
        x_c = x[:, :, :, effective_qc:] if effective_qc < self.dim else None

        # Process quantum groups
        hs = []
        if x_q is not None and num_groups > 0:
            for i in range(num_groups):
                start = 4 * i
                end = start + 4
                y = x_q[:, :, :, start:end]
                hq = apply_qnn4_pooled(y, circuits[i], params[i])  # [B,H,W,4]
                hs.append(hq)
        h_q = None
        if hs:
            hs = jnp.asarray(hs)
            h_q = jnp.concatenate(hs, axis=-1)  # [B,H,W,effective_qc]

        # Process classical channels
        if x_c is not None and x_c.shape[-1] > 0:
            h_c = WeightStandardizedConv(
                features=self.dim - effective_qc, kernel_size=(3, 3), padding=1, name='conv_0')(x_c)
        else:
            h_c = None

        # Combine
        if h_q is not None and h_c is not None:
            h = jnp.concatenate((h_q, h_c), axis=3)
        elif h_q is not None:
            h = h_q
        elif h_c is not None:
            h = h_c
        else:
            h = jnp.zeros((B, H, W, self.dim), dtype=x.dtype)
        
        # Normalize and time embedding
        # Ensure channels match dim
        if h.shape[-1] != self.dim:
            if h.shape[-1] > self.dim:
                h = h[:, :, :, :self.dim]
            else:
                pad = self.dim - h.shape[-1]
                h = jnp.concatenate((h, jnp.zeros((B, H, W, pad), dtype=h.dtype)), axis=3)

        h = nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name='norm_0')(h)
        time_emb = nn.Dense(features=2 * self.dim, dtype=self.dtype, name='time_mlp.dense_0')(nn.swish(time_emb))
        time_emb = time_emb[:, jnp.newaxis, jnp.newaxis, :]
        scale, shift = jnp.split(time_emb, 2, axis=-1)
        h = h * (1 + scale) + shift
        h = nn.swish(h)

        # Second quantum/classical stage mirroring the first
        hs2 = []
        if h_q is not None:
            # Recompute groups on h to keep dimensions consistent
            for i in range(num_groups):
                start = 4 * i
                end = start + 4
                y = h[:, :, :, start:end]
                hq2 = apply_qnn4_pooled(y, circuits[i], params[i])
                hs2.append(hq2)
        h_q2 = None
        if hs2:
            hs2 = jnp.asarray(hs2)
            h_q2 = jnp.concatenate(hs2, axis=-1)

        x_c2 = h[:, :, :, effective_qc:] if effective_qc < self.dim else None
        if x_c2 is not None and x_c2.shape[-1] > 0:
            h_c2 = WeightStandardizedConv(
                features=self.dim - effective_qc, kernel_size=(3, 3), padding=1, name='conv_1')(x_c2)
        else:
            h_c2 = None

        if h_q2 is not None and h_c2 is not None:
            h2 = jnp.concatenate((h_q2, h_c2), axis=3)
        elif h_q2 is not None:
            h2 = h_q2
        elif h_c2 is not None:
            h2 = h_c2
        else:
            h2 = h

        if h2.shape[-1] != self.dim:
            if h2.shape[-1] > self.dim:
                h2 = h2[:, :, :, :self.dim]
            else:
                pad = self.dim - h2.shape[-1]
                h2 = jnp.concatenate((h2, jnp.zeros((B, H, W, pad), dtype=h2.dtype)), axis=3)
        
        h2 = nn.swish(nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name='norm_1')(h2))

        if C != self.dim:
            x = nn.Conv(features=self.dim, kernel_size=(1,1), dtype=self.dtype, name='res_conv_0')(x)

        assert x.shape == h2.shape
        return x + h2
    
class QFullResnetBlock(nn.Module):
    """Full-quantum residual block using 4-channel pooled groups.

    Processes as many channels as possible in 4-channel quantum groups.
    Any remainder channels (dim % 4) are passed through a classical path to
    preserve shape, then concatenated back.
    """
    dim: int = None
    groups: Optional[int] = 10
    dtype: Any = jnp.float32
    name_ansatz: str = 'basic_ansatz'
    num_layer: int = 1

    def init_params(self, rng:2, num_qparams:16):
        return jax.random.uniform(rng, (num_qparams,))

    @nn.compact
    def __call__(self, x, time_emb):
        B, H, W, C = x.shape
        assert time_emb.shape[0] == B and len(time_emb.shape) == 2

        # Use all available channels in groups of 4
        effective_qc = (self.dim // 4) * 4
        num_groups = effective_qc // 4

        # Prepare parameters and circuits for groups
        params_per_layer = 4  # for basic 4-qubit ansatz
        num_qparams = params_per_layer * self.num_layer
        params = []
        circuits = quantum_circuit_creation_4ch(self.num_layer, self.name_ansatz, num_groups)
        for i in range(num_groups):
            param_name = f"param4_{i}"
            p = self.param(param_name, self.init_params, num_qparams)
            params.append(p)

        # Split channels: all-quantum groups + optional remainder classical
        x_q = x[:, :, :, :effective_qc] if effective_qc > 0 else None
        x_c = x[:, :, :, effective_qc:] if effective_qc < self.dim else None

        # Stage 1
        hs = []
        if x_q is not None and num_groups > 0:
            for i in range(num_groups):
                start = 4 * i
                end = start + 4
                y = x_q[:, :, :, start:end]
                hq = apply_qnn4_pooled(y, circuits[i], params[i])
                hs.append(hq)
        h_q = jnp.concatenate(hs, axis=-1) if hs else None

        if x_c is not None and x_c.shape[-1] > 0:
            h_c = WeightStandardizedConv(
                features=self.dim - effective_qc,
                kernel_size=(3, 3),
                padding=1,
                name='conv_0')(x_c)
        else:
            h_c = None

        if h_q is not None and h_c is not None:
            h = jnp.concatenate((h_q, h_c), axis=3)
        elif h_q is not None:
            h = h_q
        elif h_c is not None:
            h = h_c
        else:
            h = jnp.zeros((B, H, W, self.dim), dtype=x.dtype)

        # Normalize and add time embedding
        h = nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name='norm_0')(h)
        time_emb = nn.Dense(features=2 * self.dim, dtype=self.dtype, name='time_mlp.dense_0')(nn.swish(time_emb))
        time_emb = time_emb[:, jnp.newaxis, jnp.newaxis, :]
        scale, shift = jnp.split(time_emb, 2, axis=-1)
        h = h * (1 + scale) + shift
        h = nn.swish(h)

        # Stage 2 mirrors stage 1 on h
        hs2 = []
        if num_groups > 0:
            for i in range(num_groups):
                start = 4 * i
                end = start + 4
                y = h[:, :, :, start:end]
                hq2 = apply_qnn4_pooled(y, circuits[i], params[i])
                hs2.append(hq2)
        h_q2 = jnp.concatenate(hs2, axis=-1) if hs2 else None

        x_c2 = h[:, :, :, effective_qc:] if effective_qc < self.dim else None
        if x_c2 is not None and x_c2.shape[-1] > 0:
            h_c2 = WeightStandardizedConv(
                features=self.dim - effective_qc,
                kernel_size=(3, 3),
                padding=1,
                name='conv_1')(x_c2)
        else:
            h_c2 = None

        if h_q2 is not None and h_c2 is not None:
            h2 = jnp.concatenate((h_q2, h_c2), axis=3)
        elif h_q2 is not None:
            h2 = h_q2
        elif h_c2 is not None:
            h2 = h_c2
        else:
            h2 = h

        h2 = nn.swish(nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name='norm_1')(h2))

        if C != self.dim:
            x = nn.Conv(features=self.dim, kernel_size=(1, 1), dtype=self.dtype, name='res_conv_0')(x)

        assert x.shape == h2.shape
        return x + h2


class Attention(nn.Module):
    heads: int = 4
    dim_head: int = 32
    scale: int = 10
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        dim = self.dim_head * self.heads

        qkv = nn.Conv(features= dim * 3, kernel_size=(1, 1),
                      use_bias=False, dtype=self.dtype, name='to_qkv.conv_0')(x)  # [B, H, W, dim *3]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [B, H, W, dim]
        q, k, v = map(lambda t: rearrange(
            t, 'b x y (h d) -> b (x y) h d', h=self.heads), (q, k, v))

        assert q.shape == k.shape == v.shape == (
            B, H * W, self.heads, self.dim_head)

        q, k = map(l2norm, (q, k))

        sim = jnp.einsum('b i h d, b j h d -> b h i j', q, k) * self.scale
        attn = nn.softmax(sim, axis=-1)
        assert attn.shape == (B, self.heads, H * W,  H * W)

        out = jnp.einsum('b h i j , b j h d  -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=H)
        assert out.shape == (B, H, W, dim)

        out = nn.Conv(features=C, kernel_size=(1, 1), dtype=self.dtype, name='to_out.conv_0')(out)
        return (out)


class LinearAttention(nn.Module):
    heads: int = 4
    dim_head: int = 32
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        dim = self.dim_head * self.heads

        qkv = nn.Conv(
            features=dim * 3,
            kernel_size=(1, 1),
            use_bias=False,
            dtype=self.dtype,
            name='to_qkv.conv_0')(x)  # [B, H, W, dim *3]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [B, H, W, dim]
        q, k, v = map(lambda t: rearrange(
            t, 'b x y (h d) -> b (x y) h d', h=self.heads), (q, k, v))
        assert q.shape == k.shape == v.shape == (
            B, H * W, self.heads, self.dim_head)
        # compute softmax for q along its embedding dimensions
        q = nn.softmax(q, axis=-1)
        # compute softmax for k along its spatial dimensions
        k = nn.softmax(k, axis=-3)

        q = q/jnp.sqrt(self.dim_head)
        v = v / (H * W)

        context = jnp.einsum('b n h d, b n h e -> b h d e', k, v)
        out = jnp.einsum('b h d e, b n h d -> b h e n', context, q)
        out = rearrange(out, 'b h e (x y) -> b x y (h e)', x=H)
        assert out.shape == (B, H, W, dim)

        out = nn.Conv(features=C, kernel_size=(1, 1),  dtype=self.dtype, name='to_out.conv_0')(out)
        out = nn.LayerNorm(epsilon=1e-5, use_bias=False, dtype=self.dtype, name='to_out.norm_0')(out)
        return (out)

class AttnBlock(nn.Module):
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
    dtype: Any = jnp.float32


    @nn.compact
    def __call__(self, x):
      B, H, W, C = x.shape
      normed_x = nn.LayerNorm(epsilon=1e-5, use_bias=False,dtype=self.dtype)(x)
      if self.use_linear_attention:
        attn = LinearAttention(self.heads, self.dim_head, dtype=self.dtype)
      else:
        attn = Attention(self.heads, self.dim_head, dtype=self.dtype)
      out = attn(normed_x)
      assert out.shape == (B, H, W, C)
      return(out + x)
    

  
    

class DownUnet(nn.Module):
    "Encoder part of the U-Net"
    dim: int
    init_dim: Optional[int] = None # if None, same as dim
    out_dim: Optional[int] = None 
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    resnet_block_groups: int = 10
    learned_variance: bool = False
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, time):
        """
        Args:
          x: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
          h: jnp.ndarray of shape [B, H1, W1, C1]
          hs: list of jnp.ndarray
        Returns:
          x: jnp.ndarray of shape [B, H, W, C]
        """
        B, H, W, C = x.shape
        
        init_dim = self.dim if self.init_dim is None else self.init_dim
        hs = []
        h = nn.Conv(
            features= init_dim,
            kernel_size=(7, 7),
            padding=3,
            name='init.conv_0',
            dtype = self.dtype)(x)
        
        hs.append(h)
        # use sinusoidal embeddings to encode timesteps
        time_emb = SinusoidalPosEmb(self.dim, dtype=self.dtype)(time)  # [B. dim]
        time_emb = nn.Dense(features=self.dim * 4, dtype=self.dtype, name='time_mlp.dense_0')(time_emb)
        time_emb = nn.Dense(features=self.dim * 4, dtype=self.dtype, name='time_mlp.dense_1')(nn.gelu(time_emb))  # [B, 4*dim]
        
        # downsampling
        num_resolutions = len(self.dim_mults)
        for ind in range(num_resolutions):
          
          dim_in = h.shape[-1]
          
          h = ResnetBlock(dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype, name=f'down_{ind}.resblock_0')(h, time_emb)
          hs.append(h)

          h = ResnetBlock(dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype, name=f'down_{ind}.resblock_1')(h, time_emb)
          
          h = AttnBlock(dtype=self.dtype, name=f'down_{ind}.attnblock_0')(h)
          hs.append(h)

          if ind < num_resolutions -1:
            # Use consistent downsampling for all layers to ensure symmetric skip connections
                h = Downsample(dim=self.dim * self.dim_mults[ind], dtype=self.dtype, name=f'down_{ind}.downsample_0')(h)
            
        mid_dim = self.dim * self.dim_mults[-1]
        h = nn.Conv(features = mid_dim, kernel_size = (3,3), padding=1, dtype=self.dtype, name=f'down_{num_resolutions-1}.conv_0')(h)
        return h , hs ,time_emb
    

    

class UpUnet(nn.Module):
    "Decoder part of the U-Net"
    dim: int
    init_dim: Optional[int] = None # if None, same as dim
    out_dim: Optional[int] = None 
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    resnet_block_groups: int = 10
    learned_variance: bool = False
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x, h, hs, time_emb):
        """
        Args:
          x: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
          h: jnp.ndarray of shape [B, H1, W1, C1]
          hs: list of jnp.ndarray
        Returns:
          x: jnp.ndarray of shape [B, H, W, C]
        """
        B, H, W, C = x.shape
        num_resolutions = len(self.dim_mults)
        init_dim = self.dim if self.init_dim is None else self.init_dim
        
        for ind in reversed(range(num_resolutions)):
           
           dim_in = self.dim * self.dim_mults[ind]
           dim_out = self.dim * self.dim_mults[ind-1] if ind >0 else init_dim
           
           assert h.shape[-1] == dim_in
           h = jnp.concatenate([h, hs.pop()], axis=-1)
           
           assert h.shape[-1] == dim_in + dim_out
           h = ResnetBlock(dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype, name=f'up_{ind}.resblock_0')(h, time_emb)
          
           h = jnp.concatenate([h, hs.pop()], axis=-1)
           
           assert h.shape[-1] == dim_in + dim_out
           h = ResnetBlock(dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype, name=f'up_{ind}.resblock_1')(h, time_emb)
           
           h = AttnBlock(dtype=self.dtype, name=f'up_{ind}.attnblock_0')(h)
           

           assert h.shape[-1] == dim_in
           if ind > 0:
             h = Upsample(dim = dim_out, dtype=self.dtype, name = f'up_{ind}.upsample_0')(h)
             
        
        h = nn.Conv(features = init_dim, kernel_size=(3,3), padding=1, dtype=self.dtype, name=f'up_0.conv_0')(h)
        
        # final 
        h = jnp.concatenate([h, hs.pop()], axis=-1)
        
        assert h.shape[-1] == init_dim * 2
    
        out = ResnetBlock(dim=self.dim,groups=self.resnet_block_groups, dtype=self.dtype, name = 'final.resblock_0' )(h, time_emb)
        
        default_out_dim = C * (1 if not self.learned_variance else 2)
        out_dim = default_out_dim if self.out_dim is None else self.out_dim
        
        return(nn.Conv(out_dim, kernel_size=(1,1), dtype=self.dtype, name= 'final.conv_0')(out))


class Vertex(nn.Module):
    "Classical Vertex of the U-Net"
    dim: int
    init_dim: Optional[int] = None # if None, same as dim
    out_dim: Optional[int] = None 
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    resnet_block_groups: int = 10
    learned_variance: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, h, time_emb):
        """
        Args:
          h: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        Returns:
          h: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        """
        mid_dim = self.dim * self.dim_mults[-1]
        h =  ResnetBlock(dim= mid_dim, groups= self.resnet_block_groups, dtype=self.dtype, name = 'mid.resblock_0')(h, time_emb)
        h = AttnBlock(use_linear_attention=False, dtype=self.dtype, name = 'mid.attenblock_0')(h)
        h = ResnetBlock(dim= mid_dim, groups= self.resnet_block_groups, dtype=self.dtype, name = 'mid.resblock_1')(h, time_emb)
        return h,time_emb
    
class QVertex(nn.Module):
    "Hybrid Quantum Vertex of the U-Net"
    dim: int
    init_dim: Optional[int] = None # if None, same as dim
    out_dim: Optional[int] = None 
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    resnet_block_groups: int = 10
    learned_variance: bool = False
    dtype: Any = jnp.float32
    quantum_channels: int = 13
    name_ansatz: str = 'FQConv_ansatz'
    num_layer : int =3
    
    @nn.compact
    def __call__(self, h, time_emb):
        """
        Args:
          h: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        Returns:
          h: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        """
        mid_dim = self.dim * self.dim_mults[-1]
        h =  QResnetBlock(dim= mid_dim, groups= self.resnet_block_groups, dtype=self.dtype, quantum_channels= self.quantum_channels,name_ansatz= self.name_ansatz, num_layer=self.num_layer, name = 'mid.resblock_0')(h, time_emb)
        h = AttnBlock(use_linear_attention=False, dtype=self.dtype, name = 'mid.attenblock_0')(h)
        h = QResnetBlock(dim= mid_dim, groups= self.resnet_block_groups, dtype=self.dtype, quantum_channels= self.quantum_channels, name_ansatz= self.name_ansatz, num_layer=self.num_layer, name = 'mid.resblock_1')(h, time_emb)
        return h,time_emb
    
class FullQVertex(nn.Module):
    "Full Quantum Vertex of the U-Net"
    dim: int
    init_dim: Optional[int] = None # if None, same as dim
    out_dim: Optional[int] = None 
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    resnet_block_groups: int = 10
    learned_variance: bool = False
    dtype: Any = jnp.float32
    name_ansatz: str = 'FQConv_ansatz'
    num_layer : int =3
    
    @nn.compact
    def __call__(self, h, time_emb):
        """
        Args:
          h: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        Returns:
          h: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        """
        mid_dim = self.dim * self.dim_mults[-1]
        h =  QFullResnetBlock(dim= mid_dim, groups= self.resnet_block_groups, dtype=self.dtype,name_ansatz= self.name_ansatz, num_layer=self.num_layer, name = 'mid.resblock_0')(h, time_emb)
        h = AttnBlock(use_linear_attention=False, dtype=self.dtype, name = 'mid.attenblock_0')(h)
        h = QFullResnetBlock(dim= mid_dim, groups= self.resnet_block_groups, dtype=self.dtype, name_ansatz= self.name_ansatz, num_layer=self.num_layer, name = 'mid.resblock_1')(h, time_emb)
        return h,time_emb

    
class UNet(nn.Module):
    "Classical U-Net architecture "
    dim: int
    init_dim: Optional[int] = None # if None, same as dim
    out_dim: Optional[int] = None 
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    resnet_block_groups: int = 10
    learned_variance: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time):
        """
        Args:
          x: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        Returns:
          x: jnp.ndarray of shape [B, H, W, C]
        """
        h, hs, time_emb= DownUnet(dim= self.dim, init_dim= self.init_dim, out_dim= self.out_dim, dim_mults= self.dim_mults, resnet_block_groups= self.resnet_block_groups, learned_variance= self.learned_variance, dtype= self.dtype, name='DownUnet')(x,time)
        h, time_emb= Vertex(dim=self.dim, init_dim= self.init_dim, out_dim= self.out_dim, dim_mults= self.dim_mults, resnet_block_groups= self.resnet_block_groups, learned_variance= self.learned_variance, dtype= self.dtype, name='Vertexclassica')(h,time_emb)
        h= UpUnet(dim= self.dim, init_dim= self.init_dim, out_dim= self.out_dim, dim_mults= self.dim_mults, resnet_block_groups= self.resnet_block_groups, learned_variance= self.learned_variance, dtype= self.dtype, name= 'UpUnet')(x,h,hs,time_emb)
        
        return h 


class QVUNet(nn.Module):
    "Quantum Vertex U-Net Hybrid Architecture"
    dim: int
    init_dim: Optional[int] = None # if None, same as dim
    out_dim: Optional[int] = None 
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    resnet_block_groups: int = 10
    learned_variance: bool = False
    dtype: Any = jnp.float32
    quantum_channels: int = 13
    name_ansatz: str = 'FQConv_ansatz'
    num_layer : int =3

    @nn.compact
    def __call__(self, x, time):
        """
        Args:
          x: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        Returns:
          x: jnp.ndarray of shape [B, H, W, C]
        """
        h, hs, time_emb= DownUnet(dim= self.dim, init_dim= self.init_dim, out_dim= self.out_dim, dim_mults= self.dim_mults, resnet_block_groups= self.resnet_block_groups, learned_variance= self.learned_variance, dtype= self.dtype, name='DownUnet1')(x,time)
        h, time_emb= QVertex(dim=self.dim, init_dim= self.init_dim, out_dim= self.out_dim, dim_mults= self.dim_mults, resnet_block_groups= self.resnet_block_groups, learned_variance= self.learned_variance, dtype= self.dtype, quantum_channels= self.quantum_channels, name_ansatz= self.name_ansatz, num_layer= self.num_layer, name='Vertexquantum')(h,time_emb)
        h= UpUnet(dim= self.dim, init_dim= self.init_dim, out_dim= self.out_dim, dim_mults= self.dim_mults, resnet_block_groups= self.resnet_block_groups, learned_variance= self.learned_variance, dtype= self.dtype, name= 'UpUnet1')(x,h,hs,time_emb)
        
        return h 
    
class FullQVUNet(nn.Module):
    "Full Quantum Vertex U-Net Hybrid Architecture"
    dim: int
    init_dim: Optional[int] = None # if None, same as dim
    out_dim: Optional[int] = None 
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    resnet_block_groups: int = 10
    learned_variance: bool = False
    dtype: Any = jnp.float32
    name_ansatz: str = 'FQConv_ansatz'
    num_layer : int =3

    @nn.compact
    def __call__(self, x, time):
        """
        Args:
          x: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        Returns:
          x: jnp.ndarray of shape [B, H, W, C]
        """
        h, hs, time_emb= DownUnet(dim= self.dim, init_dim= self.init_dim, out_dim= self.out_dim, dim_mults= self.dim_mults, resnet_block_groups= self.resnet_block_groups, learned_variance= self.learned_variance, dtype= self.dtype, name='DownUnet1')(x,time)
        h, time_emb= FullQVertex(dim=self.dim, init_dim= self.init_dim, out_dim= self.out_dim, dim_mults= self.dim_mults, resnet_block_groups= self.resnet_block_groups, learned_variance= self.learned_variance, dtype= self.dtype, name_ansatz= self.name_ansatz, num_layer= self.num_layer, name='FullQVertexquantum')(h,time_emb)
        h= UpUnet(dim= self.dim, init_dim= self.init_dim, out_dim= self.out_dim, dim_mults= self.dim_mults, resnet_block_groups= self.resnet_block_groups, learned_variance= self.learned_variance, dtype= self.dtype, name= 'UpUnet1')(x,h,hs,time_emb)
        
        return h 


 
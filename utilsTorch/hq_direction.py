"""Standalone HQ-style direction test, distinct from QUAFFLE's residual U-Net.

This binary-segmentation adaptation retains plain U-Net encoder/decoder skips,
but has no identity, attention or gating path around its bottleneck. The exact
four-by-four quantum circuit is reused from ``unetTorch``; the two alternatives
have identical classical interfaces, including full dense reconstruction.
"""
import hashlib
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint

from .unetTorch import torch_hqunet_expvals


_CONFIG_KEYS = frozenset(('kind', 'seed', 'base_channels', 'image_size',
                          'in_channels', 'out_channels'))
_CORE_SEED_OFFSET = 700_013
_MODES = ('on', 'zero_core', 'random_core', 'zero_bottleneck')


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class _DepthwiseDoubleConv(nn.Sequential):
    """Two ordinary depthwise/pointwise convolutions, each with BN and ReLU."""
    def __init__(self, in_channels, out_channels):
        layers = []
        for source in (in_channels, out_channels):
            layers.extend((nn.Conv2d(source, source, 3, padding=1, groups=source, bias=False),
                           nn.Conv2d(source, out_channels, 1, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        super().__init__(*layers)


class _Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.MaxPool2d(2), _DepthwiseDoubleConv(in_channels, out_channels))


class _Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = _DepthwiseDoubleConv(2 * out_channels, out_channels)

    def forward(self, x, skip):
        return self.conv(torch.cat((skip, self.up(x)), dim=1))


class _QuantumCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(1, 2, 4).uniform_(-math.pi, math.pi))

    @staticmethod
    def _evaluate(angles, weights):
        return torch_hqunet_expvals(angles, weights, grid_size=4)

    def forward(self, angles):
        # Keep simulator work on CPU even when the surrounding U-Net uses CUDA.
        # Checkpoint one sample at a time rather than retaining a 16-qubit graph
        # for every item. This leaves the U-Net's batch and BN semantics intact.
        cpu_angles, cpu_weights = angles.to('cpu'), self.weights.to('cpu')
        outputs = []
        for sample in cpu_angles.split(1):
            if torch.is_grad_enabled() and (sample.requires_grad or cpu_weights.requires_grad):
                output = torch.utils.checkpoint.checkpoint(
                    self._evaluate, sample, cpu_weights, use_reentrant=False)
            else:
                output = self._evaluate(sample, cpu_weights)
            outputs.append(output)
        return torch.cat(outputs).to(device=angles.device, dtype=angles.dtype)


def _make_core(kind):
    if kind == 'quantum':
        return _QuantumCore()
    return nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(48, 64), nn.SiLU(),
                         nn.Linear(64, 32), nn.Tanh())


class _HQBottleneck(nn.Module):
    def __init__(self, kind, input_channels, output_channels, spatial_size, seed):
        super().__init__()
        self.kind = kind
        self.output_channels = output_channels
        self.spatial_size = spatial_size
        self.pool = nn.MaxPool2d(2)
        self.pre_projection = nn.Conv2d(input_channels, 3, 1)
        # Different core sizes must not change post-projection/decoder draws.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed((seed + _CORE_SEED_OFFSET) % (1 << 64))
            self.core = _make_core(kind)
        self.post_projection = nn.Linear(32, output_channels * spatial_size * spatial_size)
        self.mode = 'on'
        self._random_core_state = None

    def core_parameters(self):
        return self.core.parameters()

    def encode_angles(self, x):
        """Incoming down3 features -> [batch,16,3] angles in radians, no pi."""
        pooled = F.adaptive_avg_pool2d(self.pool(x), (4, 4))
        angles = self.pre_projection(pooled).tanh()
        return angles.permute(0, 2, 3, 1).reshape(x.shape[0], 16, 3)

    def angle_diagnostics(self, x):
        """Detached saturation summary; does not update any BatchNorm state."""
        with torch.no_grad():
            angles = self.encode_angles(x)
            return dict(n_angles=angles.numel(), min_radians=float(angles.min()),
                        max_radians=float(angles.max()), mean_abs_radians=float(angles.abs().mean()),
                        saturation_fraction=float((angles.abs() >= 0.95).float().mean()))

    def set_mode(self, mode='on', seed=0):
        if mode not in _MODES:
            raise ValueError(f'Unknown bottleneck mode {mode!r}; expected {_MODES}')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('Diagnostic seed must be an integer')
        self.mode = mode
        self._random_core_state = None
        if mode == 'random_core':
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed((seed + _CORE_SEED_OFFSET) % (1 << 64))
                replacement = _make_core(self.kind)
            self._random_core_state = {name: value.detach().clone()
                                       for name, value in replacement.state_dict().items()}
        return self

    def forward(self, x):
        if self.mode == 'zero_bottleneck':
            return x.new_zeros(x.shape[0], self.output_channels,
                               self.spatial_size, self.spatial_size)
        if self.mode == 'zero_core':
            features = x.new_zeros(x.shape[0], 32)
        else:
            angles = self.encode_angles(x)
            if self.mode == 'random_core':
                state = {name: value.to(device=x.device, dtype=x.dtype)
                         for name, value in self._random_core_state.items()}
                features = torch.func.functional_call(self.core, state, (angles,))
            else:
                features = self.core(angles)
        return self.post_projection(features).reshape(
            x.shape[0], self.output_channels, self.spatial_size, self.spatial_size)


class HQDirectionUNet(nn.Module):
    """Small plain U-Net with a replacement quantum or classical bottleneck.

    ``image_size`` is fixed for full dense reconstruction, divisible by sixteen
    and at least 64. The bottleneck module includes the final MaxPool, so its
    diagnostic methods consume the same down3 tensor as its forward method.
    Constructor seeding preserves the caller's CPU RNG and initializes all
    non-core tensors identically across kinds, including BatchNorm buffers.
    """
    def __init__(self, kind='quantum', seed=42, base_channels=8, image_size=128,
                 in_channels=3, out_channels=2):
        super().__init__()
        if kind not in ('quantum', 'mlp'):
            raise ValueError("kind must be 'quantum' or 'mlp'")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('seed must be an integer')
        for name, value in (('base_channels', base_channels), ('image_size', image_size),
                            ('in_channels', in_channels), ('out_channels', out_channels)):
            _positive_int(value, name)
        if image_size < 64 or image_size % 16:
            raise ValueError('image_size must be at least 64 and divisible by 16')
        self.kind, self.seed = kind, seed
        self.base_channels, self.image_size = base_channels, image_size
        self.in_channels, self.out_channels = in_channels, out_channels
        b = base_channels
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed % (1 << 64))
            self.stem = _DepthwiseDoubleConv(in_channels, b)
            self.down1, self.down2, self.down3 = _Down(b, 2*b), _Down(2*b, 4*b), _Down(4*b, 8*b)
            self.bottleneck = _HQBottleneck(kind, 8*b, 16*b, image_size // 16, seed)
            self.up1, self.up2 = _Up(16*b, 8*b), _Up(8*b, 4*b)
            self.up3, self.up4 = _Up(4*b, 2*b), _Up(2*b, b)
            self.head = nn.Conv2d(b, out_channels, 1)

    def forward(self, x):
        if (x.ndim != 4 or x.shape[0] < 1 or
                tuple(x.shape[1:]) != (self.in_channels, self.image_size, self.image_size)):
            raise ValueError('Expected a nonempty [B,in_channels,image_size,image_size] tensor')
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(self.bottleneck(x3), x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.head(self.up4(x, x0))


def shared_state_sha256(model):
    """Hash every non-core parameter/buffer, including BN's integer counters."""
    if not isinstance(model, HQDirectionUNet):
        raise TypeError('Expected HQDirectionUNet')
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('bottleneck.core.'):
            continue
        cpu = value.detach().cpu().contiguous()
        digest.update(f'{name}:{cpu.dtype}:{tuple(cpu.shape)}\n'.encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def model_config(model):
    if not isinstance(model, HQDirectionUNet):
        raise TypeError('Expected HQDirectionUNet')
    return {key: getattr(model, key) for key in sorted(_CONFIG_KEYS)}


def build_from_config(config):
    """Reconstruct only a complete, recognized model configuration."""
    if not isinstance(config, Mapping) or set(config) != _CONFIG_KEYS:
        raise ValueError(f'Model config must contain exactly {sorted(_CONFIG_KEYS)}')
    return HQDirectionUNet(**dict(config))

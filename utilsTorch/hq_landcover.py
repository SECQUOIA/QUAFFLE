"""Five-class HQ-UNet following the LandCover.ai architecture in Fig. 1.

This native pilot is separate from the historical binary HQ direction model.
The diagram and encoder description specify a fourth MaxPool/DoubleConv before
Pre-Q; it is retained despite the paper also calling the quantum module a
replacement for the deepest classical block. Width 64 gives 64/128/256/512/1024.
Fig. 2 is implemented with even-pair/odd-pair brickwork, not the historical
sequential edge order. Matching these architectural choices is not a claim to
reproduce the paper's reported accuracy or its unspecified training procedure.
"""
from collections.abc import Mapping
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .hq_direction import _CORE_SEED_OFFSET, _DepthwiseDoubleConv, _Down, _QuantumCore, _Up, _positive_int
from .landcover_circuit import native_hqunet_expvals


CONTROLS = ('quantum', 'frozen_quantum', 'compact_affine', 'zero_bottleneck')
FORMAT = 'hq-landcover-native-v1'
_CONFIG_KEYS = frozenset(('format', 'control', 'seed', 'base_channels', 'image_size',
                          'in_channels', 'out_channels', 'circuit_edge_order'))


class _BrickworkQuantumCore(_QuantumCore):
    """Retain the memory-bounded CPU/sample checkpointing implementation."""
    @staticmethod
    def _evaluate(angles, weights):
        return native_hqunet_expvals(angles, weights)


class _NativeBottleneck(nn.Module):
    def __init__(self, control, channels, spatial_size, seed):
        super().__init__()
        self.training_control = control
        self.mode = 'zero_bottleneck' if control == 'zero_bottleneck' else 'on'
        self.output_channels = channels
        self.spatial_size = spatial_size
        self.pre_projection = nn.Conv2d(channels, 3, 1)
        # Isolate all core draws so size changes never shift the interfaces,
        # decoder, output head or BatchNorm initialization across controls.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed((seed + _CORE_SEED_OFFSET) % (1 << 64))
            self.core = (nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(48, 32), nn.Tanh())
                         if control == 'compact_affine' else _BrickworkQuantumCore())
        self.post_projection = nn.Linear(32, channels * spatial_size * spatial_size)
        if control == 'frozen_quantum':
            self.core.requires_grad_(False)
        elif control == 'zero_bottleneck':
            self.requires_grad_(False)

    def core_parameters(self):
        return self.core.parameters()

    def encode_angles(self, x):
        """Final encoder map -> [B,16,3] tanh radians in [-1,1], without pi."""
        if x.shape[-2] % 4 == 0 and x.shape[-1] % 4 == 0:
            # The native 8x8->4x4 (or fixture4x4->4x4) adaptive bins are
            # disjoint equal windows. Ordinary average pooling is identical
            # mathematically and has deterministic CUDA backward in our runtime.
            kernel = (x.shape[-2] // 4, x.shape[-1] // 4)
            pooled = F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
        else:
            # Preserve general constructor sizes without CUDA's nondeterministic
            # adaptive-pooling backward. Device copies retain input derivatives.
            pooled = F.adaptive_avg_pool2d(x.to('cpu'), (4, 4)).to(x.device)
        angles = self.pre_projection(pooled).tanh()
        return angles.permute(0, 2, 3, 1).reshape(x.shape[0], 16, 3)

    def forward(self, x):
        if self.training_control == 'zero_bottleneck':
            # Train-time absence is independent of mutable diagnostic labels.
            return x.new_zeros(x.shape[0], self.output_channels, self.spatial_size, self.spatial_size)
        features = self.core(self.encode_angles(x))
        return self.post_projection(features).reshape(x.shape[0], self.output_channels,
                                                       self.spatial_size, self.spatial_size)


class HQNativeLandcoverUNet(nn.Module):
    """RGB -> five-class logits with a full 16-qubit replacement interface.

    Defaults follow Fig. 1: 128-pixel patches and base width 64. Four ordinary
    encoder/decoder skips connect matching resolutions; no residual or gating
    bypass is added around the quantum interface. The fourth down block runs
    before the Pre-Q projection in every control, including the zero arm.
    Constructor seeding preserves the caller's CPU RNG.
    """
    def __init__(self, control='quantum', seed=42, base_channels=64, image_size=128):
        super().__init__()
        if control not in CONTROLS:
            raise ValueError(f'Unknown native training control {control!r}; expected {CONTROLS}')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('seed must be an integer')
        _positive_int(base_channels, 'base_channels')
        _positive_int(image_size, 'image_size')
        if image_size < 64 or image_size % 16:
            raise ValueError('image_size must be at least 64 and divisible by 16')
        self.training_control = control
        self.seed, self.base_channels, self.image_size = seed, base_channels, image_size
        self.in_channels, self.out_channels = 3, 5
        self.circuit_edge_order = 'brickwork'
        b = base_channels
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed % (1 << 64))
            self.stem = _DepthwiseDoubleConv(3, b)
            self.down1, self.down2 = _Down(b, 2*b), _Down(2*b, 4*b)
            self.down3, self.down4 = _Down(4*b, 8*b), _Down(8*b, 16*b)
            self.bottleneck = _NativeBottleneck(control, 16*b, image_size // 16, seed)
            self.up1, self.up2 = _Up(16*b, 8*b), _Up(8*b, 4*b)
            self.up3, self.up4 = _Up(4*b, 2*b), _Up(2*b, b)
            self.head = nn.Conv2d(b, 5, 1)

    def forward(self, x):
        if (x.ndim != 4 or x.shape[0] < 1 or
                tuple(x.shape[1:]) != (3, self.image_size, self.image_size)):
            raise ValueError('Expected a nonempty [B,3,image_size,image_size] tensor')
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(self.bottleneck(x4), x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.head(self.up4(x, x0))


def build_model(control, seed=42, base_channels=64, image_size=128):
    return HQNativeLandcoverUNet(control=control, seed=seed, base_channels=base_channels, image_size=image_size)


def model_config(model):
    if not isinstance(model, HQNativeLandcoverUNet):
        raise TypeError('Expected HQNativeLandcoverUNet')
    return dict(format=FORMAT, control=model.training_control, seed=model.seed,
                base_channels=model.base_channels, image_size=model.image_size,
                in_channels=3, out_channels=5, circuit_edge_order='brickwork')


def build_from_config(config):
    """Restore complete native architecture and trainability before strict load."""
    if not isinstance(config, Mapping) or set(config) != _CONFIG_KEYS:
        raise ValueError(f'Native model config must contain exactly {sorted(_CONFIG_KEYS)}')
    expected = dict(format=FORMAT, in_channels=3, out_channels=5, circuit_edge_order='brickwork')
    if any(type(config[key]) is not type(value) or config[key] != value for key, value in expected.items()):
        raise ValueError('Native model format, channels or circuit edge order differs')
    return build_model(**{key: config[key] for key in ('control', 'seed', 'base_channels', 'image_size')})


def parameter_counts(model):
    """Stored/trainable counts describe flags, not functional path influence."""
    model_config(model)
    groups = {'': list(model.parameters()), 'core_': list(model.bottleneck.core_parameters()),
              'bottleneck_': list(model.bottleneck.parameters())}
    result = {}
    for prefix, parameters in groups.items():
        result[prefix+'total'] = sum(value.numel() for value in parameters)
        result[prefix+'trainable'] = sum(value.numel() for value in parameters if value.requires_grad)
    return result


def shared_state_sha256(model):
    """Hash all non-core parameters and buffers, including BN counters."""
    model_config(model)
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('bottleneck.core.'):
            continue
        value = value.detach().cpu().contiguous()
        result.update(f'{name}:{value.dtype}:{tuple(value.shape)}\n'.encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()

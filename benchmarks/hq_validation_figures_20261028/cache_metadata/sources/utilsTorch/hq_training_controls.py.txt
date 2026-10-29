"""Training-time controls for the local HQ direction experiment.

The legacy quantum/MLP constructors and their parameter/buffer names are kept
intact. Training controls are checkpoint metadata: save ``model_config(model)``
alongside the state dict, reconstruct with ``build_from_config``, then restore
the state dict strictly. PyTorch state dicts alone do not store requires_grad.
"""
from collections.abc import Mapping

import torch
from torch import nn

from .hq_direction import (
    HQDirectionUNet, _CORE_SEED_OFFSET, _HQBottleneck, _MODES,
    shared_state_sha256 as _legacy_shared_state_sha256,
)


CONTROLS = ('quantum', 'frozen_quantum', 'zero_bottleneck', 'compact_affine', 'mlp')
_CONFIG_KEYS = frozenset(('control', 'seed', 'base_channels', 'image_size'))


def _compact_core():
    return nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(48, 32), nn.Tanh())


class _ControlledBottleneck(_HQBottleneck):
    """Reuse the original modules without another draw or state-key prefix."""
    def __init__(self, original, control):
        # Do not call the legacy constructor: it would draw new interface weights.
        nn.Module.__init__(self)
        self.kind = original.kind
        self.output_channels = original.output_channels
        self.spatial_size = original.spatial_size
        self.pool = original.pool
        self.pre_projection = original.pre_projection
        self.core = original.core
        self.post_projection = original.post_projection
        self.training = original.training
        self.training_control = control
        self.mode = 'zero_bottleneck' if control == 'zero_bottleneck' else 'on'
        self._random_core_state = None

    def set_mode(self, mode='on', seed=0):
        if mode not in _MODES:
            raise ValueError(f'Unknown bottleneck mode {mode!r}; expected {_MODES}')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('Diagnostic seed must be an integer')
        if self.training_control == 'zero_bottleneck':
            # Generic diagnostic cleanup may request "on". It must never
            # reactivate the bottleneck trained as permanently absent.
            self.mode = 'zero_bottleneck'
            self._random_core_state = None
            return self
        self.mode = mode
        self._random_core_state = None
        if mode == 'random_core':
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed((seed + _CORE_SEED_OFFSET) % (1 << 64))
                replacement = _compact_core()
            self._random_core_state = {name: value.detach().clone()
                                       for name, value in replacement.state_dict().items()}
        return self

    def forward(self, x):
        if self.training_control == 'zero_bottleneck':
            # This guard is independent of mutable inference-diagnostic mode.
            return x.new_zeros(x.shape[0], self.output_channels,
                               self.spatial_size, self.spatial_size)
        return super().forward(x)


def build_model(control, seed, base_channels=8, image_size=128):
    """Build one of five binary, three-input-channel HQ training controls.

    ``frozen_quantum`` freezes only the eight circuit weights, retaining
    derivatives through circuit inputs. ``zero_bottleneck`` freezes every
    bottleneck parameter and always returns a zero reconstructed map; ordinary
    encoder/decoder skips remain. ``compact_affine`` has a 48→32 affine map
    followed by tanh (1,568 core parameters), with the same classical interfaces.
    All controls share the legacy non-core initialization, including BN buffers.
    """
    if control not in CONTROLS:
        raise ValueError(f'Unknown training control {control!r}; expected {CONTROLS}')
    kind = 'mlp' if control in ('mlp', 'compact_affine') else 'quantum'
    model = HQDirectionUNet(kind=kind, seed=seed, base_channels=base_channels,
                            image_size=image_size, in_channels=3, out_channels=2)
    model.training_control = control
    if control in ('zero_bottleneck', 'compact_affine'):
        model.bottleneck = _ControlledBottleneck(model.bottleneck, control)
    if control == 'compact_affine':
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed((seed + _CORE_SEED_OFFSET) % (1 << 64))
            model.bottleneck.core = _compact_core()
    elif control == 'frozen_quantum':
        model.bottleneck.core.requires_grad_(False)
    elif control == 'zero_bottleneck':
        model.bottleneck.requires_grad_(False)
    return model


def model_config(model):
    """Return the complete factory configuration, including training semantics."""
    if not isinstance(model, HQDirectionUNet):
        raise TypeError('Expected an HQDirectionUNet training-control model')
    control = getattr(model, 'training_control', None)
    if control not in CONTROLS or model.in_channels != 3 or model.out_channels != 2:
        raise ValueError('Expected a model built by the training-control factory')
    return dict(control=control, seed=model.seed, base_channels=model.base_channels,
                image_size=model.image_size)


def build_from_config(config):
    """Reconstruct exact control/trainability semantics before strict restore."""
    if not isinstance(config, Mapping) or set(config) != _CONFIG_KEYS:
        raise ValueError(f'Model config must contain exactly {sorted(_CONFIG_KEYS)}')
    return build_model(**dict(config))


def parameter_counts(model):
    """Count stored and trainable parameters, separately from path usefulness."""
    model_config(model)
    groups = {'': list(model.parameters()), 'core_': list(model.bottleneck.core_parameters()),
              'bottleneck_': list(model.bottleneck.parameters())}
    counts = {}
    for prefix, parameters in groups.items():
        counts[prefix + 'total'] = sum(value.numel() for value in parameters)
        counts[prefix + 'trainable'] = sum(value.numel() for value in parameters if value.requires_grad)
    return counts


def shared_state_sha256(model):
    """Hash all non-core weights and buffers using the unchanged legacy scheme."""
    return _legacy_shared_state_sha256(model)

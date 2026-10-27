"""Opt-in four-qubit filters over neighboring encoder locations.

Every core receives the same four scalar positions from a sliding 2x2 window.
Pre/post projections and their initial values are shared by construction across
the quantum and classical alternatives. This module does not replace any default
QUAFFLE bottleneck.
"""
import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint


LOCAL_SPATIAL_KERNELS = ('quantum', 'mlp', 'conv', 'sincos')
LOCAL_SPATIAL_READOUTS = ('z', 'xz_zz')
LOCAL_EDGES = ((0, 1), (2, 3), (0, 2), (1, 3))


def normalize_local_spatial_kernel(value=None):
    if value is None:
        return None
    key = str(value).strip().lower()
    if key not in LOCAL_SPATIAL_KERNELS:
        raise ValueError(f'Unknown local_spatial_kernel {value!r}; expected {LOCAL_SPATIAL_KERNELS}')
    return key


def local_spatial_options(config):
    """Canonical metadata, with absent options inactive for legacy checkpoints."""
    kind = normalize_local_spatial_kernel(config.get('local_spatial_kernel'))
    if kind is None:
        return dict(local_spatial_kernel=None, local_spatial_readout=None, local_spatial_channels=None)
    readout = str(config.get('local_spatial_readout', 'xz_zz')).strip().lower()
    if readout not in LOCAL_SPATIAL_READOUTS:
        raise ValueError(f'Unknown local_spatial_readout {readout!r}; expected {LOCAL_SPATIAL_READOUTS}')
    channels = config.get('local_spatial_channels', 4)
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
        raise ValueError('local_spatial_channels must be a positive integer')
    return dict(local_spatial_kernel=kind, local_spatial_readout=readout, local_spatial_channels=channels)


@lru_cache(maxsize=None)
def _state_circuit(layers, entangle):
    import pennylane as qml
    dev = qml.device('default.qubit', wires=4)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(tokens, weights, scales):
        for layer in range(layers):
            for wire in range(4):
                qml.RY(math.pi * torch.tanh(scales[layer, wire] * tokens[:, wire]), wires=wire)
                qml.Rot(*weights[layer, wire], wires=wire)
            if entangle:
                for edge in LOCAL_EDGES:
                    qml.CNOT(wires=edge)
        return qml.state()

    return circuit


def state_readouts(state):
    """Return [Z0..Z3, X0..X3, ZZ01, ZZ23, ZZ02, ZZ13] exactly.

    PennyLane's computational basis uses wire 0 as its most significant bit.
    The X expectation is the overlap with the state with that bit flipped.
    """
    indices = torch.arange(16, device=state.device)
    signs = torch.stack([1 - 2 * ((indices >> (3 - wire)) & 1) for wire in range(4)])
    probabilities = state.abs().square()
    z = probabilities @ signs.to(probabilities.dtype).T
    x = torch.stack([(state.conj() * state[:, indices ^ (1 << (3 - wire))]).sum(-1).real
                     for wire in range(4)], dim=-1)
    zz_signs = torch.stack([signs[first] * signs[second] for first, second in LOCAL_EDGES])
    zz = probabilities @ zz_signs.to(probabilities.dtype).T
    return torch.cat((z, x, zz), dim=-1)


class LocalSpatialBottleneckBlock(nn.Module):
    """Shared 2x2 filter with a fixed twelve-feature interface for every core.

    Input [B,C,H,W] -> pointwise C->K -> right/bottom replicate-padded 2x2
    windows -> a shared 4->12 core -> pointwise 12K->C. The core is shared over
    both spatial locations and the K projected feature channels. `conv` is an
    ordinary learned 2x2 scalar convolution with twelve output filters + SiLU;
    evaluating that convolution on extracted windows is algebraically identical.

    Z-only pads its eight missing readouts with zeros. Its post-projection keeps
    the same shape and initialization but those columns receive no data gradient.
    No-entanglement still permits classical products through the ZZ readouts.
    """
    _MAX_TOKENS = 1024

    def __init__(self, quantum_channels, kind='quantum', readout='xz_zz',
                 feature_channels=4, q_layers=1, q_training_control='trainable', seed=None):
        super().__init__()
        options = local_spatial_options(dict(local_spatial_kernel=kind,
            local_spatial_readout=readout, local_spatial_channels=feature_channels))
        self.kind = options['local_spatial_kernel']
        self.readout = options['local_spatial_readout']
        self.feature_channels = options['local_spatial_channels']
        kind, readout = self.kind, self.readout
        if isinstance(quantum_channels, bool) or not isinstance(quantum_channels, int) or quantum_channels < 1:
            raise ValueError('The local spatial block requires positive integer quantum_channels')
        if isinstance(q_layers, bool) or not isinstance(q_layers, int) or q_layers < 1:
            raise ValueError('The local spatial block requires a positive integer q_layers')
        if q_training_control not in ('trainable', 'frozen', 'no_entanglement'):
            raise ValueError('Unknown q_training_control')
        if kind != 'quantum' and q_training_control != 'trainable':
            raise ValueError('Quantum training controls apply only to the local quantum core')
        if kind != 'quantum' and readout != 'xz_zz':
            raise ValueError('Classical local controls use the full twelve-feature interface')
        self.quantum_channels = self.effective_qc = self.active_channels = quantum_channels
        self.q_layers = q_layers
        self.q_training_control = q_training_control
        self.quantum_mode = 'on'
        self._random_core = None
        self.pre_q_conv = nn.Conv2d(quantum_channels, self.feature_channels, 1)
        self.post_q_proj = nn.Conv2d(12 * self.feature_channels, quantum_channels, 1)

        # Core initialization never advances the backbone/projection RNG stream.
        core_seed = ((int(seed) if seed is not None else torch.initial_seed()) + 500_009) % (1 << 64)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(core_seed)
            if kind == 'quantum':
                self.quantum_params = nn.ParameterList([nn.Parameter(0.1 * torch.randn(q_layers, 4, 3))])
                self.input_scales = nn.ParameterList([nn.Parameter(torch.ones(q_layers, 4))])
                self.core = None
                if q_training_control == 'frozen':
                    for parameter in (*self.quantum_params, *self.input_scales):
                        parameter.requires_grad_(False)
            elif kind == 'mlp':
                self.core = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 12))
            elif kind == 'conv':
                self.core = nn.Sequential(nn.Linear(4, 12), nn.SiLU())
            else:
                self.input_scales = nn.ParameterList([nn.Parameter(torch.ones(1, 4))])
                self.core = nn.Linear(12, 12)
        self.bottleneck_actual_params = sum(parameter.numel() for parameter in self.parameters())
        self.bottleneck_target_params = (sum(parameter.numel() for layer in (self.pre_q_conv, self.post_q_proj)
                                             for parameter in layer.parameters()) + 16 * q_layers)

    def set_quantum_mode(self, mode='on', knockout_seed=0, seed=None):
        if mode not in ('on', 'zero', 'random'):
            raise ValueError('Unknown quantum mode')
        self.quantum_mode = mode
        self._random_core = None
        if mode == 'random':
            generator = torch.Generator().manual_seed(int(knockout_seed if seed is None else seed) + 1_000_003)
            if self.kind == 'quantum':
                self._random_core = dict(weights=2 * math.pi * torch.rand(self.quantum_params[0].shape,
                    generator=generator) - math.pi, scales=torch.ones_like(self.input_scales[0], device='cpu'))
            else:
                self._random_core = {name: torch.randn(parameter.shape, generator=generator) / math.sqrt(max(1, parameter.numel()))
                                     for name, parameter in self.core.named_parameters()}
        return self

    def simulator_info(self):
        quantum = self.kind == 'quantum'
        info = dict(backend='pennylane' if quantum else 'classical',
            q_backend='pennylane' if quantum else None,
            family='pennylane' if quantum else 'classical',
            device='default.qubit' if quantum else None,
            sim_device='cpu' if quantum else 'model', n_qubits=4 if quantum else None,
            local_spatial_kernel=self.kind, local_spatial_readout=self.readout,
            local_spatial_channels=self.feature_channels, q_training_control=self.q_training_control,
            core_shared_over='locations and projected channels', max_tokens=self._MAX_TOKENS)
        if quantum:
            info.update(interface='torch', diff_method='backprop',
                broadcast='batch of local 2x2 windows',
                readout_method='exact expectations derived from the statevector')
        return info

    def _tokens(self, x):
        projected = self.pre_q_conv(x)
        batch, channels, height, width = projected.shape
        windows = F.unfold(F.pad(projected, (0, 1, 0, 1), mode='replicate'), kernel_size=2)
        return windows.reshape(batch, channels, 4, height * width).permute(0, 1, 3, 2).reshape(-1, 4)

    def angle_diagnostics(self, x):
        """Input-angle saturation on this batch, without retaining a graph."""
        if self.kind not in ('quantum', 'sincos'):
            return None
        with torch.no_grad():
            tokens = self._tokens(x)
            scales = self.input_scales[0]
            bounded = torch.tanh(tokens[:, None, :] * scales.to(tokens.device)[None, :, :])
            return dict(n_angles=bounded.numel(), saturation_fraction=float((bounded.abs() >= 0.95).float().mean()),
                        mean_abs_tanh=float(bounded.abs().mean()), max_abs_tanh=float(bounded.abs().max()))

    def _evaluate_core(self, tokens):
        if self.kind == 'quantum':
            weights = self.quantum_params[0] if self._random_core is None else self._random_core['weights']
            scales = self.input_scales[0] if self._random_core is None else self._random_core['scales']
            original_device, dtype = tokens.device, tokens.dtype
            state = _state_circuit(self.q_layers, self.q_training_control != 'no_entanglement')(
                tokens.to('cpu'), weights.to('cpu'), scales.to('cpu'))
            result = state_readouts(state)
            if self.readout == 'z':
                result = torch.cat((result[:, :4], torch.zeros_like(result[:, 4:])), dim=-1)
            return result.to(device=original_device, dtype=dtype)
        if self.kind == 'sincos':
            angle = math.pi * torch.tanh(tokens * self.input_scales[0])
            cosine = torch.cos(angle)
            tokens = torch.cat((torch.sin(angle), cosine,
                                torch.stack([cosine[:, a] * cosine[:, b] for a, b in LOCAL_EDGES], dim=-1)), dim=-1)
        if self._random_core is None:
            return self.core(tokens)
        replacements = {key: value.to(device=tokens.device, dtype=tokens.dtype) for key, value in self._random_core.items()}
        return torch.func.functional_call(self.core, replacements, (tokens,))

    def forward(self, x, params=None):
        if x.ndim != 4 or x.shape[1] != self.quantum_channels:
            raise ValueError('Local spatial block expects [B, quantum_channels, H, W]')
        if self.quantum_mode == 'zero':
            return torch.zeros_like(x)
        batch, _, height, width = x.shape
        tokens = self._tokens(x)
        chunks = []
        for chunk in tokens.split(self._MAX_TOKENS):
            if self.kind == 'quantum' and torch.is_grad_enabled() and (chunk.requires_grad or any(p.requires_grad for p in self.parameters())):
                output = torch.utils.checkpoint.checkpoint(self._evaluate_core, chunk, use_reentrant=False)
            else:
                output = self._evaluate_core(chunk)
            chunks.append(output)
        features = torch.cat(chunks).reshape(batch, self.feature_channels, height, width, 12)
        features = features.permute(0, 1, 4, 2, 3).reshape(batch, 12 * self.feature_channels, height, width)
        return self.post_q_proj(features)


def build_local_spatial_block(config, model_type, seed=None):
    options = local_spatial_options(config)
    kind = options['local_spatial_kernel']
    if kind is None:
        raise ValueError('local_spatial_kernel must select a local core')
    if model_type not in ('quantum', 'ablation') or (model_type == 'quantum') != (kind == 'quantum'):
        raise ValueError('local quantum core requires model_type quantum; classical local cores require ablation')
    if (config.get('q_ansatz', 'reupload') != 'reupload' or int(config.get('n_qubits', 4)) != 4
            or config.get('q_spatial_mode', 'global') != 'global'):
        raise ValueError('Local spatial filters require reupload, n_qubits=4, and q_spatial_mode=global')
    if kind == 'quantum' and (config.get('q_backend', 'pennylane') != 'pennylane'
            or config.get('q_sim_device', 'cpu') != 'cpu' or config.get('quantum_backend', 'pennylane') != 'pennylane'):
        raise ValueError('Local quantum filters require the PennyLane CPU simulator')
    return LocalSpatialBottleneckBlock(quantum_channels=config['quantum_channels'], kind=kind,
        readout=options['local_spatial_readout'], feature_channels=options['local_spatial_channels'],
        q_layers=config.get('q_layers') if config.get('q_layers') is not None else 2,
        q_training_control=config.get('q_training_control', 'trainable'), seed=seed)

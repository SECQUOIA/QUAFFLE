#!/usr/bin/env python3
"""
Utility functions for quantum flood segmentation using QVUNet with PennyLane.
Contains common training, testing, and visualization functions.
"""

import os
import glob
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from . import unetTorch as unet

import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import flood_data
from flood_data import set_data_defaults

class FloodSegmentationDataset(Dataset):
    """Flood tiles as (CHW image in [-1, 1], 1HW binary mask).

    By default (clean mode) the committed split manifest splits/<dataset>.json
    is honored for flood_optical, flood_SAR and flood_combined, bands are
    selected by name and normalized with training-split statistics
    (splits/norm_stats.json). legacy=True (or set_data_defaults(legacy=True),
    or QUAFFLE_LEGACY_DATA=1) restores the original loader: all files, first
    three bands, per-image percentile stretch. See docs/DATA.md.
    """
    def __init__(self, images_dir, masks_dir, image_size, legacy=None, bands=None,
                 manifest=None, stats_path=None, max_samples=None, sample_seed=42,
                 data_root=None, seed=None, val_fraction=None, fit_only=None,
                 combined_input_mode=None):
        self.combined_input_mode = flood_data.normalize_combined_input_mode(combined_input_mode)
        self.samples, self.names = flood_data.load_samples(
            images_dir, masks_dir, image_size, legacy=legacy, bands=bands,
            manifest=manifest, stats_path=stats_path, progress=tqdm,
            max_samples=max_samples, sample_seed=sample_seed, data_root=data_root,
            seed=seed, val_fraction=val_fraction, fit_only=fit_only,
            combined_input_mode=self.combined_input_mode)
        self.norm_provenance = flood_data.get_last_loaded_stats_provenance(images_dir)
        self.modalities = flood_data.get_last_loaded_modalities(images_dir)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        img = torch.from_numpy(np.array(img.transpose(2, 0, 1), dtype=np.float32, copy=True))  # CHW
        mask = torch.from_numpy(np.array(mask.transpose(2, 0, 1), dtype=np.float32, copy=True))  # 1HW
        return img, mask

def get_dataloaders(images_dir, masks_dir, image_size, batch_size, num_workers=2, shuffle=True,
                    legacy=None, bands=None, manifest=None, stats_path=None,
                    max_samples=None, sample_seed=42, data_root=None,
                    seed=None, val_fraction=None, fit_only=None, combined_input_mode=None):
    dataset = FloodSegmentationDataset(
        images_dir, masks_dir, image_size, legacy=legacy, bands=bands,
        manifest=manifest, stats_path=stats_path, max_samples=max_samples,
        sample_seed=sample_seed, data_root=data_root, seed=seed,
        val_fraction=val_fraction, fit_only=fit_only,
        combined_input_mode=combined_input_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def _build_seeded(seed, build_fn):
    """Run build_fn() and return its result. If seed is None the global torch RNG
    is used as seeded by the caller. If seed is an int the whole construction
    runs under a forked CPU RNG seeded with it, so the result depends only on
    seed and the caller's CPU and CUDA RNG states are left untouched."""
    if seed is None:
        return build_fn()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        return build_fn()


def quantum_model_options(config):
    """Bottleneck options read from a run config (defaults in parentheses):
      q_ansatz ('reupload'): 'legacy' reproduces the original circuit,
          'reupload' is the data re-uploading circuit, 'hqunet' is the
          2D spatial quanvolutional grid circuit.
      q_layers (2): re-uploading / quanvolutional layers (ignored for 'legacy').
      q_share_blocks (True): qresblock1 and qresblock2 share one bottleneck
          block (original behaviour). False gives each its own parameters.
    """
    return (config.get('q_ansatz', 'reupload'),
            int(config.get('q_layers', 2)),
            bool(config.get('q_share_blocks', True)))


LOSS_CHOICES = ('ce', 'dice_ce', 'focal_dice')
DEFAULT_LOSS_MODE = 'ce'


def normalize_loss_mode(loss_mode):
    """Normalize segmentation loss mode ('ce', 'dice_ce', 'focal_dice')."""
    if loss_mode is None:
        return DEFAULT_LOSS_MODE
    key = str(loss_mode).strip().lower()
    if key not in LOSS_CHOICES:
        raise ValueError(f"Unknown loss mode {loss_mode!r}, expected one of {LOSS_CHOICES}")
    return key


def soft_dice_loss(logits, masks, smooth: float = 1e-6):
    """Differentiable foreground (class-1) soft Dice loss (#16)."""
    masks_long = masks.squeeze(1).long() if masks.dim() == 4 else masks.long()
    probs_fg = torch.softmax(logits, dim=1)[:, 1]
    targets_fg = (masks_long == 1).to(probs_fg.dtype)
    intersection = (probs_fg * targets_fg).sum()
    denom = probs_fg.sum() + targets_fg.sum()
    return 1.0 - (2.0 * intersection + float(smooth)) / (denom + float(smooth))


def focal_loss(logits, masks, gamma: float = 2.0, alpha: Optional[float] = 0.75):
    """Multi-class / binary focal loss on segmentation logits (#16)."""
    masks_long = masks.squeeze(1).long() if masks.dim() == 4 else masks.long()
    ce = torch.nn.functional.cross_entropy(logits, masks_long, reduction='none')
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** float(gamma)) * ce
    if alpha is not None:
        alpha_t = torch.where(masks_long == 1, float(alpha), 1.0 - float(alpha))
        focal = alpha_t.to(focal.dtype) * focal
    return focal.mean()


class QVUNetSegmentation(torch.nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        # Store config for lazy initialization
        self.config = config
        # seed=None: initialize from the global RNG (caller seeds it).
        # seed=int: deterministic init from a forked RNG, global RNG untouched.
        self.seed = seed
        self._model = None
    
    def _initialize_model(self):
        """Lazy initialization to avoid Ray serialization issues."""
        if self._model is None:
            # Import locally to avoid global state issues
            import torch.nn as nn
            from . import unetTorch as unet
            
            quantum_channels = self.config['quantum_channels']
            q_ansatz, q_layers, share = quantum_model_options(self.config)
            q_spatial_mode = unet.normalize_q_spatial_mode(self.config.get('q_spatial_mode'))
            hqunet_grid = unet.normalize_hqunet_grid(self.config.get('hqunet_grid'))
            use_time_emb = bool(self.config.get('use_time_emb', True))
            backbone_conv_mode = unet.normalize_backbone_conv_mode(
                self.config.get('backbone_conv_mode')
            )
            combined_input_mode = unet.normalize_combined_input_mode(
                self.config.get('combined_input_mode')
            )

            def build():
                def make_block(offset):
                    return unet.QuantumBlock(
                        backend=self.config.get('quantum_backend', 'pennylane'),
                        quantum_channels=quantum_channels,
                        ptlayer_config=self.config.get('ptlayer_config', None),
                        seed=None if self.seed is None else int(self.seed) + offset,
                        q_ansatz=q_ansatz,
                        q_layers=q_layers,
                        q_backend=self.config.get('q_backend', 'pennylane'),
                        q_sim_device=self.config.get('q_sim_device', 'cpu'),
                        q_spatial_mode=q_spatial_mode,
                        hqunet_grid=hqunet_grid,
                    )
                quantum_block = make_block(0)
                quantum_block2 = None if share else make_block(1)
                raw_skip = self.config.get('skip_modulation')
                if raw_skip is None:
                    raw_skip = self.config.get('q_residual_mode', unet.DEFAULT_SKIP_MODULATION)
                skip_modulation = unet.normalize_skip_modulation(raw_skip)
                return unet.QVUNet(
                    dim=self.config['base_channels'],
                    quantum_block=quantum_block,
                    init_dim=self.config.get('init_dim', self.config['base_channels']),
                    out_dim=self.config.get('out_dim', 2),  # Binary segmentation
                    dim_mults=self.config.get('dim_mults', (1, 2, 4, 8)),
                    resnet_block_groups=self.config.get('resnet_block_groups', 8),
                    quantum_channels=quantum_channels,
                    quantum_block2=quantum_block2,
                    skip_modulation=skip_modulation,
                    use_time_emb=use_time_emb,
                    backbone_conv_mode=backbone_conv_mode,
                    combined_input_mode=combined_input_mode,
                )

            self._model = _build_seeded(self.seed, build)

    def set_quantum_mode(self, mode='on', knockout_seed=0):
        """Quantum knockout: 'on', 'zero' or 'random'. See QuantumBlock."""
        self._initialize_model()
        self._model.set_quantum_mode(mode, knockout_seed=knockout_seed)
        return self

    def quantum_simulator_info(self):
        """How the bottleneck circuit is simulated (see QuantumBlock.simulator_info)."""
        return self.model.vertex.qresblock1.quantum_block.simulator_info()
    
    @property
    def model(self):
        """Property to ensure model is initialized before access."""
        self._initialize_model()
        return self._model
    
    def forward(self, x):
        B = x.shape[0]
        # Create dummy time steps for diffusion-style model
        dummy_time = torch.zeros((B,), dtype=torch.float32, device=x.device)
        return self.model(x, dummy_time)
    
    def parameters(self):
        """Override to ensure model is initialized."""
        self._initialize_model()
        return self._model.parameters()
    
    def named_parameters(self):
        """Override to ensure model is initialized."""
        self._initialize_model()
        return self._model.named_parameters()
    
    def state_dict(self):
        """Override to ensure model is initialized."""
        self._initialize_model()
        return self._model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Override to ensure model is initialized."""
        self._initialize_model()
        return self._model.load_state_dict(state_dict)
    
    def train(self, mode=True):
        """Override to ensure model is initialized."""
        self._initialize_model()
        self._model.train(mode)
        return super().train(mode)
    
    def eval(self):
        """Override to ensure model is initialized."""
        self._initialize_model()
        self._model.eval()
        return super().eval()
    
    def to(self, device):
        """Override to ensure model is initialized and moved to device."""
        self._initialize_model()
        self._model = self._model.to(device)
        return super().to(device)

class ClassicalAblationUNetSegmentation(torch.nn.Module):
    """Parameter-matched classical ablation of QVUNetSegmentation.

    Uses the same QVUNet architecture and the same QVertex bottleneck, but swaps
    the PQC-based QuantumBlock for a ClassicalBottleneckBlock with the same
    spatial mode and parameter budget.
    """
    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config
        self.seed = seed
        self._model = None

    def _initialize_model(self):
        if self._model is None:
            from . import unetTorch as unet
            quantum_channels = self.config['quantum_channels']
            q_ansatz, q_layers, share = quantum_model_options(self.config)
            q_spatial_mode = unet.normalize_q_spatial_mode(self.config.get('q_spatial_mode'))
            ablation_mode = unet.normalize_ablation_mode(self.config.get('ablation_mode'))
            hqunet_grid = unet.normalize_hqunet_grid(self.config.get('hqunet_grid'))
            use_time_emb = bool(self.config.get('use_time_emb', True))
            backbone_conv_mode = unet.normalize_backbone_conv_mode(
                self.config.get('backbone_conv_mode')
            )
            combined_input_mode = unet.normalize_combined_input_mode(
                self.config.get('combined_input_mode')
            )
            if q_ansatz == 'legacy':
                kind, hidden = 'linear', None
            elif q_ansatz == 'hqunet':
                kind, hidden = 'hqunet_spatial', unet.matched_mlp_hidden(16 * q_layers)
            else:
                kind, hidden = 'mlp', unet.matched_mlp_hidden(16 * q_layers)
            if ablation_mode != 'auto':
                kind = ablation_mode

            def build():
                def make_block(offset):
                    return unet.ClassicalBottleneckBlock(
                        quantum_channels=quantum_channels,
                        kind=kind,
                        hidden=hidden,
                        seed=None if self.seed is None else int(self.seed) + offset,
                        q_ansatz=q_ansatz,
                        q_layers=q_layers,
                        spatial_mode=q_spatial_mode,
                        ablation_mode=ablation_mode,
                        hqunet_grid=hqunet_grid,
                    )
                classical_bottleneck = make_block(0)
                block2 = None if share else make_block(1)
                raw_skip = self.config.get('skip_modulation')
                if raw_skip is None:
                    raw_skip = self.config.get('q_residual_mode', unet.DEFAULT_SKIP_MODULATION)
                skip_modulation = unet.normalize_skip_modulation(raw_skip)
                return unet.QVUNet(
                    dim=self.config['base_channels'],
                    quantum_block=classical_bottleneck,
                    init_dim=self.config.get('init_dim', self.config['base_channels']),
                    out_dim=self.config.get('out_dim', 2),
                    dim_mults=self.config.get('dim_mults', (1, 2, 4, 8)),
                    resnet_block_groups=self.config.get('resnet_block_groups', 8),
                    quantum_channels=quantum_channels,
                    quantum_block2=block2,
                    skip_modulation=skip_modulation,
                    use_time_emb=use_time_emb,
                    backbone_conv_mode=backbone_conv_mode,
                    combined_input_mode=combined_input_mode,
                )

            self._model = _build_seeded(self.seed, build)

    def set_quantum_mode(self, mode='on', knockout_seed=0):
        """Same knockout switch as the quantum model, applied to the classical bottleneck."""
        self._initialize_model()
        self._model.set_quantum_mode(mode, knockout_seed=knockout_seed)
        return self

    @property
    def model(self):
        self._initialize_model()
        return self._model

    def forward(self, x):
        B = x.shape[0]
        dummy_time = torch.zeros((B,), dtype=torch.float32, device=x.device)
        return self.model(x, dummy_time)

    def parameters(self):
        self._initialize_model()
        return self._model.parameters()

    def named_parameters(self):
        self._initialize_model()
        return self._model.named_parameters()

    def state_dict(self):
        self._initialize_model()
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        self._initialize_model()
        return self._model.load_state_dict(state_dict)

    def train(self, mode=True):
        self._initialize_model()
        self._model.train(mode)
        return super().train(mode)

    def eval(self):
        self._initialize_model()
        self._model.eval()
        return super().eval()

    def to(self, device):
        self._initialize_model()
        self._model = self._model.to(device)
        return super().to(device)


class UNetSegmentation(torch.nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        # Store config for lazy initialization
        self.config = config
        self.seed = seed  # same semantics as QVUNetSegmentation
        self._model = None
    
    def _initialize_model(self):
        """Lazy initialization to avoid Ray serialization issues."""
        if self._model is None:
            # Import locally to avoid global state issues
            from . import unetTorch as unet
            use_time_emb = bool(self.config.get('use_time_emb', True))
            backbone_conv_mode = unet.normalize_backbone_conv_mode(
                self.config.get('backbone_conv_mode')
            )
            combined_input_mode = unet.normalize_combined_input_mode(
                self.config.get('combined_input_mode')
            )
            
            self._model = _build_seeded(self.seed, lambda: unet.UNet(
                dim=self.config['base_channels'],
                init_dim=self.config.get('init_dim', self.config['base_channels']),
                out_dim=self.config.get('out_dim', 2),  # Binary segmentation
                dim_mults=self.config.get('dim_mults', (1, 2, 4, 8)),
                resnet_block_groups=self.config.get('resnet_block_groups', 8),
                use_time_emb=use_time_emb,
                backbone_conv_mode=backbone_conv_mode,
                combined_input_mode=combined_input_mode,
            ))
    
    @property
    def model(self):
        """Property to ensure model is initialized before access."""
        self._initialize_model()
        return self._model
    
    def forward(self, x):
        B = x.shape[0]
        # Create dummy time steps for diffusion-style model
        dummy_time = torch.zeros((B,), dtype=torch.float32, device=x.device)
        return self.model(x, dummy_time)
    
    def parameters(self):
        """Override to ensure model is initialized."""
        self._initialize_model()
        return self._model.parameters()
    
    def named_parameters(self):
        """Override to ensure model is initialized."""
        self._initialize_model()
        return self._model.named_parameters()
    
    def state_dict(self):
        """Override to ensure model is initialized."""
        self._initialize_model()
        return self._model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Override to ensure model is initialized."""
        self._initialize_model()
        return self._model.load_state_dict(state_dict)
    
    def train(self, mode=True):
        """Override to ensure model is initialized."""
        self._initialize_model()
        self._model.train(mode)
        return super().train(mode)
    
    def eval(self):
        """Override to ensure model is initialized."""
        self._initialize_model()
        self._model.eval()
        return super().eval()
    
    def to(self, device):
        """Override to ensure model is initialized and moved to device."""
        self._initialize_model()
        self._model = self._model.to(device)
        return super().to(device)


def _resolve_model_loss_mode(model, loss_mode=None):
    if loss_mode is not None:
        return normalize_loss_mode(loss_mode)
    attr_mode = getattr(model, 'loss_mode', None)
    if attr_mode is not None:
        return normalize_loss_mode(attr_mode)
    cfg = getattr(model, 'config', None)
    if isinstance(cfg, dict) and cfg.get('loss') is not None:
        return normalize_loss_mode(cfg.get('loss'))
    return DEFAULT_LOSS_MODE


def segmentation_loss(
    logits,
    masks,
    loss_mode: Optional[str] = None,
    smooth: float = 1e-6,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = 0.75,
):
    """Compute segmentation loss ('ce', 'dice_ce', or 'focal_dice'; #16)."""
    mode = normalize_loss_mode(loss_mode)
    masks_long = masks.squeeze(1).long() if masks.dim() == 4 else masks.long()
    if mode == 'ce':
        return torch.nn.functional.cross_entropy(logits, masks_long)
    if mode == 'dice_ce':
        ce = torch.nn.functional.cross_entropy(logits, masks_long)
        dice = soft_dice_loss(logits, masks_long, smooth=smooth)
        return ce + dice
    if mode == 'focal_dice':
        fl = focal_loss(logits, masks_long, gamma=focal_gamma, alpha=focal_alpha)
        dice = soft_dice_loss(logits, masks_long, smooth=smooth)
        return fl + dice
    raise ValueError(f"Unknown loss mode {loss_mode!r}")


def fedprox_proximal_loss(model, global_params, mu: float = 0.0):
    """Compute the FedProx proximal penalty (mu / 2) * sum_p ||w_p - w_p^(t)||_2^2 (#15).

    `global_params` may be a sequence of torch.Tensors or numpy ndarrays matching
    `model.parameters()`. Returns a scalar Tensor on the model's device.
    """
    mu_val = float(mu or 0.0)
    params = list(model.parameters())
    if not params:
        return torch.tensor(0.0)
    dev = params[0].device
    if mu_val <= 0.0 or global_params is None:
        return torch.zeros((), dtype=params[0].dtype, device=dev)
    prox = torch.zeros((), dtype=params[0].dtype, device=dev)
    for p, p0 in zip(params, global_params):
        if not p.requires_grad:
            continue
        ref = p0 if isinstance(p0, torch.Tensor) else torch.as_tensor(p0, dtype=p.dtype, device=p.device)
        if ref.device != p.device or ref.dtype != p.dtype:
            ref = ref.to(device=p.device, dtype=p.dtype)
        prox = prox + torch.sum((p - ref.detach()) ** 2)
    return 0.5 * mu_val * prox


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    loss_mode: Optional[str] = None,
    fedprox_mu: float = 0.0,
    global_params=None,
):
    model.train()
    eff_loss_mode = _resolve_model_loss_mode(model, loss_mode)
    mu_val = float(fedprox_mu or 0.0)
    ref_params = None
    if mu_val > 0.0 and global_params is not None:
        ref_params = [
            (p0.detach().to(device) if isinstance(p0, torch.Tensor) else torch.as_tensor(p0, device=device))
            for p0 in global_params
        ]
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    for imgs, masks in tqdm(dataloader, desc='Training'):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = segmentation_loss(logits, masks, loss_mode=eff_loss_mode)
        if mu_val > 0.0 and ref_params is not None:
            loss = loss + fedprox_proximal_loss(model, ref_params, mu_val)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == masks.squeeze(1)).sum().item()
        total_correct += correct
        total_pixels += np.prod(preds.shape)
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_pixels
    return avg_loss, accuracy


def evaluate_model(model, dataloader, device, single_class_auc=0.0, loss_mode: Optional[str] = None):
    model.eval()
    eff_loss_mode = _resolve_model_loss_mode(model, loss_mode)
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    total_intersection = 0
    total_pred_pos = 0
    total_true_pos = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc='Evaluating'):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = segmentation_loss(logits, masks, loss_mode=eff_loss_mode)
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            masks_int = masks.squeeze(1).long()
            correct = (preds == masks_int).sum().item()
            total_correct += correct
            total_pixels += np.prod(preds.shape)
            total_intersection += ((preds == 1) & (masks_int == 1)).sum().item()
            total_pred_pos += (preds == 1).sum().item()
            total_true_pos += (masks_int == 1).sum().item()
            probs = torch.softmax(logits, dim=1)[:, 1, ...]
            all_probs.append(probs.cpu().numpy().ravel())
            all_labels.append(masks.cpu().numpy().ravel())
    n_samples = len(dataloader.dataset)
    avg_loss = float(total_loss / n_samples) if n_samples > 0 else 0.0
    accuracy = float(total_correct / total_pixels) if total_pixels > 0 else 0.0
    union = total_pred_pos + total_true_pos - total_intersection
    iou = float(total_intersection / union) if union > 0 else 0.0
    dice = (
        float(2 * total_intersection / (total_pred_pos + total_true_pos))
        if (total_pred_pos + total_true_pos) > 0
        else 0.0
    )
    all_probs_arr = np.concatenate(all_probs) if all_probs else np.empty(0, dtype=np.float32)
    all_labels_arr = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.float32)
    try:
        if all_labels_arr.size == 0 or np.unique(all_labels_arr).size < 2:
            auc_score = float(single_class_auc)
        else:
            auc_score = float(roc_auc_score(all_labels_arr, all_probs_arr))
            if np.isnan(auc_score):
                auc_score = float(single_class_auc)
    except ValueError:
        auc_score = float(single_class_auc)
    return {'loss': avg_loss, 'accuracy': accuracy, 'auc': auc_score, 'iou': iou, 'dice': dice}

def save_training_curves(metrics, output_dir, prefix="", title_prefix=""):
    """Save training curves - generic version for both centralized and federated."""
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if not metrics:
        print("No metrics to save")
        return
    if os.environ.get("QUAFFLE_SKIP_CURVE_PLOTS", "0").lower() in ("1", "true", "yes"):
        return
    
    # Extract metrics - handle different key formats without shifting the x-axis origin
    is_federated = ('round' in metrics[0]) or ('fed' in f"{prefix} {title_prefix}".lower())
    if 'round' in metrics[0]:
        # Federated metrics
        x_values = [m['round'] for m in metrics]
        x_label = 'Federated Round'
    elif 'epoch' in metrics[0]:
        # Centralized metrics using epochs
        x_values = [m['epoch'] for m in metrics]
        x_label = 'Federated Round' if is_federated else 'Training Epoch'
    else:
        # Centralized/federated metrics using steps (preserve actual 1-indexed step/round values)
        x_values = [m['step'] for m in metrics]
        x_label = 'Federated Round' if is_federated else 'Training Step'
    
    losses = [m['loss'] for m in metrics]
    accuracies = [m['accuracy'] for m in metrics]
    aucs = [m['auc'] for m in metrics]
    ious = [m.get('iou', 0.0) for m in metrics]
    dices = [m.get('dice', 0.0) for m in metrics]

    # 1. Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, losses, 'b-', label='Loss', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title(f'{title_prefix}Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{prefix}loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, accuracies, 'r-', label='Accuracy', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.title(f'{title_prefix}Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{prefix}accuracy_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. AUC curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, aucs, 'g-', label='AUC', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel('AUC Score')
    plt.title(f'{title_prefix}AUC Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{prefix}auc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. IoU curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, ious, 'm-', label='IoU', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel('IoU Score')
    plt.title(f'{title_prefix}IoU Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{prefix}iou_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Dice curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, dices, 'c-', label='Dice', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel('Dice Score')
    plt.title(f'{title_prefix}Dice Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{prefix}dice_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved with prefix '{prefix}'")

def visualize_results(model, dataloader, device, output_dir, title_prefix="", filename_prefix="", num_samples=4):
    output_dir = os.path.abspath(output_dir)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    if num_samples <= 0:
        return
    model.eval()
    count = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1, ...]
            for i in range(imgs.size(0)):
                if count >= num_samples:
                    break
                raw_chw = imgs[i].cpu().numpy()
                if raw_chw.shape[0] >= 6:
                    opt_chw, sar_chw = raw_chw[:3], raw_chw[3:6]
                    raw_chw = sar_chw if np.abs(sar_chw).max() > np.abs(opt_chw).max() else opt_chw
                elif raw_chw.shape[0] > 3:
                    raw_chw = raw_chw[:3]
                image_np = np.clip((raw_chw.transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
                mask_np = masks[i, 0].cpu().numpy()
                prediction_np = preds[i].cpu().numpy()
                probabilities_np = probs[i].cpu().numpy()
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                axes[0, 0].imshow(image_np)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')
                axes[0, 1].imshow(mask_np, cmap='gray')
                axes[0, 1].set_title('True Mask')
                axes[0, 1].axis('off')
                axes[1, 0].imshow(prediction_np, cmap='gray')
                axes[1, 0].set_title('Predicted Mask')
                axes[1, 0].axis('off')
                im = axes[1, 1].imshow(probabilities_np, cmap='Blues')
                axes[1, 1].set_title('Flood Probability')
                axes[1, 1].axis('off')
                plt.colorbar(im, ax=axes[1, 1])
                plt.suptitle(f'{title_prefix}Quantum Flood Segmentation - Sample {count+1}')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'{filename_prefix}sample_{count+1}.png'), dpi=150, bbox_inches='tight')
                plt.close()
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
    print(f"Visualizations saved in {results_dir}")





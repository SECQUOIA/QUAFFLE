#!/usr/bin/env python3
"""
Utility functions for quantum flood segmentation using QVUNet with PennyLane.
Contains common training, testing, and visualization functions.
"""

import os
import glob
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
                 manifest=None, stats_path=None):
        self.samples, self.names = flood_data.load_samples(
            images_dir, masks_dir, image_size, legacy=legacy, bands=bands,
            manifest=manifest, stats_path=stats_path, progress=tqdm)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # CHW
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()  # 1HW
        return img, mask

def get_dataloaders(images_dir, masks_dir, image_size, batch_size, num_workers=2, shuffle=True,
                    legacy=None, bands=None, manifest=None, stats_path=None):
    dataset = FloodSegmentationDataset(images_dir, masks_dir, image_size, legacy=legacy,
                                       bands=bands, manifest=manifest, stats_path=stats_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def _build_seeded(seed, build_fn):
    """Run build_fn() and return its result. If seed is None the global torch RNG
    is used as seeded by the caller. If seed is an int the whole construction
    runs under a forked RNG seeded with it, so the result depends only on seed
    and the caller's global RNG state is left untouched."""
    if seed is None:
        return build_fn()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return build_fn()


def quantum_model_options(config):
    """Bottleneck options read from a run config (defaults in parentheses):
      q_ansatz ('reupload'): 'legacy' reproduces the original circuit,
          'reupload' is the data re-uploading circuit.
      q_layers (2): re-uploading layers (ignored for 'legacy').
      q_share_blocks (True): qresblock1 and qresblock2 share one bottleneck
          block (original behaviour). False gives each its own parameters.
    """
    return (config.get('q_ansatz', 'reupload'),
            int(config.get('q_layers', 2)),
            bool(config.get('q_share_blocks', True)))


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

            def build():
                def make_block(offset):
                    return unet.QuantumBlock(
                        backend=self.config.get('quantum_backend', 'pennylane'),
                        quantum_channels=quantum_channels,
                        ptlayer_config=self.config.get('ptlayer_config', None),
                        seed=None if self.seed is None else int(self.seed) + offset,
                        q_ansatz=q_ansatz,
                        q_layers=q_layers,
                        # 'torch' (exact batched simulator, default) or
                        # 'pennylane' (default.qubit QNodes). Same math.
                        q_backend=self.config.get('q_backend', 'torch'),
                    )
                quantum_block = make_block(0)
                quantum_block2 = None if share else make_block(1)
                return unet.QVUNet(
                    dim=self.config['base_channels'],
                    quantum_block=quantum_block,
                    init_dim=self.config.get('init_dim', self.config['base_channels']),
                    out_dim=self.config.get('out_dim', 2),  # Binary segmentation
                    dim_mults=self.config.get('dim_mults', (1, 2, 4, 8)),
                    resnet_block_groups=self.config.get('resnet_block_groups', 8),
                    quantum_channels=quantum_channels,
                    quantum_block2=quantum_block2,
                )

            self._model = _build_seeded(self.seed, build)

    def set_quantum_mode(self, mode='on', knockout_seed=0):
        """Quantum knockout: 'on', 'zero' or 'random'. See QuantumBlock."""
        self._initialize_model()
        self._model.set_quantum_mode(mode, knockout_seed=knockout_seed)
        return self
    
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
    pool, per-group transform, broadcast pattern. The per-group transform is
    matched to the configured q_ansatz:
      'legacy'   -> Linear(4, 4) per group (the original ablation)
      'reupload' -> Linear(4, h)-Tanh-Linear(h, 4)-Tanh per group with h chosen
                    so the per-group count is closest to the circuit's 16*q_layers
    q_share_blocks is honoured the same way as in QVUNetSegmentation, so totals
    match the quantum model to within a few hundred weights (see docs/MODEL.md).
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
            if q_ansatz == 'legacy':
                kind, hidden = 'linear', None
            else:
                kind, hidden = 'mlp', unet.matched_mlp_hidden(16 * q_layers)

            def build():
                def make_block():
                    return unet.ClassicalBottleneckBlock(quantum_channels=quantum_channels,
                                                         kind=kind, hidden=hidden)
                classical_bottleneck = make_block()
                block2 = None if share else make_block()
                return unet.QVUNet(
                    dim=self.config['base_channels'],
                    quantum_block=classical_bottleneck,
                    init_dim=self.config.get('init_dim', self.config['base_channels']),
                    out_dim=self.config.get('out_dim', 2),
                    dim_mults=self.config.get('dim_mults', (1, 2, 4, 8)),
                    resnet_block_groups=self.config.get('resnet_block_groups', 8),
                    quantum_channels=quantum_channels,
                    quantum_block2=block2,
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
            
            self._model = _build_seeded(self.seed, lambda: unet.UNet(
                dim=self.config['base_channels'],
                init_dim=self.config.get('init_dim', self.config['base_channels']),
                out_dim=self.config.get('out_dim', 2),  # Binary segmentation
                dim_mults=self.config.get('dim_mults', (1, 2, 4, 8)),
                resnet_block_groups=self.config.get('resnet_block_groups', 8)
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

def segmentation_loss(logits, masks):
    # logits: [B, num_classes, H, W], masks: [B, 1, H, W] or [B, H, W]
    masks = masks.squeeze(1).long()  # [B, H, W]
    return torch.nn.functional.cross_entropy(logits, masks)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    for imgs, masks in tqdm(dataloader, desc='Training'):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = segmentation_loss(logits, masks)
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

def evaluate_model(model, dataloader, device):
    model.eval()
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
            loss = segmentation_loss(logits, masks)
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
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(masks.cpu().numpy().flatten())
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_pixels
    union = total_pred_pos + total_true_pos - total_intersection
    iou = total_intersection / union if union > 0 else 0.0
    dice = 2 * total_intersection / (total_pred_pos + total_true_pos) if (total_pred_pos + total_true_pos) > 0 else 0.0
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except:
        auc_score = 0.0
    return {'loss': avg_loss, 'accuracy': accuracy, 'auc': auc_score, 'iou': iou, 'dice': dice}

def save_training_curves(metrics, output_dir, prefix="", title_prefix=""):
    """Save training curves - generic version for both centralized and federated."""
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if not metrics:
        print("No metrics to save")
        return
    
    # Extract metrics - handle different key formats
    if 'round' in metrics[0]:
        # Federated metrics
        x_values = [m['round'] for m in metrics]
        x_label = 'Federated Round'
    elif 'epoch' in metrics[0]:
        # Centralized metrics using epochs
        x_values = [m['epoch'] for m in metrics]
        x_label = 'Training Epoch'
    else:
        # Centralized metrics using steps
        x_values = [m['step'] for m in metrics]
        x_label = 'Training Step'
        # Normalize steps to start from 0
        if x_values:
            first_step = x_values[0]
            x_values = [x - first_step for x in x_values]
    
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
                image_np = (imgs[i].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
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
    print(f"Visualizations saved in {results_dir}")





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
import unetTorch as unet

class FloodSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size):
        image_paths = glob.glob(os.path.join(images_dir, "*.tif")) + \
                      glob.glob(os.path.join(images_dir, "*.tiff"))
        if not image_paths:
            raise ValueError(f"No TIFF images found in {images_dir}")
        self.samples = []
        for img_path in tqdm(image_paths, desc="Loading data"):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = None
            for ext in ['.png', '.jpg', '.tif']:
                potential_mask = os.path.join(masks_dir, img_name + ext)
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break
            if mask_path is None:
                continue
            try:
                try:
                    import rasterio
                    with rasterio.open(img_path) as src:
                        img_data = src.read()[:3]
                        img_array = np.transpose(img_data, (1, 2, 0))
                        if img_array.dtype == np.uint8:
                            img_array = img_array.astype(np.float32) / 255.0
                        elif img_array.dtype == np.uint16:
                            img_array = img_array.astype(np.float32) / 65535.0
                        else:
                            img_array = img_array.astype(np.float32)
                            for c in range(3):
                                channel = img_array[:, :, c]
                                low, high = np.percentile(channel, [2, 98])
                                if high > low:
                                    img_array[:, :, c] = np.clip((channel - low) / (high - low), 0, 1)
                        img = Image.fromarray((img_array * 255).astype(np.uint8))
                        img = img.resize((image_size, image_size), Image.LANCZOS)
                        img_array = np.array(img, dtype=np.float32) / 255.0
                except ImportError:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((image_size, image_size), Image.LANCZOS)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = img_array * 2.0 - 1.0
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((image_size, image_size), Image.NEAREST)
                mask_array = np.array(mask, dtype=np.float32) / 255.0
                mask_array = (mask_array > 0.001).astype(np.float32)[..., np.newaxis]
                self.samples.append((img_array, mask_array))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        if not self.samples:
            raise ValueError("No valid image-mask pairs loaded")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # CHW
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()  # 1HW
        return img, mask

def get_dataloaders(images_dir, masks_dir, image_size, batch_size, num_workers=2, shuffle=True):
    dataset = FloodSegmentationDataset(images_dir, masks_dir, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

class QVUNetSegmentation(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        quantum_channels = config['quantum_channels']
        
        # Use the new QuantumBlock API with ptlayer_config instead of ptlayer
        quantum_block = unet.QuantumBlock(
            backend=config.get('quantum_backend', 'pennylane'),
            quantum_channels=quantum_channels,
            ptlayer_config=config.get('ptlayer_config', None)
        )
        
        # Use the new QVUNet API
        self.model = unet.QVUNet(
            dim=config['base_channels'],
            quantum_block=quantum_block,
            init_dim=config.get('init_dim', config['base_channels']),
            out_dim=config.get('out_dim', 2),  # Binary segmentation
            dim_mults=config.get('dim_mults', (1, 2, 4, 8)),
            resnet_block_groups=config.get('resnet_block_groups', 8),
            quantum_channels=quantum_channels
        )
    
    def forward(self, x):
        B = x.shape[0]
        # Create dummy time steps for diffusion-style model
        dummy_time = torch.zeros((B,), dtype=torch.float32, device=x.device)
        return self.model(x, dummy_time)

class UNetSegmentation(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Use the new classical UNet API
        self.model = unet.UNet(
            dim=config['base_channels'],
            init_dim=config.get('init_dim', config['base_channels']),
            out_dim=config.get('out_dim', 2),  # Binary segmentation
            dim_mults=config.get('dim_mults', (1, 2, 4, 8)),
            resnet_block_groups=config.get('resnet_block_groups', 8)
        )
    
    def forward(self, x):
        B = x.shape[0]
        # Create dummy time steps for diffusion-style model
        dummy_time = torch.zeros((B,), dtype=torch.float32, device=x.device)
        return self.model(x, dummy_time)

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
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc='Evaluating'):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = segmentation_loss(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == masks.squeeze(1)).sum().item()
            total_correct += correct
            total_pixels += np.prod(preds.shape)
            probs = torch.softmax(logits, dim=1)[:, 1, ...]  # Flood probability
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(masks.cpu().numpy().flatten())
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_pixels
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except:
        auc_score = 0.0
    return {'loss': avg_loss, 'accuracy': accuracy, 'auc': auc_score}

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

def split_data_among_clients(images, masks, num_clients, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(images))
    client_data = []
    samples_per_client = len(images) // num_clients
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            end_idx = len(images)
        else:
            end_idx = (i + 1) * samples_per_client
        client_indices = indices[start_idx:end_idx]
        client_images = images[client_indices]
        client_masks = masks[client_indices]
        client_data.append((client_images, client_masks))
        print(f"Client {i}: {len(client_images)} samples")
    return client_data

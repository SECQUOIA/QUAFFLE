#!/usr/bin/env python3
"""
Utility functions for quantum flood segmentation using QVUNet with PennyLane.
Contains common training, testing, and visualization functions.
"""

import os
import glob
import numpy as np
from flax.training import train_state
from flax import linen as nn
import optax
import tensorflow as tf
from PIL import Image
import ml_collections
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from typing import List

# Import quantum model
from . import unetJAX as unet

# Disable TensorFlow GPU usage
tf.config.experimental.set_visible_devices([], "GPU")

def load_dataset(images_dir, masks_dir, config):
    """Load images and masks for segmentation."""
    # Get TIFF files
    image_paths = glob.glob(os.path.join(images_dir, "*.tif")) + \
                  glob.glob(os.path.join(images_dir, "*.tiff"))
    
    if not image_paths:
        raise ValueError(f"No TIFF images found in {images_dir}")
    
    images, masks = [], []
    
    for img_path in tqdm(image_paths, desc="Loading data"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Find corresponding mask
        mask_path = None
        for ext in ['.png', '.jpg', '.tif']:
            potential_mask = os.path.join(masks_dir, img_name + ext)
            if os.path.exists(potential_mask):
                mask_path = potential_mask
                break
        
        if mask_path is None:
            continue
        
        try:
            # Load image with rasterio for multi-band TIFF
            try:
                import rasterio
                with rasterio.open(img_path) as src:
                    img_data = src.read()[:3]  # Take first 3 bands as RGB
                    img_array = np.transpose(img_data, (1, 2, 0))
                    
                    # Normalize based on data type
                    if img_array.dtype == np.uint8:
                        img_array = img_array.astype(np.float32) / 255.0
                    elif img_array.dtype == np.uint16:
                        img_array = img_array.astype(np.float32) / 65535.0
                    else:
                        # Percentile normalization for float data
                        img_array = img_array.astype(np.float32)
                        for c in range(3):
                            channel = img_array[:, :, c]
                            low, high = np.percentile(channel, [2, 98])
                            if high > low:
                                img_array[:, :, c] = np.clip((channel - low) / (high - low), 0, 1)
                    
                    # Resize
                    img = Image.fromarray((img_array * 255).astype(np.uint8))
                    img = img.resize((config.image_size, config.image_size), Image.LANCZOS)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
            except ImportError:
                # Fallback to PIL
                img = Image.open(img_path).convert('RGB')
                img = img.resize((config.image_size, config.image_size), Image.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Normalize to [-1, 1]
            img_array = img_array * 2.0 - 1.0
            
            # Load mask
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((config.image_size, config.image_size), Image.NEAREST)
            mask_array = np.array(mask, dtype=np.float32) / 255.0
            mask_array = (mask_array > 0.001).astype(np.float32)[..., np.newaxis]
            
            images.append(img_array)
            masks.append(mask_array)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid image-mask pairs loaded")
    
    return np.stack(images), np.stack(masks)

def create_data_iterator(images, masks, config):
    """Create data iterator for training."""
    import jax
    batch_size = config.batch_size // jax.process_count()
    local_device_count = jax.local_device_count()
    
    def data_generator():
        while True:
            indices = np.random.permutation(len(images))
            shuffled_images = images[indices]
            shuffled_masks = masks[indices]
            
            for i in range(0, len(shuffled_images), batch_size):
                batch_images = shuffled_images[i:i + batch_size]
                batch_masks = shuffled_masks[i:i + batch_size]
                
                # Pad last batch if needed
                if len(batch_images) < batch_size:
                    padding = batch_size - len(batch_images)
                    batch_images = np.concatenate([batch_images, batch_images[-padding:]], axis=0)
                    batch_masks = np.concatenate([batch_masks, batch_masks[-padding:]], axis=0)
                
                # Reshape for devices
                batch_images = batch_images.reshape((local_device_count, -1) + batch_images.shape[1:])
                batch_masks = batch_masks.reshape((local_device_count, -1) + batch_masks.shape[1:])
                
                yield {'image': batch_images, 'mask': batch_masks}
    
    return data_generator()

class QVUNet_Segmentation(nn.Module):
    """Quantum segmentation model wrapper."""
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x):
        # Use QVUNet from unet.py
        qvunet = unet.QVUNet(
            dim=self.config.base_channels,
            out_dim=self.config.num_classes,
            dim_mults=self.config.dim_mults,
            resnet_block_groups=self.config.resnet_block_groups,
            quantum_channels=self.config.quantum_channels,
            name_ansatz=self.config.name_ansatz,
            num_layer=self.config.num_layer
        )
        
        # Dummy time input for segmentation
        import jax.numpy as jnp
        B = x.shape[0]
        dummy_time = jnp.zeros((B,), dtype=jnp.int32)
        
        return qvunet(x, dummy_time)

def create_train_state(rng, config):
    """Create training state."""
    import jax
    import jax.numpy as jnp
    model = QVUNet_Segmentation(config=config)
    dummy_input = jnp.ones((1, config.image_size, config.image_size, config.channels))
    params = model.init(rng, dummy_input)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")
    
    tx = optax.adam(learning_rate=config.learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def segmentation_loss(logits, masks):
    """Compute cross entropy loss for segmentation."""
    import jax
    import jax.numpy as jnp
    masks_one_hot = jax.nn.one_hot(masks.astype(jnp.int32).squeeze(-1), num_classes=2)
    return optax.softmax_cross_entropy(logits, masks_one_hot).mean()

def train_step(state, batch):
    """Training step."""
    import jax
    import jax.numpy as jnp
    
    def loss_fn(params):
        logits = state.apply_fn(params, batch['image'])
        loss = segmentation_loss(logits, batch['mask'])
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    predictions = jnp.argmax(logits, axis=-1)
    true_labels = batch['mask'].astype(jnp.int32).squeeze(-1)
    accuracy = jnp.mean(predictions == true_labels)
    
    return state, {'loss': loss, 'accuracy': accuracy}

# JIT compiled version for performance
_jitted_train_step = None

def get_jitted_train_step():
    """Get JIT compiled train_step function."""
    global _jitted_train_step
    if _jitted_train_step is None:
        import jax
        _jitted_train_step = jax.jit(train_step)
    return _jitted_train_step

def evaluate_model(state, images, masks, config):
    """Evaluate model."""
    import jax
    import jax.numpy as jnp
    batch_size = config.batch_size
    total_loss = total_accuracy = num_batches = 0
    all_probs = []
    all_labels = []
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_masks = masks[i:i + batch_size]
        
        if len(batch_images) < batch_size:
            break
        
        logits = state.apply_fn(state.params, batch_images)
        loss = segmentation_loss(logits, batch_masks)
        
        predictions = jnp.argmax(logits, axis=-1)
        probabilities = jax.nn.softmax(logits, axis=-1)[..., 1]  # Flood probability
        true_labels = batch_masks.astype(jnp.int32).squeeze(-1)
        accuracy = jnp.mean(predictions == true_labels)
        
        # Collect for AUC calculation
        all_probs.extend(np.array(probabilities).flatten())
        all_labels.extend(np.array(true_labels).flatten())
        
        total_loss += loss
        total_accuracy += accuracy
        num_batches += 1
    
    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except:
        auc_score = 0.0  # Fallback if AUC calculation fails
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'accuracy': total_accuracy / num_batches if num_batches > 0 else 0.0,
        'auc': auc_score
    }

def save_training_curves(metrics, output_dir, prefix="", title_prefix=""):
    """Save training curves - generic version for both centralized and federated."""
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if not metrics:
        print("No metrics to save")
        return
    
    # Extract metrics
    if 'round' in metrics[0]:
        # Federated metrics
        x_values = [m['round'] for m in metrics]
        x_label = 'Federated Round'
    else:
        # Centralized metrics
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

def visualize_results(state, test_images, test_masks, config, output_dir, title_prefix="", filename_prefix="", num_samples=4):
    """Create visualizations - generic version."""
    output_dir = os.path.abspath(output_dir)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(test_images))):
        image = test_images[i:i+1]
        mask = test_masks[i:i+1]
        
        # Predict
        import jax
        import jax.numpy as jnp
        logits = state.apply_fn(state.params, image)
        prediction = jnp.argmax(logits[0], axis=-1)
        probabilities = jax.nn.softmax(logits[0])[..., 1]
        
        # Convert for visualization
        image_np = (image[0] + 1.0) / 2.0  # Denormalize
        mask_np = mask[0, ..., 0]
        prediction_np = np.array(prediction)
        probabilities_np = np.array(probabilities)
        
        # Create visualization
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
        
        plt.suptitle(f'{title_prefix}Quantum Flood Segmentation - Sample {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{filename_prefix}sample_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved in {results_dir}")



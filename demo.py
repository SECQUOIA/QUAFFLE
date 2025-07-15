#!/usr/bin/env python3
import os
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from flax import jax_utils
import ml_collections
from tqdm import tqdm

# Import utilities
from utils import (
    load_dataset, create_data_iterator, create_train_state, 
    train_step, evaluate_model, save_training_curves, visualize_results
)

def get_config():
    """Get configuration for flood segmentation."""
    config = ml_collections.ConfigDict()
    
    # Data
    config.image_size = 128
    config.channels = 3
    config.num_classes = 2
    config.batch_size = 8
    
    # Model
    config.dim = 64
    config.dim_mults = (1, 2, 4, 8)
    config.resnet_block_groups = 8
    config.quantum_channel = 4
    config.name_ansatz = 'FQConv_ansatz'
    config.num_layer = 2
    
    # Training
    config.num_train_steps = 500
    config.learning_rate = 1e-4
    config.log_every = 10
    config.eval_every = 100
    config.seed = 42
    
    return config

def train_model(config, train_images, train_masks, val_images, val_masks, output_dir):
    """Train the model."""
    # Convert to absolute path to fix Orbax checkpointing issue
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize
    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(rng, config)
    
    # Try to load checkpoint with proper error handling
    start_step = 0
    try:
        restored_state = checkpoints.restore_checkpoint(checkpoint_dir, state)
        if restored_state is not None and hasattr(restored_state, 'step'):
            state = restored_state
            start_step = int(state.step)
            print(f"Checkpoint loaded successfully! Resuming from step {start_step}")
        else:
            print("No valid checkpoint found, starting from scratch")
    except Exception as e:
        print(f" Failed to load checkpoint: {e}")
        print("Starting from scratch")
    
    # Skip training if already completed
    if start_step >= config.num_train_steps:
        print(f"Training already completed at step {start_step}/{config.num_train_steps}")
        return state, [], []
    
    # Replicate for pmap
    pmap_state = jax_utils.replicate(state)
    data_iter = create_data_iterator(train_images, train_masks, config)
    p_train_step = jax.pmap(train_step, axis_name='batch')
    
    # Training loop - resume from start_step
    remaining_steps = config.num_train_steps - start_step
    print(f"Training {remaining_steps} remaining steps (from {start_step} to {config.num_train_steps})")
    
    train_metrics = []
    val_metrics = []

    for step_offset in tqdm(range(remaining_steps), desc="Training"):
        current_step = start_step + step_offset
        batch = next(data_iter)
        pmap_state, metrics = p_train_step(pmap_state, batch)
        
        if (current_step + 1) % config.log_every == 0:
            train_metrics.append({'step': current_step + 1, **jax.tree_util.tree_map(lambda x: x.mean(), metrics)})
            print(f"Step {current_step + 1}: Loss = {train_metrics[-1]['loss']:.4f}, Accuracy = {train_metrics[-1]['accuracy']:.4f}")
        
        if (current_step + 1) % config.eval_every == 0:
            # Update single-device state
            current_params = jax.tree_util.tree_map(lambda x: x[0], pmap_state.params)
            state = state.replace(params=current_params, step=current_step + 1)
            
            # Evaluate
            val_metrics.append({'step': current_step + 1, **evaluate_model(state, val_images, val_masks, config)})
            print(f"Validation: Loss = {val_metrics[-1]['loss']:.4f}, Accuracy = {val_metrics[-1]['accuracy']:.4f}, AUC = {val_metrics[-1]['auc']:.4f}")
            
            # Save checkpoint
            checkpoints.save_checkpoint(checkpoint_dir, state, current_step + 1, keep=3)
            print(f"Checkpoint saved at step {current_step + 1}")
        
        # Save training curves every 50 steps (if we have validation metrics)
        if (current_step + 1) % 50 == 0 and val_metrics:
            print(f"Saving training curves at step {current_step + 1}...")
            save_training_curves(val_metrics, output_dir, title_prefix="Training and Validation ")
    
    # Return single-device state
    final_params = jax.tree_util.tree_map(lambda x: x[0], pmap_state.params)
    final_step = start_step + remaining_steps
    return state.replace(params=final_params, step=final_step), train_metrics, val_metrics

def main():
    """Run quantum flood segmentation demo."""
    # Data paths
    base_dir = "/anvil/projects/x-chm250024/data/flood_optical"
    train_images_dir = os.path.join(base_dir, "Training", "images")
    train_masks_dir = os.path.join(base_dir, "Training", "labels")
    test_images_dir = os.path.join(base_dir, "Testing", "images")
    test_masks_dir = os.path.join(base_dir, "Testing", "labels")
    output_dir = "results_optical"
    
    print("Quantum Flood Segmentation Demo")
    print("Using QVUNet with PennyLane quantum circuits")
    
    # Check data
    if not os.path.exists(train_images_dir):
        print(f"Training images not found: {train_images_dir}")
        return
    
    config = get_config()
    print(f"Configuration: {config.quantum_channel} quantum channels, {config.name_ansatz} ansatz")
    
    # Load data
    print("Loading training data...")
    train_images, train_masks = load_dataset(train_images_dir, train_masks_dir, config)
    
    # Split train/validation
    split_idx = int(0.8 * len(train_images))
    val_images = train_images[split_idx:]
    val_masks = train_masks[split_idx:]
    train_images = train_images[:split_idx]
    train_masks = train_masks[:split_idx]
    
    print(f"Training: {len(train_images)}, Validation: {len(val_images)}")
    
    # Train
    print("Training model...")
    trained_state, train_metrics, val_metrics = train_model(config, train_images, train_masks, val_images, val_masks, output_dir)
    
    # Test and visualize
    if os.path.exists(test_images_dir):
        print("Loading test data...")
        test_images, test_masks = load_dataset(test_images_dir, test_masks_dir, config)
        visualize_results(trained_state, test_images, test_masks, config, output_dir)
    else:
        print("Using validation data for visualization...")
        visualize_results(trained_state, val_images, val_masks, config, output_dir)
    
    # Final save of training curves
    if val_metrics:
        print("Saving final training curves...")
        save_training_curves(val_metrics, output_dir, title_prefix="Training and Validation ")

    print(f"Demo completed! Results in {output_dir}")

if __name__ == "__main__":
    main() 
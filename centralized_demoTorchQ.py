#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utilsTorch import (
    QVUNet, QuantumBlock, QVUNetSegmentation,
    FloodSegmentationDataset, get_dataloaders,
    train_one_epoch, evaluate_model, save_training_curves, visualize_results
)
from torch.utils.data import DataLoader
from results_tracker import save_results_to_excel
from tools.provenance import collect_provenance, format_provenance, print_provenance
from tools import training_common as tc

def parse_args():
    parser = argparse.ArgumentParser(description="Centralized Quantum Flood Segmentation")
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--dataset', choices=['optical', 'sar', 'combined'], default='optical',
                        help='Dataset to use (default: optical)')
    tc.add_common_args(parser, quantum=True)
    return parser.parse_args()

def get_config():
    config = {
        'image_size': 128,
        'base_channels': 32,  # Main dimension parameter for QVUNet
        'init_dim': 32,       # Initial dimension (can be different from base_channels)
        'out_dim': 2,         # Output channels for binary segmentation
        'dim_mults': (1, 2, 4, 8),  # Dimension multipliers for each resolution level
        'resnet_block_groups': 4,   # Number of groups for group normalization
        'quantum_channels': 32,      # Number of channels processed by quantum circuits
        'batch_size': 8,
        'num_train_steps': 100,
        'learning_rate': 1e-4,
        'log_every': 10,
        'eval_every': 10,
        'seed': 42,
        'quantum_backend': 'pennylane',  # pennylane or ptlayer
        # Results saving settings
        'save_samples': True,
        'num_train_samples': 6,    # Number of final training samples to save
        'num_val_samples': 6,      # Number of final validation samples to save
        'num_test_samples': 20,     # Number of test samples to save
    }
    return config

def train_model(config, train_loader, val_loader, model, optimizer, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')
    # Create logs directory for evaluation scores
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    eval_log_path = os.path.join(logs_dir, 'evaluation_scores.txt')
    
    start_step = 0
    train_metrics = []
    val_metrics = []
    # Try to load checkpoint
    checkpoint = None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not tc.checkpoint_compatible(checkpoint):
            checkpoint = None
    if checkpoint is not None:
        checkpoint_loaded = False
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_loaded = True
        except RuntimeError as e:
            if "Missing key(s)" in str(e) and ("model." in str(e) or "qvunet." in str(e)):
                print("Checkpoint has incompatible model structure.")
                # Reset to starting from scratch
                start_step = 0
                train_metrics = []
                val_metrics = []
            else:
                raise e
        
        if checkpoint_loaded:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint.get('step', 0)
            train_metrics = checkpoint.get('train_metrics', [])
            val_metrics = checkpoint.get('val_metrics', [])
            print(f"Checkpoint loaded successfully! Resuming from step {start_step}")
        else:
            print("Starting training from scratch due to incompatible checkpoint")
    else:
        print("No checkpoint found, starting from scratch")
    if start_step >= config['num_train_steps']:
        print(f"Training already completed at step {start_step}/{config['num_train_steps']}")
        return model, train_metrics, val_metrics
    print(f"Training {config['num_train_steps'] - start_step} remaining steps (from {start_step} to {config['num_train_steps']})")
    step = start_step
    
    # Initial evaluation at step 0 (or starting step) if no validation metrics exist
    if not val_metrics:
        print("Performing initial evaluation at step 0...")
        val_result = evaluate_model(model, val_loader, device)
        val_result['step'] = step
        val_metrics.append(val_result)
        print(f"Initial Validation: Loss = {val_result['loss']:.4f}, Accuracy = {val_result['accuracy']:.4f}, AUC = {val_result['auc']:.4f}, IoU = {val_result['iou']:.4f}, Dice = {val_result['dice']:.4f}")

        # Log initial evaluation scores to text file
        with open(eval_log_path, 'a') as f:
            f.write(f"Initial Step {step}: Loss={val_result['loss']:.4f}, Accuracy={val_result['accuracy']:.4f}, AUC={val_result['auc']:.4f}, IoU={val_result['iou']:.4f}, Dice={val_result['dice']:.4f}\n")
        
        # Save initial training curves
        save_training_curves(val_metrics, output_dir, title_prefix="Training and Validation ")
    
    while step < config['num_train_steps']:
        avg_loss, accuracy = train_one_epoch(model, train_loader, optimizer, device)
        step += 1
        train_metrics.append({'step': step, 'loss': avg_loss, 'accuracy': accuracy, 'auc': 0.0})
        if step % config['log_every'] == 0:
            print(f"Step {step}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        if step % config['eval_every'] == 0:
            val_result = evaluate_model(model, val_loader, device)
            val_result['step'] = step
            val_metrics.append(val_result)
            print(f"Validation: Loss = {val_result['loss']:.4f}, Accuracy = {val_result['accuracy']:.4f}, AUC = {val_result['auc']:.4f}, IoU = {val_result['iou']:.4f}, Dice = {val_result['dice']:.4f}")

            # Log evaluation scores to text file
            with open(eval_log_path, 'a') as f:
                f.write(f"Step {step}: Loss={val_result['loss']:.4f}, Accuracy={val_result['accuracy']:.4f}, AUC={val_result['auc']:.4f}, IoU={val_result['iou']:.4f}, Dice={val_result['dice']:.4f}\n")
            
            # Save regular checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'run_format': tc.RUN_FORMAT,
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"Checkpoint saved at step {step}")
            
            # Save unique checkpoint every 100 steps
            if step % 100 == 0:
                unique_checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{step}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'run_format': tc.RUN_FORMAT,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, unique_checkpoint_path)
                print(f"Unique checkpoint saved: checkpoint_step_{step}.pt")
        if step % 50 == 0 and val_metrics:
            print(f"Saving training curves at step {step}...")
            save_training_curves(val_metrics, output_dir, title_prefix="Training and Validation ")
    return model, train_metrics, val_metrics

DATA_DIRS = {'optical': 'data/flood_optical', 'sar': 'data/flood_SAR', 'combined': 'data/flood_combined'}

def main():
    args = parse_args()
    base_dir = DATA_DIRS[args.dataset]
    train_images_dir = os.path.join(base_dir, "Training", "images")
    train_masks_dir = os.path.join(base_dir, "Training", "labels")
    test_images_dir = os.path.join(base_dir, "Testing", "images")
    test_masks_dir = os.path.join(base_dir, "Testing", "labels")
    print("=" * 60)
    print("Quantum Flood Segmentation Demo")
    print("=" * 60)

    if not os.path.exists(train_images_dir):
        tc.fail_run(None, f"Training images not found: {train_images_dir}")

    config = get_config()
    config['seed'] = args.seed
    config['dataset'] = args.dataset
    ds_suffix = "" if args.dataset == "optical" else f"_{args.dataset}"
    output_dir = f"results/flood_central_optical_pytorch_pennylane{ds_suffix}_seed{config['seed']}"
    print(f"Using QVUNet with PyTorch and {config['quantum_backend']} quantum circuits")
    
    tc.apply_common_args(config, args)

    # Seed python, numpy, torch and cuda once, before any data or model work.
    tc.seed_everything(config['seed'])
    provenance = collect_provenance(seed=config['seed'])
    config['provenance'] = provenance
    print_provenance(provenance)
    
    print(f"Configuration:")
    print(f"  - Base channels: {config['base_channels']}")
    print(f"  - Quantum channels: {config['quantum_channels']}")
    print(f"  - Quantum backend: {config['quantum_backend']}")
    print(f"  - Image size: {config['image_size']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Training steps: {config['num_train_steps']}")
    print(f"  - Save samples: {config['save_samples']}")
    
    # Data: global stratified validation holdout, identical to the one used by
    # the federated and ablation scripts for the same seed.
    print("\nLoading training data...")
    full_dataset = tc.load_dataset(train_images_dir, train_masks_dir, config)
    full_dataset = tc.cap_samples(full_dataset, config.get('max_samples'), config['seed'])
    train_indices, val_indices, _ = tc.holdout_validation(
        full_dataset, config['seed'], config['val_fraction'])

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = tc.make_loader(train_subset, config['batch_size'], True, config['seed'])
    val_loader = tc.make_loader(val_subset, config['batch_size'], False, config['seed'])

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    
    # Model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nInitializing QVUNet model...")
    try:
        # Use QVUNetSegmentation wrapper from utils for consistent checkpoint structure
        model = tc.build_model(QVUNetSegmentation, config).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        tc.fail_run(output_dir, f"model initialization failed: {e}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train
    print("\nStarting training...")
    model, train_metrics, val_metrics = train_model(config, train_loader, val_loader, model, optimizer, device, output_dir)
    
    print("\nTraining completed! Now saving results...")
    
    # Save final training curves
    if val_metrics:
        print("Saving training curves...")
        save_training_curves(val_metrics, output_dir, title_prefix="Training and Validation ")
    
    # Save final results and samples
    if config['save_samples']:
        print("Saving final training samples...")
        visualize_results(model, train_loader, device, output_dir, 
                         title_prefix="Final Training - ", 
                         filename_prefix="final_train_", 
                         num_samples=config['num_train_samples'])
        
        print("Saving final validation samples...")
        visualize_results(model, val_loader, device, output_dir, 
                         title_prefix="Final Validation - ", 
                         filename_prefix="final_val_", 
                         num_samples=config['num_val_samples'])
    
    # Test and save test results
    if os.path.exists(test_images_dir):
        print("Loading test data for evaluation and visualization...")
        test_dataset = tc.load_dataset(test_images_dir, test_masks_dir, config)
        test_dataset = tc.cap_samples(test_dataset, config.get('max_samples'), config['seed'])
        test_loader = tc.make_loader(test_dataset, config['batch_size'], False, config['seed'])
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_result = evaluate_model(model, test_loader, device)
        print(f"Test Results - Loss: {test_result['loss']:.4f}, Accuracy: {test_result['accuracy']:.4f}, AUC: {test_result['auc']:.4f}, IoU: {test_result['iou']:.4f}, Dice: {test_result['dice']:.4f}")

        knockout = None
        if config.get('quantum_knockout_eval'):
            print("Quantum knockout evaluation on the test set...")
            knockout = tc.knockout_eval(model, test_loader, device, evaluate_model,
                                       knockout_seed=config['seed'])
        
        # Save test samples
        if config['save_samples']:
            print("Saving test samples...")
            visualize_results(model, test_loader, device, output_dir, 
                             title_prefix="Test - ", 
                             filename_prefix="test_", 
                             num_samples=config['num_test_samples'])
    else:
        test_result = None
        if config['save_samples']:
            print("Saving additional validation samples...")
            visualize_results(model, val_loader, device, output_dir, 
                             title_prefix="Additional Validation - ", 
                             filename_prefix="additional_val_", 
                             num_samples=config['num_test_samples'])
    
    if test_result is None:
        tc.fail_run(output_dir, f"no test set at {test_images_dir}, refusing to write a summary or tracker row")

    # Save a comprehensive summary of results (completed runs only)
    if test_result is not None:
        print("Saving results summary...")
        summary_path = os.path.join(output_dir, "results", "summary.txt")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            f.write("Quantum Flood Segmentation Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: QVUNet with {config['quantum_backend']} backend\n")
            f.write(f"Quantum channels: {config['quantum_channels']}\n")
            f.write(f"Base channels: {config['base_channels']}\n")
            f.write(f"Image size: {config['image_size']}\n")
            f.write(f"Training steps: {config['num_train_steps']}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Validation: global held-out {len(val_subset)} samples "
                    f"({config['val_fraction']} of the training pool, stratified on flood label)\n")
            f.write(f"Training samples: {len(train_subset)}\n\n")
            
            if val_metrics:
                final_val = val_metrics[-1]
                f.write("Final Validation Results:\n")
                f.write(f"  Loss: {final_val['loss']:.4f}\n")
                f.write(f"  Accuracy: {final_val['accuracy']:.4f}\n")
                f.write(f"  AUC: {final_val['auc']:.4f}\n")
                f.write(f"  IoU: {final_val['iou']:.4f}\n")
                f.write(f"  Dice: {final_val['dice']:.4f}\n\n")

            if test_result:
                f.write("Test Results:\n")
                f.write(f"  Loss: {test_result['loss']:.4f}\n")
                f.write(f"  Accuracy: {test_result['accuracy']:.4f}\n")
                f.write(f"  AUC: {test_result['auc']:.4f}\n")
                f.write(f"  IoU: {test_result['iou']:.4f}\n")
                f.write(f"  Dice: {test_result['dice']:.4f}\n\n")
            if config.get('quantum_knockout_eval'):
                f.write(tc.format_knockout(knockout))
            
            f.write("Saved Samples:\n")
            f.write(f"  Final training samples: {config['num_train_samples']}\n")
            f.write(f"  Final validation samples: {config['num_val_samples']}\n")
            if os.path.exists(test_images_dir):
                f.write(f"  Test samples: {config['num_test_samples']}\n")
            else:
                f.write(f"  Additional validation samples: {config['num_test_samples']}\n")
        
            f.write("\n" + format_provenance(provenance))

        print(f"Results summary saved to: {summary_path}")
    
    # Save results to shared Excel summary
    excel_config = {k: v for k, v in config.items() if k != 'provenance'}
    excel_config['total_params'] = total_params
    final_val = val_metrics[-1] if val_metrics else None
    save_results_to_excel(
        demo_name="Centralized Quantum Torch",
        framework="torch",
        training="centralized",
        variant="quantum",
        val_metrics=final_val,
        test_metrics=test_result,
        config=excel_config,
        dataset=args.dataset,
        provenance=provenance,
    )

    print(f"\nDemo completed successfully! All results saved in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
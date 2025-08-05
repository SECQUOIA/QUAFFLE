#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly.
"""

import sys
import os

# Add utils to path
sys.path.append('utils')

def test_jax_imports():
    """Test JAX imports."""
    try:
        from utilsJAX import (
            load_dataset, create_data_iterator, create_train_state, 
            train_step, evaluate_model, save_training_curves, visualize_results
        )
        from unetJAX import QVUNet
        print("✓ JAX imports successful")
        return True
    except Exception as e:
        print(f"✗ JAX imports failed: {e}")
        return False

def test_torch_imports():
    """Test PyTorch imports."""
    try:
        from utilsTorch import (
            FloodSegmentationDataset, get_dataloaders, QVUNetSegmentation, UNetSegmentation,
            train_one_epoch, evaluate_model, save_training_curves, visualize_results
        )
        from unetTorch import QVUNet, UNet
        print("✓ PyTorch imports successful")
        return True
    except Exception as e:
        print(f"✗ PyTorch imports failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    try:
        # Test JAX config
        import ml_collections
        config = ml_collections.ConfigDict()
        config.image_size = 128
        config.channels = 3
        config.num_classes = 2
        config.batch_size = 8
        config.dim = 64
        config.dim_mults = (1, 2, 4, 8)
        config.resnet_block_groups = 8
        config.quantum_channel = 4
        config.name_ansatz = 'FQConv_ansatz'
        config.num_layer = 2
        config.learning_rate = 1e-4
        config.seed = 42
        print("✓ JAX config creation successful")
        
        # Test PyTorch config
        torch_config = {
            'image_size': 128,
            'base_channels': 32,
            'init_dim': 32,
            'out_dim': 2,
            'dim_mults': (1, 2, 4, 8),
            'resnet_block_groups': 4,
            'quantum_channels': 8,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'quantum_backend': 'pennylane',
            'seed': 42,
        }
        print("✓ PyTorch config creation successful")
        return True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing imports and configurations...")
    print("=" * 50)
    
    jax_ok = test_jax_imports()
    torch_ok = test_torch_imports()
    config_ok = test_config_creation()
    
    print("=" * 50)
    if jax_ok and torch_ok and config_ok:
        print("✓ All tests passed! The demos should work correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return jax_ok and torch_ok and config_ok

if __name__ == "__main__":
    main() 
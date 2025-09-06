#!/bin/bash

set -e

MODE=${1:-gpu}

echo "Installing QFedUNet requirements in $MODE mode..."

# Install core dependencies that are common to both CPU and GPU
echo "Installing core dependencies..."
pip install \
    numpy==1.26.4 \
    scipy==1.10.1 \
    einops==0.8.1 \
    tqdm==4.67.1 \
    matplotlib==3.10.3 \
    rasterio==1.4.3 \
    affine==2.4.0 \
    packaging==25.0 \
    typing_extensions==4.14.1 \
    ml_collections==1.1.0 \
    tensorflow==2.19.0 \
    scikit-learn==1.7.0 \
    flwr==1.19.0 \
    flwr[simulation]==1.19.0

# Install JAX ecosystem dependencies
echo "Installing JAX ecosystem dependencies..."
pip install \
    flax==0.8.4 \
    optax==0.2.5 \
    chex==0.1.89 \
    orbax-checkpoint==0.5.15

# Install Quantum Computing dependencies
echo "Installing quantum computing dependencies..."
pip install \
    autoray==0.6.12 \
    PennyLane==0.41.1 \
    PennyLane_Lightning==0.41.1

# Install mode-specific packages
if [ "$MODE" = "gpu" ]; then
    echo "Installing GPU-specific packages..."
    
    # Install JAX with CUDA support
    echo "Installing JAX with CUDA 12 support..."
    pip install jax[cuda12_pip]==0.4.28 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    # Install PyTorch with CUDA support
    echo "Installing PyTorch with CUDA support..."
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
    
    echo "GPU installation complete!"
    
elif [ "$MODE" = "cpu" ]; then
    echo "Installing CPU-only packages..."
    
    # Install JAX CPU version
    echo "Installing JAX CPU version..."
    pip install jax[cpu]==0.4.28
    
    # Install PyTorch CPU version
    echo "Installing PyTorch CPU version..."
    pip install torch==2.1.2 torchvision==0.16.2
    
    echo "CPU installation complete!"
    
fi

echo ""
echo "Installation completed successfully!"

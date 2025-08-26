# QUAFFLE: Quantum U-Net Assisted Federated Flood Learning and Estimation

> **NASA Beyond the Algorithm Challenge Submission**

Flood detection using satellite imagery faces challenges due to the large-scale, heterogeneous nature of the data and the computational intensity required. We propose QUAFFLE (Quantum U-Net Assisted Federated Flood Learning and Estimation), a hybrid framework that integrates quantum-enhanced U-Net architectures with federated learning to enable distributed and accurate flood mapping. QUAFFLE combines SAR and optical satellite data in a federated setting, reducing the need to transmit raw images while maintaining performance. A variational quantum layer at the U-Net bottleneck reduces parameter count and improves feature extraction. We implemented QUAFFLE using PennyLane for gate-based and ORCA-SDK photonic quantum simulations, Flower for federated learning, and PyTorch for model development. Evaluations on the IEEE DFC24 SAR and Optical datasets show improved AUC and accuracy with fewer parameters than classical counterparts. The framework is compatible with real and simulated quantum hardware, offering a resource-efficient approach to enhance flood mapping accuracy under practical constraints. 

## Installation

### GPU Installation (Default)
```bash
./install_requirements.sh gpu
```

### CPU Installation
```bash
./install_requirements.sh cpu
```

**Orca**: Orca's PTSeries is downloaded from their repository. Follow their instructions to setup the PT Library.

**Requirements**: CUDA 12.0.1, Python 3.11

## Dataset Setup

Download the required datasets:

**Optical Dataset:**
https://drive.google.com/drive/folders/1UHE8pRFGZkHiOwmwc8tFI77c620U59eD

**SAR Dataset:**
https://drive.google.com/drive/folders/1WvKc7AxSiFknl1EuICnlLY2hU487qtB-

### Dataset Organization
After downloading, organize your datasets in the `data` folder as follows:

```
data/
├── flood_optical/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
└── flood_sar/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

### Combined Dataset Setup
For combined training using both optical and SAR data, simply merge the datasets into a single folder:

```
data/
└── flood_combined/
    ├── train/
    │   ├── images/  # Contains both optical and SAR images
    │   └── masks/   # Contains corresponding masks
    └── test/
        ├── images/  # Contains both optical and SAR images
        └── masks/   # Contains corresponding masks
```

**Note**: The combined dataset is created by copying all images and masks from both `flood_optical` and `flood_sar` datasets into a single `flood_combined` folder. The model will automatically process both data types during training.

## Running the Demos

**Note**: The PyTorch implementation is more developed and feature-complete compared to the JAX version. The results were obtained from the PyTorch implmentation.

### Centralized Training (PyTorch)

#### Classical U-Net
```bash
python centralized_demoTorchClassical.py
```

#### Quantum U-Net (PennyLane)
```bash
python centralized_demoTorchQ.py
```

### Federated Learning (PyTorch)

#### Classical U-Net Federated
```bash
python federated_demoTorchClassical.py
```

#### Quantum U-Net Federated
```bash
python federated_demoTorchQ.py
```

### JAX Implementation (Experimental)

#### Centralized JAX
```bash
python centralized_demoJAX.py
```

#### Federated JAX
```bash
python federated_demoJAX.py
```

## Configuration

All demos use configuration dictionaries that can be modified in the respective files:

- `image_size`: Input image resolution (default: 128)
- `base_channels`: Base number of channels (default: 32)
- `quantum_channels`: Number of channels processed by quantum circuits (default: 32)
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Learning rate (default: 1e-4)
- `quantum_backend`: 'pennylane' or 'ptlayer' (PyTorch only)

## Output

Results are saved in the `results/` directory with the following structure:
- Training curves (loss, accuracy, AUC)
- Model checkpoints
- Sample visualizations
- Evaluation metrics

## Quantum Backends

- **PennyLane**: Quantum machine learning library with various simulators
- **PTLayer**: Photonic quantum computing backend (requires Orca setup)

## Related Work

Our QVUNet architectures are inspired by the quantum hybrid diffusion models proposed in:

```bibtex
@article{defalco2024quantum,
  title={Towards Efficient Quantum Hybrid Diffusion Models},
  author={De Falco, Francesca and Ceschini, Andrea and Sebastianelli, Alessandro and Le Saux, Bertrand and Panella, Massimo},
  journal={KI - Künstliche Intelligenz},
  volume={38},
  number={4},
  pages={311--326},
  year={2024},
  publisher={Springer},
  doi={10.1007/s13218-024-00858-5}
}
```

**Reference**: [arXiv:2402.16147](https://arxiv.org/abs/2402.16147) - This paper provided the foundation for our QVUNet architectures, particularly the quantum ResNet blocks and hybridization schemes.


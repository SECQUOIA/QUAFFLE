# QUAFFLE: Quantum U-Net Assisted Federated Flood Learning and Estimation

**QUAFFLE** is a hybrid quantum–classical framework that combines federated learning and quantum-enhanced U-Net architectures to enable distributed, resource-efficient flood mapping using satellite imagery.

---

##  Highlights

- **Finalist** in NASA’s *Beyond the Algorithm Challenge: Novel Computing Architectures for Flood Analysis*.  
- Efficient federated flood detection across SAR and optical satellite data, with reduced parameter count and improved AUC compared to classical baselines.  
- Compatible with both gate-based and photonic quantum simulators.

Flood detection using satellite imagery faces challenges due to the large-scale, heterogeneous nature of the data and the computational intensity required. We propose QUAFFLE (Quantum U-Net Assisted Federated Flood Learning and Estimation), a hybrid framework that integrates quantum-enhanced U-Net architectures with federated learning to enable distributed and accurate flood mapping. QUAFFLE combines SAR and optical satellite data in a federated setting, reducing the need to transmit raw images while maintaining performance. A variational quantum layer at the U-Net bottleneck reduces parameter count and improves feature extraction. We implemented QUAFFLE using PennyLane for gate-based and ORCA-SDK photonic quantum simulations, Flower for federated learning, and PyTorch for model development. Evaluations on the IEEE DFC24 SAR and Optical datasets show improved AUC and accuracy with fewer parameters than classical counterparts. The framework is compatible with real and simulated quantum hardware, offering a resource-efficient approach to enhance flood mapping accuracy under practical constraints. 

##  Architecture and Implementation

- **Quantum U-Net**: Incorporates a variational quantum layer at the U-Net bottleneck for enhanced feature extraction and parameter efficiency.  
- **Federated Learning**: Built upon the Flower framework for distributed collaborative training without centralized data exchange.  
- **Quantum Backends**: Supports PennyLane (gate-based) and ORCA-SDK (photonic) simulations.  
- **Modeling Framework**: Implementation relies on PyTorch for U-Net structure and training workflows.  
- **Datasets**: Evaluated on the IEEE DFC24 SAR and Optical datasets with strong performance.

---

##  Acknowledgments

- **NASA ESTO** for hosting and recognizing QUAFFLE as a finalist in the *Beyond the Algorithm Challenge: Novel Computing Architectures for Flood Analysis* (https://www.nasa-beyond-challenge.org/).  
- The **SECQUOIA research group** at Purdue University for support and collaboration.

---

##  Contact

For questions or collaboration, reach out to the SECQUOIA team (Purdue University) or check group resources at [SECQUOIA website](https://secquoia.github.io/).



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
### Towards Efficient Quantum Hybrid Diffusion Models
Our QVUNet architectures are inspired by the quantum hybrid diffusion models proposed in this paper. The authors demonstrate how leveraging quantum resources can improve generative model performance and scalability.
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

## Previous Work

### Federated Learning in Chemical Engineering: A Tutorial on a Framework for Privacy-Preserving Collaboration across Distributed Data Sources

This tutorial provides the chemical engineering community with a hands-on framework for federated learning using Flower and TensorFlow Federated. It demonstrates applications in manufacturing optimization and multimodal biomedical tasks, emphasizing privacy and collaboration across distributed datasets.

```bibtex
@article{dutta2025federated,
  title={Federated Learning in Chemical Engineering: A Tutorial on a Framework for Privacy-Preserving Collaboration across Distributed Data Sources},
  author={Dutta, Siddhant and de Freitas, Iago Leal and Xavier, Pedro Maciel and de Farias, Claudio Miceli and Bernal Neira, David E.},
  journal={Industrial \& Engineering Chemistry Research},
  volume={64},
  number={15},
  pages={7767--7783},
  year={2025},
  publisher={ACS Publications},
  doi={10.1021/acs.iecr.4c03805}
}
```

### Federated Learning with Quantum Computing and Fully Homomorphic Encryption

This work introduces quantum computing with fully homomorphic encryption in federated learning, mitigating accuracy loss during aggregation while preserving privacy. It lays the foundation for privacy-enhanced, quantum-augmented FL frameworks.

```bibtex
@article{dutta2024federated,
  title={Federated learning with quantum computing and fully homomorphic encryption: A novel computing paradigm shift in privacy-preserving ML},
  author={Dutta, Siddhant and Karanth, P. P. and Xavier, P. M. and de Freitas, I. L. and Innan, Nouhaila and Yahia, Sadok Ben and Shafique, Muhammad and Bernal Neira, David E.},
  journal={arXiv preprint arXiv:2409.11430},
  year={2024}
}
```

#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn

from utilsTorch import QVUNetSegmentation, FloodSegmentationDataset, get_dataloaders, segmentation_loss, train_one_epoch, evaluate_model, save_training_curves, visualize_results
from torch.utils.data import DataLoader, Subset
import copy

def split_dataset_among_clients(dataset, num_clients, seed=42):
    """Split a PyTorch Dataset among multiple clients for federated learning."""
    import random
    random.seed(seed)
    
    # Get all indices
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Split indices among clients
    client_datasets = []
    samples_per_client = total_size // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else total_size
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets

def torch_params_to_numpy(model) -> List[np.ndarray]:
    """Convert PyTorch model parameters to numpy arrays."""
    return [param.detach().cpu().numpy() for param in model.parameters()]

def numpy_to_torch_params(model, numpy_params: List[np.ndarray]):
    """Load numpy arrays into PyTorch model parameters."""
    for param, numpy_param in zip(model.parameters(), numpy_params):
        param.data = torch.from_numpy(numpy_param).to(param.device)

def get_config():
    """Get configuration for federated quantum flood segmentation."""
    config = {
        'image_size': 128,
        'base_channels': 32,  # Main dimension parameter for QVUNet
        'init_dim': 32,       # Initial dimension (can be different from base_channels)
        'out_dim': 2,         # Output channels for binary segmentation
        'dim_mults': (1, 2, 4, 8),  # Dimension multipliers for each resolution level
        'resnet_block_groups': 4,   # Number of groups for group normalization
        'quantum_channels': 32,     # Number of channels processed by quantum circuits
        'batch_size': 8,
        'learning_rate': 1e-4,
        'seed': 42,
        'quantum_backend': 'ptlayer',  # pennylane or ptlayer
        
        # Federated Learning specific
        'num_clients': 4,
        'local_epochs': 4,
        'fed_rounds': 25,  # Total federated rounds
        'clients_per_round': 4,  # Number of clients participating per round
        'fraction_evaluate': 1, 
        'log_every': 5,  # Log every N federated rounds
        'eval_every': 5,  # Evaluate every N federated rounds
        
        # Results saving settings
        'save_samples': True,
        'num_train_samples': 6,
        'num_val_samples': 6,
        'num_test_samples': 20,
    }
    return config

class QuantumFlowerClient(fl.client.NumPyClient):
    """Flower client for quantum model training."""
    
    def __init__(self, client_id: int, client_dataset: Subset, config: Dict, device_str: str):

        import torch
        
        self.client_id = client_id
        self.client_dataset = client_dataset
        self.config = config
        
        # Set seeds for this client to ensure reproducibility
        client_seed = config['seed'] + client_id  # Different seed for each client
        torch.manual_seed(client_seed)
        torch.cuda.manual_seed(client_seed)
        torch.cuda.manual_seed_all(client_seed)
        np.random.seed(client_seed)
        
        # Check if GPU is available in this worker process
        self.device = torch.device(device_str)
        if self.device.type == 'cuda' and torch.cuda.is_available():
            pass  # Use GPU
        else:
            self.device = torch.device('cpu')
            print(f"Client {client_id}: GPU not available, using CPU")
            
        self.template_model = None
        self.local_model = None
        
        # Create data loader for this client with reproducible shuffling
        self.client_loader = DataLoader(
            client_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(client_seed)
        )
    
        print(f"Client {client_id} initialized with {len(client_dataset)} samples on {self.device}")
    
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """Return model parameters as numpy arrays."""
        import torch
        
        model_to_use = self.local_model if self.local_model is not None else self.template_model
        params = torch_params_to_numpy(model_to_use)
        return params
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model locally and return updated parameters."""
        import torch
        
        print(f"Client {self.client_id} starting local training...")
        
        # Set seeds and CudnnModule settings for this training session
        training_seed = self.config['seed'] + self.client_id + 1000  # Different seed for training
        torch.manual_seed(training_seed)
        torch.cuda.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
        np.random.seed(training_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if self.template_model is None:
            self.template_model = QVUNetSegmentation(self.config)
            self.template_model.to(self.device)
        
        # Create local model copy and load parameters
        self.local_model = copy.deepcopy(self.template_model)
        self.local_model.to(self.device)
        
        # Update parameters if provided
        if parameters:
            numpy_to_torch_params(self.local_model, parameters)
        
        # Create optimizer for local training
        optimizer = torch.optim.Adam(
            self.local_model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Local training for specified epochs
        self.local_model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0
        
        for epoch in range(self.config['local_epochs']):
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_batches = 0
            
            for batch_idx, (images, masks) in enumerate(self.client_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                # Create dummy time steps for the model
                B = images.shape[0]
                dummy_time = torch.zeros((B,), dtype=torch.float32, device=self.device)
                logits = self.local_model.model(images, dummy_time)
                
                # Compute loss
                masks_long = masks.squeeze(1).long()
                loss = torch.nn.functional.cross_entropy(logits, masks_long)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Compute accuracy
                preds = torch.argmax(logits, dim=1)
                correct = (preds == masks_long).sum().item()
                total_pixels = np.prod(preds.shape)
                accuracy = correct / total_pixels
                
                epoch_loss += loss.item()
                epoch_acc += accuracy
                epoch_batches += 1
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            avg_epoch_acc = epoch_acc / epoch_batches if epoch_batches > 0 else 0.0
            
            total_loss += avg_epoch_loss
            total_acc += avg_epoch_acc
            total_batches += 1
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        avg_acc = total_acc / total_batches if total_batches > 0 else 0.0
        
        print(f"Client {self.client_id} completed training: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
        
        # Return updated parameters, number of examples, and metrics
        updated_params = torch_params_to_numpy(self.local_model)
        num_examples = len(self.client_dataset)
        metrics = {"loss": float(avg_loss), "accuracy": float(avg_acc)}
        
        return (updated_params, num_examples, metrics)
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict]:
        """Evaluate model locally."""
        import torch
        
        if self.template_model is None:
            self.template_model = QVUNetSegmentation(self.config)
            self.template_model.to(self.device)
        
        # Create evaluation model and load parameters
        eval_model = copy.deepcopy(self.template_model)
        eval_model.to(self.device)
        
        if parameters:
            numpy_to_torch_params(eval_model, parameters)
        
        # Evaluate on local data
        metrics = evaluate_model(eval_model, self.client_loader, self.device)
        
        loss = float(metrics['loss'])
        num_examples = len(self.client_dataset)
        eval_metrics = {
            "accuracy": float(metrics['accuracy']),
            "auc": float(metrics['auc'])
        }
        
        return (loss, num_examples, eval_metrics)

class QuantumFedAvg(FedAvg):
    """Custom FedAvg strategy with quantum model evaluation."""
    
    def __init__(self, val_loader, config, device_str, output_dir, **kwargs):
        super().__init__(**kwargs)
        self.val_loader = val_loader
        self.config = config
        self.template_model = None  # Will be created when needed
        self.device_str = device_str
        self.device = None  # Will be created when needed
        self.output_dir = output_dir
        self.round_metrics = []
        self.best_performance = 0.0
        
        # Create logs directory for evaluation scores
        self.logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        self.eval_log_path = os.path.join(self.logs_dir, 'evaluation_scores.txt')
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results and perform server-side evaluation."""
        import torch
        
        # Call parent method
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        # Server-side evaluation if we have validation data
        if self.val_loader is not None and len(results) > 0:
            # Get current global parameters
            if hasattr(self, '_current_parameters') and self._current_parameters is not None:
                if self.device is None:
                    self.device = torch.device(self.device_str)
                    if self.device.type == 'cuda' and not torch.cuda.is_available():
                        self.device = torch.device('cpu')
                
                if self.template_model is None:
                    self.template_model = QVUNetSegmentation(self.config)
                    self.template_model.to(self.device)
                
                # Create evaluation model and load parameters
                eval_model = copy.deepcopy(self.template_model)
                eval_model.to(self.device)
                numpy_params = parameters_to_ndarrays(self._current_parameters)
                numpy_to_torch_params(eval_model, numpy_params)
                
                # Evaluate global model
                metrics = evaluate_model(eval_model, self.val_loader, self.device)
                metrics['step'] = server_round
                
                # Track best performance
                if metrics['auc'] > self.best_performance:
                    self.best_performance = metrics['auc']
                
                self.round_metrics.append(metrics)
                
                print(f"Global evaluation - Round {server_round}: "
                      f"Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
                
                # Log evaluation scores to text file
                with open(self.eval_log_path, 'a') as f:
                    f.write(f"Round {server_round}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}\n")
                
                # Save unique checkpoint every 25 rounds
                if server_round % 25 == 0:
                    unique_checkpoint_path = os.path.join(self.output_dir, f'checkpoint_round_{server_round}.pt')
                    torch.save({
                        'model_state_dict': eval_model.state_dict(),
                        'round': server_round,
                        'round_metrics': self.round_metrics,
                        'best_performance': self.best_performance
                    }, unique_checkpoint_path)
                    print(f"Unique checkpoint saved: checkpoint_round_{server_round}.pt")
        
        return aggregated_result
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results and store parameters."""
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        # Store current parameters for evaluation
        if aggregated_result is not None:
            self._current_parameters = aggregated_result[0]
        
        return aggregated_result

# Use the utility function for client dataset creation

def create_flower_client_fn(client_datasets: List[Subset], config: Dict, device_str: str):
    """Factory function to create Flower clients."""
    def client_fn(cid: str) -> QuantumFlowerClient:
        client_id = int(cid)
        client_dataset = client_datasets[client_id]
        return QuantumFlowerClient(client_id, client_dataset, config, device_str)
    
    return client_fn

def main():
    """Run federated quantum flood segmentation demo with Flower."""
    # Data paths
    base_dir = "data/flood_optical"
    train_images_dir = os.path.join(base_dir, "Training", "images")
    train_masks_dir = os.path.join(base_dir, "Training", "labels")
    test_images_dir = os.path.join(base_dir, "Testing", "images")
    test_masks_dir = os.path.join(base_dir, "Testing", "labels")
    output_dir = "results/flood_federated_optical_pytorch_ptlayer"
    
    print("=" * 60)
    print("Federated Quantum Flood Segmentation Demo with Flower")
    print(f"Using QVUNet with PyTorch and {config['quantum_backend']} quantum circuits")
    print("=" * 60)
    
    # Check data
    if not os.path.exists(train_images_dir):
        print(f"Training images not found: {train_images_dir}")
        return
    
    config = get_config()
    
    # Set seeds for reproducibility (excluding CudnnModule settings to avoid Ray serialization issues)
    # Note: We'll set seeds in the worker processes to avoid CudnnModule serialization
    
    print(f"Configuration:")
    print(f"  - Base channels: {config['base_channels']}")
    print(f"  - Quantum channels: {config['quantum_channels']}")
    print(f"  - Quantum backend: {config['quantum_backend']}")
    print(f"  - Image size: {config['image_size']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Clients: {config['num_clients']}")
    print(f"  - Fed rounds: {config['fed_rounds']}")
    print(f"  - Local epochs: {config['local_epochs']}")
    print(f"  - Clients per round: {config['clients_per_round']}")
    
    # Device setup - pass device string instead of device object
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    
    # Load full dataset
    print("\nLoading training data...")
    full_dataset = FloodSegmentationDataset(
        train_images_dir, 
        train_masks_dir, 
        config['image_size']
    )
    
    # Create client datasets using utility function
    print(f"\nSplitting data among {config['num_clients']} clients...")
    client_datasets = split_dataset_among_clients(full_dataset, config['num_clients'], config['seed'])
    
    # Create validation set from a portion of the last client's data
    val_size = len(client_datasets[-1]) // 5  # 20% for validation
    val_indices = list(range(len(client_datasets[-1]) - val_size, len(client_datasets[-1])))
    train_indices = list(range(len(client_datasets[-1]) - val_size))
    
    # Create validation dataset
    val_dataset = Subset(client_datasets[-1].dataset, 
                        [client_datasets[-1].indices[i] for i in val_indices])
    
    # Update last client dataset to exclude validation data
    client_datasets[-1] = Subset(client_datasets[-1].dataset,
                                [client_datasets[-1].indices[i] for i in train_indices])
    
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Initialize a temporary model to get parameter counts
    print("\nInitializing temporary quantum model for parameter counting...")
    try:
        # Set seed for model initialization
        torch.manual_seed(config['seed'])
        temp_model = QVUNetSegmentation(config)
        total_params = sum(p.numel() for p in temp_model.parameters())
        trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        print(f"Model architecture initialized successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        # Clean up temporary model
        del temp_model
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create initial parameters for the strategy
    print("\nCreating initial parameters for federated strategy...")
    temp_model = QVUNetSegmentation(config)
    initial_parameters = ndarrays_to_parameters(torch_params_to_numpy(temp_model))
    del temp_model  # Clean up temporary model
    
    # Create Flower strategy
    strategy = QuantumFedAvg(
        val_loader=val_loader,
        config=config,
        device_str=device_str,
        output_dir=output_dir,
        fraction_fit=config['clients_per_round'] / config['num_clients'],
        fraction_evaluate=1,  
        min_fit_clients=config['clients_per_round'],
        min_evaluate_clients=1,
        min_available_clients=config['num_clients'],
        initial_parameters=initial_parameters
    )
    
    # Create client function
    client_fn = create_flower_client_fn(client_datasets, config, device_str)
    
    print(f"\nStarting Flower federated training for {config['fed_rounds']} rounds...")
    
    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config['num_clients'],
        config=fl.server.ServerConfig(num_rounds=config['fed_rounds']),
        strategy=strategy,
        client_resources={"num_cpus": 1.0, "num_gpus": 0.2}  # Allocate GPU resources to clients
    )
    
    print(f"\nFederated training completed!")
    print(f"Best AUC achieved: {strategy.best_performance:.4f}")
    
    # Get final global model for testing
    if hasattr(strategy, '_current_parameters') and strategy._current_parameters is not None:
        final_model = QVUNetSegmentation(config)
        final_model.to(torch.device(device_str))
        final_params = parameters_to_ndarrays(strategy._current_parameters)
        numpy_to_torch_params(final_model, final_params)
        
        # Save final training curves
        if strategy.round_metrics:
            print("Saving final training curves...")
            
            # Ensure all metrics have the 'step' key for compatibility
            for i, metrics in enumerate(strategy.round_metrics):
                if 'step' not in metrics:
                    metrics['step'] = i + 1
                    
            save_training_curves(strategy.round_metrics, output_dir, 
                               prefix="flower_fed_", title_prefix="Flower Federated Learning - ")
        
        # Initialize test_result to None in case test data doesn't exist
        test_result = None
        
        # Test and visualize with final global model
        if config['save_samples']:
            if os.path.exists(test_images_dir):
                print("Loading test data for evaluation and visualization...")
                test_loader = get_dataloaders(test_images_dir, test_masks_dir, 
                                            config['image_size'], config['batch_size'], shuffle=False)
                
                # Evaluate on test set
                print("Evaluating on test set...")
                test_result = evaluate_model(final_model, test_loader, torch.device(device_str))
                print(f"Test Results - Loss: {test_result['loss']:.4f}, "
                      f"Accuracy: {test_result['accuracy']:.4f}, AUC: {test_result['auc']:.4f}")
                
                # Log test results to evaluation log file
                with open(strategy.eval_log_path, 'a') as f:
                    f.write(f"Final Test Results: Loss={test_result['loss']:.4f}, Accuracy={test_result['accuracy']:.4f}, AUC={test_result['auc']:.4f}\n")
                
                # Save test samples
                print("Saving test samples...")
                visualize_results(final_model, test_loader, torch.device(device_str), output_dir, 
                                 title_prefix="Flower Federated Test - ", 
                                 filename_prefix="flower_fed_test_", 
                                 num_samples=config['num_test_samples'])
            else:
                print("Test directory not found, using validation data for visualization...")
                # Save validation samples
                print("Saving validation samples...")
                visualize_results(final_model, val_loader, torch.device(device_str), output_dir,
                                 title_prefix="Flower Federated Validation - ", 
                                 filename_prefix="flower_fed_val_", 
                                 num_samples=config['num_val_samples'])
        
        # Save comprehensive summary
        if config['save_samples']:
            print("Saving results summary...")
            summary_path = os.path.join(output_dir, "results", "summary.txt")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            
            with open(summary_path, 'w') as f:
                f.write("Flower Federated Quantum Flood Segmentation Results Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Model: QVUNet with {config['quantum_backend']} backend\n")
                f.write(f"Quantum channels: {config['quantum_channels']}\n")
                f.write(f"Base channels: {config['base_channels']}\n")
                f.write(f"Image size: {config['image_size']}\n")
                f.write(f"Federated rounds: {config['fed_rounds']}\n")
                f.write(f"Number of clients: {config['num_clients']}\n")
                f.write(f"Clients per round: {config['clients_per_round']}\n")
                f.write(f"Local epochs: {config['local_epochs']}\n")
                f.write(f"Total parameters: {total_params:,}\n")
                f.write(f"Trainable parameters: {trainable_params:,}\n")
                f.write(f"Best AUC achieved: {strategy.best_performance:.4f}\n\n")
                
                if strategy.round_metrics:
                    final_metrics = strategy.round_metrics[-1]
                    f.write("Final Validation Results:\n")
                    f.write(f"  Loss: {final_metrics['loss']:.4f}\n")
                    f.write(f"  Accuracy: {final_metrics['accuracy']:.4f}\n")
                    f.write(f"  AUC: {final_metrics['auc']:.4f}\n\n")
                
                # Add test results if available
                if test_result is not None:
                    f.write("Test Results:\n")
                    f.write(f"  Loss: {test_result['loss']:.4f}\n")
                    f.write(f"  Accuracy: {test_result['accuracy']:.4f}\n")
                    f.write(f"  AUC: {test_result['auc']:.4f}\n\n")
                
                f.write("Training completed successfully with Flower federated framework.\n")
            
            print(f"Results summary saved to: {summary_path}")
    
    print(f"\nFlower federated demo completed successfully! All results saved in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
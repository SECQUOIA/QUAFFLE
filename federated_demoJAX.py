#!/usr/bin/env python3
import os
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import sys
sys.path.append('utils')

from utilsJAX import (
    load_dataset, create_train_state, train_step, evaluate_model, 
    save_training_curves, visualize_results, split_data_among_clients,
    jax_params_to_numpy, numpy_to_jax_params
)
from unetJAX import QVUNet

def get_config():
    """Get configuration for federated quantum flood segmentation."""
    config = ml_collections.ConfigDict()
    
    # Data
    config.image_size = 128
    config.channels = 3
    config.num_classes = 2
    config.batch_size = 8
    
    # Model
    config.dim = 32
    config.dim_mults = (1, 2, 4, 8)
    config.resnet_block_groups = 4
    config.quantum_channel = 32
    config.name_ansatz = 'FQConv_ansatz'
    config.num_layer = 2
    
    # Federated Learning
    config.num_clients = 4
    config.local_epochs = 4
    config.fed_rounds = 25  # Total federated rounds
    config.clients_per_round = 4  # Number of clients participating per round
    
    # Training
    config.learning_rate = 1e-4
    config.log_every = 5  # Log every N federated rounds
    config.eval_every = 5  # Evaluate every N federated rounds
    config.seed = 42
    
    return config



class QuantumFlowerClient(fl.client.NumPyClient):
    """Flower client for quantum model training."""
    
    def __init__(self, client_id: int, client_images: np.ndarray, client_masks: np.ndarray, 
                 config: ml_collections.ConfigDict, template_state):
        self.client_id = client_id
        self.client_images = client_images
        self.client_masks = client_masks
        self.config = config
        self.template_state = template_state
        self.local_state = None
        
        print(f"Client {client_id} initialized with {len(client_images)} samples")
    
    def create_data_iterator(self):
        """Create local data iterator for client."""
        def data_generator():
            while True:
                indices = np.random.permutation(len(self.client_images))
                shuffled_images = self.client_images[indices]
                shuffled_masks = self.client_masks[indices]
                
                batch_size = int(self.config.batch_size)  # Convert to int
                for i in range(0, len(shuffled_images), batch_size):
                    batch_images = shuffled_images[i:i + batch_size]
                    batch_masks = shuffled_masks[i:i + batch_size]
                    
                    # Pad last batch if needed
                    if len(batch_images) < batch_size:
                        padding = batch_size - len(batch_images)
                        batch_images = np.concatenate([batch_images, batch_images[-padding:]], axis=0)
                        batch_masks = np.concatenate([batch_masks, batch_masks[-padding:]], axis=0)
                    
                    yield {'image': batch_images, 'mask': batch_masks}
        
        return data_generator()
    
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """Return model parameters as numpy arrays."""
        if self.local_state is None:
            return jax_params_to_numpy(self.template_state.params)
        return jax_params_to_numpy(self.local_state.params)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model locally and return updated parameters."""
        print(f"Client {self.client_id} starting local training...")
        
        # Convert numpy parameters back to JAX
        jax_params = numpy_to_jax_params(parameters, self.template_state.params)
        self.local_state = self.template_state.replace(params=jax_params)
        
        # Create local data iterator
        data_iter = self.create_data_iterator()
        
        # Local training
        batch_size = int(self.config.batch_size)  # Convert to int
        local_epochs = int(self.config.local_epochs)  # Convert to int
        batches_per_epoch = max(1, len(self.client_images) // batch_size)
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch_idx in range(batches_per_epoch):
                batch = next(data_iter)
                self.local_state, metrics = train_step(self.local_state, batch)
                
                epoch_loss += metrics['loss']
                epoch_acc += metrics['accuracy']
                total_batches += 1
            
            total_loss += epoch_loss
            total_acc += epoch_acc
        
        avg_loss = total_loss / (local_epochs * batches_per_epoch)
        avg_acc = total_acc / (local_epochs * batches_per_epoch)
        
        print(f"Client {self.client_id} completed training: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
        
        # Return updated parameters, number of examples, and metrics
        return (
            jax_params_to_numpy(self.local_state.params),
            len(self.client_images),
            {"loss": float(avg_loss), "accuracy": float(avg_acc)}
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict]:
        """Evaluate model locally."""
        # Convert numpy parameters back to JAX
        jax_params = numpy_to_jax_params(parameters, self.template_state.params)
        eval_state = self.template_state.replace(params=jax_params)
        
        # Evaluate on local data
        metrics = evaluate_model(eval_state, self.client_images, self.client_masks, self.config)
        
        return (
            float(metrics['loss']),
            len(self.client_images),
            {
                "accuracy": float(metrics['accuracy']),
                "auc": float(metrics['auc'])
            }
        )

def create_flower_client_fn(client_data: List[Tuple], config: ml_collections.ConfigDict, template_state):
    """Factory function to create Flower clients."""
    def client_fn(cid: str) -> QuantumFlowerClient:
        client_id = int(cid)
        client_images, client_masks = client_data[client_id]
        return QuantumFlowerClient(client_id, client_images, client_masks, config, template_state)
    
    return client_fn

class QuantumFedAvg(FedAvg):
    """Custom FedAvg strategy with quantum model evaluation."""
    
    def __init__(self, val_images, val_masks, config, template_state, **kwargs):
        super().__init__(**kwargs)
        self.val_images = val_images
        self.val_masks = val_masks
        self.config = config
        self.template_state = template_state
        self.round_metrics = []
        self.best_auc = 0.0
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results and perform server-side evaluation."""
        # Call parent method
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        # Server-side evaluation if we have validation data
        if self.val_images is not None and len(results) > 0:
            # Get current global parameters
            if hasattr(self, '_current_parameters') and self._current_parameters is not None:
                # Convert parameters back to JAX
                numpy_params = parameters_to_ndarrays(self._current_parameters)
                jax_params = numpy_to_jax_params(numpy_params, self.template_state.params)
                eval_state = self.template_state.replace(params=jax_params)
                
                # Evaluate global model
                metrics = evaluate_model(eval_state, self.val_images, self.val_masks, self.config)
                metrics['round'] = server_round
                
                # Track best performance
                if metrics['auc'] > self.best_auc:
                    self.best_auc = metrics['auc']
                
                self.round_metrics.append(metrics)
                
                print(f"Global evaluation - Round {server_round}: "
                      f"Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
        
        return aggregated_result
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results and store parameters."""
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        # Store current parameters for evaluation
        if aggregated_result is not None:
            self._current_parameters = aggregated_result[0]
        
        return aggregated_result

def main():
    """Run federated quantum flood segmentation demo with Flower."""
    # Data paths
    base_dir = "data/flood_optical"
    train_images_dir = os.path.join(base_dir, "Training", "images")
    train_masks_dir = os.path.join(base_dir, "Training", "labels")
    test_images_dir = os.path.join(base_dir, "Testing", "images")
    test_masks_dir = os.path.join(base_dir, "Testing", "labels")
    output_dir = "results/flood_federated_optical_jax_pennylane"
    
    print("Federated Quantum Flood Segmentation Demo with Flower")
    print("Using QVUNet with PennyLane quantum circuits in Flower federated setting")
    
    # Check data
    if not os.path.exists(train_images_dir):
        print(f"Training images not found: {train_images_dir}")
        return
    
    config = get_config()
    print(f"Configuration: {config.num_clients} clients, {config.fed_rounds} rounds")
    print(f"Quantum: {config.quantum_channel} channels, {config.name_ansatz} ansatz")
    
    # Load and split data
    print("Loading training data...")
    train_images, train_masks = load_dataset(train_images_dir, train_masks_dir, config)
    
    # Split data for federated learning
    print(f"Splitting data among {config.num_clients} clients...")
    client_data = split_data_among_clients(train_images, train_masks, int(config.num_clients), int(config.seed))
    
    # Create validation set from last client's data (for global evaluation)
    val_split = int(0.2 * len(client_data[-1][0]))
    val_images = client_data[-1][0][-val_split:]
    val_masks = client_data[-1][1][-val_split:]
    # Remove validation data from last client
    client_data[-1] = (client_data[-1][0][:-val_split], client_data[-1][1][:-val_split])
    
    print(f"Validation set: {len(val_images)} samples")
    
    # Initialize template model (for parameter structure)
    print("Initializing global quantum model template...")
    rng = jax.random.PRNGKey(int(config.seed))
    template_state = create_train_state(rng, config)
    
    # Create Flower strategy
    strategy = QuantumFedAvg(
        val_images=val_images,
        val_masks=val_masks,
        config=config,
        template_state=template_state,
        fraction_fit=int(config.clients_per_round) / int(config.num_clients),
        fraction_evaluate=0.5,  # Evaluate on half the clients
        min_fit_clients=int(config.clients_per_round),
        min_evaluate_clients=1,
        min_available_clients=int(config.num_clients),
        initial_parameters=ndarrays_to_parameters(jax_params_to_numpy(template_state.params))
    )
    
    # Create client function
    client_fn = create_flower_client_fn(client_data, config, template_state)
    
    print(f"\nStarting Flower federated training for {config.fed_rounds} rounds...")
    
    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(config.num_clients),
        config=fl.server.ServerConfig(num_rounds=int(config.fed_rounds)),
        strategy=strategy,
        client_resources={"num_cpus": 1.0}  # Adjust based on your resources
    )
    
    print(f"\nFederated training completed!")
    print(f"Best AUC achieved: {strategy.best_auc:.4f}")
    
    # Get final global model for testing
    if hasattr(strategy, '_current_parameters') and strategy._current_parameters is not None:
        final_params = parameters_to_ndarrays(strategy._current_parameters)
        final_jax_params = numpy_to_jax_params(final_params, template_state.params)
        final_state = template_state.replace(params=final_jax_params)
        
        # Test and visualize with final global model
        if os.path.exists(test_images_dir):
            print("Loading test data for visualization...")
            test_images, test_masks = load_dataset(test_images_dir, test_masks_dir, config)
            visualize_results(final_state, test_images, test_masks, config, output_dir, 
                             title_prefix="Flower Federated ", filename_prefix="flower_fed_")
        else:
            print("Using validation data for visualization...")
            visualize_results(final_state, val_images, val_masks, config, output_dir,
                             title_prefix="Flower Federated ", filename_prefix="flower_fed_")
        
        # Save federated training curves
        if strategy.round_metrics:
            save_training_curves(strategy.round_metrics, output_dir, prefix="flower_fed_", 
                                title_prefix="Flower Federated Learning - ")
    
    print(f"\nFlower federated demo completed! Results in {output_dir}")

if __name__ == "__main__":
    main() 
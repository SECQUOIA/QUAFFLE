#!/usr/bin/env python3
import os
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from tqdm import tqdm

# Import utilities
from utils import (
    load_dataset, create_train_state, train_step, evaluate_model, 
    save_training_curves, visualize_results, split_data_among_clients
)

def get_config():
    """Get configuration for federated quantum flood segmentation."""
    config = ml_collections.ConfigDict()
    
    # Data
    config.image_size = 128
    config.channels = 3
    config.num_classes = 2
    config.batch_size = 4
    
    # Model
    config.dim = 32
    config.dim_mults = (1, 2, 4, 8)
    config.resnet_block_groups = 8
    config.quantum_channel = 2
    config.name_ansatz = 'FQConv_ansatz'
    config.num_layer = 1
    
    # Federated Learning
    config.num_clients = 4
    config.local_epochs = 2
    config.fed_rounds = 25  # Total federated rounds
    config.clients_per_round = 2  # Number of clients participating per round
    
    # Training
    config.learning_rate = 1e-4
    config.log_every = 5  # Log every N federated rounds
    config.eval_every = 5  # Evaluate every N federated rounds
    config.seed = 42
    
    return config

class FederatedServer:
    """Federated server for quantum model aggregation."""
    
    def __init__(self, config, global_state):
        self.config = config
        self.global_state = global_state
        self.round = 0
        self.best_performance = 0.0
        self.fed_metrics = []
    
    def aggregate_weights(self, client_states, client_data_sizes):
        """Federated averaging of client model weights."""
        total_samples = sum(client_data_sizes)
        
        # Initialize aggregated parameters
        aggregated_params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self.global_state.params)
        
        # Weighted average based on client data sizes
        for client_state, data_size in zip(client_states, client_data_sizes):
            weight = data_size / total_samples
            aggregated_params = jax.tree_util.tree_map(
                lambda global_p, client_p: global_p + weight * client_p,
                aggregated_params, client_state.params
            )
        
        # Update global state
        self.global_state = self.global_state.replace(params=aggregated_params)
        self.round += 1
        
        return self.global_state
    
    def evaluate_global_model(self, val_images, val_masks):
        """Evaluate global model on validation data."""
        metrics = evaluate_model(self.global_state, val_images, val_masks, self.config)
        metrics['round'] = self.round
        
        # Track best performance
        if metrics['auc'] > self.best_performance:
            self.best_performance = metrics['auc']
        
        self.fed_metrics.append(metrics)
        return metrics

class FederatedClient:
    """Federated client for local quantum model training."""
    
    def __init__(self, client_id, client_images, client_masks, config):
        self.client_id = client_id
        self.client_images = client_images
        self.client_masks = client_masks
        self.config = config
        self.local_state = None
    
    def create_data_iterator(self):
        """Create local data iterator for client."""
        def data_generator():
            while True:
                indices = np.random.permutation(len(self.client_images))
                shuffled_images = self.client_images[indices]
                shuffled_masks = self.client_masks[indices]
                
                batch_size = self.config.batch_size
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
    
    def local_train(self, global_state):
        """Perform local training on client data."""
        # Copy global state to local state
        self.local_state = global_state.replace()
        
        # Create local data iterator
        data_iter = self.create_data_iterator()
        
        # Local training for specified epochs
        batches_per_epoch = max(1, len(self.client_images) // self.config.batch_size)
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch_idx in range(batches_per_epoch):
                batch = next(data_iter)
                self.local_state, metrics = train_step(self.local_state, batch)
                
                epoch_loss += metrics['loss']
                epoch_acc += metrics['accuracy']
            
            avg_loss = epoch_loss / batches_per_epoch
            avg_acc = epoch_acc / batches_per_epoch
            
            if epoch == self.config.local_epochs - 1:  # Log final epoch
                print(f"  Client {self.client_id} final epoch: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
        
        return self.local_state

def main():
    """Run federated quantum flood segmentation demo."""
    # Data paths
    base_dir = "/anvil/projects/x-chm250024/data/flood_optical"
    train_images_dir = os.path.join(base_dir, "Training", "images")
    train_masks_dir = os.path.join(base_dir, "Training", "labels")
    test_images_dir = os.path.join(base_dir, "Testing", "images")
    test_masks_dir = os.path.join(base_dir, "Testing", "labels")
    output_dir = "Output_federated_segmentation"
    
    print("🌐 Federated Quantum Flood Segmentation Demo")
    print("Using QVUNet with PennyLane quantum circuits in federated setting")
    
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
    client_data = split_data_among_clients(train_images, train_masks, config.num_clients, config.seed)
    
    # Create validation set from last client's data (for global evaluation)
    val_split = int(0.2 * len(client_data[-1][0]))
    val_images = client_data[-1][0][-val_split:]
    val_masks = client_data[-1][1][-val_split:]
    # Remove validation data from last client
    client_data[-1] = (client_data[-1][0][:-val_split], client_data[-1][1][:-val_split])
    
    print(f"Validation set: {len(val_images)} samples")
    
    # Initialize global model
    print("Initializing global quantum model...")
    rng = jax.random.PRNGKey(config.seed)
    global_state = create_train_state(rng, config)
    
    # Create federated server
    server = FederatedServer(config, global_state)
    
    # Create federated clients
    clients = []
    for i, (client_images, client_masks) in enumerate(client_data):
        client = FederatedClient(i, client_images, client_masks, config)
        clients.append(client)
    
    print(f"\nStarting federated training for {config.fed_rounds} rounds...")
    
    # Federated training loop
    for round_num in tqdm(range(config.fed_rounds), desc="Federated Rounds"):
        print(f"\n🔄 Round {round_num + 1}/{config.fed_rounds}")
        
        # Select clients for this round
        selected_clients = np.random.choice(
            clients, 
            size=min(config.clients_per_round, len(clients)), 
            replace=False
        )
        
        # Local training on selected clients
        client_states = []
        client_data_sizes = []
        
        for client in selected_clients:
            print(f"  Training client {client.client_id}...")
            local_state = client.local_train(server.global_state)
            client_states.append(local_state)
            client_data_sizes.append(len(client.client_images))
        
        # Aggregate models on server
        print(f"  Aggregating {len(client_states)} client models...")
        server.aggregate_weights(client_states, client_data_sizes)
        
        # Evaluate global model
        if (round_num + 1) % config.eval_every == 0:
            print(f"  Evaluating global model...")
            metrics = server.evaluate_global_model(val_images, val_masks)
            print(f"  Round {round_num + 1}: Loss={metrics['loss']:.4f}, "
                  f"Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
    
    print(f"\n Federated training completed!")
    print(f"Best AUC achieved: {server.best_performance:.4f}")
    
    # Test and visualize with global model
    if os.path.exists(test_images_dir):
        print("Loading test data for visualization...")
        test_images, test_masks = load_dataset(test_images_dir, test_masks_dir, config)
        visualize_results(server.global_state, test_images, test_masks, config, output_dir, 
                         title_prefix="Federated ", filename_prefix="fed_")
    else:
        print("Using validation data for visualization...")
        visualize_results(server.global_state, val_images, val_masks, config, output_dir,
                         title_prefix="Federated ", filename_prefix="fed_")
    
    # Save federated training curves
    save_training_curves(server.fed_metrics, output_dir, prefix="fed_", title_prefix="Federated Learning - ")
    
    print(f"\n🎉 Federated demo completed! Results in {output_dir}")

if __name__ == "__main__":
    main() 
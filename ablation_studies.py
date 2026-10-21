#!/usr/bin/env python3
"""
ablation_studies.py

Parameter-matched classical UNet ablation against quantum (QVUNet) models.
Supports both PyTorch (Torch) and JAX (Flax/PennyLane) frameworks, and both
centralized and federated training modes.

For each framework × mode combination, the script:
  1. Initializes the reference quantum model to count its parameters.
  2. Builds a parameter-matched classical ablation that uses the same QVUNet
     architecture, swapping only the QuantumBlock at the bottleneck for a
     lightweight ClassicalBottleneckBlock (1x1 conv). This isolates the effect
     of the quantum circuit itself — matching params to within ~0.01% instead
     of the ~18% gap the old binary-search-on-base_channels approach produced.
  3. Trains that ablation model under identical conditions.

Usage:
    python ablation_studies.py --framework torch --mode centralized
    python ablation_studies.py --framework torch --mode centralized --quantum-backend ptlayer
    python ablation_studies.py --framework torch --mode federated
    python ablation_studies.py --framework torch --mode federated --partition dirichlet --alpha 0.5
    python ablation_studies.py --framework jax  --mode centralized
    python ablation_studies.py --framework jax  --mode federated
    python ablation_studies.py --framework jax  --mode federated --partition dirichlet --alpha 0.5
"""

import os
import argparse
import copy
import numpy as np
from typing import List, Tuple, Dict
import flwr as fl

from tools.provenance import collect_provenance, format_provenance, print_provenance
from tools import training_common as tc

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter-matched classical ablation vs QVUNet (Torch & JAX)"
    )
    parser.add_argument('--framework', choices=['torch', 'jax'], required=True,
                        help='ML framework to use')
    parser.add_argument('--mode', choices=['centralized', 'federated'], required=True,
                        help='Training mode')
    parser.add_argument('--quantum-backend', choices=['pennylane', 'ptlayer'],
                        default='pennylane',
                        help='Quantum backend for reference QVUNet (Torch only, default: pennylane)')
    parser.add_argument('--partition', choices=['random', 'dirichlet'], default='random',
                        help='Federated data partitioning strategy (default: random)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha, lower = more non-IID (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--dataset', choices=['optical', 'sar', 'combined'], default='optical',
                        help='Dataset to use (default: optical)')
    # Torch-only shared flags. --q-ansatz / --q-layers configure the reference
    # quantum model that the ablation is parameter-matched against.
    tc.add_common_args(parser, quantum=True, federated=True)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override centralized training epochs (debug)')
    return parser.parse_args()


# ===========================================================================
#  TORCH SECTION
# ===========================================================================

def _torch_quantum_config(backend='pennylane'):
    return {
        'image_size': 128,
        'base_channels': 32,
        'init_dim': 32,
        'out_dim': 2,
        'dim_mults': (1, 2, 4, 8),
        'resnet_block_groups': 4,
        'quantum_channels': 32,
        'quantum_backend': backend,
        'batch_size': 8,
        'seed': 42,
    }


def _torch_classical_base_config():
    return {
        'image_size': 128,
        'base_channels': 32,
        'init_dim': 32,
        'out_dim': 2,
        'dim_mults': (1, 2, 4, 8),
        'resnet_block_groups': 4,
        'batch_size': 8,
        'seed': 42,
    }


def _count_torch_params(model):
    return sum(p.numel() for p in model.parameters())


def _find_matched_torch_config(backend, overrides=None):
    """Parameter-matched ablation config.

    The ablation uses ClassicalAblationUNetSegmentation: the same QVUNet
    architecture as the quantum model, with the PQC-based QuantumBlock swapped
    for a lightweight 1x1-conv ClassicalBottleneckBlock. Matching the
    parameter count by searching over base_channels instead cannot get closer
    than about 18% because of GroupNorm divisibility constraints.
    """
    from utilsTorch import QVUNetSegmentation, ClassicalAblationUNetSegmentation

    qcfg = _torch_quantum_config(backend)
    if overrides:
        qcfg.update(overrides)
    q_params = _count_torch_params(tc.build_model(QVUNetSegmentation, qcfg))
    c_params = _count_torch_params(tc.build_model(ClassicalAblationUNetSegmentation, qcfg))
    print(f"Quantum params:   {q_params:,}")
    print(f"Ablation params:  {c_params:,}  ({100*(c_params-q_params)/q_params:+.2f}% vs quantum)")
    return dict(qcfg), c_params, q_params


def _torch_overrides(args):
    """Config entries from the shared CLI flags (seed, data, quantum model)."""
    cfg = {'seed': args.seed}
    tc.apply_common_args(cfg, args)
    return cfg


# ---------------------------------------------------------------------------
# Torch data helpers
# ---------------------------------------------------------------------------

def _torch_split_random(dataset, num_clients, seed=42):
    """Legacy split, unused by the Torch path (see tc.build_federated_split)."""
    import random
    from torch.utils.data import Subset
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    per = len(dataset) // num_clients
    result = []
    for i in range(num_clients):
        s = i * per
        e = s + per if i < num_clients - 1 else len(dataset)
        result.append(Subset(dataset, indices[s:e]))
    return result


def _torch_flood_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        labels.append(1 if mask.mean().item() > 0.05 else 0)
    return np.array(labels)


def _torch_split_dirichlet(dataset, num_clients, alpha, seed=42):
    """Legacy split, unused by the Torch path (see tc.build_federated_split)."""
    from torch.utils.data import Subset
    rng = np.random.default_rng(seed)
    labels = _torch_flood_labels(dataset)
    client_indices = [[] for _ in range(num_clients)]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(alpha * np.ones(num_clients))
        splits = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, splits)):
            client_indices[cid].extend(chunk.tolist())
    print(f"Dirichlet (alpha={alpha}) client sizes: {[len(x) for x in client_indices]}")
    return [Subset(dataset, idxs) for idxs in client_indices]


# ---------------------------------------------------------------------------
# Torch centralized
# ---------------------------------------------------------------------------

def run_centralized_torch(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from utilsTorch import ClassicalAblationUNetSegmentation, get_dataloaders, evaluate_model, \
        save_training_curves, visualize_results
    from results_tracker import save_results_to_excel

    _DATA_DIRS = {'optical': 'data/flood_optical', 'sar': 'data/flood_SAR', 'combined': 'data/flood_combined'}
    backend = args.quantum_backend
    base_dir = _DATA_DIRS[args.dataset]

    print("=" * 60)
    print("Ablation (Torch): Parameter-Matched Classical UNet (Centralized)")
    print(f"Reference quantum backend: {backend}")
    print("=" * 60)

    train_images_dir = os.path.join(base_dir, "Training", "images")
    train_masks_dir  = os.path.join(base_dir, "Training", "labels")
    test_images_dir  = os.path.join(base_dir, "Testing", "images")
    test_masks_dir   = os.path.join(base_dir, "Testing", "labels")

    if not os.path.exists(train_images_dir):
        tc.fail_run(None, f"Training images not found: {train_images_dir}")

    overrides = _torch_overrides(args)
    tc.seed_everything(args.seed)
    provenance = collect_provenance(seed=args.seed)
    print_provenance(provenance)

    matched_config, c_params, q_params = _find_matched_torch_config(backend, overrides)
    config = {
        **matched_config,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'log_every': 10,
        'eval_every': 10,
        'save_samples': True,
        'num_train_samples': 6,
        'num_val_samples': 6,
        'num_test_samples': 20,
        'quantum_backend': backend,
        'dataset': args.dataset,
    }
    config['seed'] = args.seed
    ds_suffix = "" if args.dataset == "optical" else f"_{args.dataset}"
    vtag = tc.variant_tag(config)  # matched quantum variant, e.g. _reupload_L2
    output_dir = f"results/ablation_centralized_torch_bottleneck_matched_{backend}{ds_suffix}{vtag}_seed{config['seed']}"
    print(f"Output directory: {output_dir}")

    if getattr(args, 'epochs', None) is not None:
        config['num_epochs'] = int(args.epochs)

    # Reseed after the parameter-count probes so the ablation model init only
    # depends on the seed.
    tc.seed_everything(config['seed'])

    print(f"Matched base_channels={config['base_channels']}  "
          f"({c_params:,} params vs {q_params:,} quantum params)")

    # Same global stratified validation holdout as the main demos.
    full_dataset = tc.load_dataset(train_images_dir, train_masks_dir, config)
    full_dataset = tc.cap_samples(full_dataset, config.get('max_samples'), config['seed'])
    tr_idx, va_idx, _ = tc.holdout_validation(full_dataset, config['seed'], config['val_fraction'])
    train_loader = tc.make_loader(Subset(full_dataset, tr_idx), config['batch_size'], True, config['seed'])
    val_loader   = tc.make_loader(Subset(full_dataset, va_idx), config['batch_size'], False, config['seed'])
    print(f"Train: {len(tr_idx)}, Val: {len(va_idx)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = tc.build_model(ClassicalAblationUNetSegmentation, config).to(device)
    total_params = _count_torch_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    eval_log = os.path.join(logs_dir, 'evaluation_scores.txt')
    ckpt_path = os.path.join(output_dir, 'checkpoint.pt')

    start_epoch = 0
    train_metrics, val_metrics = [], []
    ckpt = tc.safe_torch_load(ckpt_path, map_location=device)
    if ckpt is not None and not tc.checkpoint_compatible(ckpt):
        ckpt = None
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        train_metrics = ckpt.get('train_metrics', [])
        val_metrics   = ckpt.get('val_metrics', [])
        tc.restore_rng_state(ckpt.get('rng_state'), train_loader)
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch")

    if not val_metrics:
        r = evaluate_model(model, val_loader, device)
        r['epoch'] = start_epoch
        val_metrics.append(r)
        print(f"Initial Val: Loss={r['loss']:.4f} Acc={r['accuracy']:.4f} "
              f"AUC={r['auc']:.4f} IoU={r['iou']:.4f} Dice={r['dice']:.4f}")
        with open(eval_log, 'a') as f:
            f.write(f"Initial Epoch {start_epoch}: Loss={r['loss']:.4f} "
                    f"Accuracy={r['accuracy']:.4f} AUC={r['auc']:.4f} "
                    f"IoU={r['iou']:.4f} Dice={r['dice']:.4f}\n")
        save_training_curves(val_metrics, output_dir,
                             title_prefix="Ablation Centralized Torch Classical ")

    from utilsTorch import train_one_epoch
    epoch = start_epoch
    while epoch < config['num_epochs']:
        avg_loss, accuracy = train_one_epoch(model, train_loader, optimizer, device)
        epoch += 1
        train_metrics.append({'epoch': epoch, 'loss': avg_loss, 'accuracy': accuracy,
                               'auc': 0.0, 'iou': 0.0, 'dice': 0.0})
        if epoch % config['log_every'] == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f} Acc={accuracy:.4f}")
        if epoch % config['eval_every'] == 0:
            r = evaluate_model(model, val_loader, device)
            r['epoch'] = epoch
            val_metrics.append(r)
            print(f"Val Epoch {epoch}: Loss={r['loss']:.4f} Acc={r['accuracy']:.4f} "
                  f"AUC={r['auc']:.4f} IoU={r['iou']:.4f} Dice={r['dice']:.4f}")
            with open(eval_log, 'a') as f:
                f.write(f"Epoch {epoch}: Loss={r['loss']:.4f} "
                        f"Accuracy={r['accuracy']:.4f} AUC={r['auc']:.4f} "
                        f"IoU={r['iou']:.4f} Dice={r['dice']:.4f}\n")
            tc.atomic_torch_save({'model_state_dict': model.state_dict(),
                                  'run_format': tc.RUN_FORMAT,
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'epoch': epoch, 'train_metrics': train_metrics,
                                  'val_metrics': val_metrics,
                                  'rng_state': tc.rng_state(train_loader)}, ckpt_path)
            if epoch % 100 == 0:
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch},
                           os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt'))
        if epoch % 50 == 0 and val_metrics:
            save_training_curves(val_metrics, output_dir,
                                 title_prefix="Ablation Centralized Torch Classical ")

    if val_metrics:
        save_training_curves(val_metrics, output_dir,
                             title_prefix="Ablation Centralized Torch Classical ")

    test_result = None
    if os.path.exists(test_images_dir):
        test_dataset = tc.load_dataset(test_images_dir, test_masks_dir, config)
        test_dataset = tc.cap_samples(test_dataset, config.get('max_samples'), config['seed'])
        test_loader = tc.make_loader(test_dataset, config['batch_size'], False, config['seed'])
        test_result = evaluate_model(model, test_loader, device)
        print(f"Test: Loss={test_result['loss']:.4f} Acc={test_result['accuracy']:.4f} "
              f"AUC={test_result['auc']:.4f} IoU={test_result['iou']:.4f} "
              f"Dice={test_result['dice']:.4f}")
        if config['save_samples']:
            visualize_results(model, test_loader, device, output_dir,
                              title_prefix="Ablation Test - ", filename_prefix="ablation_test_",
                              num_samples=config['num_test_samples'])

    if test_result is None:
        tc.fail_run(output_dir, f"no test set at {test_images_dir}, refusing to write a summary or tracker row")
    _write_summary_torch(output_dir, backend, q_params, config['base_channels'],
                         total_params, val_metrics, test_result, mode='centralized',
                         extra_lines=[f"Validation: global held-out {len(va_idx)} samples "
                                      f"({config['val_fraction']} of the training pool, stratified)"],
                         provenance=provenance)
    tc.write_metrics_json(
        output_dir, script=os.path.basename(__file__), training="centralized",
        model_type="ablation", config=config,
        val_metrics=val_metrics[-1] if val_metrics else None, test_metrics=test_result,
        history=val_metrics, total_params=total_params, provenance=provenance,
        extra={'reference_quantum_params': q_params})
    tc.safe_tracker_call(
        save_results_to_excel,
        demo_name=f"Ablation Centralized Torch Bottleneck (vs {backend})",
        framework="torch", training="centralized",
        variant=f"classical_matched_{config['base_channels']}ch",
        val_metrics=val_metrics[-1] if val_metrics else None,
        test_metrics=test_result, config={**config, 'total_params': total_params},
        dataset=args.dataset, provenance=provenance,
    )
    print(f"\nAblation centralized Torch complete. Results in: {output_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Torch federated
# ---------------------------------------------------------------------------

def _torch_params_to_numpy(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]


def _numpy_to_torch_params(model, numpy_params):
    import torch
    for p, np_p in zip(model.parameters(), numpy_params):
        p.data = torch.from_numpy(np_p).to(p.device)


class _TorchClassicalFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient for the parameter-matched classical Torch model."""

    def __init__(self, client_id, client_dataset, config, device_str):
        import torch
        import flwr as fl
        self._fl = fl
        self.client_id = client_id
        self.config = config
        self.client_dataset = client_dataset
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.template_model = None
        self.local_model = None
        # Shuffle order is re-seeded every round in fit() from
        # (seed, client_id, server_round), see tc.client_round_seed.
        self.loader = tc.make_loader(client_dataset, config['batch_size'], True,
                                     tc.client_round_seed(config['seed'], client_id, 0))
        self.eval_loader = tc.make_loader(client_dataset, config['batch_size'], False,
                                          config['seed'])
        print(f"Client {client_id}: {len(client_dataset)} samples on {self.device}")

    def get_parameters(self, config):
        m = self.local_model if self.local_model is not None else self.template_model
        return _torch_params_to_numpy(m)

    def fit(self, parameters, config):
        import torch
        from utilsTorch import ClassicalAblationUNetSegmentation
        server_round = int(config.get('server_round', 0)) if config else 0
        round_seed = tc.client_round_seed(self.config['seed'], self.client_id, server_round)
        torch.manual_seed(round_seed)
        np.random.seed(round_seed)
        self.loader = tc.make_loader(self.client_dataset, self.config['batch_size'], True, round_seed)
        if self.template_model is None:
            self.template_model = tc.build_model(ClassicalAblationUNetSegmentation, self.config).to(self.device)
        self.local_model = copy.deepcopy(self.template_model).to(self.device)
        if parameters:
            _numpy_to_torch_params(self.local_model, parameters)
        opt = torch.optim.Adam(self.local_model.parameters(), lr=self.config['learning_rate'])
        self.local_model.train()
        total_loss = total_acc = n = 0
        for _ in range(self.config['local_epochs']):
            for imgs, masks in self.loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                opt.zero_grad()
                B = imgs.shape[0]
                dummy_t = torch.zeros((B,), device=self.device)
                logits = self.local_model.model(imgs, dummy_t)
                masks_long = masks.squeeze(1).long()
                loss = torch.nn.functional.cross_entropy(logits, masks_long)
                loss.backward()
                opt.step()
                preds = torch.argmax(logits, dim=1)
                total_loss += loss.item()
                total_acc += (preds == masks_long).float().mean().item()
                n += 1
        avg_loss = total_loss / n if n else 0.0
        avg_acc  = total_acc  / n if n else 0.0
        print(f"Client {self.client_id}: Loss={avg_loss:.4f} Acc={avg_acc:.4f}")
        return (_torch_params_to_numpy(self.local_model), len(self.loader.dataset),
                {"loss": float(avg_loss), "accuracy": float(avg_acc)})

    def evaluate(self, parameters, config):
        from utilsTorch import ClassicalAblationUNetSegmentation, evaluate_model
        if self.template_model is None:
            self.template_model = tc.build_model(ClassicalAblationUNetSegmentation, self.config).to(self.device)
        m = copy.deepcopy(self.template_model).to(self.device)
        if parameters:
            _numpy_to_torch_params(m, parameters)
        metrics = evaluate_model(m, self.eval_loader, self.device)
        return (float(metrics['loss']), len(self.loader.dataset),
                {"accuracy": float(metrics['accuracy']), "auc": float(metrics['auc']),
                 "iou": float(metrics['iou']), "dice": float(metrics['dice'])})


class _TorchAblationFedAvg:
    """FedAvg strategy that evaluates the global classical model on a validation set."""

    def __init__(self, val_loader, config, device_str, output_dir, **kwargs):
        import flwr as fl
        from flwr.server.strategy import FedAvg
        self._base = FedAvg(**kwargs)
        self.val_loader = val_loader
        self.config = config
        self.device_str = device_str
        self.device = None
        self.template_model = None
        self.output_dir = output_dir
        self.round_metrics = []
        self.best_performance = 0.0
        self._current_parameters = None
        self.failed_rounds = []
        # Resume support: Flower counts rounds from 1 in every process, a
        # resumed run shifts them by round_offset (see tc.RoundOffsetMixin).
        self.round_offset = 0
        self.resumed_from_round = 0
        self.fingerprint = None
        logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        self.eval_log = os.path.join(logs_dir, 'evaluation_scores.txt')
        self._fl = fl
        # expose required attributes
        self.fraction_fit       = self._base.fraction_fit
        self.fraction_evaluate  = self._base.fraction_evaluate

    @property
    def initial_parameters(self):
        return self._base.initial_parameters

    @initial_parameters.setter
    def initial_parameters(self, value):
        self._base.initial_parameters = value

    def initialize_parameters(self, client_manager):
        return self._base.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self._base.configure_fit(server_round + self.round_offset, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self._base.configure_evaluate(server_round + self.round_offset, parameters,
                                             client_manager)

    def aggregate_fit(self, server_round, results, failures):
        server_round += self.round_offset
        if failures:
            self.failed_rounds.append((server_round, len(failures), repr(failures[0])[:500]))
            print(f"Round {server_round}: {len(failures)} client fit failures, first: {repr(failures[0])[:500]}")
        agg = self._base.aggregate_fit(server_round, results, failures)
        if agg is not None:
            self._current_parameters = agg[0]
        return agg

    def aggregate_evaluate(self, server_round, results, failures):
        import torch
        from flwr.common import parameters_to_ndarrays
        from utilsTorch import ClassicalAblationUNetSegmentation, evaluate_model
        server_round += self.round_offset
        agg = self._base.aggregate_evaluate(server_round, results, failures)
        if self.val_loader is not None and self._current_parameters is not None:
            if self.device is None:
                self.device = torch.device(self.device_str if torch.cuda.is_available() else 'cpu')
            if self.template_model is None:
                self.template_model = tc.build_model(ClassicalAblationUNetSegmentation, self.config).to(self.device)
            m = copy.deepcopy(self.template_model).to(self.device)
            _numpy_to_torch_params(m, parameters_to_ndarrays(self._current_parameters))
            metrics = evaluate_model(m, self.val_loader, self.device)
            metrics['step'] = server_round
            if metrics['auc'] > self.best_performance:
                self.best_performance = metrics['auc']
            self.round_metrics.append(metrics)
            print(f"Global Round {server_round}: Loss={metrics['loss']:.4f} "
                  f"Acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f} "
                  f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f}")
            with open(self.eval_log, 'a') as f:
                f.write(f"Round {server_round}: Loss={metrics['loss']:.4f} "
                        f"Accuracy={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f} "
                        f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f}\n")
            if server_round % 25 == 0:
                torch.save({'model_state_dict': m.state_dict(), 'round': server_round,
                            'round_metrics': self.round_metrics},
                           os.path.join(self.output_dir, f'checkpoint_round_{server_round}.pt'))
        tc.fed_round_end(self, server_round)
        return agg

    def evaluate(self, server_round, parameters):
        r = server_round + self.round_offset if server_round > 0 else server_round
        return self._base.evaluate(r, parameters)


def run_federated_torch(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from utilsTorch import ClassicalAblationUNetSegmentation, FloodSegmentationDataset, \
        get_dataloaders, evaluate_model, save_training_curves, visualize_results
    from results_tracker import save_results_to_excel

    _DATA_DIRS = {'optical': 'data/flood_optical', 'sar': 'data/flood_SAR', 'combined': 'data/flood_combined'}
    backend = args.quantum_backend
    suffix  = f"_dirichlet{args.alpha}" if args.partition == 'dirichlet' else "_random"

    print("=" * 60)
    print("Ablation (Torch): Parameter-Matched Classical UNet (Federated)")
    print(f"Reference quantum backend: {backend}  |  Partition: {args.partition}"
          + (f"  alpha={args.alpha}" if args.partition == 'dirichlet' else ""))
    print("=" * 60)

    base_dir        = _DATA_DIRS[args.dataset]
    train_images_dir = os.path.join(base_dir, "Training", "images")
    train_masks_dir  = os.path.join(base_dir, "Training", "labels")
    test_images_dir  = os.path.join(base_dir, "Testing", "images")
    test_masks_dir   = os.path.join(base_dir, "Testing", "labels")

    if not os.path.exists(train_images_dir):
        tc.fail_run(None, f"Training images not found: {train_images_dir}")

    overrides = _torch_overrides(args)
    tc.seed_everything(args.seed, deterministic=False)
    provenance = collect_provenance(seed=args.seed)
    print_provenance(provenance)

    matched_config, c_params, q_params = _find_matched_torch_config(backend, overrides)
    config = {
        **matched_config,
        'learning_rate': 1e-4,
        'num_clients': 4,
        'local_epochs': 4,
        'fed_rounds': 25,
        'clients_per_round': 4,
        'log_every': 5,
        'eval_every': 5,
        'save_samples': True,
        'num_val_samples': 6,
        'num_test_samples': 20,
        'partition': args.partition,
        'dirichlet_alpha': args.alpha,
        'quantum_backend': backend,
        'dataset': args.dataset,
    }
    config['seed'] = args.seed
    tc.apply_common_args(config, args)
    tc.seed_everything(config['seed'], deterministic=False)
    ds_suffix = "" if args.dataset == "optical" else f"_{args.dataset}"
    vtag = tc.variant_tag(config)  # matched quantum variant, e.g. _reupload_L2
    output_dir = f"results/ablation_federated_torch_bottleneck_matched_{backend}{suffix}{ds_suffix}{vtag}_seed{config['seed']}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"Matched base_channels={config['base_channels']}  "
          f"({c_params:,} params vs {q_params:,} quantum params)")

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")

    full_dataset = tc.load_dataset(train_images_dir, train_masks_dir, config)
    full_dataset = tc.cap_samples(full_dataset, config.get('max_samples'), config['seed'])
    # Global validation holdout first, then random IID or label-skew
    # Dirichlet partitioning (shared with the main federated demos).
    client_index_lists, val_indices, split_info = tc.build_federated_split(full_dataset, config)
    client_datasets = [Subset(full_dataset, idxs) for idxs in client_index_lists]
    val_dataset = Subset(full_dataset, val_indices)
    val_loader = tc.make_loader(val_dataset, config['batch_size'], False, config['seed'])
    print(f"Validation set: {len(val_dataset)} samples")

    torch.manual_seed(config['seed'])
    temp = tc.build_model(ClassicalAblationUNetSegmentation, config)
    total_params = _count_torch_params(temp)
    initial_parameters = ndarrays_to_parameters(_torch_params_to_numpy(temp))
    del temp
    print(f"Model initialized: {total_params:,} parameters")

    strategy = _TorchAblationFedAvg(
        val_loader=val_loader, config=config, device_str=device_str, output_dir=output_dir,
        fraction_fit=config['clients_per_round'] / config['num_clients'],
        fraction_evaluate=1.0,
        min_fit_clients=config['clients_per_round'],
        min_evaluate_clients=1,
        min_available_clients=config['num_clients'],
        initial_parameters=initial_parameters,
        accept_failures=False,
        on_fit_config_fn=tc.fit_config_fn,
    )

    def client_fn(cid):
        return _TorchClassicalFlowerClient(
            int(cid), client_datasets[int(cid)], config, device_str).to_client()

    # Resume from the last resumable checkpoint (global parameters, history)
    # and run only the remaining rounds with the original round numbering.
    strategy.fingerprint = tc.run_fingerprint(config, 'ablation')
    resume_ckpt = tc.load_fed_checkpoint(output_dir, strategy.fingerprint, config.get('resume', True))
    resumed_params, remaining_rounds = tc.apply_fed_resume(strategy, resume_ckpt)
    if resumed_params is not None:
        strategy.initial_parameters = resumed_params
    del resume_ckpt

    if remaining_rounds > 0:
        print(f"\nStarting Flower federated training: rounds {strategy.round_offset + 1}.."
              f"{config['fed_rounds']} ({remaining_rounds} of {config['fed_rounds']})...")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config['num_clients'],
            config=fl.server.ServerConfig(num_rounds=remaining_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1.0, "num_gpus": 0.1 if device_str == 'cuda' else 0.0},
        )
    else:
        print(f"\nAll {config['fed_rounds']} rounds already completed in the checkpoint, "
              f"skipping training")

    if strategy.failed_rounds:
        tc.fail_run(output_dir, f"{len(strategy.failed_rounds)} federated rounds had client failures "
                                f"(first: round {strategy.failed_rounds[0][0]}, {strategy.failed_rounds[0][2]})")
    if strategy._current_parameters is None:
        tc.fail_run(output_dir, "no aggregated global parameters (every round failed)")
    if not strategy.round_metrics:
        tc.fail_run(output_dir, "no server-side validation round completed")
    print(f"\nFederated training complete. Best AUC: {strategy.best_performance:.4f}")

    test_result = None
    if strategy._current_parameters is not None:
        final_model = tc.build_model(ClassicalAblationUNetSegmentation, config)
        final_model.to(torch.device(device_str))
        _numpy_to_torch_params(final_model, parameters_to_ndarrays(strategy._current_parameters))

        if strategy.round_metrics:
            save_training_curves(strategy.round_metrics, output_dir,
                                 prefix="ablation_fed_",
                                 title_prefix="Ablation Federated Torch Classical ")

        if os.path.exists(test_images_dir):
            test_dataset = tc.load_dataset(test_images_dir, test_masks_dir, config)
            test_dataset = tc.cap_samples(test_dataset, config.get('max_samples'), config['seed'])
            test_loader = tc.make_loader(test_dataset, config['batch_size'], False, config['seed'])
            test_result = evaluate_model(final_model, test_loader, torch.device(device_str))
            print(f"Test: Loss={test_result['loss']:.4f} Acc={test_result['accuracy']:.4f} "
                  f"AUC={test_result['auc']:.4f} IoU={test_result['iou']:.4f} "
                  f"Dice={test_result['dice']:.4f}")
            with open(strategy.eval_log, 'a') as f:
                f.write(f"Final Test: Loss={test_result['loss']:.4f} "
                        f"Accuracy={test_result['accuracy']:.4f} AUC={test_result['auc']:.4f} "
                        f"IoU={test_result['iou']:.4f} Dice={test_result['dice']:.4f}\n")
            if config['save_samples']:
                visualize_results(final_model, test_loader, torch.device(device_str), output_dir,
                                  title_prefix="Ablation Fed Test - ",
                                  filename_prefix="ablation_fed_test_",
                                  num_samples=config['num_test_samples'])

    if test_result is None:
        tc.fail_run(output_dir, f"no test evaluation (test set missing at {test_images_dir})")
    _write_summary_torch(output_dir, backend, q_params, config['base_channels'],
                         total_params, strategy.round_metrics, test_result,
                         mode='federated',
                         extra_lines=split_info + [
                             f"Clients: {config['num_clients']}, Rounds: {config['fed_rounds']}, "
                             f"Local epochs: {config['local_epochs']}",
                         ],
                         provenance=provenance)
    tc.write_metrics_json(
        output_dir, script=os.path.basename(__file__), training="federated",
        model_type="ablation", config=config,
        val_metrics=strategy.round_metrics[-1] if strategy.round_metrics else None,
        test_metrics=test_result, history=strategy.round_metrics,
        total_params=total_params, best_val_auc=strategy.best_performance,
        provenance=provenance,
        extra={'reference_quantum_params': q_params,
               'resumed_from_round': strategy.resumed_from_round})
    tc.safe_tracker_call(
        save_results_to_excel,
        demo_name=f"Ablation Federated Torch Bottleneck (vs {backend})",
        framework="torch", training="federated",
        variant=f"classical_matched_{config['base_channels']}ch",
        val_metrics=strategy.round_metrics[-1] if strategy.round_metrics else None,
        test_metrics=test_result,
        config={**config, 'total_params': total_params},
        dataset=args.dataset, provenance=provenance,
    )
    print(f"\nAblation federated Torch complete. Results in: {output_dir}")
    print("=" * 60)


def _write_summary_torch(output_dir, backend, q_params, base_channels,
                         total_params, val_metrics, test_result,
                         mode='centralized', extra_lines=None, provenance=None):
    summary_path = os.path.join(output_dir, "results", "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(f"Ablation Study (Torch) — Parameter-Matched Classical UNet ({mode.title()})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Reference quantum backend: {backend}\n")
        if extra_lines:
            for line in extra_lines:
                f.write(line + "\n")
        f.write(f"Quantum model params:    {q_params:,}\n")
        f.write(f"Classical base_channels: {base_channels}\n")
        f.write(f"Classical model params:  {total_params:,}\n\n")
        if val_metrics:
            v = val_metrics[-1]
            f.write("Final Validation:\n")
            f.write(f"  Loss: {v['loss']:.4f}\n  Accuracy: {v['accuracy']:.4f}\n")
            f.write(f"  AUC: {v['auc']:.4f}\n  IoU: {v['iou']:.4f}\n  Dice: {v['dice']:.4f}\n\n")
        if test_result:
            f.write("Test Results:\n")
            f.write(f"  Loss: {test_result['loss']:.4f}\n  Accuracy: {test_result['accuracy']:.4f}\n")
            f.write(f"  AUC: {test_result['auc']:.4f}\n  IoU: {test_result['iou']:.4f}\n")
            f.write(f"  Dice: {test_result['dice']:.4f}\n")
        if provenance:
            f.write("\n" + format_provenance(provenance))


# ===========================================================================
#  JAX SECTION
# ===========================================================================

def _jax_quantum_config():
    import ml_collections
    config = ml_collections.ConfigDict()
    config.image_size = 128
    config.channels = 3
    config.num_classes = 2
    config.batch_size = 8
    config.base_channels = 32
    config.dim_mults = (1, 2, 4, 8)
    config.resnet_block_groups = 4
    config.quantum_channels = 32
    config.name_ansatz = 'basic_ansatz'
    config.num_layer = 1
    config.learning_rate = 1e-4
    config.seed = 42
    return config


def _make_jax_classical_config(base_channels):
    """Return a ConfigDict for the classical JAX model with the given base_channels."""
    import ml_collections
    config = ml_collections.ConfigDict()
    config.image_size = 128
    config.channels = 3
    config.num_classes = 2
    config.batch_size = 8
    config.base_channels = base_channels
    config.dim_mults = (1, 2, 4, 8)
    config.resnet_block_groups = 4
    config.learning_rate = 1e-4
    config.seed = 42
    return config


def _count_jax_params(params):
    import jax
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def _create_jax_classical_state(rng, config):
    """Create a Flax TrainState wrapping the classical UNet (JAX)."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    import flax.linen as nn
    import ml_collections
    from utilsJAX import unetJAX as jax_unet

    class UNet_Segmentation(nn.Module):
        cfg: ml_collections.ConfigDict

        @nn.compact
        def __call__(self, x):
            unet_model = jax_unet.UNet(
                dim=self.cfg.base_channels,
                out_dim=self.cfg.num_classes,
                dim_mults=self.cfg.dim_mults,
                resnet_block_groups=self.cfg.resnet_block_groups,
            )
            B = x.shape[0]
            dummy_time = jnp.zeros((B,), dtype=jnp.int32)
            return unet_model(x, dummy_time)

    model = UNet_Segmentation(cfg=config)
    dummy_input = jnp.ones((1, config.image_size, config.image_size, config.channels))
    params = model.init(rng, dummy_input)

    param_count = _count_jax_params(params)
    print(f"Classical JAX model parameters: {param_count:,}")

    tx = optax.adam(learning_rate=config.learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def _find_matched_jax_config():
    import jax
    from utilsJAX import create_train_state

    print("Counting JAX quantum model parameters...")
    rng = jax.random.PRNGKey(42)
    q_config = _jax_quantum_config()
    q_state = create_train_state(rng, q_config)
    q_params_count = _count_jax_params(q_state.params)
    del q_state
    print(f"JAX quantum model parameters: {q_params_count:,}")
    print("Searching for parameter-matched classical base_channels (JAX)...")

    lo, hi, best_dim = 1, 512, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        cfg = _make_jax_classical_config(mid)
        try:
            rng_test = jax.random.PRNGKey(0)
            state = _create_jax_classical_state(rng_test, cfg)
            c = _count_jax_params(state.params)
            del state
        except Exception:
            hi = mid - 1
            continue
        if c <= q_params_count:
            best_dim = mid
            lo = mid + 1
        else:
            hi = mid - 1

    matched_config = _make_jax_classical_config(best_dim)
    rng_final = jax.random.PRNGKey(0)
    state = _create_jax_classical_state(rng_final, matched_config)
    c_params = _count_jax_params(state.params)
    del state
    print(f"Matched JAX classical base_channels={best_dim}  "
          f"({c_params:,} classical params vs {q_params_count:,} quantum params)")
    return matched_config, c_params, q_params_count


# ---------------------------------------------------------------------------
# JAX data helpers
# ---------------------------------------------------------------------------

def _jax_split_random(images, masks, num_clients, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    per = len(images) // num_clients
    result = []
    for i in range(num_clients):
        s = i * per
        e = s + per if i < num_clients - 1 else len(images)
        result.append((images[idx[s:e]], masks[idx[s:e]]))
    return result


def _jax_flood_labels(masks):
    return (masks.mean(axis=(1, 2, 3)) > 0.05).astype(int)


def _jax_split_dirichlet(images, masks, num_clients, alpha, seed=42):
    rng = np.random.default_rng(seed)
    labels = _jax_flood_labels(masks)
    client_indices = [[] for _ in range(num_clients)]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(alpha * np.ones(num_clients))
        splits = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, splits)):
            client_indices[cid].extend(chunk.tolist())
    sizes = [len(x) for x in client_indices]
    print(f"Dirichlet (alpha={alpha}) client sizes: {sizes}")
    return [(images[idxs], masks[idxs]) for idxs in client_indices]


def _jax_client_data_iter(images, masks, batch_size, rng_key):
    """Simple infinite iterator yielding {'image': ..., 'mask': ...} batches."""
    import jax
    while True:
        rng_key, sub = jax.random.split(rng_key)
        idx = np.array(jax.random.permutation(sub, len(images)))
        imgs_s = images[idx]
        msks_s = masks[idx]
        for i in range(0, len(imgs_s), batch_size):
            bi = imgs_s[i:i + batch_size]
            bm = msks_s[i:i + batch_size]
            if len(bi) < batch_size:
                pad = batch_size - len(bi)
                bi = np.concatenate([bi, bi[-pad:]], axis=0)
                bm = np.concatenate([bm, bm[-pad:]], axis=0)
            yield {'image': bi, 'mask': bm}


# ---------------------------------------------------------------------------
# JAX centralized
# ---------------------------------------------------------------------------

def run_centralized_jax(args):
    import jax
    import jax.numpy as jnp
    from flax import jax_utils
    from flax.training import checkpoints
    from utilsJAX import (load_dataset, create_data_iterator,
                           train_step, evaluate_model,
                           save_training_curves, visualize_results)
    from results_tracker import save_results_to_excel

    base_dir         = "data/flood_optical"
    train_images_dir = os.path.join(base_dir, "Training", "images")
    train_masks_dir  = os.path.join(base_dir, "Training", "labels")
    test_images_dir  = os.path.join(base_dir, "Testing", "images")
    test_masks_dir   = os.path.join(base_dir, "Testing", "labels")

    print("=" * 60)
    print("Ablation (JAX): Parameter-Matched Classical UNet (Centralized)")
    print("=" * 60)

    if not os.path.exists(train_images_dir):
        print(f"Training images not found: {train_images_dir}"); return

    matched_config, c_params, q_params = _find_matched_jax_config()
    config = matched_config.copy_and_resolve_references()

    # Add training hyperparameters
    import ml_collections
    config_dict = ml_collections.ConfigDict(config)
    config_dict.num_train_steps = 100
    config_dict.log_every = 10
    config_dict.eval_every = 10
    config = config_dict

    output_dir = f"results/ablation_centralized_jax_classical_matched_seed{config.seed}"
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    eval_log = os.path.join(logs_dir, 'evaluation_scores.txt')

    print(f"Matched base_channels={config.base_channels}  "
          f"({c_params:,} params vs {q_params:,} quantum params)")

    print("Loading training data...")
    train_images, train_masks = load_dataset(train_images_dir, train_masks_dir, config)
    split_idx = int(0.8 * len(train_images))
    val_images, val_masks     = train_images[split_idx:], train_masks[split_idx:]
    train_images, train_masks = train_images[:split_idx], train_masks[:split_idx]
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    rng = jax.random.PRNGKey(int(config.seed))
    state = _create_jax_classical_state(rng, config)

    # Try to resume from checkpoint
    start_step = 0
    try:
        restored = checkpoints.restore_checkpoint(ckpt_dir, state)
        if restored is not None and hasattr(restored, 'step'):
            state = restored
            start_step = int(state.step)
            print(f"Checkpoint loaded, resuming from step {start_step}")
        else:
            print("No valid checkpoint, starting from scratch")
    except Exception as e:
        print(f"Checkpoint load failed ({e}), starting from scratch")

    if start_step >= config.num_train_steps:
        print(f"Training already complete ({start_step}/{config.num_train_steps})")
    else:
        pmap_state = jax_utils.replicate(state)
        data_iter  = create_data_iterator(train_images, train_masks, config)
        p_train_step = jax.pmap(train_step, axis_name='batch')

        remaining = config.num_train_steps - start_step
        print(f"Training {remaining} remaining steps...")

        train_metrics, val_metrics = [], []

        for step_off in range(remaining):
            current_step = start_step + step_off
            batch = next(data_iter)
            pmap_state, metrics = p_train_step(pmap_state, batch)

            if (current_step + 1) % config.log_every == 0:
                m = jax.tree_util.tree_map(lambda x: float(x.mean()), metrics)
                train_metrics.append({'step': current_step + 1, **m})
                print(f"Step {current_step + 1}: Loss={m['loss']:.4f} Acc={m['accuracy']:.4f}")

            if (current_step + 1) % config.eval_every == 0:
                cur_params = jax.tree_util.tree_map(lambda x: x[0], pmap_state.params)
                state = state.replace(params=cur_params, step=current_step + 1)
                vm = {'step': current_step + 1,
                      **evaluate_model(state, val_images, val_masks, config)}
                val_metrics.append(vm)
                print(f"Val Step {current_step + 1}: Loss={vm['loss']:.4f} "
                      f"Acc={vm['accuracy']:.4f} AUC={vm['auc']:.4f} "
                      f"IoU={vm['iou']:.4f} Dice={vm['dice']:.4f}")
                with open(eval_log, 'a') as f:
                    f.write(f"Step {current_step + 1}: Loss={vm['loss']:.4f} "
                            f"Accuracy={vm['accuracy']:.4f} AUC={vm['auc']:.4f} "
                            f"IoU={vm['iou']:.4f} Dice={vm['dice']:.4f}\n")
                checkpoints.save_checkpoint(ckpt_dir, state, current_step + 1, keep=3)

            if (current_step + 1) % 50 == 0 and val_metrics:
                save_training_curves(val_metrics, output_dir,
                                     title_prefix="Ablation Centralized JAX Classical ")

        # Recover final single-device state
        final_params = jax.tree_util.tree_map(lambda x: x[0], pmap_state.params)
        state = state.replace(params=final_params, step=config.num_train_steps)

        if val_metrics:
            save_training_curves(val_metrics, output_dir,
                                 title_prefix="Ablation Centralized JAX Classical ")

    # Test
    test_result = None
    if os.path.exists(test_images_dir):
        test_images, test_masks = load_dataset(test_images_dir, test_masks_dir, config)
        test_result = evaluate_model(state, test_images, test_masks, config)
        print(f"Test: Loss={test_result['loss']:.4f} Acc={test_result['accuracy']:.4f} "
              f"AUC={test_result['auc']:.4f} IoU={test_result['iou']:.4f} "
              f"Dice={test_result['dice']:.4f}")
        with open(eval_log, 'a') as f:
            f.write(f"Final Test: Loss={test_result['loss']:.4f} "
                    f"Accuracy={test_result['accuracy']:.4f} AUC={test_result['auc']:.4f} "
                    f"IoU={test_result['iou']:.4f} Dice={test_result['dice']:.4f}\n")
        visualize_results(state, test_images, test_masks, config, output_dir)

    _write_summary_jax(output_dir, q_params, config.base_channels, c_params,
                       val_metrics if 'val_metrics' in dir() else [], test_result,
                       mode='centralized')
    save_results_to_excel(
        demo_name="Ablation Centralized JAX Classical (matched)",
        framework="jax", training="centralized",
        variant=f"classical_matched_{config.base_channels}ch",
        val_metrics=val_metrics[-1] if ('val_metrics' in dir() and val_metrics) else None,
        test_metrics=test_result, config=config,
    )
    print(f"\nAblation centralized JAX complete. Results in: {output_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# JAX federated
# ---------------------------------------------------------------------------

def _jax_params_to_numpy(params):
    import jax
    return [np.array(p) for p in jax.tree_util.tree_flatten(params)[0]]


def _numpy_to_jax_params(numpy_params, template_params):
    import jax
    import jax.numpy as jnp
    leaves, treedef = jax.tree_util.tree_flatten(template_params)
    new_leaves = [jnp.array(np_p) for np_p in numpy_params]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


class _JAXClassicalFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient for the parameter-matched classical JAX model."""

    def __init__(self, client_id, client_images, client_masks, config, template_state):
        import jax
        self.client_id = client_id
        self.client_images = client_images
        self.client_masks  = client_masks
        self.config        = config
        self.template_state = template_state
        self.local_state    = None
        client_seed = int(config.seed) + client_id
        self.client_key = jax.random.PRNGKey(client_seed)
        print(f"Client {client_id} initialized with {len(client_images)} samples")

    def get_parameters(self, config):
        params = (self.local_state.params if self.local_state is not None
                  else self.template_state.params)
        return _jax_params_to_numpy(params)

    def fit(self, parameters, config):
        import jax
        from utilsJAX import train_step as jax_train_step
        jax_params = _numpy_to_jax_params(parameters, self.template_state.params)
        self.local_state = self.template_state.replace(params=jax_params)

        batch_size = int(self.config.batch_size)
        local_epochs = int(self.config.local_epochs)
        n_per_epoch = max(1, len(self.client_images) // batch_size)

        data_iter = _jax_client_data_iter(
            self.client_images, self.client_masks, batch_size, self.client_key)

        total_loss = total_acc = 0.0
        for _ in range(local_epochs):
            for _ in range(n_per_epoch):
                batch = next(data_iter)
                self.local_state, m = jax_train_step(self.local_state, batch)
                total_loss += float(m['loss'])
                total_acc  += float(m['accuracy'])

        n_total = local_epochs * n_per_epoch
        avg_loss = total_loss / n_total
        avg_acc  = total_acc  / n_total
        print(f"Client {self.client_id}: Loss={avg_loss:.4f} Acc={avg_acc:.4f}")
        return (_jax_params_to_numpy(self.local_state.params), len(self.client_images),
                {"loss": float(avg_loss), "accuracy": float(avg_acc)})

    def evaluate(self, parameters, config):
        from utilsJAX import evaluate_model as jax_evaluate
        jax_params = _numpy_to_jax_params(parameters, self.template_state.params)
        eval_state = self.template_state.replace(params=jax_params)
        metrics = jax_evaluate(eval_state, self.client_images, self.client_masks, self.config)
        return (float(metrics['loss']), len(self.client_images),
                {"accuracy": float(metrics['accuracy']), "auc": float(metrics['auc'])})


class _JAXAblationFedAvg:
    """FedAvg strategy that evaluates the global classical JAX model each round."""

    def __init__(self, val_images, val_masks, config, template_state, output_dir, **kwargs):
        import flwr as fl
        from flwr.server.strategy import FedAvg
        self._base = FedAvg(**kwargs)
        self.val_images = val_images
        self.val_masks  = val_masks
        self.config     = config
        self.template_state = template_state
        self.output_dir = output_dir
        self.round_metrics = []
        self.best_auc = 0.0
        self._current_parameters = None
        logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        self.eval_log = os.path.join(logs_dir, 'evaluation_scores.txt')

    def initialize_parameters(self, client_manager):
        return self._base.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self._base.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self._base.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        agg = self._base.aggregate_fit(server_round, results, failures)
        if agg is not None:
            self._current_parameters = agg[0]
        return agg

    def aggregate_evaluate(self, server_round, results, failures):
        from flwr.common import parameters_to_ndarrays
        from utilsJAX import evaluate_model as jax_evaluate
        agg = self._base.aggregate_evaluate(server_round, results, failures)
        if self.val_images is not None and self._current_parameters is not None:
            np_params = parameters_to_ndarrays(self._current_parameters)
            jax_params = _numpy_to_jax_params(np_params, self.template_state.params)
            eval_state = self.template_state.replace(params=jax_params)
            metrics = jax_evaluate(eval_state, self.val_images, self.val_masks, self.config)
            metrics['round'] = server_round
            if metrics['auc'] > self.best_auc:
                self.best_auc = metrics['auc']
            self.round_metrics.append(metrics)
            print(f"Global Round {server_round}: Loss={metrics['loss']:.4f} "
                  f"Acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f} "
                  f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f}")
            with open(self.eval_log, 'a') as f:
                f.write(f"Round {server_round}: Loss={metrics['loss']:.4f} "
                        f"Accuracy={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f} "
                        f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f}\n")
        return agg

    def evaluate(self, server_round, parameters):
        return self._base.evaluate(server_round, parameters)


def run_federated_jax(args):
    import jax
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from utilsJAX import (load_dataset, evaluate_model as jax_evaluate,
                           save_training_curves, visualize_results)
    from results_tracker import save_results_to_excel

    suffix = f"_dirichlet{args.alpha}" if args.partition == 'dirichlet' else "_random"

    print("=" * 60)
    print("Ablation (JAX): Parameter-Matched Classical UNet (Federated)")
    print(f"Partition: {args.partition}" +
          (f"  alpha={args.alpha}" if args.partition == 'dirichlet' else ""))
    print("=" * 60)

    base_dir         = "data/flood_optical"
    train_images_dir = os.path.join(base_dir, "Training", "images")
    train_masks_dir  = os.path.join(base_dir, "Training", "labels")
    test_images_dir  = os.path.join(base_dir, "Testing", "images")
    test_masks_dir   = os.path.join(base_dir, "Testing", "labels")

    if not os.path.exists(train_images_dir):
        print(f"Training images not found: {train_images_dir}"); return

    matched_config, c_params, q_params = _find_matched_jax_config()

    import ml_collections
    config = ml_collections.ConfigDict(matched_config)
    config.num_clients       = 4
    config.local_epochs      = 4
    config.fed_rounds        = 25
    config.clients_per_round = 4
    config.log_every         = 5
    config.eval_every        = 5
    config.partition         = args.partition
    config.dirichlet_alpha   = args.alpha

    output_dir = f"results/ablation_federated_jax_classical_matched{suffix}_seed{config.seed}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Matched base_channels={config.base_channels}  "
          f"({c_params:,} params vs {q_params:,} quantum params)")

    print("Loading training data...")
    train_images, train_masks = load_dataset(train_images_dir, train_masks_dir, config)

    print(f"Splitting among {config.num_clients} clients ({config.partition})...")
    if config.partition == 'dirichlet':
        client_data = _jax_split_dirichlet(
            train_images, train_masks, int(config.num_clients),
            float(config.dirichlet_alpha), int(config.seed))
    else:
        client_data = _jax_split_random(
            train_images, train_masks, int(config.num_clients), int(config.seed))

    # Carve validation from last client
    val_split  = max(1, int(0.2 * len(client_data[-1][0])))
    val_images = client_data[-1][0][-val_split:]
    val_masks  = client_data[-1][1][-val_split:]
    client_data[-1] = (client_data[-1][0][:-val_split], client_data[-1][1][:-val_split])
    print(f"Validation set: {len(val_images)} samples")

    rng = jax.random.PRNGKey(int(config.seed))
    template_state = _create_jax_classical_state(rng, config)

    strategy = _JAXAblationFedAvg(
        val_images=val_images, val_masks=val_masks, config=config,
        template_state=template_state, output_dir=output_dir,
        fraction_fit=int(config.clients_per_round) / int(config.num_clients),
        fraction_evaluate=1.0,
        min_fit_clients=int(config.clients_per_round),
        min_evaluate_clients=1,
        min_available_clients=int(config.num_clients),
        initial_parameters=ndarrays_to_parameters(_jax_params_to_numpy(template_state.params)),
    )

    def client_fn(cid):
        cid = int(cid)
        imgs, msks = client_data[cid]
        return _JAXClassicalFlowerClient(cid, imgs, msks, config, template_state).to_client()

    print(f"\nStarting Flower federated training for {config.fed_rounds} rounds...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(config.num_clients),
        config=fl.server.ServerConfig(num_rounds=int(config.fed_rounds)),
        strategy=strategy,
        client_resources={"num_cpus": 1.0, "num_gpus": 0.2},
    )

    print(f"\nFederated training complete. Best AUC: {strategy.best_auc:.4f}")

    if strategy.round_metrics:
        save_training_curves(strategy.round_metrics, output_dir,
                             prefix="ablation_fed_",
                             title_prefix="Ablation Federated JAX Classical ")

    test_result = None
    if strategy._current_parameters is not None and os.path.exists(test_images_dir):
        np_params = parameters_to_ndarrays(strategy._current_parameters)
        jax_params = _numpy_to_jax_params(np_params, template_state.params)
        final_state = template_state.replace(params=jax_params)

        test_images, test_masks = load_dataset(test_images_dir, test_masks_dir, config)
        test_result = jax_evaluate(final_state, test_images, test_masks, config)
        print(f"Test: Loss={test_result['loss']:.4f} Acc={test_result['accuracy']:.4f} "
              f"AUC={test_result['auc']:.4f} IoU={test_result['iou']:.4f} "
              f"Dice={test_result['dice']:.4f}")
        with open(strategy.eval_log, 'a') as f:
            f.write(f"Final Test: Loss={test_result['loss']:.4f} "
                    f"Accuracy={test_result['accuracy']:.4f} AUC={test_result['auc']:.4f} "
                    f"IoU={test_result['iou']:.4f} Dice={test_result['dice']:.4f}\n")
        visualize_results(final_state, test_images, test_masks, config, output_dir,
                          title_prefix="Ablation Fed JAX ", filename_prefix="ablation_fed_jax_")

    _write_summary_jax(output_dir, q_params, config.base_channels, c_params,
                       strategy.round_metrics, test_result, mode='federated',
                       extra_lines=[
                           f"Partitioning: {config.partition}" +
                           (f", alpha={config.dirichlet_alpha}" if config.partition == 'dirichlet' else ""),
                           f"Clients: {config.num_clients}, Rounds: {config.fed_rounds}, "
                           f"Local epochs: {config.local_epochs}",
                       ])
    save_results_to_excel(
        demo_name="Ablation Federated JAX Classical (matched)",
        framework="jax", training="federated",
        variant=f"classical_matched_{config.base_channels}ch",
        val_metrics=strategy.round_metrics[-1] if strategy.round_metrics else None,
        test_metrics=test_result, config=config,
    )
    print(f"\nAblation federated JAX complete. Results in: {output_dir}")
    print("=" * 60)


def _write_summary_jax(output_dir, q_params, base_channels, c_params,
                        val_metrics, test_result, mode='centralized', extra_lines=None):
    summary_path = os.path.join(output_dir, "results", "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(f"Ablation Study (JAX) — Parameter-Matched Classical UNet ({mode.title()})\n")
        f.write("=" * 60 + "\n\n")
        if extra_lines:
            for line in extra_lines:
                f.write(line + "\n")
        f.write(f"Quantum model params:    {q_params:,}\n")
        f.write(f"Classical base_channels: {base_channels}\n")
        f.write(f"Classical model params:  {c_params:,}\n\n")
        if val_metrics:
            v = val_metrics[-1]
            f.write("Final Validation:\n")
            f.write(f"  Loss: {v['loss']:.4f}\n  Accuracy: {v['accuracy']:.4f}\n")
            f.write(f"  AUC: {v['auc']:.4f}\n  IoU: {v['iou']:.4f}\n  Dice: {v['dice']:.4f}\n\n")
        if test_result:
            f.write("Test Results:\n")
            f.write(f"  Loss: {test_result['loss']:.4f}\n  Accuracy: {test_result['accuracy']:.4f}\n")
            f.write(f"  AUC: {test_result['auc']:.4f}\n  IoU: {test_result['iou']:.4f}\n")
            f.write(f"  Dice: {test_result['dice']:.4f}\n")


# ===========================================================================
#  Entry point
# ===========================================================================

def main():
    args = parse_args()
    dispatch = {
        ('torch', 'centralized'): run_centralized_torch,
        ('torch', 'federated'):   run_federated_torch,
        ('jax',   'centralized'): run_centralized_jax,
        ('jax',   'federated'):   run_federated_jax,
    }
    dispatch[(args.framework, args.mode)](args)


if __name__ == "__main__":
    main()

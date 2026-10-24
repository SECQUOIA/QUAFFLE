#!/usr/bin/env python3
"""Shared training plumbing for the Torch demos and ablation_studies.py.

Keeps the parts that must be identical across the quantum, classical and
ablation runs in one place:

  * seeding (python, numpy, torch, cuda) and seeded DataLoaders
  * the global held-out validation split, carved from the training pool
    before any client partitioning, stratified on the flood label
  * the label-skew Dirichlet partition with a deterministic
    minimum-samples-per-client guard
  * per (seed, client, round) shuffle seeds for federated clients
  * model construction that forwards optional kwargs (seed, q_ansatz,
    q_layers) only when the model class accepts them
  * common CLI flags and the failed-run exit path

Every function only depends on torch / numpy so it can be imported inside
Ray workers.
"""
import argparse
import inspect
import json
import os
import random
import socket
import sys
import time
from datetime import datetime

import numpy as np

VAL_FRACTION = 0.15
FLOOD_LABEL_THRESHOLD = 0.05
DEFAULT_MIN_CLIENT_SAMPLES = 16
DEFAULT_CKPT_EVERY = 5
FED_CKPT_NAME = 'fed_resume.pt'

# Quantum circuit simulator (q_backend) of the QVUNet bottleneck.
Q_BACKEND_CHOICES = ('pennylane', 'pennylane_pergroup', 'torch')
DEFAULT_Q_BACKEND = 'pennylane'
Q_SIM_DEVICE_CHOICES = ('cpu', 'model')
DEFAULT_Q_SIM_DEVICE = 'cpu'
# Runs written before the q_backend option was recorded have no q_backend
# record. The model always used the torch statevector simulator then.
LEGACY_Q_BACKEND = 'torch'


DATASET_FOLDER_NAMES = {
    'optical': 'flood_optical',
    'sar': 'flood_SAR',
    'combined': 'flood_combined',
    'flood_optical': 'flood_optical',
    'flood_SAR': 'flood_SAR',
    'flood_combined': 'flood_combined',
}


def resolve_data_dirs(dataset, data_root=None):
    """Return (base_dir, train_images_dir, train_masks_dir, test_images_dir, test_masks_dir)
    under `data_root` (default: 'data')."""
    folder = DATASET_FOLDER_NAMES.get(dataset, dataset)
    root = data_root if data_root is not None else os.environ.get('QUAFFLE_DATA_ROOT', 'data')
    cand = os.path.join(root, folder)
    if not os.path.exists(cand) and os.path.exists(os.path.join(root, 'Training', 'images')):
        base_dir = root
    else:
        base_dir = cand
    train_images_dir = os.path.join(base_dir, 'Training', 'images')
    train_masks_dir = os.path.join(base_dir, 'Training', 'labels')
    test_images_dir = os.path.join(base_dir, 'Testing', 'images')
    test_masks_dir = os.path.join(base_dir, 'Testing', 'labels')
    return base_dir, train_images_dir, train_masks_dir, test_images_dir, test_masks_dir


def resolve_output_dir(default_rel_path, config=None, args=None):
    """Resolve run output directory, honoring --results-dir or QUAFFLE_RESULTS_DIR."""
    results_root = None
    if args is not None and getattr(args, 'results_dir', None):
        results_root = args.results_dir
    elif config is not None and config.get('results_dir'):
        results_root = config['results_dir']
    elif os.environ.get('QUAFFLE_RESULTS_DIR'):
        results_root = os.environ['QUAFFLE_RESULTS_DIR']
    if not results_root:
        return default_rel_path
    norm = os.path.normpath(default_rel_path)
    parts = norm.split(os.sep)
    if parts and parts[0] == 'results':
        rel = os.sep.join(parts[1:]) if len(parts) > 1 else ''
        return os.path.join(results_root, rel) if rel else results_root
    return os.path.join(results_root, os.path.basename(norm))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(parser, quantum=True, federated=False):
    """Add the flags shared by every Torch entry point.

    Defaults select the current behaviour. Every flag is passed on to the
    data and model code only when that code accepts it (signature checks).
    """
    parser.add_argument('--data-root', type=str, default=None,
                        help='Root directory containing flood_<dataset>/ folders and optional splits/ '
                             '(default: data)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Root directory for run output folders (default: results)')
    parser.add_argument('--base-channels', type=int, default=None,
                        help='Override model base_channels / init_dim (default: script config)')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Override input image spatial size (default: script config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override training and evaluation batch size (default: script config)')
    parser.add_argument('--eval-every', type=int, default=None,
                        help='Override validation and checkpoint interval (default: script config)')
    parser.add_argument('--no-save-samples', action='store_true',
                        help='Skip saving sample prediction visualizations at the end of the run')
    parser.add_argument('--legacy-data', action='store_true',
                        help='Use the pre-fix data loader behaviour (if the loader supports it)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Debug: cap the training pool and the test set at N samples')
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION,
                        help='Fraction of the training pool held out as the global, '
                             'stratified validation set (default: 0.15)')
    if quantum:
        parser.add_argument('--q-ansatz', choices=['reupload', 'legacy'], default='reupload',
                            help='Quantum bottleneck ansatz (default: reupload)')
        parser.add_argument('--q-layers', type=int, default=None,
                            help='Number of variational layers in the quantum bottleneck '
                                 '(default: model default)')
        parser.add_argument('--quantum-knockout-eval', action='store_true',
                            help="Also evaluate the test set with the quantum block in "
                                 "mode 'zero' and 'random' and write both to summary.txt")
        parser.add_argument('--q-backend', choices=list(Q_BACKEND_CHOICES),
                            default=DEFAULT_Q_BACKEND,
                            help="Quantum circuit simulator (default: pennylane, one broadcast "
                                 "default.qubit QNode call per block). 'pennylane_pergroup' is "
                                 "the old per-group PennyLane loop, 'torch' is the exact torch "
                                 "statevector cross-check.")
        parser.add_argument('--q-sim-device', choices=list(Q_SIM_DEVICE_CHOICES),
                            default=DEFAULT_Q_SIM_DEVICE,
                            help='Where the PennyLane circuit runs (default: cpu)')
    if federated:
        parser.add_argument('--rounds', type=int, default=None,
                            help='Override the number of federated rounds (debug)')
        parser.add_argument('--local-epochs', type=int, default=None,
                            help='Override local epochs per round (debug)')
        parser.add_argument('--num-clients', type=int, default=None,
                            help='Override the number of federated clients (debug)')
        parser.add_argument('--clients-per-round', type=int, default=None,
                            help='Override the number of clients sampled per federated round (debug)')
    else:
        parser.add_argument('--epochs', type=int, default=None,
                            help='Override the number of training epochs/steps (debug)')
    if federated:
        parser.add_argument('--min-client-samples', type=int, default=DEFAULT_MIN_CLIENT_SAMPLES,
                            help='Label-skew Dirichlet guard: minimum training samples per '
                                 'client (default: 16)')
        parser.add_argument('--ckpt-every', type=int, default=DEFAULT_CKPT_EVERY,
                            help='Write the resumable federated checkpoint every N rounds '
                                 f'(default: {DEFAULT_CKPT_EVERY}; the last round is always saved)')
        parser.add_argument('--no-resume', action='store_true',
                            help='Ignore an existing resumable federated checkpoint and start '
                                 'from round 0 (the old checkpoint is moved aside)')
        # Debug: simulate a crash right after the checkpoint of round K has
        # been written. Also settable via QUAFFLE_DEBUG_STOP_AFTER_ROUND.
        parser.add_argument('--debug-stop-after-round', type=int, default=None,
                            help=argparse.SUPPRESS)
    # Debug: simulate an interrupted centralized job right after the
    # checkpoint of epoch/step K has been written.
    parser.add_argument('--debug-stop-after-epoch', type=int, default=None,
                        help=argparse.SUPPRESS)
    return parser


def _compatible_group_count(base_channels, current_groups):
    g = max(1, min(int(current_groups), int(base_channels)))
    while g > 1 and int(base_channels) % g != 0:
        g -= 1
    return g


def apply_common_args(config, args):
    """Copy the common CLI flags into the config dict."""
    config['legacy_data'] = bool(getattr(args, 'legacy_data', False))
    config['max_samples'] = getattr(args, 'max_samples', None)
    config['val_fraction'] = float(getattr(args, 'val_fraction', VAL_FRACTION))
    if getattr(args, 'data_root', None) is not None:
        config['data_root'] = args.data_root
    elif os.environ.get('QUAFFLE_DATA_ROOT') and not config.get('data_root'):
        config['data_root'] = os.environ['QUAFFLE_DATA_ROOT']
    else:
        config.setdefault('data_root', None)
    if getattr(args, 'results_dir', None) is not None:
        config['results_dir'] = args.results_dir
    if getattr(args, 'base_channels', None) is not None:
        bc = int(args.base_channels)
        config['base_channels'] = bc
        config['init_dim'] = bc
        if 'resnet_block_groups' in config:
            config['resnet_block_groups'] = _compatible_group_count(bc, config['resnet_block_groups'])
        if 'quantum_channels' in config and getattr(args, 'quantum_channels', None) is None:
            mults = config.get('dim_mults', (1, 2, 4, 8))
            mid_dim = bc * int(mults[-1]) if mults else bc * 8
            if int(config['quantum_channels']) > mid_dim:
                config['quantum_channels'] = max(4, (mid_dim // 4) * 4)
    if getattr(args, 'image_size', None) is not None:
        config['image_size'] = int(args.image_size)
    if getattr(args, 'batch_size', None) is not None:
        config['batch_size'] = int(args.batch_size)
    if getattr(args, 'no_save_samples', False):
        config['save_samples'] = False
    if hasattr(args, 'q_ansatz'):
        config['q_ansatz'] = args.q_ansatz
    if getattr(args, 'q_layers', None) is not None:
        config['q_layers'] = args.q_layers
    if hasattr(args, 'quantum_knockout_eval'):
        config['quantum_knockout_eval'] = bool(args.quantum_knockout_eval)
    if hasattr(args, 'q_backend'):
        config['q_backend'] = args.q_backend
        config['q_sim_device'] = getattr(args, 'q_sim_device', DEFAULT_Q_SIM_DEVICE)
    if hasattr(args, 'min_client_samples'):
        config['min_client_samples'] = int(args.min_client_samples)
    if getattr(args, 'epochs', None) is not None:
        ep = int(args.epochs)
        for key in ('num_epochs', 'num_train_steps'):
            if key in config:
                config[key] = ep
        if getattr(args, 'eval_every', None) is None and 'eval_every' in config and ep > 0:
            config['eval_every'] = min(int(config['eval_every']), ep)
        if 'log_every' in config and ep > 0:
            config['log_every'] = min(int(config['log_every']), ep)
    if getattr(args, 'eval_every', None) is not None:
        config['eval_every'] = max(1, int(args.eval_every))
    if getattr(args, 'rounds', None) is not None:
        config['fed_rounds'] = int(args.rounds)
    if getattr(args, 'local_epochs', None) is not None:
        config['local_epochs'] = int(args.local_epochs)
    if getattr(args, 'num_clients', None) is not None:
        nc = int(args.num_clients)
        config['num_clients'] = nc
        if getattr(args, 'clients_per_round', None) is None and 'clients_per_round' in config:
            config['clients_per_round'] = min(int(config['clients_per_round']), nc)
    if getattr(args, 'clients_per_round', None) is not None:
        config['clients_per_round'] = int(args.clients_per_round)
    if hasattr(args, 'ckpt_every'):
        config['ckpt_every'] = max(1, int(args.ckpt_every))
        config['resume'] = not bool(getattr(args, 'no_resume', False))
        stop = getattr(args, 'debug_stop_after_round', None)
        if stop is None and os.environ.get('QUAFFLE_DEBUG_STOP_AFTER_ROUND'):
            stop = int(os.environ['QUAFFLE_DEBUG_STOP_AFTER_ROUND'])
        config['debug_stop_after_round'] = stop
    if hasattr(args, 'debug_stop_after_epoch'):
        stop_ep = getattr(args, 'debug_stop_after_epoch', None)
        if stop_ep is None and os.environ.get('QUAFFLE_DEBUG_STOP_AFTER_EPOCH'):
            stop_ep = int(os.environ['QUAFFLE_DEBUG_STOP_AFTER_EPOCH'])
        config['debug_stop_after_epoch'] = stop_ep
    return config


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed, deterministic=True):
    """Seed python, numpy, torch and cuda once at the start of a run."""
    import torch
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(*parts):
    """Deterministic 32-bit seed from integer parts (seed, client, round, ...)."""
    ss = np.random.SeedSequence([int(p) for p in parts])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _seed_worker(worker_id):
    import torch
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, seed, num_workers=0):
    """DataLoader with an explicit seeded generator (and worker seeding)."""
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(int(seed))
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                  generator=g)
    if num_workers > 0:
        kwargs['worker_init_fn'] = _seed_worker
    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

_OPTIONAL_MODEL_KWARGS = ('seed', 'q_ansatz', 'q_layers')


def build_model(model_cls, config):
    """Instantiate model_cls(config, **optional) where optional only holds the
    kwargs (seed, q_ansatz, q_layers) that the constructor actually accepts."""
    try:
        params = inspect.signature(model_cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs = {}
    for name in _OPTIONAL_MODEL_KWARGS:
        if name in params and config.get(name) is not None:
            kwargs[name] = config[name]
    return model_cls(config, **kwargs)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _supported_kwargs(fn, config):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    kwargs = {}
    legacy = bool(config.get('legacy_data', False))
    for name in ('legacy_data', 'legacy'):
        if name in params:
            kwargs[name] = legacy
            break
    if 'max_samples' in params and config.get('max_samples') is not None:
        kwargs['max_samples'] = int(config['max_samples'])
    if 'sample_seed' in params and config.get('seed') is not None:
        kwargs['sample_seed'] = int(config['seed'])
    if 'data_root' in params and config.get('data_root') is not None:
        kwargs['data_root'] = config['data_root']
    if 'stats_path' in params and config.get('stats_path') is not None:
        kwargs['stats_path'] = config['stats_path']
    if 'seed' in params and config.get('seed') is not None:
        kwargs['seed'] = int(config['seed'])
    if 'val_fraction' in params and config.get('val_fraction') is not None:
        kwargs['val_fraction'] = float(config['val_fraction'])
    if 'fit_only' in params and config.get('fit_only_norm') is not None:
        kwargs['fit_only'] = bool(config['fit_only_norm'])
    return kwargs


def load_dataset(images_dir, masks_dir, config):
    """FloodSegmentationDataset with --legacy-data, --max-samples, --data-root,
    and fit-only normalization parameters forwarded if supported."""
    from utilsTorch import FloodSegmentationDataset
    kwargs = _supported_kwargs(FloodSegmentationDataset.__init__, config)
    if config.get('legacy_data') and not any(k in kwargs for k in ('legacy', 'legacy_data')):
        print("WARNING: --legacy-data requested but this FloodSegmentationDataset "
              "has no legacy switch, the current loader is used as is.")
    ds = FloodSegmentationDataset(images_dir, masks_dir, config['image_size'], **kwargs)
    if getattr(ds, 'norm_provenance', None) is not None:
        config['_resolved_norm_provenance'] = dict(ds.norm_provenance)
    return ds


def canonical_order(dataset):
    """Index order that does not depend on filesystem listing order when the
    dataset exposes per-sample paths, else the dataset's own order."""
    for attr in ('names', 'image_paths', 'paths', 'img_paths', 'files'):
        paths = getattr(dataset, attr, None)
        if paths is not None and len(paths) == len(dataset):
            return sorted(range(len(dataset)), key=lambda i: os.path.basename(str(paths[i])))
    return list(range(len(dataset)))


def cap_samples(dataset, max_samples, seed):
    """Debug subset: a seeded random subset of at most max_samples items."""
    from torch.utils.data import Subset
    if not max_samples or max_samples >= len(dataset):
        return dataset
    order = canonical_order(dataset)
    rng = np.random.default_rng(derive_seed(seed, 7))
    keep = sorted(rng.permutation(order)[:max_samples].tolist())
    print(f"--max-samples: using {len(keep)} of {len(dataset)} samples")
    return Subset(dataset, keep)


def flood_labels(dataset, threshold=FLOOD_LABEL_THRESHOLD):
    """1 if the mask flood fraction exceeds threshold, else 0."""
    labels = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        labels.append(1 if float(mask.float().mean()) > threshold else 0)
    return np.array(labels, dtype=np.int64)


def split_train_val(n_or_dataset, val_fraction=VAL_FRACTION, seed=42, labels=None):
    """Deterministic stratified train/val index split.

    Accepts either an integer sample count `n` or a Dataset/sequence `dataset`.
    Returns `(train_indices, val_indices)`.
    """
    if isinstance(n_or_dataset, (int, np.integer)):
        n = int(n_or_dataset)
        order = np.arange(n, dtype=np.int64)
        if labels is None:
            labels = np.zeros(n, dtype=np.int64)
        else:
            labels = np.asarray(labels, dtype=np.int64)
    else:
        n = len(n_or_dataset)
        order = np.array(canonical_order(n_or_dataset), dtype=np.int64)
        if labels is None:
            if (
                hasattr(n_or_dataset, '__getitem__')
                and n > 0
                and isinstance(n_or_dataset[0], tuple)
                and len(n_or_dataset[0]) == 2
            ):
                labels = flood_labels(n_or_dataset)
            else:
                labels = np.zeros(n, dtype=np.int64)
        else:
            labels = np.asarray(labels, dtype=np.int64)

    if n <= 1 or val_fraction is None or float(val_fraction) <= 0.0:
        return sorted(order.tolist()), []

    rng = np.random.default_rng(derive_seed(seed, 1))
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = order[labels[order] == c]
        idx = rng.permutation(idx)
        n_val = int(round(float(val_fraction) * len(idx)))
        if len(idx) > 1:
            n_val = min(max(n_val, 1), len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def holdout_validation(dataset, seed, val_fraction=VAL_FRACTION, labels=None):
    """Global held-out validation split carved from the training pool BEFORE
    any client partitioning.

    Stratified on the binarised flood label and determined only by
    (dataset contents order, seed, val_fraction), so the quantum, classical
    and ablation runs with the same seed see exactly the same validation set
    in centralized, random and label-skew Dirichlet modes.

    Returns (train_indices, val_indices, labels) as positions in `dataset`.
    """
    if labels is None:
        labels = flood_labels(dataset)
    train_idx, val_idx = split_train_val(
        dataset, val_fraction=val_fraction, seed=seed, labels=labels
    )
    pos = sum(labels[i] for i in val_idx)
    print(f"Global held-out validation: {len(val_idx)} samples "
          f"({pos} flood-dominant, {len(val_idx) - pos} other), "
          f"training pool: {len(train_idx)} samples (seed={seed}, fraction={val_fraction})")
    return train_idx, val_idx, labels


def split_random(pool_indices, num_clients, seed):
    """IID split of pool_indices into num_clients near-equal shards."""
    rng = np.random.default_rng(derive_seed(seed, 2))
    perm = rng.permutation(np.asarray(pool_indices, dtype=np.int64))
    shards = np.array_split(perm, num_clients)
    sizes = [len(s) for s in shards]
    print(f"Random IID client sizes: {sizes}")
    return [sorted(s.tolist()) for s in shards]


def split_label_skew_dirichlet(pool_indices, labels, num_clients, alpha, seed,
                               min_samples=DEFAULT_MIN_CLIENT_SAMPLES):
    """Label-skew Dirichlet partition over the binarised flood label.

    For each label class the class indices are split across clients with
    proportions drawn from Dir(alpha). This is label skew only (no feature
    or geographic skew). Guard: every client receives at least min_samples
    training samples. If a draw violates the guard, samples are moved one at
    a time from the currently largest client (taking that client's most
    represented class, deterministic order) until the guard holds. The whole
    procedure is a function of (pool, labels, seed, alpha, min_samples).
    """
    pool = np.asarray(pool_indices, dtype=np.int64)
    if num_clients * min_samples > len(pool):
        raise ValueError(f"min_samples={min_samples} x {num_clients} clients exceeds the "
                         f"training pool ({len(pool)} samples)")
    rng = np.random.default_rng(derive_seed(seed, 3))
    client = [[] for _ in range(num_clients)]
    pool_labels = labels[pool]
    for c in np.unique(pool_labels):
        idx = rng.permutation(pool[pool_labels == c])
        props = rng.dirichlet(alpha * np.ones(num_clients))
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, cuts)):
            client[cid].extend(chunk.tolist())
    raw_sizes = [len(c) for c in client]

    moved = 0
    for cid in range(num_clients):
        while len(client[cid]) < min_samples:
            donor = max(range(num_clients), key=lambda k: (len(client[k]), -k))
            donor_labels = labels[np.asarray(client[donor])]
            major = int(np.bincount(donor_labels, minlength=2).argmax())
            cand = sorted(i for i in client[donor] if labels[i] == major)
            take = cand[-1]
            client[donor].remove(take)
            client[cid].append(take)
            moved += 1

    sizes = [len(c) for c in client]
    flood = [int(sum(labels[i] for i in c)) for c in client]
    print(f"Label-skew Dirichlet (alpha={alpha}) raw client sizes: {raw_sizes}")
    if moved:
        print(f"  min-samples guard ({min_samples}) moved {moved} samples")
    print(f"Label-skew Dirichlet final client sizes: {sizes}, "
          f"flood-dominant per client: {flood}")
    return [sorted(c) for c in client]


def build_federated_split(dataset, config):
    """Global validation holdout followed by client partitioning.

    Returns (client_index_lists, val_indices, info_lines) with indices into
    `dataset`. info_lines are human readable lines for summary.txt.
    """
    seed = config['seed']
    train_idx, val_idx, labels = holdout_validation(
        dataset, seed, config.get('val_fraction', VAL_FRACTION))
    n = config['num_clients']
    if config.get('partition') == 'dirichlet':
        clients = split_label_skew_dirichlet(
            train_idx, labels, n, config['dirichlet_alpha'], seed,
            config.get('min_client_samples', DEFAULT_MIN_CLIENT_SAMPLES))
        part = f"label-skew Dirichlet (alpha={config['dirichlet_alpha']}, " \
               f"min {config.get('min_client_samples', DEFAULT_MIN_CLIENT_SAMPLES)} per client)"
    else:
        clients = split_random(train_idx, n, seed)
        part = "random IID"
    info = [
        f"Partitioning: {part}",
        f"Client training sizes: {[len(c) for c in clients]}",
        f"Global validation set: {len(val_idx)} samples "
        f"({config.get('val_fraction', VAL_FRACTION)} of the training pool, stratified)",
    ]
    return clients, val_idx, info


# ---------------------------------------------------------------------------
# Federated helpers
# ---------------------------------------------------------------------------

def fit_config_fn(server_round):
    """on_fit_config_fn: lets clients reseed their shuffle every round."""
    return {"server_round": int(server_round)}


def client_round_seed(seed, client_id, server_round):
    return derive_seed(seed, 1000 + int(client_id), int(server_round))


def seed_client_round(seed, client_id, fit_config=None):
    """Seed python, numpy, torch, cuda and CuDNN deterministically for one
    federated client fit() call, and return the derived round_seed."""
    import torch
    server_round = int(fit_config.get('server_round', 0)) if fit_config else 0
    if server_round == 0:
        print(f"Client {client_id}: no server_round in fit config, "
              f"shuffle seed falls back to round 0")
    round_seed = client_round_seed(seed, client_id, server_round)
    random.seed(round_seed)
    np.random.seed(round_seed)
    torch.manual_seed(round_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(round_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return round_seed


# ---------------------------------------------------------------------------
# Quantum knockout evaluation
# ---------------------------------------------------------------------------

def _call_set_mode(obj, mode, knockout_seed):
    try:
        params = inspect.signature(obj.set_quantum_mode).parameters
    except (TypeError, ValueError):
        params = {}
    if 'knockout_seed' in params:
        obj.set_quantum_mode(mode, knockout_seed=knockout_seed)
    else:
        obj.set_quantum_mode(mode)


def set_quantum_mode(model, mode, knockout_seed=0):
    """Switch the quantum bottleneck between 'on', 'zero' and 'random'.

    Uses set_quantum_mode() on the wrapper or inner model when it has one,
    else on any submodule, falling back to a `quantum_mode` attribute.
    Returns True if the switch was applied somewhere.
    """
    for obj in (model, getattr(model, '_model', None)):
        if obj is not None and hasattr(obj, 'set_quantum_mode'):
            _call_set_mode(obj, mode, knockout_seed)
            return True
    inner = getattr(model, '_model', None) or model
    applied = False
    if hasattr(inner, 'modules'):
        for m in inner.modules():
            if m is inner:
                continue
            if hasattr(m, 'set_quantum_mode'):
                _call_set_mode(m, mode, knockout_seed)
                applied = True
            elif hasattr(m, 'quantum_mode'):
                m.quantum_mode = mode
                applied = True
    return applied


def knockout_eval(model, loader, device, evaluate_fn, knockout_seed=0):
    """Evaluate with the quantum block in 'zero' and 'random' mode, then
    restore 'on'.

    Returns {mode: metrics or None}. None means the model build has no
    knockout switch.
    """
    out = {}
    for mode in ('zero', 'random'):
        if not set_quantum_mode(model, mode, knockout_seed):
            print("Quantum knockout eval: this model build has no quantum mode switch, skipped")
            return {'zero': None, 'random': None}
        out[mode] = evaluate_fn(model, loader, device)
        print(f"Knockout '{mode}' test: IoU={out[mode]['iou']:.4f} Dice={out[mode]['dice']:.4f} "
              f"AUC={out[mode]['auc']:.4f}")
    set_quantum_mode(model, 'on', knockout_seed)
    return out


def format_knockout(knock):
    if not knock:
        return ""
    lines = []
    for mode in ('zero', 'random'):
        m = knock.get(mode)
        if m is None:
            lines.append(f"Quantum Knockout ({mode}): unavailable in this model build\n")
            continue
        # Deliberately not "Test Results:" so analyze_results.py never parses
        # a knockout block as the run's headline test metrics.
        lines.append(f"Quantum Knockout ({mode}) test metrics:\n")
        for k, name in (('loss', 'Loss'), ('accuracy', 'Accuracy'), ('auc', 'AUC'),
                        ('iou', 'IoU'), ('dice', 'Dice')):
            lines.append(f"  {name}: {m[k]:.4f}\n")
    return "".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------

# Bumped whenever the split, seeding or data pipeline changes so that a rerun
# never silently resumes a checkpoint produced by older code in the same
# results directory.
RUN_FORMAT = "split-v3"


def checkpoint_compatible(ckpt):
    fmt = ckpt.get('run_format') if isinstance(ckpt, dict) else None
    if fmt != RUN_FORMAT:
        print(f"Ignoring existing checkpoint (run_format={fmt!r}, expected {RUN_FORMAT!r}), "
              f"starting from scratch")
        return False
    return True


def normalization_provenance(config):
    """Structured normalization provenance recorded in checkpoint fingerprints."""
    legacy = bool(config.get('legacy_data', False))
    if legacy:
        return {
            'mode': 'legacy_per_image_p2_p98',
            'fit_only': False,
            'seed': None,
            'val_fraction': None,
            'stats_path': None,
        }
    fit_only_cfg = config.get('fit_only_norm')
    eff_seed = int(config['seed'])
    val_frac = float(config.get('val_fraction', VAL_FRACTION))
    img_sz = int(config['image_size']) if config.get('image_size') is not None else 128
    max_s = int(config['max_samples']) if config.get('max_samples') is not None else None
    stats_path = (
        os.path.abspath(config['stats_path'])
        if config.get('stats_path') is not None
        else None
    )

    resolved = config.get('_resolved_norm_provenance')
    if (
        isinstance(resolved, dict)
        and resolved.get('seed') == eff_seed
        and (resolved.get('val_fraction') is None or abs(float(resolved['val_fraction']) - val_frac) < 1e-6)
        and (resolved.get('image_size') in (None, img_sz))
        and resolved.get('max_samples') == max_s
    ):
        out = dict(resolved)
        out['stats_path'] = stats_path
        return out

    if config.get('data_root') is not None or stats_path is not None or fit_only_cfg is False:
        try:
            import flood_data
            ds_folder = DATASET_FOLDERS.get(config.get('dataset', 'optical'), 'flood_optical')
            st = flood_data.load_stats(
                ds_folder,
                stats_path=stats_path,
                seed=eff_seed if fit_only_cfg is not False else None,
                val_fraction=val_frac if fit_only_cfg is not False else None,
                fit_only=fit_only_cfg,
                data_root=config.get('data_root'),
                image_size=img_sz,
                max_samples=max_s,
                sample_seed=eff_seed,
            )
            out = flood_data.stats_provenance(st)
            out['stats_path'] = stats_path
            return out
        except Exception:
            if fit_only_cfg is True and (config.get('data_root') is not None or stats_path is not None):
                return {
                    'mode': 'unresolved_fit_only',
                    'fit_only': False,
                    'seed': eff_seed,
                    'val_fraction': val_frac,
                    'image_size': img_sz,
                    'max_samples': max_s,
                    'stats_path': stats_path,
                }

    fit_only = True if fit_only_cfg is None else bool(fit_only_cfg)
    return {
        'mode': 'fit_only_train_split' if (fit_only and val_frac > 0.0) else 'full_train_split',
        'fit_only': bool(fit_only and val_frac > 0.0),
        'seed': eff_seed,
        'val_fraction': val_frac,
        'image_size': img_sz,
        'max_samples': max_s,
        'stats_path': stats_path,
    }


def maybe_stop_after_centralized_step(config, current, total, label='epoch'):
    """Simulate an interrupted centralized job right after saving checkpoint `current`."""
    stop = config.get('debug_stop_after_epoch')
    if stop is not None and int(current) == int(stop) and int(current) < int(total):
        print(f"DEBUG: stopping after {label} {current} (--debug-stop-after-epoch / "
              f"QUAFFLE_DEBUG_STOP_AFTER_EPOCH), simulating an interrupted job")
        sys.stdout.flush()
        raise SystemExit(75)


def centralized_fingerprint(config, model_type):
    """Settings that must match for a centralized checkpoint to be resumed."""
    ansatz, layers = effective_q_variant(config)
    quantum = model_type in ('quantum', 'ablation')
    epochs_or_steps = config.get('num_epochs', config.get('num_train_steps'))
    base_channels = config.get('base_channels')
    init_dim = config.get('init_dim', base_channels)
    out_dim = config.get('out_dim', 2)
    dim_mults = [int(m) for m in config.get('dim_mults', (1, 2, 4, 8))]
    resnet_block_groups = int(config.get('resnet_block_groups', 8))
    data_root = config.get('data_root')
    return {
        'model_type': model_type,
        'dataset': config.get('dataset', 'optical'),
        'data_root': os.path.abspath(data_root) if data_root is not None else None,
        'seed': int(config['seed']),
        'q_ansatz': ansatz if quantum else None,
        'q_layers': layers if quantum else None,
        'quantum_channels': (int(config['quantum_channels'])
                             if quantum and config.get('quantum_channels') is not None else None),
        'quantum_backend': (config.get('quantum_backend', 'pennylane')
                            if model_type == 'quantum' else None),
        'ptlayer_config': (config.get('ptlayer_config', None)
                           if model_type == 'quantum' else None),
        'q_backend': (config.get('q_backend', DEFAULT_Q_BACKEND)
                      if model_type == 'quantum' else None),
        'q_sim_device': (config.get('q_sim_device', DEFAULT_Q_SIM_DEVICE)
                         if model_type == 'quantum' else None),
        'q_share_blocks': bool(config.get('q_share_blocks', True)) if quantum else None,
        'epochs_or_steps': int(epochs_or_steps) if epochs_or_steps is not None else None,
        'batch_size': int(config['batch_size']),
        'learning_rate': float(config['learning_rate']),
        'base_channels': int(base_channels) if base_channels is not None else None,
        'init_dim': int(init_dim) if init_dim is not None else None,
        'out_dim': int(out_dim) if out_dim is not None else None,
        'dim_mults': dim_mults,
        'resnet_block_groups': resnet_block_groups,
        'image_size': int(config['image_size']) if config.get('image_size') is not None else None,
        'max_samples': config.get('max_samples'),
        'val_fraction': float(config.get('val_fraction', VAL_FRACTION)),
        'legacy_data': bool(config.get('legacy_data', False)),
        'norm_provenance': normalization_provenance(config),
    }


def checkpoint_matches_centralized(ckpt, config, model_type, path=None):
    """Validate run_format, q_backend (if quantum), and centralized_fingerprint.
    Moves incompatible checkpoints aside when `path` is provided."""
    if not isinstance(ckpt, dict):
        if path and os.path.exists(path):
            move_aside(path, 'incompatible')
        return False
    if not checkpoint_compatible(ckpt):
        if path and os.path.exists(path):
            move_aside(path, 'incompatible')
        return False
    if model_type == 'quantum' and not checkpoint_matches_q_backend(ckpt, config):
        if path and os.path.exists(path):
            move_aside(path, 'incompatible')
        return False
    want = centralized_fingerprint(config, model_type)
    saved = ckpt.get('fingerprint')
    if not isinstance(saved, dict):
        print("Rejecting centralized checkpoint (missing fingerprint from older code), "
              "starting from scratch")
        if path and os.path.exists(path):
            move_aside(path, 'incompatible')
        return False
    diff = {k: (saved.get(k), v) for k, v in want.items() if saved.get(k) != v}
    if diff:
        print("Rejecting centralized checkpoint, config differs (saved, current): "
              + ", ".join(f"{k}={a!r}->{b!r}" for k, (a, b) in sorted(diff.items())))
        if path and os.path.exists(path):
            move_aside(path, 'incompatible')
        return False
    return True


def checkpoint_q_backend(ckpt):
    """q_backend recorded in a checkpoint dict. A checkpoint without the key
    was written before the q_backend option was recorded, when the model
    always used the torch statevector simulator."""
    if not isinstance(ckpt, dict):
        return None
    return ckpt.get('q_backend', LEGACY_Q_BACKEND)


def checkpoint_matches_q_backend(ckpt, config):
    """True if a quantum checkpoint was produced with the circuit simulator
    of this run (config['q_backend']). A missing key counts as 'torch'."""
    want = config.get('q_backend', DEFAULT_Q_BACKEND)
    have = checkpoint_q_backend(ckpt)
    if have != want:
        print(f"Ignoring existing checkpoint (q_backend={have!r}"
              f"{' (not recorded, older code)' if 'q_backend' not in ckpt else ''}, "
              f"this run uses {want!r}), starting from scratch")
        return False
    return True


# ---------------------------------------------------------------------------
# Run naming (variant tag)
# ---------------------------------------------------------------------------

_DEFAULT_Q_LAYERS = 2


def default_q_layers():
    """Default q_layers of the quantum block constructor (falls back to 2)."""
    try:
        from utilsTorch.unetTorch import QuantumBlock
        p = inspect.signature(QuantumBlock.__init__).parameters.get('q_layers')
        if p is not None and p.default is not inspect.Parameter.empty:
            return int(p.default)
    except Exception:
        pass
    return _DEFAULT_Q_LAYERS


def effective_q_variant(config):
    """(q_ansatz, q_layers) actually used by the quantum bottleneck.

    q_layers is None for the legacy ansatz (it has a single fixed layer)."""
    ansatz = config.get('q_ansatz') or 'reupload'
    if ansatz == 'legacy':
        return ansatz, None
    layers = config.get('q_layers')
    return ansatz, int(layers) if layers is not None else default_q_layers()


def variant_tag(config):
    """Directory suffix that separates quantum ansatz variants, e.g.
    '_reupload_L2' or '_legacy'. Used for quantum runs and the ablation
    (whose classical bottleneck is matched to the quantum variant)."""
    ansatz, layers = effective_q_variant(config)
    if ansatz == 'legacy':
        return '_legacy'
    return f'_{ansatz}_L{layers}'


_BACKEND_TAGS = {'pennylane': '', 'torch': '_qtorch', 'pennylane_pergroup': '_qpergroup'}


def backend_tag(config):
    """Directory suffix for the quantum circuit simulator, appended after
    variant_tag in the quantum demos only. '' for the default 'pennylane' so
    PennyLane runs keep the canonical directory names; '_qtorch' and
    '_qpergroup' keep cross-check runs (and their checkpoints) apart."""
    qb = config.get('q_backend', DEFAULT_Q_BACKEND)
    if qb not in _BACKEND_TAGS:
        raise ValueError(f"Unknown q_backend {qb!r}, expected one of {sorted(_BACKEND_TAGS)}")
    return _BACKEND_TAGS[qb]


def quantum_simulator_info(model, config=None):
    """Dict describing the circuit simulator the model was actually built with.

    Uses model.quantum_simulator_info() (QVUNetSegmentation) or
    simulator_info() of the first QuantumBlock when the model code provides
    them. Older model code without those methods falls back to the q_backend
    attribute of the instantiated quantum block, then to config['q_backend'],
    and marks the dict with 'source' so the summary says where it came from.
    """
    config = config or {}
    for obj in (model, getattr(model, '_model', None)):
        fn = getattr(obj, 'quantum_simulator_info', None) if obj is not None else None
        if callable(fn):
            try:
                info = fn()
                if info:
                    return dict(info)
            except Exception as e:
                print(f"WARNING: quantum_simulator_info() failed ({type(e).__name__}: {e})")
    inner = getattr(model, 'model', None) if model is not None else None
    inner = inner if inner is not None else model
    if inner is not None and hasattr(inner, 'modules'):
        for m in inner.modules():
            fn = getattr(m, 'simulator_info', None)
            if callable(fn):
                try:
                    info = fn()
                    if info:
                        return dict(info)
                except Exception as e:
                    print(f"WARNING: simulator_info() failed ({type(e).__name__}: {e})")
                break
        for m in inner.modules():
            if hasattr(m, 'q_backend') and getattr(m, 'backend', None) is not None:
                info = {'q_backend': getattr(m, 'q_backend'), 'family': getattr(m, 'backend'),
                        'source': 'quantum block attribute (model code has no simulator_info)'}
                if info['q_backend'] == 'torch':
                    info.update(device='exact torch statevector simulator',
                                diff_method='torch autograd', sim_device='model')
                return info
    return {'q_backend': config.get('q_backend', DEFAULT_Q_BACKEND),
            'family': config.get('quantum_backend'),
            'source': 'config (model code does not report its simulator)'}


def format_quantum_simulator(info):
    """One-line description of quantum_simulator_info() for summary.txt."""
    if not info:
        return 'unknown'
    qb = info.get('q_backend')
    if qb is None:
        return f"none (quantum family: {info.get('family')})"
    if qb == 'torch':
        text = 'exact torch statevector simulator (q_backend=torch)'
    elif info.get('device'):
        parts = [f"PennyLane {info['device']}"]
        if info.get('diff_method'):
            parts.append(str(info['diff_method']))
        if info.get('interface'):
            parts.append(f"{info['interface']} interface")
        if info.get('broadcast'):
            parts.append(f"broadcast over {info['broadcast']}"
                         if info['broadcast'].startswith('batch') else str(info['broadcast']))
        if info.get('sim_device'):
            parts.append(f"on {info['sim_device']}")
        text = ', '.join(parts) + f" (q_backend={qb}"
        if info.get('pennylane_version'):
            text += f", PennyLane {info['pennylane_version']}"
        text += ')'
    else:
        text = f"q_backend={qb}"
    if info.get('source'):
        text += f" [from {info['source']}]"
    return text


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _tmp_name(path):
    return f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"


def atomic_torch_save(obj, path):
    """torch.save to a temp file in the same directory, then os.replace, so a
    kill during the write never leaves a truncated checkpoint behind."""
    import torch
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = _tmp_name(path)
    try:
        with open(tmp, 'wb') as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(d)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, 'item'):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


def atomic_write_json(obj, path):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = _tmp_name(path)
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def move_aside(path, tag):
    """Rename path to path.<tag>-<timestamp> (kept for inspection)."""
    base = f"{path}.{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dst, n = base, 1
    while os.path.exists(dst):
        dst = f"{base}.{n}"
        n += 1
    try:
        os.replace(path, dst)
        print(f"Moved {path} aside to {dst}")
    except OSError as e:
        print(f"WARNING: could not move {path} aside: {e}")
    return dst


def safe_torch_load(path, map_location=None):
    """torch.load that returns None (and keeps the file aside) when the
    checkpoint is unreadable instead of crashing the run."""
    import torch
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"WARNING: checkpoint {path} is unreadable ({type(e).__name__}: {e}), "
              f"starting without it")
        move_aside(path, 'corrupt')
        return None


def rng_state(loader=None):
    """Python / numpy / torch (and loader generator) RNG state for checkpoints."""
    import torch
    st = {'python': random.getstate(), 'numpy': np.random.get_state(),
          'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        try:
            st['cuda'] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    gen = getattr(loader, 'generator', None) if loader is not None else None
    if gen is not None:
        st['loader'] = gen.get_state()
    return st


def restore_rng_state(st, loader=None):
    """Best effort inverse of rng_state()."""
    import torch
    if not st:
        return
    try:
        random.setstate(st['python'])
        np.random.set_state(st['numpy'])
        torch.set_rng_state(st['torch'].cpu() if hasattr(st['torch'], 'cpu') else st['torch'])
        if 'cuda' in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in st['cuda']])
        gen = getattr(loader, 'generator', None) if loader is not None else None
        if gen is not None and 'loader' in st:
            gen.set_state(st['loader'].cpu())
        print("Restored RNG and data loader state from checkpoint")
    except Exception as e:
        print(f"WARNING: could not restore RNG state from checkpoint ({e})")


# ---------------------------------------------------------------------------
# Federated checkpoint / resume
# ---------------------------------------------------------------------------

def run_fingerprint(config, model_type):
    """Settings that must match for a federated checkpoint to be resumed."""
    ansatz, layers = effective_q_variant(config)
    quantum = model_type in ('quantum', 'ablation')
    data_root = config.get('data_root')
    fp = {
        'model_type': model_type,
        'dataset': config.get('dataset', 'optical'),
        'data_root': os.path.abspath(data_root) if data_root is not None else None,
        'partition': config.get('partition', 'random'),
        'dirichlet_alpha': (float(config['dirichlet_alpha'])
                            if config.get('partition') == 'dirichlet' else None),
        'seed': int(config['seed']),
        'q_ansatz': ansatz if quantum else None,
        'q_layers': layers if quantum else None,
        'quantum_channels': config.get('quantum_channels') if quantum else None,
        'quantum_backend': config.get('quantum_backend') if quantum else None,
        # Circuit simulator: a checkpoint written with another simulator (or
        # before q_backend was recorded) is never resumed.
        'q_backend': (config.get('q_backend', DEFAULT_Q_BACKEND)
                      if model_type == 'quantum' else None),
        'fed_rounds': int(config['fed_rounds']),
        'local_epochs': int(config['local_epochs']),
        'num_clients': int(config['num_clients']),
        'clients_per_round': int(config.get('clients_per_round', config['num_clients'])),
        'batch_size': int(config['batch_size']),
        'learning_rate': float(config['learning_rate']),
        'base_channels': config.get('base_channels'),
        'image_size': int(config['image_size']) if config.get('image_size') is not None else None,
        'max_samples': config.get('max_samples'),
        'val_fraction': float(config.get('val_fraction', VAL_FRACTION)),
        'min_client_samples': config.get('min_client_samples'),
        'legacy_data': bool(config.get('legacy_data', False)),
        'norm_provenance': normalization_provenance(config),
    }
    return fp


def _run_inprocess_flower_simulation(*, client_fn, num_clients, config, strategy, **kwargs):
    """Run Flower's Server + SimpleClientManager in-process without spawning Ray actors.

    Exercises the exact same Flower Server.fit(), Strategy (configure_fit,
    aggregate_fit, configure_evaluate, aggregate_evaluate, evaluate),
    RoundOffsetMixin, client_fn(cid).to_client(), and checkpointing logic.
    """
    from flwr.common import DisconnectRes
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.server import Server

    class _InProcessClientProxy(ClientProxy):
        def __init__(self, cid):
            super().__init__(str(cid))
            self._cid = str(cid)

        def _client(self):
            c = client_fn(self._cid)
            return c.to_client() if hasattr(c, 'to_client') else c

        def get_properties(self, ins, timeout, group_id):
            return self._client().get_properties(ins)

        def get_parameters(self, ins, timeout, group_id):
            return self._client().get_parameters(ins)

        def fit(self, ins, timeout, group_id):
            return self._client().fit(ins)

        def evaluate(self, ins, timeout, group_id):
            return self._client().evaluate(ins)

        def reconnect(self, ins, timeout, group_id):
            return DisconnectRes(reason='UNKNOWN')

    cm = SimpleClientManager()
    for cid in range(int(num_clients)):
        cm.register(_InProcessClientProxy(str(cid)))
    server = Server(client_manager=cm, strategy=strategy)
    if hasattr(server, 'set_max_workers'):
        server.set_max_workers(1)
    num_rounds = int(getattr(config, 'num_rounds', 1))
    round_timeout = getattr(config, 'round_timeout', None)
    res = server.fit(num_rounds=num_rounds, timeout=round_timeout)
    return res[0] if isinstance(res, tuple) else res


def start_flower_simulation(**kwargs):
    """Run flwr.simulation.start_simulation, optionally using an in-process
    Flower Server (QUAFFLE_INPROCESS_FED=1) or keeping a single-worker Ray
    runtime initialized across calls (QUAFFLE_KEEP_RAY=1)."""
    if os.environ.get('QUAFFLE_INPROCESS_FED', '0').lower() in ('1', 'true', 'yes'):
        return _run_inprocess_flower_simulation(**kwargs)
    import flwr as fl
    os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')
    if os.environ.get('QUAFFLE_KEEP_RAY', '0').lower() in ('1', 'true', 'yes'):
        try:
            params = inspect.signature(fl.simulation.start_simulation).parameters
            if 'keep_initialised' in params:
                kwargs.setdefault('keep_initialised', True)
            if 'ray_init_args' in params:
                num_gpus = 1 if kwargs.get('client_resources', {}).get('num_gpus', 0) > 0 else 0
                kwargs.setdefault(
                    'ray_init_args',
                    {
                        'num_cpus': 1,
                        'num_gpus': num_gpus,
                        'include_dashboard': False,
                        'ignore_reinit_error': True,
                    },
                )
        except (TypeError, ValueError):
            pass
    return fl.simulation.start_simulation(**kwargs)


def fed_checkpoint_path(output_dir):
    return os.path.join(output_dir, FED_CKPT_NAME)


def load_fed_checkpoint(output_dir, fingerprint, resume=True):
    """Return a compatible resumable checkpoint dict or None.

    Incompatible (other run_format or fingerprint) or unreadable checkpoints
    are moved aside, never deleted, and the run starts from round 0."""
    path = fed_checkpoint_path(output_dir)
    if not os.path.exists(path):
        print(f"No resumable federated checkpoint at {path}, starting from round 0")
        return None
    if not resume:
        print("--no-resume given, ignoring the existing federated checkpoint")
        move_aside(path, 'ignored')
        return None
    ckpt = safe_torch_load(path, map_location='cpu')
    if ckpt is None:
        return None
    if not checkpoint_compatible(ckpt):
        move_aside(path, 'incompatible')
        return None
    saved = ckpt.get('fingerprint') or {}
    diff = {k: (saved.get(k), v) for k, v in fingerprint.items() if saved.get(k) != v}
    if diff:
        print("Rejecting federated checkpoint, config differs (saved, current): "
              + ", ".join(f"{k}={a!r}->{b!r}" for k, (a, b) in sorted(diff.items())))
        move_aside(path, 'incompatible')
        return None
    print(f"Resuming federated run from checkpoint: {ckpt['completed_round']}/"
          f"{fingerprint['fed_rounds']} rounds completed")
    return ckpt


def apply_fed_resume(strategy, ckpt):
    """Restore history into a strategy before start_simulation.

    Returns (initial_parameters or None, remaining_rounds)."""
    from flwr.common import ndarrays_to_parameters
    total = int(strategy.config['fed_rounds'])
    if ckpt is None:
        strategy.round_offset = 0
        strategy.resumed_from_round = 0
        return None, total
    done = int(ckpt['completed_round'])
    strategy.round_offset = done
    strategy.resumed_from_round = done
    strategy.round_metrics = list(ckpt.get('round_metrics', []))
    strategy.best_performance = float(ckpt.get('best_performance', 0.0))
    params = ndarrays_to_parameters(ckpt['parameters'])
    strategy._current_parameters = params
    log = getattr(strategy, 'eval_log_path', None) or getattr(strategy, 'eval_log', None)
    if log:
        with open(log, 'a') as f:
            f.write(f"Resumed from checkpoint after round {done}\n")
    return params, max(0, total - done)


def fed_round_end(strategy, server_round):
    """Call at the end of aggregate_evaluate (server_round already offset).

    Writes the resumable checkpoint every config['ckpt_every'] rounds and on
    the last round, unless a round has failed: then the last good checkpoint
    is kept so a rerun can retry from before the failure while this process
    still exits non-zero via fail_run."""
    cfg = strategy.config
    every = int(cfg.get('ckpt_every', DEFAULT_CKPT_EVERY))
    total = int(cfg['fed_rounds'])
    if strategy._current_parameters is None:
        return
    if strategy.failed_rounds:
        print(f"Round {server_round}: not writing a resumable checkpoint after a failed round")
    elif server_round % every == 0 or server_round == total:
        from flwr.common import parameters_to_ndarrays
        atomic_torch_save({
            'run_format': RUN_FORMAT,
            'fingerprint': strategy.fingerprint,
            'completed_round': int(server_round),
            'parameters': parameters_to_ndarrays(strategy._current_parameters),
            'round_metrics': strategy.round_metrics,
            'best_performance': strategy.best_performance,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }, fed_checkpoint_path(strategy.output_dir))
        print(f"Resumable checkpoint saved after round {server_round}/{total}")
    stop = cfg.get('debug_stop_after_round')
    if stop is not None and int(server_round) == int(stop) and server_round < total:
        print(f"DEBUG: stopping after round {server_round} (--debug-stop-after-round / "
              f"QUAFFLE_DEBUG_STOP_AFTER_ROUND), simulating an interrupted job")
        sys.stdout.flush()
        # SystemExit is not caught by Flower's `except Exception`, so the
        # simulation unwinds and the process exits with this code.
        raise SystemExit(75)


class RoundOffsetMixin:
    """Mixin for FedAvg subclasses: Flower always counts rounds from 1, so a
    resumed run shifts every server_round by round_offset. This keeps the
    on_fit_config_fn round (and therefore the per (seed, client, round)
    shuffle seed), logs and checkpoints numbered as in an uninterrupted run.
    aggregate_fit / aggregate_evaluate apply the offset in the subclass."""
    round_offset = 0

    def configure_fit(self, server_round, parameters, client_manager):
        return super().configure_fit(server_round + self.round_offset, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return super().configure_evaluate(server_round + self.round_offset, parameters,
                                          client_manager)

    def evaluate(self, server_round, parameters):
        r = server_round + self.round_offset if server_round > 0 else server_round
        return super().evaluate(r, parameters)


# ---------------------------------------------------------------------------
# Per-run metrics.json and tracker isolation
# ---------------------------------------------------------------------------

def _clean_metrics(m):
    if not m:
        return None
    return {k: _json_default(v) if not isinstance(v, (int, float, str, type(None))) else v
            for k, v in dict(m).items()}


def write_metrics_json(output_dir, *, script, training, model_type, config, val_metrics,
                       test_metrics, history=None, total_params=None, best_val_auc=None,
                       knockout=None, provenance=None, extra=None,
                       q_simulator=None):
    """Canonical machine-readable result: <output_dir>/results/metrics.json.

    Written right after summary.txt and before the Excel tracker so analysis
    never depends on the shared workbook."""
    ansatz, layers = effective_q_variant(config)
    quantum = model_type in ('quantum', 'ablation')
    payload = {
        'schema': 'quaffle-metrics-v1',
        'run_dir': os.path.basename(os.path.normpath(output_dir)),
        'script': script,
        'training': training,
        'model_type': model_type,
        'dataset': config.get('dataset', 'optical'),
        'partition': config.get('partition', 'centralized') if training == 'federated'
        else 'centralized',
        'dirichlet_alpha': config.get('dirichlet_alpha') if config.get('partition') == 'dirichlet'
        else None,
        'seed': int(config['seed']),
        'q_ansatz': ansatz if quantum else None,
        'q_layers': layers if quantum else None,
        'quantum_channels': config.get('quantum_channels') if quantum else None,
        # Circuit simulator (quantum runs only). The ablation has no circuit.
        'q_backend': (config.get('q_backend', DEFAULT_Q_BACKEND)
                      if model_type == 'quantum' else None),
        'q_simulator': q_simulator if model_type == 'quantum' else None,
        'total_params': total_params,
        'fed_rounds': config.get('fed_rounds') if training == 'federated' else None,
        'local_epochs': config.get('local_epochs') if training == 'federated' else None,
        'num_clients': config.get('num_clients') if training == 'federated' else None,
        'epochs_or_steps': (config.get('num_epochs') or config.get('num_train_steps'))
        if training != 'federated' else None,
        'max_samples': config.get('max_samples'),
        'val': _clean_metrics(val_metrics),
        'test': _clean_metrics(test_metrics),
        'best_val_auc': best_val_auc,
        'knockout': {k: _clean_metrics(v) for k, v in knockout.items()} if knockout else None,
        'history': [_clean_metrics(h) for h in (history or [])],
        'provenance': provenance,
        'written_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    if extra:
        payload.update(extra)
    path = os.path.join(output_dir, 'results', 'metrics.json')
    atomic_write_json(payload, path)
    print(f"Metrics JSON saved to: {path}")
    return path


def safe_tracker_call(fn, *args, **kwargs):
    """Run the shared Excel tracker. Any failure is logged as a warning and
    never propagates: the run's summary.txt and metrics.json are already on
    disk, so the process must still exit 0."""
    if os.environ.get('QUAFFLE_SKIP_EXCEL_TRACKER', '0').lower() in ('1', 'true', 'yes'):
        return True
    try:
        fn(*args, **kwargs)
        return True
    except Exception as e:
        import traceback
        print(f"WARNING: results tracker failed ({type(e).__name__}: {e}). "
              f"summary.txt and metrics.json are written, the run is still complete.")
        traceback.print_exc()
        sys.stdout.flush()
        return False


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def fail_run(output_dir, reason):
    """Record a failed run and exit non-zero so SLURM marks the job FAILED.

    No summary.txt and no tracker row are written for failed runs.
    """
    msg = f"RUN FAILED: {reason}"
    print("=" * 60)
    print(msg)
    print("No summary.txt or tracker row written.")
    print("=" * 60)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "FAILED.txt"), "a") as f:
                f.write(msg + "\n")
        except OSError:
            pass
    sys.stdout.flush()
    sys.exit(1)

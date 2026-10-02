#!/usr/bin/env python3
"""Shared training plumbing for the Torch demos and abalation_studies.py.

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
import inspect
import os
import random
import sys

import numpy as np

VAL_FRACTION = 0.15
FLOOD_LABEL_THRESHOLD = 0.05
DEFAULT_MIN_CLIENT_SAMPLES = 16


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(parser, quantum=True, federated=False):
    """Add the flags shared by every Torch entry point.

    Defaults select the new behaviour. Every flag is passed on to the data
    and model code only when that code accepts it (signature checks), so the
    scripts run both before and after the data and model branches merge.
    """
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
    if federated:
        parser.add_argument('--rounds', type=int, default=None,
                            help='Override the number of federated rounds (debug)')
        parser.add_argument('--local-epochs', type=int, default=None,
                            help='Override local epochs per round (debug)')
    else:
        parser.add_argument('--epochs', type=int, default=None,
                            help='Override the number of training epochs/steps (debug)')
    if federated:
        parser.add_argument('--min-client-samples', type=int, default=DEFAULT_MIN_CLIENT_SAMPLES,
                            help='Label-skew Dirichlet guard: minimum training samples per '
                                 'client (default: 16)')
    return parser


def apply_common_args(config, args):
    """Copy the common CLI flags into the config dict."""
    config['legacy_data'] = bool(getattr(args, 'legacy_data', False))
    config['max_samples'] = getattr(args, 'max_samples', None)
    config['val_fraction'] = float(getattr(args, 'val_fraction', VAL_FRACTION))
    if hasattr(args, 'q_ansatz'):
        config['q_ansatz'] = args.q_ansatz
    if getattr(args, 'q_layers', None) is not None:
        config['q_layers'] = args.q_layers
    if hasattr(args, 'quantum_knockout_eval'):
        config['quantum_knockout_eval'] = bool(args.quantum_knockout_eval)
    if hasattr(args, 'min_client_samples'):
        config['min_client_samples'] = int(args.min_client_samples)
    if getattr(args, 'epochs', None) is not None:
        for key in ('num_epochs', 'num_train_steps'):
            if key in config:
                config[key] = int(args.epochs)
    if getattr(args, 'rounds', None) is not None:
        config['fed_rounds'] = int(args.rounds)
    if getattr(args, 'local_epochs', None) is not None:
        config['local_epochs'] = int(args.local_epochs)
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
    return kwargs


def load_dataset(images_dir, masks_dir, config):
    """FloodSegmentationDataset with --legacy-data forwarded if supported."""
    from utilsTorch import FloodSegmentationDataset
    kwargs = _supported_kwargs(FloodSegmentationDataset.__init__, config)
    if config.get('legacy_data') and not kwargs:
        print("WARNING: --legacy-data requested but this FloodSegmentationDataset "
              "has no legacy switch, the current loader is used as is.")
    return FloodSegmentationDataset(images_dir, masks_dir, config['image_size'], **kwargs)


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
    order = np.array(canonical_order(dataset), dtype=np.int64)
    rng = np.random.default_rng(derive_seed(seed, 1))
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = order[labels[order] == c]
        idx = rng.permutation(idx)
        n_val = int(round(val_fraction * len(idx)))
        if len(idx) > 1:
            n_val = min(max(n_val, 1), len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
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

    Uses set_quantum_mode() on the wrapper or inner model (model branch API),
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
RUN_FORMAT = "split-v2"


def checkpoint_compatible(ckpt):
    fmt = ckpt.get('run_format') if isinstance(ckpt, dict) else None
    if fmt != RUN_FORMAT:
        print(f"Ignoring existing checkpoint (run_format={fmt!r}, expected {RUN_FORMAT!r}), "
              f"starting from scratch")
        return False
    return True


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

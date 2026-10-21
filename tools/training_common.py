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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(parser, quantum=True, federated=False):
    """Add the flags shared by every Torch entry point.

    Defaults select the current behaviour. Every flag is passed on to the
    data and model code only when that code accepts it (signature checks).
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
    if hasattr(args, 'q_backend'):
        config['q_backend'] = args.q_backend
        config['q_sim_device'] = getattr(args, 'q_sim_device', DEFAULT_Q_SIM_DEVICE)
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
    if hasattr(args, 'ckpt_every'):
        config['ckpt_every'] = max(1, int(args.ckpt_every))
        config['resume'] = not bool(getattr(args, 'no_resume', False))
        stop = getattr(args, 'debug_stop_after_round', None)
        if stop is None and os.environ.get('QUAFFLE_DEBUG_STOP_AFTER_ROUND'):
            stop = int(os.environ['QUAFFLE_DEBUG_STOP_AFTER_ROUND'])
        config['debug_stop_after_round'] = stop
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
RUN_FORMAT = "split-v2"


def checkpoint_compatible(ckpt):
    fmt = ckpt.get('run_format') if isinstance(ckpt, dict) else None
    if fmt != RUN_FORMAT:
        print(f"Ignoring existing checkpoint (run_format={fmt!r}, expected {RUN_FORMAT!r}), "
              f"starting from scratch")
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
    fp = {
        'model_type': model_type,
        'dataset': config.get('dataset', 'optical'),
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
        'max_samples': config.get('max_samples'),
        'val_fraction': float(config.get('val_fraction', VAL_FRACTION)),
        'min_client_samples': config.get('min_client_samples'),
        'legacy_data': bool(config.get('legacy_data', False)),
    }
    return fp


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

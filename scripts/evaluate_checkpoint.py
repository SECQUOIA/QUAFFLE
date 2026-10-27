#!/usr/bin/env python3
"""Evaluate a saved model on pooled and explicit sensor subsets without retraining."""

import argparse
import hashlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import training_common as tc
from tools.checkpoint_evaluation import load_checkpoint_model
from utilsTorch.utilsTorch import evaluate_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='JSON architecture defaults for older checkpoints')
    parser.add_argument('--model-type', choices=['classical', 'ablation', 'quantum'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    protected = [args.checkpoint, *([args.config] if args.config else [])]
    if any(args.out.resolve() == p.resolve() for p in protected):
        parser.error('--out must not overwrite a checkpoint or its configuration')
    model, config, kind = load_checkpoint_model(
        args.checkpoint, data_root=args.data_root, device=args.device,
        config_path=args.config, model_type=args.model_type,
    )
    _, _, _, images, masks = tc.resolve_data_dirs(config['dataset'], config['data_root'])
    dataset = tc.load_dataset(images, masks, config)
    loader = tc.make_loader(dataset, args.batch_size, False, int(config['seed']))
    metrics = evaluate_model(model, loader, args.device)
    payload = dict(
        checkpoint=args.checkpoint.name,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        model_type=kind, dataset=config['dataset'], seed=int(config['seed']),
        image_size=int(config['image_size']), n_test_tiles=len(dataset),
        test=metrics,
    )
    tc.atomic_write_json(payload, str(args.out))
    print(f'Wrote pooled and sensor-subset metrics to {args.out}')
    return payload


if __name__ == '__main__':
    main()

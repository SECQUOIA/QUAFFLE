#!/usr/bin/env python3
"""Compare saved segmentation checkpoints on fixed-rule test examples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from textwrap import fill

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import training_common as tc
from tools.checkpoint_evaluation import load_checkpoint_model

KINDS = ('classical', 'ablation', 'quantum')
LABELS = ('Classical U-Net', 'Classical ablation', 'QUAFFLE')
COLORS = {'TP': '#18a558', 'FP': '#e34aab', 'FN': '#00b9ef'}
PREPROCESSING_DEFAULTS = dict(image_size=128, dataset=None, seed=42, bands=None,
    legacy_data=False, fit_only_norm=True, val_fraction=0.15, max_samples=None,
    sample_seed=None, stats_path=None, manifest=None, combined_input_mode='shared3')
NORMALIZATION_FIELDS = ('mode', 'fit_only', 'seed', 'val_fraction', 'image_size',
    'max_samples', 'n_manifest_train_tiles', 'n_fit_train_tiles',
    'n_excluded_val_tiles', 'fit_names_sha256', 'stats_sha256', 'modality_fallbacks')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _saved_normalization(config):
    provenance = config.get('norm_provenance') or config.get('_resolved_norm_provenance')
    if provenance is None:
        return None
    if not isinstance(provenance, dict):
        raise ValueError('Saved normalization provenance must be an object')
    if str(provenance.get('mode', '')).startswith('unresolved_'):
        return None
    return {key: provenance[key] for key in NORMALIZATION_FIELDS if key in provenance}


def select_examples(records, min_flood_fraction=0.05):
    """Lower median and worst baseline IoU; dataset index breaks exact ties."""
    eligible = sorted((r for r in records if r['flood_fraction'] >= min_flood_fraction),
                      key=lambda r: (r['iou']['classical'], r['index']))
    if not eligible:
        return None
    return dict(worst=eligible[0]['index'], median=eligible[(len(eligible) - 1) // 2]['index'])


def pixel_iou(predicted, target):
    union = np.count_nonzero(predicted | target)
    return float(np.count_nonzero(predicted & target) / union) if union else 0.0


def _boundary(mask):
    padded = np.pad(mask, 1, constant_values=False)
    interior = mask.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    return mask & ~interior


def overlay_prediction(image, target, predicted):
    """White prediction fill with TP/FP/FN region boundaries on the input."""
    canvas = np.clip(image, 0, 1).copy()
    canvas[predicted] = 0.75 * canvas[predicted] + 0.25
    regions = {'TP': predicted & target, 'FP': predicted & ~target, 'FN': ~predicted & target}
    for name, region in regions.items():
        canvas[_boundary(region)] = matplotlib.colors.to_rgb(COLORS[name])
    return canvas


def evaluate_group(group, data_root, checkpoint_root, device, batch_size):
    models, configs, checkpoint_meta = {}, {}, {}
    for kind in KINDS:
        entry = group['checkpoints'][kind]
        entry = {'path': entry} if isinstance(entry, str) else entry
        path = (checkpoint_root / entry['path']).resolve()
        config_path = (checkpoint_root / entry['config']).resolve() if entry.get('config') else None
        models[kind], configs[kind], _ = load_checkpoint_model(
            path, data_root=data_root, device=device, config_path=config_path, model_type=kind)
        checkpoint_meta[kind] = dict(file=path.name, sha256=_sha256(path),
            config_sha256=_sha256(config_path) if config_path else None,
            total_params=sum(p.numel() for p in models[kind].parameters()))
    signatures = [{key: cfg.get(key, default) for key, default in PREPROCESSING_DEFAULTS.items()}
                  for cfg in configs.values()]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError('Compared checkpoints must have identical dataset, seed and preprocessing settings')
    saved_normalization = {kind: _saved_normalization(config) for kind, config in configs.items()}
    for key in NORMALIZATION_FIELDS:
        known = [value[key] for value in saved_normalization.values() if value is not None and key in value]
        if known and any(value != known[0] for value in known[1:]):
            raise ValueError('Saved normalization differs across checkpoints; use models trained with matching statistics')
    cfg = configs['classical']
    _, _, _, images, masks = tc.resolve_data_dirs(cfg['dataset'], str(data_root))
    dataset = tc.load_dataset(images, masks, cfg)
    actual_normalization = dataset.norm_provenance or {}
    for kind, identity in saved_normalization.items():
        if identity is not None and any(actual_normalization.get(key) != value for key, value in identity.items()):
            raise ValueError('Saved normalization differs from reloaded data; supply a matching config with the original stats_path')
        checkpoint_meta[kind]['saved_normalization'] = identity
        checkpoint_meta[kind]['normalization_verified'] = bool(identity and (
            identity.get('stats_sha256') or identity.get('mode') == 'legacy_per_image_p2_p98'))
    if not len(dataset):
        raise ValueError('The test dataset is empty')
    loader = tc.make_loader(dataset, batch_size, False, cfg['seed'])
    predictions = {kind: [] for kind in KINDS}
    with torch.no_grad():
        for images, _ in loader:
            for kind, model in models.items():
                logits = model(images.to(device))
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite model output: ' + kind)
                predictions[kind].extend(logits.argmax(dim=1).cpu().numpy().astype(bool))
    records, display, targets = [], [], []
    for index in range(len(dataset)):
        image, mask = dataset[index]
        target = mask.squeeze(0).numpy().astype(bool)
        sensor = dataset.modalities[index]
        # Six-channel combined inputs have one active sensor triplet per tile.
        channels = image[3:6] if image.shape[0] == 6 and sensor == 'sar' else image[:3]
        display.append(np.clip((channels.permute(1, 2, 0).numpy() + 1) / 2, 0, 1))
        targets.append(target)
        records.append(dict(index=index, name=dataset.names[index], modality=sensor,
            flood_fraction=float(target.mean()),
            iou={kind: pixel_iou(predictions[kind][index], target) for kind in KINDS}))
    return dict(id=group['id'], dataset=cfg['dataset'], seed=cfg['seed'],
                preprocessing=signatures[0], normalization=dataset.norm_provenance,
                checkpoints=checkpoint_meta, tiles=records,
                selected=select_examples(records)), display, targets, predictions


def render(groups, out_dir):
    figure, axes = plt.subplots(2 * len(groups), 5, figsize=(15, 5.4 * len(groups)), squeeze=False)
    for row, (meta, images, targets, predictions) in enumerate(groups):
        for offset, rule in enumerate(('median', 'worst')):
            panels = axes[2 * row + offset]
            if meta['selected'] is None:
                for panel in panels:
                    panel.text(0.5, 0.5, 'No test tile has ≥5% flood', ha='center')
                    panel.axis('off')
                continue
            index = meta['selected'][rule]
            tile = meta['tiles'][index]
            panels[0].imshow(images[index])
            title = fill(f"{meta['id']} {rule}: {tile['name']}", width=26)
            panels[0].set_title(title + f"\nInput ({tile['modality']})", fontsize=10)
            panels[1].imshow(targets[index], cmap='gray', vmin=0, vmax=1)
            panels[1].set_title(f"Ground truth\nFlood {100 * tile['flood_fraction']:.1f}%")
            for panel, kind, label in zip(panels[2:], KINDS, LABELS):
                panel.imshow(overlay_prediction(images[index], targets[index], predictions[kind][index]))
                panel.set_title(f"{label}\nIoU {tile['iou'][kind]:.3f}")
            for panel in panels:
                panel.axis('off')
    figure.legend(handles=[Patch(color=color, label=name) for name, color in COLORS.items()],
                  loc='lower center', ncol=3)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    for extension in ('png', 'pdf'):
        figure.savefig(out_dir / ('qualitative_median_worst_fp_fn.' + extension), dpi=180, bbox_inches='tight')
    plt.close(figure)
    figure, axes = plt.subplots(len(groups), 2, figsize=(9, 4 * len(groups)), squeeze=False)
    for row, (meta, _, _, _) in enumerate(groups):
        for panel, reference, label in zip(axes[row], KINDS[:2], LABELS[:2]):
            for sensor, marker in (('optical', 'o'), ('sar', '^')):
                tiles = [tile for tile in meta['tiles'] if tile['modality'] == sensor]
                if tiles:
                    panel.scatter([t['iou'][reference] for t in tiles], [t['iou']['quantum'] for t in tiles],
                                  s=20, alpha=0.6, marker=marker, label=sensor)
            panel.plot([0, 1], [0, 1], '--', color='gray')
            panel.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel=label + ' IoU',
                      ylabel='QUAFFLE IoU', title=fill(meta['id'] + ' — all test tiles', width=40), aspect='equal')
            panel.legend()
    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(out_dir / ('per_tile_iou_scatter.' + extension), dpi=180, bbox_inches='tight')
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, help='Relative checkpoint/config root; defaults to manifest directory')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    manifest = json.loads(args.manifest.read_text())
    entries = manifest['datasets']
    if not entries or len({entry['id'] for entry in entries}) != len(entries):
        parser.error('Manifest datasets require nonempty unique ids')
    checkpoint_root = args.checkpoint_dir or args.manifest.resolve().parent
    inputs = {args.manifest.resolve()}
    for group in entries:
        for kind in KINDS:
            entry = group['checkpoints'][kind]
            entry = {'path': entry} if isinstance(entry, str) else entry
            inputs.update((checkpoint_root / entry[key]).resolve() for key in ('path', 'config') if entry.get(key))
    outputs = {(args.out_dir / name).resolve() for name in (
        'qualitative_eval_meta.json', 'qualitative_median_worst_fp_fn.png',
        'qualitative_median_worst_fp_fn.pdf', 'per_tile_iou_scatter.png', 'per_tile_iou_scatter.pdf')}
    if inputs & outputs:
        parser.error('Output files cannot overwrite the manifest, checkpoints or configurations')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.12)
    groups = [evaluate_group(group, args.data_root, checkpoint_root, device, args.batch_size) for group in entries]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render(groups, args.out_dir)
    payload = dict(manifest_sha256=_sha256(args.manifest),
        selection_rule='Classical IoU; flood fraction >=0.05; lower median; ties by dataset index',
        empty_union_iou=0.0, datasets=[group[0] for group in groups])
    (args.out_dir / 'qualitative_eval_meta.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return payload


if __name__ == '__main__':
    main()

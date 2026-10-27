"""Render sensor-specific examples from the same combined-trained checkpoints."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE = Path('/tmp/quaffle-real-spatial-results-20261027')
BASE = SOURCE / 'benchmarks/real_spatial_local_20261027'
RUN = Path('/tmp/quaffle-real-spatial-runs-60ep-20261027')
DATA = Path('/tmp/quaffle-dfc24-real')
OUT = Path('/tmp/quaffle-real-spatial-sensor-figures-20261027')
sys.path.insert(0, str(SOURCE))
import torch
from misc.build_qualitative_error_figure import evaluate_group, render, select_examples

if OUT.exists():
    raise RuntimeError('Use a new output directory; preserve preceding attempts')
for name, expected in json.loads((BASE / 'artifact_sha256.json').read_text()).items():
    assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.cuda.set_per_process_memory_fraction(0.12)
OUT.mkdir()
audit = []
for mode in ('global', 'patch4x4'):
    manifest = json.loads((BASE / 'qualitative' / (mode + '_manifest.json')).read_text())
    original = json.loads((BASE / 'qualitative' / mode / 'qualitative_eval_meta.json').read_text())
    evaluated = []
    for entry, previous in zip(manifest['datasets'], original['datasets']):
        group = evaluate_group(entry, DATA, RUN, torch.device('cuda'), 8)
        assert group[0]['tiles'] == previous['tiles'], 'Per-tile results differ from the retained original figure'
        assert group[0]['checkpoints'] == previous['checkpoints']
        evaluated.append(group)
    for sensor, expected_count in (('optical', 71), ('sar', 134)):
        groups = []
        for meta, images, targets, predictions in evaluated:
            indices = [tile['index'] for tile in meta['tiles'] if tile['modality'] == sensor]
            assert len(indices) == expected_count
            subset = copy.deepcopy(meta)
            subset['id'] = sensor.upper() + '; ' + mode + '; seed ' + str(meta['seed'])
            subset['example_modality'] = sensor
            subset['tiles'] = [dict(meta['tiles'][index], index=j, dataset_index=index) for j, index in enumerate(indices)]
            subset['selected'] = select_examples(subset['tiles'])
            assert subset['selected'] is not None
            eligible = sorted((tile for tile in subset['tiles'] if tile['flood_fraction'] >= 0.05),
                              key=lambda tile: (tile['iou']['classical'], tile['dataset_index']))
            assert subset['tiles'][subset['selected']['worst']]['dataset_index'] == eligible[0]['dataset_index']
            assert subset['tiles'][subset['selected']['median']]['dataset_index'] == eligible[(len(eligible)-1)//2]['dataset_index']
            groups.append((subset, [images[i] for i in indices], [targets[i] for i in indices],
                           {kind: [values[i] for i in indices] for kind, values in predictions.items()}))
        directory = OUT / mode / sensor
        directory.mkdir(parents=True)
        render(groups, directory)
        payload = dict(selection_rule='Per-sensor classical lower median/worst IoU among >=5% flood tiles; original dataset index breaks ties',
            scope='Sensor test subsets of combined-trained checkpoints', empty_union_iou=0,
            datasets=[group[0] for group in groups])
        (directory / 'qualitative_eval_meta.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        audit.append(dict(mode=mode, sensor=sensor, groups=len(groups), tiles_per_group=expected_count,
                          original_prediction_parity=True, selection_checked=True))
        print('VERIFIED SENSOR FIGURES', mode, sensor, flush=True)
provenance = dict(source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
    source_hashes={name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in (
        'misc/build_qualitative_error_figure.py', 'utilsTorch/unetTorch.py', 'utilsTorch/utilsTorch.py',
        'tools/training_common.py', 'tools/checkpoint_evaluation.py', 'flood_data.py')},
    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), audit=audit,
    device='cuda', batch_size=8, memory_max='3G', memory_swap_max=0,
    note='Original pooled figures retained separately; every per-tile IoU and checkpoint fingerprint matched the original before subset selection')
(OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

"""Render every paired seed/mode with the budget-matched ordinary U-Net."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE = Path('/tmp/quaffle-real-spatial-results-20261027')
RUN = Path('/tmp/quaffle-real-spatial-runs-60ep-20261027')
DATA = Path('/tmp/quaffle-dfc24-real')
OUT = Path('/tmp/quaffle-real-spatial-figures-20261027')
sys.path.insert(0, str(SOURCE))
from misc.build_qualitative_error_figure import main

summary = json.loads((RUN / 'summary.json').read_text())
records = {(row['id'], row['seed']): row for row in summary['records']}
OUT.mkdir(exist_ok=True)
for mode in ('global', 'patch4x4'):
    groups = []
    for seed in summary['plan']['settings']['seeds']:
        mapping = dict(classical='unet_width7', ablation='mlp_' + mode, quantum='quantum_' + mode)
        groups.append(dict(id=mode + '; seed ' + str(seed) + '; width-7 U-Net',
            checkpoints={kind: records[name, seed]['checkpoint'] for kind, name in mapping.items()}))
    manifest = OUT / (mode + '_manifest.json')
    manifest.write_text(json.dumps(dict(datasets=groups), indent=2) + '\n')
    payload = main(['--manifest', str(manifest), '--checkpoint-dir', str(RUN),
        '--data-root', str(DATA), '--out-dir', str(OUT / mode), '--device', 'cuda', '--batch-size', '8'])
    for group in payload['datasets']:
        assert len(group['tiles']) == 205
        assert sum(tile['modality'] == 'optical' for tile in group['tiles']) == 71
        assert sum(tile['modality'] == 'sar' for tile in group['tiles']) == 134
        assert all(value['normalization_verified'] for value in group['checkpoints'].values())
        for value in group['checkpoints'].values():
            assert value['sha256'] in {
                hashlib.sha256((RUN / row['checkpoint']).read_bytes()).hexdigest()
                for row in summary['records']}
    print('VERIFIED FIGURES', mode, len(payload['datasets']), flush=True)
provenance = dict(source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
    figure_source_sha256=hashlib.sha256((SOURCE / 'misc/build_qualitative_error_figure.py').read_bytes()).hexdigest(),
    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    checkpoint_root=str(RUN), data_root=str(DATA), device='cuda', batch_size=8,
    classical_baseline='unet_width7; nearest ordinary U-Net count, +2.05% vs quantum',
    selection='All three seeds, both spatial modes; median/worst selected by fixed classical rule; no best-test-seed selection',
    all_groups_tiles=205, optical_tiles=71, sar_tiles=134, normalization_verified=True,
    note='Per-tile empty-union IoU is zero; pooled test IoU is calculated from pixel totals, not the average of tile IoUs')
(OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

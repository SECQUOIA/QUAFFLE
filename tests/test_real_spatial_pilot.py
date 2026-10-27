"""Selection is validation-only; the capped optimization partition stays paired."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_real_spatial_pilot as pilot


def test_validation_rate_selection_ignores_test_scores_and_ties_are_fixed():
    rows = [dict(id='model', learning_rate=rate, val={'iou': val}, test={'iou': test})
            for rate, val, test in [(0.003, 0.4, 0.99), (0.001, 0.5, 0), (0.0005, 0.5, 0.1)]]
    assert pilot.choose_rates(rows)['model']['learning_rate'] == 0.0005
    fit = pilot.selected_fit_indices(list(range(20)), 8, 42)
    assert len(fit) == len(set(fit)) == 8
    assert fit == pilot.selected_fit_indices(list(range(20)), 8, 42)
    assert set(fit) <= set(range(20))
    assert fit != pilot.selected_fit_indices(list(range(20)), 8, 77)


def test_paired_aggregation_rejects_duplicate_and_unpaired_cohorts():
    def row(name, seed, iou):
        return dict(id=name, seed=seed, total_params=4, fit_names=['a'], val_names=['b'],
                    test_names=['c'], normalization={'stats_sha256': 'fixed'},
                    test=dict(iou=iou, optical_subset_iou=iou, sar_subset_iou=iou))
    records = [row('quantum_patch4x4', 42, 0.8), row('mlp_patch4x4', 42, 0.3),
               row('quantum_patch4x4', 77, 0.4), row('mlp_patch4x4', 77, 0.6)]
    contrast = pilot.aggregate(records)['paired_iou'][0]
    assert contrast['mean'] == pytest.approx(0.15)
    assert contrast['std'] == pytest.approx(0.7 / 2**0.5)
    with pytest.raises(ValueError, match='Duplicate'):
        pilot.aggregate(records + [records[0]])
    records[-1]['fit_names'] = ['unpaired']
    with pytest.raises(ValueError, match='Unpaired'):
        pilot.aggregate(records)


def test_bounded_checkpoint_smoke_keeps_full_test_and_records_learning_diagnostics(tmp_path):
    from tools.make_mini_flood_dataset import generate_mini_flood_dataset
    data = tmp_path / 'data'
    generate_mini_flood_dataset(str(data), n_train=8, n_test=3, seed=42, seeds=[42, 77])
    args = ['--data-root', str(data), '--work-dir', str(tmp_path / 'runs'), '--device', 'cpu',
            '--seeds', '42', '77', '--learning-rates', '0.001', '0.003', '--epochs', '1',
            '--eval-every', '1', '--fit-cap', '8', '--batch-size', '2', '--image-size', '32',
            '--specs', 'unet_width7', 'mlp_patch4x4', 'quantum_patch4x4']
    with (tmp_path / 'cli.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(pilot.__file__)), *args], check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    result = json.loads((tmp_path / 'runs/summary.json').read_text())
    assert len(result['records']) == 6
    assert result['plan']['input_identity']['verified_files'] > 0
    assert all(len(row['test_names']) == 6 and len(row['fit_names']) == 8 for row in result['records'])
    assert all(set(row['fit_names']).isdisjoint(row['val_names']) for row in result['records'])
    assert all(row['n_global_fit'] > len(row['fit_names']) for row in result['records'])
    candidates = json.loads((tmp_path / 'runs/tuning_candidates.json').read_text())
    assert len(candidates) == 6 and all('test' not in row for row in candidates)
    selected = pilot.choose_rates(candidates)
    assert result['selection'] == {name: row['learning_rate'] for name, row in selected.items()}
    quantum = [row for row in result['records'] if row['model_type'] == 'quantum']
    assert all(row['diagnostics']['gradient_norms']['circuit_angles']['nonzero_steps'] > 0 for row in quantum)
    assert all(row['diagnostics']['initial_bottleneck_output'] for row in quantum)
    assert all(Path(tmp_path / 'runs' / row['checkpoint']).is_file() for row in result['records'])
    assert (tmp_path / 'runs/report.md').is_file()
    from scripts.collect_real_spatial_results import audit_run
    _, fits, _, checkpoints = audit_run(tmp_path / 'runs')
    assert len(fits) == 9 and len(checkpoints) == 6
    summary_path = tmp_path / 'runs/summary.json'
    original_summary = summary_path.read_text()
    try:
        result['records'][0]['seed'] = 999
        summary_path.write_text(json.dumps(result))
        with pytest.raises(ValueError, match='planned model/seed'):
            audit_run(tmp_path / 'runs')
        result['records'][0]['seed'] = 42
        result['records'][0]['elapsed_s'] += 1
        summary_path.write_text(json.dumps(result))
        with pytest.raises(ValueError, match='saved validation-only fit record'):
            audit_run(tmp_path / 'runs')
    finally:
        summary_path.write_text(original_summary)
    # Cached candidate records remain validation-only after test replay.
    assert all('test' not in json.loads(path.read_text()) for path in (tmp_path / 'runs/tuning').rglob('metrics.json'))
    before = {path: path.stat().st_mtime_ns for path in (tmp_path / 'runs').rglob('metrics.json')}
    with (tmp_path / 'resume.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(pilot.__file__)), *args, '--resume'], check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    assert before == {path: path.stat().st_mtime_ns for path in before}
    image = data / 'flood_combined/Training/images' / (result['records'][0]['fit_names'][0] + '.tif')
    original = image.read_bytes()
    try:
        image.write_bytes(b'changed raw input')
        with pytest.raises(ValueError, match='Raw file differs'):
            pilot.verify_inputs(data)
    finally:
        image.write_bytes(original)

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, '/tmp/quaffle-real-spatial-pilot-20261027')
import torch
from tools.checkpoint_evaluation import load_checkpoint_model

torch.set_num_threads(2)
run = Path('/tmp/quaffle-real-spatial-runs-60ep-20261027')
out = Path('/tmp/quaffle-spatial-checkpoint-progress-audit.json')
audited = json.loads(out.read_text()) if out.is_file() else []
previous = {item['path']: item for item in audited}
paths = sorted(run.glob('tuning/*/*/seed42/metrics.json')) + sorted(run.glob('replication/*/seed*/metrics.json'))
added = 0
for path in paths:
    identity = str(path.relative_to(run))
    if identity in previous:
        assert hashlib.sha256((path.parent / 'checkpoint.pt').read_bytes()).hexdigest() == previous[identity]['checkpoint_sha256']
        continue
    row = json.loads(path.read_text())
    model, cfg, kind = load_checkpoint_model(path.parent / 'checkpoint.pt', data_root='/tmp/quaffle-dfc24-real')
    checkpoint = torch.load(path.parent / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert kind == row['model_type']
    assert cfg['_resolved_norm_provenance'] == row['normalization']
    assert cfg['seed'] == row['seed'] and cfg['learning_rate'] == row['learning_rate']
    assert checkpoint['best_epoch'] == row['best_epoch']
    expected = max((point for point in row['history'] if 'val' in point), key=lambda point: point['val']['iou'])
    assert expected['epoch'] == row['best_epoch'] and expected['val'] == row['val']
    assert set(row['fit_names']).isdisjoint(row['val_names'])
    assert sum(parameter.numel() for parameter in model.parameters()) == row['total_params']
    state = model.state_dict()
    assert state.keys() == checkpoint['best_state_dict'].keys()
    assert all(torch.equal(value, checkpoint['best_state_dict'][name]) for name, value in state.items())
    audited.append(dict(path=identity, best_epoch=row['best_epoch'], params=row['total_params'],
        checkpoint_sha256=hashlib.sha256((path.parent / 'checkpoint.pt').read_bytes()).hexdigest(),
        checks='strict selected-weight replay, validation selection, disjoint fit/validation, full normalization identity'))
    out.write_text(json.dumps(audited, indent=2) + '\n')
    added += 1
    del model, checkpoint, state
print(json.dumps(dict(audited=len(audited), added=added)))

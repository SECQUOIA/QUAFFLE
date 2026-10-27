"""Strictly restore every control checkpoint with its producing control semantics."""
import hashlib
import json
from pathlib import Path
import sys

SOURCE = Path('/tmp/quaffle-real-spatial-controls-20261027')
RUN = Path('/tmp/quaffle-real-quantum-controls-20261027')
sys.path.insert(0, str(SOURCE))
import torch
from tools.checkpoint_evaluation import load_checkpoint_model
from scripts.run_real_spatial_pilot import parameter_groups

torch.set_num_threads(2)
plan = json.loads((RUN / 'plan.json').read_text())
for name, expected in plan['source_hashes'].items():
    assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
paths = []
for spec in plan['specs']:
    paths.extend(RUN / 'tuning' / spec['id'] / str(rate) / 'seed42' for rate in plan['settings']['learning_rates'])
    paths.extend(RUN / 'replication' / spec['id'] / ('seed' + str(seed)) for seed in plan['settings']['seeds'][1:])
rows = []
for directory in paths:
    record = json.loads((directory / 'metrics.json').read_text())
    path = directory / 'checkpoint.pt'
    model, cfg, kind = load_checkpoint_model(path, data_root='/tmp/quaffle-dfc24-real')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert kind == 'quantum'
    assert saved['settings'] == record['settings']
    assert cfg['_resolved_norm_provenance'] == record['normalization']
    assert saved['best_epoch'] == record['best_epoch']
    best = max((point for point in record['history'] if 'val' in point), key=lambda point: point['val']['iou'])
    assert best['epoch'] == record['best_epoch'] and best['val'] == record['val']
    state = model.state_dict()
    assert state.keys() == saved['best_state_dict'].keys()
    assert all(torch.equal(value, saved['best_state_dict'][name]) and torch.isfinite(value).all()
               for name, value in state.items())
    assert sum(p.numel() for p in model.parameters()) == record['total_params']
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == record['trainable_params']
    groups, blocks = parameter_groups(model)
    control = record['settings']['q_training_control']
    assert all(block.q_training_control == control for block in blocks)
    assert all(block.q_entangle == (control != 'no_entanglement') for block in blocks)
    assert all(p.requires_grad for _, p in groups['backbone'])
    assert all(p.requires_grad == (control != 'frozen') for _, p in groups['bottleneck'])
    rows.append(dict(directory=str(directory.relative_to(RUN)), q_training_control=control,
        seed=record['seed'], best_epoch=record['best_epoch'], total_params=record['total_params'],
        trainable_params=record['trainable_params'],
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        selected_for_test=(directory / 'selected_test.json').is_file(),
        checks='strict selected finite weights; configuration/normalization; best validation epoch; parameter counts; frozen flags; CNOT-control semantics'))
    del model, saved, state, groups, blocks
result = dict(producing_source_revision=plan['source_revision'],
    audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
Path('/tmp/quaffle-quantum-controls-checkpoint-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print('STRICT CONTROL RESTORES', len(rows))

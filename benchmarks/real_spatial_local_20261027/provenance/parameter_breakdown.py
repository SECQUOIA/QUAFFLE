"""Count unique parameters in each component of the frozen pilot models."""
import hashlib
import json
from pathlib import Path
import sys

SOURCE = Path('/tmp/quaffle-real-spatial-pilot-20261027')
RUN = Path('/tmp/quaffle-real-spatial-runs-60ep-20261027')
sys.path.insert(0, str(SOURCE))
import torch
from tools import training_common as tc
from scripts.run_real_spatial_pilot import parameter_groups
from utilsTorch.utilsTorch import UNetSegmentation, QVUNetSegmentation, ClassicalAblationUNetSegmentation

torch.set_num_threads(2)
plan = json.loads((RUN / 'plan.json').read_text())
for name, expected in plan['source_hashes'].items():
    assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
classes = dict(classical=UNetSegmentation, quantum=QVUNetSegmentation, ablation=ClassicalAblationUNetSegmentation)

def count(module):
    return sum(p.numel() for p in module.parameters())

def state_hash(module):
    return hashlib.sha256(b''.join(name.encode() + value.detach().cpu().numpy().tobytes()
        for name, value in module.named_parameters())).hexdigest()

rows = []
for spec in plan['specs']:
    model = tc.build_model(classes[spec['model_type']], dict(spec['config'], seed=42))
    body = model.model
    groups, _ = parameter_groups(model)
    row = dict(id=spec['id'], total=count(model), encoder=count(body.down), decoder=count(body.up),
        bottleneck_total=count(body.vertex), bottleneck_attention=count(body.vertex.attn),
        processing_core=sum(p.numel() for _, p in groups['bottleneck']),
        circuit_angles=sum(p.numel() for _, p in groups['circuit_angles']),
        input_scales=sum(p.numel() for _, p in groups['input_scales']),
        classical_remainder_initial_parameter_sha256=hashlib.sha256(b''.join(
            name.encode() + value.detach().cpu().numpy().tobytes()
            for name, value in groups['backbone'])).hexdigest(),
        encoder_initial_parameter_sha256=state_hash(body.down),
        decoder_initial_parameter_sha256=state_hash(body.up))
    row['bottleneck_residual_processing'] = row['bottleneck_total'] - row['bottleneck_attention']
    assert row['total'] == row['encoder'] + row['decoder'] + row['bottleneck_total']
    rows.append(row)
report = dict(producing_source_revision=plan['source_revision'], seed=42, models=rows,
    audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    semantics='Counts are unique Parameter objects, including shared blocks only once; Q/MLP processing_core excludes classical attention. Initial hashes describe the common seed, not identical initial tensors across different model classes.')
Path('/tmp/quaffle-spatial-parameter-breakdown.json').write_text(json.dumps(report, indent=2) + '\n')
for row in rows:
    print(json.dumps({key: value for key, value in row.items() if not key.endswith('sha256')}))

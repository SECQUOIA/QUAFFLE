"""Compute paired, descriptive contrasts from the retained local sweep records."""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
summary = json.loads((root / 'summary.json').read_text())
assert summary['seeds'] == [42, 77, 123]
panels = {p['panel_id']: p['rows'] for p in summary['panels']}
assert len(panels[5]) == 14 and len(panels[6]) == 12
assert sum(len(r['runs']) for rows in panels.values() for r in rows) == 78
contrasts = []

def compare(name, left, right):
    a = {r['seed']: r['test']['iou'] for r in left['runs']}
    b = {r['seed']: r['test']['iou'] for r in right['runs']}
    assert set(a) == set(b) == {42, 77, 123}
    values = {str(s): a[s] - b[s] for s in sorted(a)}
    result = {'comparison': name, 'left': left['label'], 'right': right['label'],
              'unit': 'IoU', 'paired_differences': values,
              'mean': statistics.mean(values.values()),
              'sample_std': statistics.stdev(values.values()), 'n': len(values)}
    contrasts.append(result)
    return result

def get_arch(model, qc, gate, mlp_mode=None):
    return next(row for row in panels[5] if (
        row['runs'][0]['settings']['model_type'] == model
        and row['runs'][0]['settings'].get('quantum_channels') == qc
        and row['runs'][0]['settings']['decoder_skip_gate'] == gate
        and row['runs'][0]['settings'].get('ablation_mlp_mode') == mlp_mode))

for gate in ('none', 'affine'):
    for qc in (32, 64):
        for mode in ('param_matched', 'full_rank'):
            compare('quantum_minus_mlp', get_arch('quantum', qc, gate), get_arch('ablation', qc, gate, mode))
    for model, mode in (('quantum', None), ('ablation', 'param_matched'), ('ablation', 'full_rank')):
        compare('full_minus_partial', get_arch(model, 64, gate, mode), get_arch(model, 32, gate, mode))
for qc in (32, 64):
    for model, mode in (('quantum', None), ('ablation', 'param_matched'), ('ablation', 'full_rank')):
        compare('affine_minus_none', get_arch(model, qc, 'affine', mode), get_arch(model, qc, 'none', mode))
compare('affine_minus_none', get_arch('classical', None, 'affine'), get_arch('classical', None, 'none'))
for model in ('classical', 'ablation', 'quantum'):
    rows = {row['runs'][0]['settings']['comm_compression']: row for row in panels[6]
            if row['runs'][0]['settings']['model_type'] == model}
    for mode in ('fp16', 'int8', 'topk10'):
        compare('compressed_minus_uncompressed', rows[mode], rows['none'])
(root / 'paired_comparisons.json').write_text(json.dumps(contrasts, indent=2) + '\n')

print('ARCHITECTURE')
for row in panels[5]:
    st = row['stats']
    print(row['label'], 'params', st['total_params'], 'IoU', st['test_iou']['mean'],
          'zero_delta', st['ko_zero_delta_iou']['mean'], 'random_delta', st['ko_random_delta_iou']['mean'])
print('COMPRESSION')
for row in panels[6]:
    print(row['label'], row['stats']['test_iou']['mean'], row['stats']['compression_ratio']['mean'])
print('PAIRED CONTRASTS (IoU percentage points)')
for r in contrasts:
    print(r['left'], '-', r['right'], 100*r['mean'], '+/-',100*r['sample_std'])

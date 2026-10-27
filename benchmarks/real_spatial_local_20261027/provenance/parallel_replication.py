"""One independent, unchanged fit from an already-frozen spatial study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE = Path('/tmp/quaffle-real-spatial-pilot-20261027')
RUN = Path('/tmp/quaffle-real-spatial-runs-60ep-20261027')
DATA = Path('/tmp/quaffle-dfc24-real')
sys.path.insert(0, str(SOURCE))
import torch
from scripts import run_real_spatial_pilot as pilot

plan = json.loads((RUN / 'plan.json').read_text())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() == plan['source_revision']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE, text=True).strip()
for name, expected in plan['source_hashes'].items():
    assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
assert pilot.verify_inputs(DATA) == plan['input_identity']
model_id, seed = 'quantum_patch4x4', 123
args = argparse.Namespace(**plan['settings'], data_root=DATA, work_dir=RUN)
assert seed in args.seeds[1:]
spec = next(spec for spec in plan['specs'] if spec['id'] == model_id)
rate = json.loads((RUN / 'validation_selection.json').read_text())[model_id]['learning_rate']
target = RUN / 'replication' / model_id / ('seed' + str(seed))
if target.exists():
    raise RuntimeError('Refusing to share a fit directory with another attempt')
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.cuda.set_per_process_memory_fraction(plan['gpu_allocator_fraction'])
data = pilot.load_training_data(args, seed)
reference = json.loads((RUN / 'replication/unet_width7/seed123/metrics.json').read_text())
assert [data['dataset'].names[i] for i in data['fit']] == reference['fit_names']
assert [data['dataset'].names[i] for i in data['validation']] == reference['val_names']
assert data['normalization'] == reference['normalization']
provenance = dict(started_utc=datetime.now(timezone.utc).isoformat(),
    source_revision=plan['source_revision'], source_hashes=plan['source_hashes'],
    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    model_id=model_id, seed=seed, learning_rate=rate, status='running',
    target=str(target.relative_to(RUN)), threads=2, memory_max='4G', memory_swap_max=0,
    gpu_allocator_fraction=plan['gpu_allocator_fraction'],
    purpose='Overlap one independent replication; main driver later reuses identical completed fit; no test replay in this process',
    limitations='Fit durations overlap other models and are not controlled timing benchmarks')
pilot.write_json(RUN / 'parallel_replication.json', provenance)
result = pilot.fit_one(spec, seed, rate, args, data, target, torch.device(args.device))
provenance.update(status='complete', ended_utc=datetime.now(timezone.utc).isoformat(),
    best_epoch=result['best_epoch'], metrics_sha256=hashlib.sha256((target / 'metrics.json').read_bytes()).hexdigest(),
    checkpoint_sha256=hashlib.sha256((target / 'checkpoint.pt').read_bytes()).hexdigest())
pilot.write_json(RUN / 'parallel_replication.json', provenance)
print('INDEPENDENT REPLICATION COMPLETE', flush=True)

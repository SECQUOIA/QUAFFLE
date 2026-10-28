"""Exercise rendered launch jobs without contacting SLURM or loading a model."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from misc import generate_hq_campaign_scripts as launch


ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith('QUAFFLE_') or key == 'HQ_ACCEPTED_PROFILE_SHA256':
            monkeypatch.delenv(key)
    # Shell metacharacters/spaces must remain ordinary pathname bytes.
    root = tmp_path/"repo ' $(touch UNSAFE)"
    for name in (launch.GENERATOR, 'misc/generate_grid.py', 'config/seeds.py'):
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    direction = root/'scripts/run_hq_dfc24_direction.py'
    direction.parent.mkdir(parents=True, exist_ok=True)
    direction.write_text("SOURCE_FILES = ('scripts/run_hq_dfc24_direction.py', 'native_dependency.py')\n")
    (root/'native_dependency.py').write_text('NATIVE_DEPENDENCY = 1\n')
    # Native stand-ins reject wrong flags and provide a durable dispatch trace.
    # They do not simulate model accuracy or the native scientific audit.
    (root/launch.RUNNER).write_text('''import argparse, json, os
from pathlib import Path
class hq:
    SOURCE_FILES = ('scripts/run_hq_dfc24_direction.py', 'native_dependency.py')
SOURCE_FILES = (*hq.SOURCE_FILES, 'scripts/run_hq_dfc24_fresh.py')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('profile','run'))
    p.add_argument('--mode', choices=('full',), required=True)
    for key in ('data-root','work-dir','device','gpu-memory-fraction','budget-seconds'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--profile'); p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    assert a.device == 'cuda'
    if a.phase == 'run': assert a.profile
    with Path(os.environ['HQ_TEST_CALLS']).open('a') as f: f.write(json.dumps(vars(a))+'\\n')
''')
    (root/launch.COLLECTOR).write_text('''import argparse, json, os
from pathlib import Path
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('work-dir','data-root','output-dir'): p.add_argument('--'+key, required=True)
    a = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    with Path(os.environ['HQ_TEST_CALLS']).open('a') as f: f.write(json.dumps(dict(phase='collect',**vars(a)))+'\\n')
    if not (Path(a.work_dir)/'COMPLETE').is_file(): p.error('incomplete native grid')
    Path(a.output_dir).mkdir(parents=True)
''')
    protocol = root/launch.PROTOCOL
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text('Frozen full four-arm execution protocol.\n')
    config = tmp_path/'cluster config.json'
    dump(config, dict(account='SYNTHETIC_TEST_ACCOUNT', partition='SYNTHETIC_GPU',
                     python=sys.executable, mem='8G', cpus_per_task=2,
                     output_log_dir='stdout logs', error_log_dir='stderr logs'))
    bin_dir = tmp_path/'bin'; bin_dir.mkdir()
    contacted = tmp_path/'scheduler_contacted'
    for name in ('sbatch', 'ssh', 'squeue', 'sinfo'):
        path = bin_dir/name
        path.write_text('#!/bin/sh\ntouch '+str(contacted)+'\nexit 97\n')
        path.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir)+os.pathsep+os.environ['PATH'])
    calls = tmp_path/'native_calls.jsonl'
    monkeypatch.setenv('HQ_TEST_CALLS', str(calls))
    args = [sys.executable, str(root/launch.GENERATOR), 'generate', '--repo-root', str(root),
            '--data-root', str(tmp_path/'data not present'), '--profile-dir', str(tmp_path/'profile'),
            '--run-dir', str(tmp_path/'raw'), '--report-dir', str(tmp_path/'report'),
            '--output-dir', str(tmp_path/'package'), '--gpu-memory-fraction', '.23',
            '--cluster-config', str(config)]
    for phase in launch.PHASES:
        args += ['--'+phase+'-budget-seconds', '20', '--'+phase+'-walltime', '00:02:00']
    return dict(root=root, args=args, package=tmp_path/'package', config=config,
                calls=calls, contacted=contacted, tmp=tmp_path)


def generate(campaign, extra=(), success=True):
    result = subprocess.run([*campaign['args'], *extra], text=True, capture_output=True)
    if success:
        assert result.returncode == 0, result.stderr
        return json.loads((campaign['package']/'launch_manifest.json').read_text())
    assert result.returncode != 0
    return result


def execute(campaign, phase, env=None):
    return subprocess.run(['bash', str(campaign['package']/launch.FILENAMES[phase])],
                          env=env, capture_output=True, text=True)


def profile(campaign, manifest, change=None):
    value = dict(format='hq-dfc24-fresh-profile-v1', mode='full', status='passed',
                 settings=launch.FULL_SETTINGS.copy(), source_hashes=manifest['producer_source_hashes'],
                 protocol_sha256=manifest['protocol_sha256'],
                 runtime=dict(gpu_allocator_fraction=.23), profile_learning_rate=.001,
                 cases=[dict(kind=kind, batch_size=8, strict_restore=True, frozen_unchanged=True)
                        for kind in sorted(launch.CONTROLS)])
    if change:
        change(value)
    path = Path(manifest['paths']['profile'])/'resource_profile.json'
    dump(path, value)
    env = os.environ.copy()
    env['HQ_ACCEPTED_PROFILE_SHA256'] = launch.digest(path)
    return path, env


def calls(campaign):
    path = campaign['calls']
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_three_separate_jobs_native_commands_manual_acceptance_and_no_scheduler(campaign):
    manifest = generate(campaign)
    assert set(manifest['phases']) == {'profile', 'run', 'collect'}
    assert not campaign['contacted'].exists()
    assert not (campaign['package']/'submit_all.sh').exists()
    for phase in launch.PHASES:
        script = campaign['package']/launch.FILENAMES[phase]
        assert subprocess.run(['bash', '-n', str(script)], capture_output=True).returncode == 0
        text = script.read_text()
        assert 'set -euo pipefail' in text and 'timeout --signal=TERM --kill-after=30s' in text
        assert 'sbatch' not in text and 'ssh ' not in text and 'eval ' not in text
    assert execute(campaign, 'profile').returncode == 0
    assert calls(campaign)[0]['phase'] == 'profile'
    assert execute(campaign, 'run').returncode != 0  # No manual acceptance.
    assert len(calls(campaign)) == 1
    path, env = profile(campaign, manifest)
    wrong = {**env, 'HQ_ACCEPTED_PROFILE_SHA256': '0'*64}
    assert execute(campaign, 'run', wrong).returncode != 0
    assert len(calls(campaign)) == 1
    assert execute(campaign, 'run', env).returncode == 0
    row = calls(campaign)[1]
    assert row['phase'] == 'run' and row['mode'] == 'full' and row['profile'] == str(path)
    assert row['gpu_memory_fraction'] == '0.23' and row['budget_seconds'] == '20.0'
    assert row['data_root'] == str(campaign['tmp']/'data not present')
    assert not row['resume']
    # Collection still relies on the native complete-grid guard.
    result = execute(campaign, 'collect')
    assert result.returncode != 0 and 'incomplete native grid' in result.stderr
    Path(manifest['paths']['run']).mkdir()
    (Path(manifest['paths']['run'])/'COMPLETE').touch()
    assert execute(campaign, 'collect').returncode == 0
    assert not campaign['contacted'].exists() and not (campaign['root']/'UNSAFE').exists()
    assert '--gpus-per-node' not in (campaign['package']/launch.FILENAMES['collect']).read_text()
    assert set(manifest['source_hashes']) >= {launch.COLLECTOR, launch.PROTOCOL, launch.GENERATOR,
                                             'misc/generate_grid.py', 'native_dependency.py'}


@pytest.mark.parametrize('change', [
    lambda p: p.update(mode='rehearsal'),
    lambda p: p.update(status='failed'),
    lambda p: p.update(format='other'),
    lambda p: p['settings'].update(fit_cap=128),
    lambda p: p['settings'].update(epochs=2),
    lambda p: p['runtime'].update(gpu_allocator_fraction=.12),
    lambda p: p.update(protocol_sha256='0'*64),
    lambda p: p['source_hashes'].update({'native_dependency.py': '0'*64}),
    lambda p: p['cases'].pop(),
    lambda p: p['cases'][0].update(strict_restore=False),
])
def test_accepted_digest_cannot_bypass_full_profile_contract(campaign, change):
    manifest = generate(campaign)
    _, env = profile(campaign, manifest, change)
    assert execute(campaign, 'run', env).returncode != 0
    assert calls(campaign) == []


@pytest.mark.parametrize('phase', launch.PHASES)
@pytest.mark.parametrize('target', ['native_dependency.py', launch.COLLECTOR, launch.PROTOCOL, 'config'])
def test_changed_bound_source_or_config_blocks_every_dispatch(campaign, phase, target):
    manifest = generate(campaign)
    _, env = profile(campaign, manifest)
    path = campaign['config'] if target == 'config' else campaign['root']/target
    path.write_text(path.read_text()+'\n')
    assert execute(campaign, phase, env).returncode != 0
    assert calls(campaign) == []


def test_cluster_cli_environment_json_precedence_and_resume_opt_in(campaign, monkeypatch):
    monkeypatch.setenv('QUAFFLE_ACCOUNT', 'ENV_ACCOUNT')
    monkeypatch.setenv('QUAFFLE_MEM', '12G')
    manifest = generate(campaign, ['--account', 'CLI_ACCOUNT', '--resume'])
    assert manifest['cluster']['account'] == 'CLI_ACCOUNT'
    assert manifest['cluster']['mem'] == '12G'
    assert manifest['cluster']['partition'] == 'SYNTHETIC_GPU'
    assert manifest['cluster']['gpus'] == 1
    assert '--resume' in manifest['phases']['run']['command']
    assert '--resume' not in manifest['phases']['profile']['command']
    assert manifest['resume'] is True


@pytest.mark.parametrize('extra', [
    ['--run-budget-seconds', 'nan'], ['--profile-budget-seconds', 'inf'],
    ['--collect-budget-seconds', '0'], ['--run-walltime', '00:01:00'],
    ['--profile-walltime', '00:99:00'], ['--gpu-memory-fraction', '0'],
    ['--gpu-memory-fraction', 'nan'], ['--gpu-memory-fraction', '1.1'],
    ['--account', 'account\n#SBATCH --time=99:00:00'],
    ['--partition', 'gpu --reservation=other'], ['--gpus', '2'],
    ['--cpus-per-task', '1'], ['--mem', '8G --exclusive'],
    ['--output-log-dir', '../outside'],
])
def test_invalid_resource_or_header_values_refused_before_output(campaign, extra):
    generate(campaign, extra, success=False)
    assert not campaign['package'].exists() and not campaign['contacted'].exists()


@pytest.mark.parametrize('path_case', ['same_phase', 'nested_phase', 'data_alias', 'symlink_alias', 'repo_parent'])
def test_path_aliases_rejected(campaign, path_case):
    tmp = campaign['tmp']
    if path_case == 'same_phase':
        extra = ['--run-dir', str(tmp/'profile')]
    elif path_case == 'nested_phase':
        extra = ['--report-dir', str(tmp/'raw'/'report')]
    elif path_case == 'data_alias':
        extra = ['--output-dir', str(tmp/'data not present')]
    elif path_case == 'symlink_alias':
        (tmp/'alias').symlink_to(tmp/'raw', target_is_directory=True)
        extra = ['--report-dir', str(tmp/'alias')]
    else:
        extra = ['--run-dir', str(tmp)]
    generate(campaign, extra, success=False)
    assert not campaign['package'].exists()


def test_nonempty_package_and_modified_manifest_are_preserved(campaign):
    manifest = generate(campaign)
    before = {str(path): path.read_bytes() for path in campaign['package'].rglob('*') if path.is_file()}
    generate(campaign, success=False)
    assert before == {str(path): path.read_bytes() for path in campaign['package'].rglob('*') if path.is_file()}
    path, env = profile(campaign, manifest)
    manifest_path = campaign['package']/'launch_manifest.json'
    value = json.loads(manifest_path.read_text()); value['gpu_memory_fraction'] = .99
    dump(manifest_path, value)
    assert execute(campaign, 'run', env).returncode != 0
    assert calls(campaign) == []
    assert path.exists()


def test_native_inventory_matches_actual_runner_without_model_imports():
    binding = launch.source_binding(ROOT)
    assert launch.RUNNER in binding['producer_source_hashes']
    assert 'utilsTorch/hq_direction.py' in binding['producer_source_hashes']
    assert launch.COLLECTOR in binding['source_hashes']
    assert launch.PROTOCOL in binding['source_hashes']


def test_missing_explicit_budgets_are_cli_errors(campaign):
    args = campaign['args'][:]
    index = args.index('--run-budget-seconds')
    del args[index:index+2]
    result = subprocess.run(args, text=True, capture_output=True)
    assert result.returncode != 0 and '--run-budget-seconds' in result.stderr
    assert not campaign['package'].exists()

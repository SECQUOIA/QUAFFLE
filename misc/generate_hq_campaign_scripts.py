#!/usr/bin/env python3
"""Generate three independently submitted SLURM jobs for the full HQ campaign.

This stdlib-only generator never contacts a scheduler or imports a model. Cluster
values use generate_grid's flag > QUAFFLE_* > JSON > default precedence. Budgets
are operator limits in seconds, not estimates of cluster runtime.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from misc import generate_grid as grid

GENERATOR = 'misc/generate_hq_campaign_scripts.py'
RUNNER = 'scripts/run_hq_dfc24_fresh.py'
COLLECTOR = 'scripts/collect_hq_dfc24_fresh.py'
PROTOCOL = 'docs/HQ_DFC24_FRESH_PROTOCOL.md'
FORMAT = 'hq-dfc24-slurm-package-v1'
PHASES = ('profile', 'run', 'collect')
FILENAMES = dict(zip(PHASES, ('01_profile.sbatch', '02_run.sbatch', '03_collect.sbatch')))
CONTROLS = {'quantum', 'frozen_quantum', 'zero_bottleneck', 'compact_affine'}
FULL_SETTINGS = dict(mode='full', device='cuda', batch_size=8, image_size=128,
                     base_channels=8, augment=True, epochs=60, fit_cap=None,
                     validation_cap=None, eval_every=2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def clean(value):
    value = str(value)
    require(not any(ord(char) < 32 or ord(char) == 127 for char in value),
            'Control characters are forbidden in paths and cluster values')
    return value


def overlap(left, right):
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def wall_seconds(value):
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', clean(value))
    require(match is not None, 'Walltime must be [D-]HH:MM:SS')
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    require(minutes < 60 and seconds < 60 and (not match[1] or hours < 24), 'Invalid walltime')
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    require(total > 0, 'Walltime must be finite and positive')
    return total


def _source_list(path, inherited=()):
    """Read only the native declarative inventory; never execute model imports."""
    tree = ast.parse(path.read_text())
    value = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'SOURCE_FILES'
                         for target in node.targets))
    require(isinstance(value, (ast.Tuple, ast.List)), 'Unsupported native source inventory')
    result = []
    for item in value.elts:
        if isinstance(item, ast.Starred):
            require(ast.dump(item.value) == ast.dump(ast.parse('hq.SOURCE_FILES', mode='eval').body)
                    and inherited, 'Unsupported inherited source inventory')
            result.extend(inherited)
        else:
            name = ast.literal_eval(item)
            require(isinstance(name, str), 'Invalid source inventory path')
            result.append(name)
    require(len(result) == len(set(result)), 'Duplicate native source paths')
    return result


def source_binding(root):
    inherited = _source_list(root/'scripts/run_hq_dfc24_direction.py')
    producer = _source_list(root/RUNNER, inherited)
    names = set(producer) | {COLLECTOR, PROTOCOL, GENERATOR,
                             'misc/generate_grid.py', 'config/seeds.py'}
    for name in names:
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'Missing or escaped source: ' + name)
    return dict(source_hashes={name: digest(root/name) for name in sorted(names)},
                producer_source_hashes={name: digest(root/name) for name in producer},
                protocol_sha256=digest(root/PROTOCOL))


def _cluster(args):
    cluster = grid.resolve_cluster({key: getattr(args, key, None) for key in grid.CLUSTER_DEFAULTS},
                                   args.cluster_config)
    for key, value in cluster.items():
        for item in value if isinstance(value, list) else [value]:
            if item is not None:
                clean(item)
    require(cluster['account'], 'An explicit SLURM account is required')
    for key in ('account', 'partition', 'mail_user', 'mail_type'):
        if cluster[key] is not None:
            require(re.fullmatch(r'[A-Za-z0-9_.@+,/:=-]+', cluster[key]), 'Unsafe SLURM header: ' + key)
    require(cluster['gpus'] == 1 and cluster['cpus_per_task'] >= 2,
            'The native campaign requires one GPU and at least two CPU threads')
    require(re.fullmatch(r'[1-9]\d*(?:[KMGTP])?', cluster['mem'], re.I), 'Invalid SLURM memory limit')
    chosen = args.cluster_config or os.environ.get('QUAFFLE_CLUSTER_CONFIG') or grid.CLUSTER_CONFIG_DEFAULT
    path = Path(chosen).resolve()
    config = dict(path=str(path), sha256=digest(path)) if path.exists() else None
    return dict(cluster), config


def _header(phase, args, cluster, paths):
    stdout = paths['output_logs']/'%x-%j.out'
    stderr = paths['error_logs']/'%x-%j.err'
    lines = ['#!/bin/bash', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
             '#SBATCH --account=' + cluster['account'], '#SBATCH --job-name=hq-full-' + phase,
             '#SBATCH --cpus-per-task=' + str(cluster['cpus_per_task']),
             '#SBATCH --mem=' + cluster['mem'], '#SBATCH --time=' + getattr(args, phase+'_walltime'),
             '#SBATCH --output=' + shlex.quote(str(stdout)), '#SBATCH --error=' + shlex.quote(str(stderr))]
    if phase != 'collect':
        lines.append('#SBATCH --gpus-per-node=1')
    if cluster['partition']:
        lines.append('#SBATCH --partition=' + cluster['partition'])
    if cluster['mail_user']:
        lines.append('#SBATCH --mail-user=' + cluster['mail_user'])
        if cluster['mail_type']:
            lines.append('#SBATCH --mail-type=' + cluster['mail_type'])
    lines += ['', 'set -euo pipefail', '# Operator-specified limits; no cluster runtime estimate.',
              'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2',
              'export PYTHONDONTWRITEBYTECODE=1']
    if phase == 'collect':
        lines.append("export CUDA_VISIBLE_DEVICES=''")
    if cluster['modules']:
        lines.append('module purge')
        lines += ['module load ' + shlex.quote(name) for name in cluster['modules']]
    if cluster['conda_sh']:
        lines.append('source ' + shlex.quote(cluster['conda_sh']))
    if cluster['conda_env']:
        lines.append('conda activate ' + shlex.quote(cluster['conda_env']))
    if cluster['git_safe_directory']:
        lines.append('export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=' +
                     shlex.quote(str(paths['repo'])))
    lines += ['cd ' + shlex.quote(str(paths['repo'])),
              'export PYTHONPATH=' + shlex.quote(str(paths['repo'])) + '${PYTHONPATH:+:$PYTHONPATH}']
    return lines


def generate(args):
    cluster, config = _cluster(args)
    paths = {key: Path(clean(getattr(args, key+'_dir' if key != 'data' else 'data_root'))).resolve()
             for key in ('profile', 'run', 'report', 'output', 'data')}
    paths['repo'] = Path(clean(args.repo_root)).resolve()
    destinations = ('profile', 'run', 'report', 'output')
    for index, key in enumerate(destinations):
        require(not overlap(paths[key], paths['data']), 'Output path aliases data: ' + key)
        require(not paths['repo'].is_relative_to(paths[key]), 'Output contains repository: ' + key)
        for other in destinations[index+1:]:
            require(not overlap(paths[key], paths[other]), 'Phase/output directories overlap')
    require(not paths['output'].exists() or (paths['output'].is_dir() and not any(paths['output'].iterdir())),
            'Use a new or empty package output directory')
    fraction = args.gpu_memory_fraction
    require(math.isfinite(fraction) and 0 < fraction <= 1, 'Explicit allocator fraction must be in (0,1]')
    phases = {}
    for phase in PHASES:
        budget = getattr(args, phase+'_budget_seconds')
        require(math.isfinite(budget) and budget > 0, 'Each phase needs a positive finite seconds budget')
        outer_timeout = budget + 30 if phase != 'collect' else budget
        headroom = 90 if phase != 'collect' else 60
        require(wall_seconds(getattr(args, phase+'_walltime')) >= budget + headroom,
                'Walltime must allow 90 seconds beyond profile/run budgets or 60 beyond collection')
        if phase == 'collect':
            command = [cluster['python'], str(paths['repo']/COLLECTOR), '--work-dir', str(paths['run']),
                       '--data-root', str(paths['data']), '--output-dir', str(paths['report'])]
        else:
            command = [cluster['python'], str(paths['repo']/RUNNER), phase, '--mode', 'full',
                       '--data-root', str(paths['data']), '--work-dir', str(paths[phase]),
                       '--device', 'cuda', '--gpu-memory-fraction', str(fraction),
                       '--budget-seconds', str(budget)]
            if phase == 'run':
                command += ['--profile', str(paths['profile']/'resource_profile.json')]
                if args.resume:
                    command.append('--resume')
        phases[phase] = dict(script=FILENAMES[phase], command=command, budget_seconds=budget,
                             outer_timeout_seconds=outer_timeout, kill_grace_seconds=30,
                             walltime=getattr(args, phase+'_walltime'))
    for key in ('output', 'error'):
        log_path = Path(cluster[key+'_log_dir'])
        paths[key+'_logs'] = (paths['output']/log_path).resolve()
        require(paths[key+'_logs'].is_relative_to(paths['output']) and paths[key+'_logs'] != paths['output'],
                'Log directories must be relative paths inside the package')
        require(paths[key+'_logs'].parts[len(paths['output'].parts)] not in
                {'launch_manifest.json', 'OPERATIONS.md', *FILENAMES.values()}, 'Log path aliases package file')
    binding = source_binding(paths['repo'])
    require(Path(__file__).resolve() == paths['repo']/GENERATOR,
            'Generate using the generator inside --repo-root')
    manifest = dict(format=FORMAT, paths={key: str(value) for key, value in paths.items()},
                    cluster=cluster, cluster_config=config, phases=phases,
                    gpu_memory_fraction=fraction, resume=args.resume, **binding)
    encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    manifest_sha = hashlib.sha256(encoded).hexdigest()
    manifest_path = paths['output']/'launch_manifest.json'
    rendered = {}
    for phase in PHASES:
        verify = [cluster['python'], str(paths['repo']/GENERATOR), 'verify', '--manifest', str(manifest_path),
                  '--manifest-sha256', manifest_sha, '--phase', phase]
        lines = _header(phase, args, cluster, paths)
        if phase == 'run':
            lines.append(': "${HQ_ACCEPTED_PROFILE_SHA256:?Inspect the full profile and explicitly export its accepted SHA256}"')
        lines.append(shlex.join(verify))
        lines.append(shlex.join(['timeout', '--signal=TERM', '--kill-after=30s',
                                  str(phases[phase]['outer_timeout_seconds'])+'s', *phases[phase]['command']]))
        rendered[FILENAMES[phase]] = '\n'.join(lines)+'\n'
    paths['output'].mkdir(parents=True, exist_ok=True)
    for key in ('output_logs', 'error_logs'):
        paths[key].mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(encoded)
    for name, content in rendered.items():
        target = paths['output']/name
        target.write_text(content)
        target.chmod(0o755)
    (paths['output']/'OPERATIONS.md').write_text(instructions(manifest, manifest_sha))
    return manifest


def verify(manifest_path, expected_sha, phase, accepted_sha=None):
    require(re.fullmatch(r'[0-9a-f]{64}', expected_sha or ''), 'Expected immutable manifest SHA256')
    require(digest(manifest_path) == expected_sha, 'Launch manifest bytes changed; regenerate the package')
    value = read_json(manifest_path)
    require(value['format'] == FORMAT and phase in PHASES, 'Invalid launch manifest/phase')
    root = Path(value['paths']['repo'])
    for name, expected in value['source_hashes'].items():
        path = (root/name).resolve()
        require(path.is_relative_to(root) and digest(path) == expected, 'Bound source changed: ' + name)
    config = value['cluster_config']
    if config is not None:
        require(digest(config['path']) == config['sha256'], 'Bound cluster config changed')
    if phase == 'run':
        require(re.fullmatch(r'[0-9a-f]{64}', accepted_sha or ''), 'Explicit accepted profile SHA256 required')
        profile_path = Path(value['paths']['profile'])/'resource_profile.json'
        require(digest(profile_path) == accepted_sha, 'Accepted profile SHA256 differs from actual bytes')
        profile = read_json(profile_path)
        require(profile['format'] == 'hq-dfc24-fresh-profile-v1' and profile['status'] == 'passed'
                and profile['mode'] == 'full' and profile['settings'] == FULL_SETTINGS,
                'Acceptance requires a passed native FULL profile with uncapped full settings')
        require(profile['source_hashes'] == value['producer_source_hashes']
                and profile['protocol_sha256'] == value['protocol_sha256'], 'Profile source/protocol binding differs')
        require(profile['runtime']['gpu_allocator_fraction'] == value['gpu_memory_fraction']
                and profile['profile_learning_rate'] == .001, 'Profile allocator/rate binding differs')
        cases = profile['cases']
        require(len(cases) == 4 and {case['kind'] for case in cases} == CONTROLS
                and all(case['batch_size'] == 8 and case['strict_restore'] is True
                        and case['frozen_unchanged'] is True for case in cases), 'Incomplete actual four-arm profile')
    return value


def instructions(value, manifest_sha):
    paths, phases = value['paths'], value['phases']
    package = Path(paths['output'])
    profile = Path(paths['profile'])/'resource_profile.json'
    return '\n'.join([
        '# Full HQ campaign handoff', '',
        'Alan submits each job separately after checking the preceding result. This package submits nothing.',
        'These are operator-selected resource limits, not Anvil runtime or allocation recommendations. The generator has not contacted or validated a cluster.', '',
        f'Immutable launch manifest SHA256: `{manifest_sha}`. Source/protocol and selected cluster-config bytes are checked before every phase; changes require a new package.', '',
        '1. Alan regenerates this package from his actual source checkout and cluster configuration; data and Python paths need not exist on the generator host. Put the exact checkout, dataset, package and configured environment at the recorded execution paths. Log folders are inside this package. Confirm scheduler account/partition, Python, modules, memory, walltimes and allocator fraction with local policy. The collection job uses the same configured partition without requesting a GPU; confirm that this is permitted or regenerate a separately configured collection package.',
        '2. Submit only the profile job:', '', '```bash', shlex.join(['sbatch', str(package/FILENAMES['profile'])]), '```', '',
        '3. Inspect its exit status and full `resource_profile.json`: all four actual batch8 cases must pass. Decide whether the explicitly bounded full campaign is affordable. A laptop profile is not a cluster profile. Acceptance is manual, and binds the exact inspected bytes:', '',
        '```bash', 'sha256sum -- ' + shlex.quote(str(profile)),
        "export HQ_ACCEPTED_PROFILE_SHA256='PASTE_THE_INSPECTED_64_CHARACTER_SHA256'",
        shlex.join(['sbatch', '--export=ALL,HQ_ACCEPTED_PROFILE_SHA256', str(package/FILENAMES['run'])]), '```', '',
        '4. After the run exits successfully and all 16 fresh fits and 12 saved test replays are complete, submit collection separately:', '',
        '```bash', shlex.join(['sbatch', str(package/FILENAMES['collect'])]), '```', '',
        'The native runner rechecks full settings, exact runtime/source/data/profile/cohorts and strict checkpoints. The native collector refuses incomplete grids. The 205 test tiles were already exposed; this is not a new holdout or a quantum-superiority test.', '',
        f"Resume opt-in for this package: `{value['resume']}`. No failed/partial fit or replay is silently retried. Native incomplete-run deadlines include downtime/queue time; completed resumes verify without new fitting or inference. Do not erase or edit preserved attempts to obtain a retry. A different resume choice requires a new package using the original run budget/profile/paths.", '',
        '| Phase | Native/phase budget (seconds) | Outer TERM deadline (seconds) | SLURM walltime |', '|---|---:|---:|---|',
        *[f"| {phase} | {phases[phase]['budget_seconds']} | {phases[phase]['outer_timeout_seconds']} | {phases[phase]['walltime']} |" for phase in PHASES], '',
        'All commands have an outer TERM deadline with a 30-second KILL grace. Profile/run receive the native persisted learning budget and an outer deadline 30 seconds longer, allowing native failure preservation. Collection has an outer execution limit and performs CPU restoration/audits only. Walltimes allow at least 90 seconds beyond profile/run budgets and 60 beyond collection; these limits do not guarantee campaign completion.', '',
        f"CUDA allocator fraction: `{value['gpu_memory_fraction']}`. This is a process allocator cap, not total GPU/host memory accounting. Numerical threads are 2. Collection requests no GPU and hides CUDA. The four-arm model remains the production 16-qubit full bottleneck at image size 128 and base width 8; there are no geometry or epoch overrides.", ''])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    make = commands.add_parser('generate', help='Write a new package; never submit jobs')
    make.add_argument('--repo-root', type=Path, default=ROOT)
    for key in ('data-root', 'profile-dir', 'run-dir', 'report-dir', 'output-dir'):
        make.add_argument('--'+key, type=Path, required=True)
    make.add_argument('--gpu-memory-fraction', type=float, required=True)
    make.add_argument('--resume', action='store_true', help='Explicit native run resume; preserves attempts')
    for phase in PHASES:
        make.add_argument('--'+phase+'-budget-seconds', type=float, required=True)
        make.add_argument('--'+phase+'-walltime', required=True, help='Explicit [D-]HH:MM:SS operator limit')
    make.add_argument('--cluster-config')
    for key in grid.CLUSTER_DEFAULTS:
        if key == 'git_safe_directory':
            make.add_argument('--git-safe-directory', action=argparse.BooleanOptionalAction, default=None)
        else:
            options = dict(default=None)
            if key in grid.CLUSTER_INT_KEYS:
                options['type'] = int
            if key == 'modules':
                options['nargs'] = '*'
            make.add_argument('--'+key.replace('_', '-'), **options)
    check = commands.add_parser('verify', help='Read-only pre-dispatch integrity/acceptance check')
    check.add_argument('--manifest', type=Path, required=True)
    check.add_argument('--manifest-sha256', required=True)
    check.add_argument('--phase', choices=PHASES, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == 'generate':
            generate(args)
            print('Wrote three scripts, an immutable manifest and operator instructions; no jobs submitted.')
        else:
            verify(args.manifest, args.manifest_sha256, args.phase,
                   os.environ.get('HQ_ACCEPTED_PROFILE_SHA256'))
    except (ValueError, KeyError, OSError, StopIteration, SyntaxError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

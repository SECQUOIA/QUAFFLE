import copy

import pytest

from scripts import run_local_spatial_study as study


def complete_summary():
    records = []
    for index, spec in enumerate(study.study_specs()):
        for seed in (42, 77, 123):
            records.append(dict(id=spec['id'], seed=seed, learning_rate=0.001,
                total_params=100 + index, val={'iou': 0.5}, test={'iou': 0.3},
                initial_parameter_sha256='same-initial-' + str(seed),
                fit_names=['fit'], val_names=['val'], test_names=['test'], normalization={'fit_only': True}))
    return dict(records=records)


def test_finalists_use_validation_and_ignore_test_scores():
    summary = complete_summary()
    for row in summary['records']:
        if row['id'] == 'local_quantum_xz_zz':
            row['val']['iou'] = 0.7
        if row['id'] == 'local_conv':
            row['val']['iou'] = 0.6
        row['test']['iou'] = 1.0 if row['id'] in ('local_quantum_z', 'local_sincos') else 0.0
    assert [spec['id'] for spec in study.finalist_specs(summary)] == ['local_quantum_xz_zz', 'local_conv']


def test_finalist_ties_prefer_z_and_smaller_classical_core():
    summary = complete_summary()
    for row in summary['records']:
        if row['id'] == 'local_sincos':
            row['total_params'] = 90
    assert [spec['id'] for spec in study.finalist_specs(summary)] == ['local_quantum_z', 'local_sincos']


@pytest.mark.parametrize('damage', ['missing_seed', 'missing_model', 'duplicate'])
def test_incomplete_finalist_cohort_rejected(damage):
    summary = complete_summary()
    if damage == 'missing_seed':
        summary['records'].pop()
    elif damage == 'missing_model':
        summary['records'] = [row for row in summary['records'] if row['id'] != 'local_sincos']
    else:
        summary['records'].append(copy.deepcopy(summary['records'][0]))
    with pytest.raises(ValueError):
        study.finalist_specs(summary)


def test_paired_statistics_preserve_seed_differences_and_rate_confounds():
    summary = complete_summary()
    for row in summary['records']:
        if row['id'] == 'local_quantum_xz_zz':
            row['test']['iou'] += {42: 0.01, 77: -0.01, 123: 0.03}[row['seed']]
            row['learning_rate'] = 0.003
    result = next(row for row in study.paired_analysis(summary['records'])
                  if row['left'] == 'local_quantum_xz_zz' and row['right'] == 'local_quantum_z' and row['split'] == 'test')
    assert result['mean'] == pytest.approx(1.0)
    assert result['std'] == pytest.approx(2.0)
    assert result['wins'] == 2
    assert result['n'] == 3
    assert not result['same_learning_rates']


@pytest.mark.parametrize('key', ['fit_names', 'val_names', 'test_names', 'normalization'])
def test_analysis_rejects_unpaired_data(key):
    records = complete_summary()['records']
    records[0][key] = {'changed': True} if key == 'normalization' else ['changed']
    with pytest.raises(ValueError, match='Unpaired'):
        study.paired_analysis(records)


def test_duration_analysis_uses_only_selected_quantum_classical_pair():
    selected = {'local_quantum_z', 'local_conv'}
    records = [row for row in complete_summary()['records'] if row['id'] in selected]
    analysis = study.paired_analysis(records)
    assert len(analysis) == 2
    assert {row['split'] for row in analysis} == {'val', 'test'}
    assert all((row['left'], row['right']) == ('local_quantum_z', 'local_conv') for row in analysis)


def tuning_records():
    records = []
    for row in complete_summary()['records']:
        if row['seed'] == 42:
            for rate in (0.001, 0.003):
                candidate = copy.deepcopy(row)
                candidate['learning_rate'] = rate
                if candidate['id'] == 'local_quantum_xz_zz':
                    candidate['val']['iou'] += 0.01 if rate == 0.001 else -0.02
                records.append(candidate)
    return records


def test_common_rate_readout_contrasts_use_only_seed42_validation():
    records = tuning_records()
    for row in records:
        row['test']['iou'] = 1.0 if row['id'] == 'local_quantum_xz_zz' else 0.0
    result = study.fixed_rate_readout_contrasts(records)
    assert [row['difference_pp'] for row in result] == pytest.approx([1.0, -2.0])
    assert [row['learning_rate'] for row in result] == [0.001, 0.003]
    assert all(row['seed'] == 42 and row['n'] == 1 for row in result)


@pytest.mark.parametrize('damage', ['missing', 'whole_readout_missing', 'duplicate', 'cohort', 'initial'])
def test_common_rate_readout_contrasts_reject_invalid_pairing(damage):
    records = tuning_records()
    target = next(row for row in records if row['id'] == 'local_quantum_xz_zz')
    if damage == 'missing':
        records.remove(target)
    elif damage == 'whole_readout_missing':
        records = [row for row in records if row['id'] != 'local_quantum_xz_zz']
    elif damage == 'duplicate':
        records.append(copy.deepcopy(target))
    elif damage == 'cohort':
        target['val_names'] = ['different']
    else:
        target['initial_parameter_sha256'] = 'different'
    with pytest.raises(ValueError):
        study.fixed_rate_readout_contrasts(records)


def test_duration_extension_explicitly_has_only_one_readout():
    records = [row for row in tuning_records() if row['id'] != 'local_quantum_xz_zz']
    assert study.fixed_rate_readout_contrasts(records, require_both=False) == []

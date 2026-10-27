"""Additional-control selection must remain independent of test outcomes."""
import pytest

from scripts.check_spatial_control_trigger import decide


def rows(global_gains=(0.01, 0.01, 0.01), patch_gains=(0.009, 0.009, 0.009)):
    result = []
    for mode, gains in [('global', global_gains), ('patch4x4', patch_gains)]:
        for seed, gain in zip((42, 77, 123), gains):
            result.append(dict(id='mlp_' + mode, seed=seed, val={'iou': 0.5}))
            result.append(dict(id='quantum_' + mode, seed=seed, val={'iou': 0.5 + gain},
                               diagnostics={'movement': {'circuit_angles': {'l2': 0.1}}}))
    return result


def test_largest_qualified_validation_gain_selects_control_mode():
    decision = decide(rows())
    assert decision['ready'] is True
    assert decision['selected_mode'] == 'global'
    assert decision['modes']['global']['mean'] == pytest.approx(0.01)


def test_mean_alone_cannot_override_win_count_or_missing_learning():
    records = rows(global_gains=(0.05, -0.01, -0.01), patch_gains=(0.0, 0.0, 0.0))
    assert decide(records)['selected_mode'] is None
    records = rows()
    for record in records:
        if record['id'].startswith('quantum_') and record['seed'] == 77:
            record['diagnostics']['movement']['circuit_angles']['l2'] = 0
    assert decide(records)['selected_mode'] is None


def test_missing_pairs_wait_and_test_input_is_rejected():
    records = rows()
    assert decide(records[:-1])['ready'] is False
    records[0]['test'] = {'iou': 0.99}
    with pytest.raises(ValueError, match='validation-only'):
        decide(records)


def test_fixed_three_seed_rule_rejects_another_study_design():
    with pytest.raises(ValueError, match='three study seeds'):
        decide(rows(), seeds=(42, 77))

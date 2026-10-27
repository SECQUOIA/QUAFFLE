"""Paired control contrasts must keep interventions, cohorts and arithmetic honest."""
import copy

import pytest

from scripts.summarize_quantum_training_controls import compare, report


def summaries():
    settings = dict(seeds=[42, 77, 123], learning_rates=[0.001, 0.003], epochs=60,
                    eval_every=2, fit_cap=128, batch_size=16, image_size=64, device='cuda')
    plan = dict(settings=settings, source_revision='original', input_identity={'raw': 'same'})
    decision = dict(ready=True, selected_mode='global')
    control_plan = dict(settings=copy.deepcopy(settings), experiment_context=dict(
        validation_control_decision=decision, baseline_source_revision='original', baseline_input_identity={'raw': 'same'}))
    environment = dict(python='fixture', packages={}, device='cuda', gpu='fixture',
                       gpu_allocator_fraction=0.12, threads=2)
    plan.update(copy.deepcopy(environment))
    control_plan.update(copy.deepcopy(environment))
    baseline, controls = dict(plan=plan, records=[]), dict(plan=control_plan, records=[])
    for seed, gain in zip(settings['seeds'], [0.03, 0.06, 0.0]):
        cfg = dict(quantum_channels=4, n_qubits=4, q_layers=2, q_share_blocks=True, q_spatial_mode='global')
        for name, value, control in [('quantum_global', 0.5 + gain, None), ('mlp_global', 0.4, None),
                ('quantum_frozen_global', 0.5, 'frozen'), ('quantum_no_entanglement_global', 0.5, 'no_entanglement')]:
            row = dict(id=name, seed=seed, total_params=1000, trainable_params=968 if control == 'frozen' else 1000,
                learning_rate=0.003 if control == 'no_entanglement' else 0.001,
                initial_parameter_sha256='initial-' + str(seed), fit_names=['fit-' + str(seed)],
                val_names=['val-' + str(seed)], test_names=['heldout'], normalization={'seed': seed},
                settings=dict(cfg, **({'q_training_control': control} if control else {})),
                diagnostics={'movement': {'circuit_angles': {'l2': 0}, 'input_scales': {'l2': 0}, 'backbone': {'l2': 1}}},
                test=dict(iou=value, optical_subset_iou=value, sar_subset_iou=value))
            (controls if control else baseline)['records'].append(row)
    return baseline, controls


def test_control_report_uses_paired_differences_and_keeps_trainable_counts_separate():
    result = compare(*summaries())
    contrast = result['paired_iou'][0]
    assert contrast['mean'] == pytest.approx(0.03)
    assert contrast['std'] == pytest.approx(0.03)
    assert contrast['wins'] == 2
    assert '968' in report(result)
    assert '| 0.003 |' in report(result)
    assert '+3.000 ± 3.000' in report(result)


@pytest.mark.parametrize('change, message', [
    ('cohort', 'Unpaired'), ('initial', 'initialization'), ('movement', 'Frozen circuit'),
    ('config', 'configuration'), ('budget', 'budget'), ('duplicate', 'Duplicate'), ('runtime', 'environment'),
    ('rate', 'Selected rate'),
])
def test_control_report_rejects_broken_comparisons(change, message):
    baseline, controls = summaries()
    row = controls['records'][0]
    if change == 'cohort':
        row['fit_names'] = ['different']
    elif change == 'initial':
        row['initial_parameter_sha256'] = 'different'
    elif change == 'movement':
        row['diagnostics']['movement']['circuit_angles']['l2'] = 0.1
    elif change == 'config':
        row['settings']['q_spatial_mode'] = 'patch4x4'
    elif change == 'budget':
        controls['plan']['settings']['epochs'] = 120
    elif change == 'duplicate':
        controls['records'].append(row)
    elif change == 'runtime':
        controls['plan']['packages']['torch'] = 'different'
    elif change == 'rate':
        row['learning_rate'] = 0.003
    with pytest.raises(ValueError, match=message):
        compare(baseline, controls)

#!/usr/bin/env python3
"""Per-angle diagnostics on a fixed optimization-only probe, never test images.

Run after both study phases finish, in a CPU memory scope with CUDA hidden.
Gradients are a local derivative of Dice+CE on the first eight saved optimization
tiles, not accumulated training gradients or proof of a parameter's usefulness.
Frozen parameters retain absent gradients; their movement must be exactly zero.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_local_spatial_study as collection
from scripts import run_real_spatial_pilot as pilot


def gradient_record(parameter, initial, *, threshold=1e-10):
    """Preserve every coefficient; distinguish absent from numerical-zero grads."""
    movement = parameter.detach().cpu() - initial.detach().cpu()
    gradient = parameter.grad.detach().cpu() if parameter.grad is not None else None
    return dict(initial=initial.detach().cpu().tolist(), selected=parameter.detach().cpu().tolist(),
        movement=movement.tolist(), movement_l2=float(movement.square().sum().sqrt()),
        requires_grad=parameter.requires_grad, gradient=None if gradient is None else gradient.tolist(),
        gradient_nonzero_exact=None if gradient is None else (gradient != 0).tolist(),
        gradient_above_threshold=None if gradient is None else (gradient.abs() > threshold).tolist(),
        gradient_max_abs=None if gradient is None else float(gradient.abs().max()),
        threshold=threshold)


def probe_model(model, images, masks, initial_core):
    import torch
    from utilsTorch.local_spatial import LocalSpatialBottleneckBlock
    from utilsTorch.utilsTorch import segmentation_loss
    blocks = [block for block in model.modules() if isinstance(block, LocalSpatialBottleneckBlock)]
    collection.require(len(blocks) == 1 and blocks[0].kind == 'quantum', 'Expected one shared quantum local block')
    block = blocks[0]
    model.eval()
    model.zero_grad()
    saturations = []
    def capture(module, inputs, output):
        saturations.append(module.angle_diagnostics(inputs[0]))
    handle = block.register_forward_hook(capture)
    try:
        loss = segmentation_loss(model(images), masks, loss_mode='dice_ce')
        collection.require(bool(torch.isfinite(loss)), 'Nonfinite optimization-probe loss')
        loss.backward()
    finally:
        handle.remove()
    parameters = dict(block.named_parameters())
    names = ('quantum_params.0', 'input_scales.0')
    values = {name: gradient_record(parameters[name], initial_core[name]) for name in names}
    collection.require(all(torch.isfinite(parameters[name]).all() and
        (parameters[name].grad is None or torch.isfinite(parameters[name].grad).all()) for name in names),
        'Nonfinite individual-angle diagnostic')
    if block.q_training_control == 'frozen':
        collection.require(all(value['movement_l2'] == 0 and value['gradient'] is None for value in values.values()),
                           'Frozen circuit moved or acquired a training gradient')
    return dict(loss=float(loss.detach()), parameter_diagnostics=values, input_angle_saturation=saturations,
        angle_axes=['layer', 'spatial_wire_row_major', 'Rot_phi_theta_omega'],
        scale_axes=['layer', 'spatial_wire_row_major'],
        gradient_policy='Eval-mode Dice+CE derivative on fixed first-eight optimization tiles; frozen gradients absent, not counterfactually enabled')


def load_probe_dataset(row, data_root):
    """Use the complete original pool for its frozen fit-only normalization."""
    import torch
    from tools import training_common as tc
    cfg = copy.deepcopy(row['settings'])
    cfg['data_root'] = str(data_root)
    _, images_dir, masks_dir, _, _ = tc.resolve_data_dirs('combined', str(data_root))
    dataset = tc.load_dataset(images_dir, masks_dir, cfg)
    collection.require(dataset.norm_provenance == row['normalization'], 'Probe dataset normalization differs from checkpoint')
    indices = row['fit_indices'][:8]
    names = [dataset.names[index] for index in indices]
    collection.require(names == row['fit_names'][:8] and not set(names) & set(row['val_names']),
                       'Probe is not the first eight frozen optimization tiles')
    images, masks = zip(*(dataset[index] for index in indices))
    return torch.stack(images), torch.stack(masks), names


def audit_angles(pilot_dir, extension_dir, data_root):
    import torch
    from tools import training_common as tc
    from utilsTorch.utilsTorch import QVUNetSegmentation
    result = collection.audit_study(pilot_dir, extension_dir)
    identity = pilot.verify_inputs(data_root)
    collection.require(identity == result['pilot60']['summary']['plan']['input_identity'], 'Raw probe data differ from frozen study inputs')
    records = []
    probes = {}
    for phase, directory in (('pilot60', Path(pilot_dir)), ('extend120', Path(extension_dir))):
        for row in result[phase]['summary']['records']:
            if row['model_type'] != 'quantum':
                continue
            if row['seed'] not in probes:
                probes[row['seed']] = load_probe_dataset(row, data_root)
            images, masks, names = probes[row['seed']]
            collection.require(names == row['fit_names'][:8], 'Probe cohort differs between study phases/configurations')
            tc.seed_everything(row['seed'])
            model = tc.build_model(QVUNetSegmentation, copy.deepcopy(row['settings'])).to('cpu')
            collection.require(collection.initial_parameter_digest(model) == row['initial_parameter_sha256'], 'Diagnostic initialization differs')
            block = model.model.vertex.q_block
            initial_core = {name: value.detach().clone() for name, value in block.named_parameters()
                            if name in ('quantum_params.0', 'input_scales.0')}
            checkpoint = collection.safe_relative(directory, row['checkpoint'])
            expected_hash = result[phase]['checkpoints'][row['checkpoint']]['sha256']
            collection.require(collection.digest(checkpoint) == expected_hash, 'Checkpoint changed during angle audit')
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['best_state_dict'])
            diagnostics = probe_model(model, images, masks, initial_core)
            records.append(dict(phase=phase, id=row['id'], seed=row['seed'], learning_rate=row['learning_rate'],
                best_epoch=row['best_epoch'], checkpoint_sha256=expected_hash,
                initial_parameter_sha256=row['initial_parameter_sha256'], probe_names=names,
                probe_tensor_sha256=hashlib.sha256(images.numpy().tobytes() + masks.numpy().tobytes()).hexdigest(),
                **diagnostics))
    collection.require(pilot.verify_inputs(data_root) == identity, 'Raw inputs changed during the read-only diagnostic')
    return dict(summary_sha256={phase: result[phase]['summary_sha256'] for phase in ('pilot60', 'extend120')},
        source_hashes=result['pilot60']['summary']['plan']['source_hashes'], input_identity=identity,
        runtime=dict(**collection.current_runtime(), device='cpu', threads=2),
        diagnostic_source_sha256=collection.digest(__file__), collector_source_sha256=collection.digest(collection.__file__),
        records=records, interpretation='Movement and one probe-batch derivative per parameter; not accumulated training gradients, general usefulness, or accuracy evidence')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--extension-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    collection.require(not args.output.exists(), 'Refusing to overwrite an existing angle audit')
    for directory in (args.pilot_dir, args.extension_dir, args.data_root):
        collection.require(not args.output.resolve().is_relative_to(directory.resolve()), 'Angle audit must be written outside immutable inputs')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    result = audit_angles(args.pilot_dir, args.extension_dir, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    collection.write(args.output, result)
    print('Individual-angle optimization-probe records:', len(result['records']))
    return result


if __name__ == '__main__':
    main()

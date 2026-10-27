#!/usr/bin/env python3
"""Compare a uniformly narrowed ordinary U-Net with a full quantum bottleneck.

This is a fixed, small synthetic experiment. Width selection uses parameter counts
before training. One-group controls isolate the GroupNorm change required by the
nearest integer U-Net width; four-group references connect to the earlier sweep.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def base_config(width=8, groups=4):
    return dict(
        base_channels=width, init_dim=width, dim_mults=(1, 2, 4, 8), out_dim=2,
        resnet_block_groups=groups, quantum_channels=64, quantum_backend="pennylane",
        q_ansatz="reupload", q_layers=2, n_qubits=4, q_backend="pennylane",
        q_sim_device="cpu", q_share_blocks=True, q_spatial_mode="global",
        skip_modulation="affine", decoder_skip_gate="none", use_time_emb=False,
        backbone_conv_mode="ws_conv", combined_input_mode="shared3",
        ablation_mode="mlp", ablation_mlp_mode="full_rank", loss="dice_ce",
        dataset="optical", val_fraction=0.15, fit_only_norm=True, augment=False,
        select_metric="iou", max_samples=None, legacy_data=False,
    )


def choose_width(candidates, target):
    """Choose by absolute count difference, preferring the smaller width on ties."""
    if not candidates or target <= 0:
        raise ValueError("Positive target and a nonempty candidate set are required")
    return min(candidates, key=lambda width: (abs(candidates[width] - target), width))


def make_plan():
    from tools import training_common as tc
    from utilsTorch.utilsTorch import UNetSegmentation, QVUNetSegmentation

    def count(cls, cfg):
        model = tc.build_model(cls, dict(cfg, seed=42))
        return sum(p.numel() for p in model.parameters())

    target = count(QVUNetSegmentation, base_config())
    candidates = {width: count(UNetSegmentation, base_config(width, 1)) for width in range(4, 9)}
    width = choose_width(candidates, target)
    specs = []
    for groups in (1, 4):
        for kind in ("classical", "quantum", "ablation"):
            specs.append(dict(id=f"{kind}_bc8_g{groups}", model_type=kind,
                              config=base_config(8, groups)))
    specs.insert(0, dict(id=f"classical_bc{width}_g1", model_type="classical",
                         config=base_config(width, 1)))
    return dict(target_params=target, candidate_params=candidates, selected_width=width,
                selected_params=candidates[width],
                relative_parameter_gap=(candidates[width] - target) / target, specs=specs)


def _stat(values):
    return dict(mean=statistics.mean(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.0, n=len(values))


def summarize(records):
    groups = {}
    for record in records:
        group = groups.setdefault(record["id"], {})
        if record["seed"] in group:
            raise ValueError("Duplicate model/seed result")
        group[record["seed"]] = record
    seed_sets = [set(g) for g in groups.values()]
    if not seed_sets or any(s != seed_sets[0] for s in seed_sets):
        raise ValueError("All models must have identical seed coverage")
    for seed in seed_sets[0]:
        matched = [group[seed] for group in groups.values()]
        for key in ("train_indices", "val_indices", "train_names", "test_names"):
            if any(r[key] != matched[0][key] for r in matched):
                raise ValueError(f"Unpaired {key} for seed {seed}")
    rows = []
    for name, group in groups.items():
        runs = list(group.values())
        rows.append(dict(id=name, total_params=runs[0]["total_params"],
                         test_iou=_stat([r["test"]["iou"] for r in runs]),
                         test_dice=_stat([r["test"]["dice"] for r in runs]),
                         zero_delta_iou=_stat([r["test"]["iou"] - r["knockout"]["zero"]["iou"]
                                              for r in runs]) if runs[0]["knockout"] else None))
    slim = next(name for name in groups if name.startswith("classical_bc") and name != "classical_bc8_g1" and name.endswith("_g1"))
    pairs = [(name, slim) for name in ("classical_bc8_g1", "quantum_bc8_g1", "ablation_bc8_g1")]
    pairs += [(f"{kind}_bc8_g1", f"{kind}_bc8_g4") for kind in ("classical", "quantum", "ablation")]
    pairs += [("quantum_bc8_g4", "ablation_bc8_g4"), ("quantum_bc8_g1", "ablation_bc8_g1")]
    contrasts = []
    for left, right in pairs:
        differences = {str(seed): groups[left][seed]["test"]["iou"] - groups[right][seed]["test"]["iou"]
                       for seed in sorted(groups[left])}
        contrasts.append(dict(left=left, right=right, unit="IoU", differences=differences,
                              **_stat(list(differences.values()))))
    return dict(rows=rows, paired_comparisons=contrasts)


def format_report(summary):
    plan = summary["plan"]
    lines = ["# Ordinary U-Net parameter-budget comparison", "",
             f"Seeds: {summary['seeds']}. One synthetic dataset, generated with seed {summary['data_seed']}.", "",
             f"Target: {plan['target_params']:,} parameters. Selected U-Net: {plan['selected_params']:,} "
             f"({100 * plan['relative_parameter_gap']:+.2f}%). Width chosen by counts before training.", "",
             "| Model | Parameters | Test IoU mean ± sample SD | On − zero IoU |",
             "| --- | ---: | ---: | ---: |"]
    for row in summary["analysis"]["rows"]:
        ko = row["zero_delta_iou"]
        ko_text = f"{ko['mean']:+.4f}" if ko else "—"
        st = row["test_iou"]
        lines.append(f"| {row['id']} | {row['total_params']:,} | {st['mean']:.4f} ± {st['std']:.4f} | {ko_text} |")
    lines += ["", "Paired differences below are IoU percentage points (left minus right).", "",
              "| Left | Right | Mean ± sample SD |", "| --- | --- | ---: |"]
    for c in summary["analysis"]["paired_comparisons"]:
        lines.append(f"| {c['left']} | {c['right']} | {100*c['mean']:+.3f} ± {100*c['std']:.3f} |")
    lines += ["", "`g1`/`g4` indicate GroupNorm groups; `bc` is the base channel width.",
              "The one-group comparisons control normalization across widths. Different-width models cannot share identical initial tensors.",
              "All fits use Adam, Dice + cross-entropy loss, the same paired splits/shuffles, and validation-IoU checkpoint selection.",
              "This tiny synthetic experiment does not establish real-data superiority, statistical equivalence, or quantum advantage."]
    return "\n".join(lines) + "\n"


def run_one(spec, seed, args, data_root, run_dir, device):
    import torch
    from tools import training_common as tc
    from utilsTorch.utilsTorch import (
        UNetSegmentation, QVUNetSegmentation, ClassicalAblationUNetSegmentation,
        train_one_epoch, evaluate_model,
    )

    started = time.perf_counter()
    cfg = dict(spec["config"], seed=seed, data_root=str(data_root), image_size=args.image_size,
               batch_size=args.batch_size, learning_rate=args.learning_rate, num_epochs=args.epochs)
    tc.seed_everything(seed)
    _, train_img, train_mask, test_img, test_mask = tc.resolve_data_dirs("optical", str(data_root))
    full = tc.load_dataset(train_img, train_mask, cfg)
    train_idx, val_idx, _ = tc.holdout_validation(full, seed, cfg["val_fraction"])
    train = tc.make_loader(torch.utils.data.Subset(full, train_idx), args.batch_size, True, seed)
    val = tc.make_loader(torch.utils.data.Subset(full, val_idx), args.batch_size, False, seed)
    test_data = tc.load_dataset(test_img, test_mask, cfg)
    test = tc.make_loader(test_data, args.batch_size, False, seed)
    classes = dict(classical=UNetSegmentation, quantum=QVUNetSegmentation, ablation=ClassicalAblationUNetSegmentation)
    model = tc.build_model(classes[spec["model_type"]], cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_score, best_state, best_epoch, best_metrics = -float("inf"), None, None, None
    for epoch in range(args.epochs + 1):
        if epoch:
            train_one_epoch(model, train, optimizer, device)
        metrics = evaluate_model(model, val, device)
        history.append(dict(epoch=epoch, **metrics))
        if tc.is_better_val_score(metrics["iou"], best_score, "iou"):
            best_score, best_epoch, best_metrics = metrics["iou"], epoch, dict(metrics)
            best_state = tc.clone_state_dict_cpu(model.state_dict())
    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test, device)
    knockout = tc.knockout_eval(model, test, device, evaluate_model, knockout_seed=seed) if spec["model_type"] != "classical" else None
    count = sum(p.numel() for p in model.parameters())
    record = dict(id=spec["id"], seed=seed, model_type=spec["model_type"], total_params=count,
                  settings={k: v for k, v in cfg.items() if k not in ("data_root", "_resolved_norm_provenance")},
                  train_indices=list(map(int, train_idx)), val_indices=list(map(int, val_idx)),
                  train_names=list(full.names), test_names=list(test_data.names), val=best_metrics,
                  test=test_metrics, knockout=knockout, best_epoch=best_epoch, history=history,
                  elapsed_s=time.perf_counter() - started)
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    tc.atomic_torch_save(dict(model_state_dict=best_state, settings=record["settings"],
                             model_type=spec["model_type"], best_epoch=best_epoch), str(run_dir / "checkpoint.pt"))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="New empty experiment directory; existing data is never removed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 77, 123])
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--n-train", type=int, default=16)
    parser.add_argument("--n-test", type=int, default=6)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or args.epochs < 1:
        parser.error("Seeds must be unique and epochs must be positive")
    if args.work_dir.exists() and (not args.work_dir.is_dir() or any(args.work_dir.iterdir())):
        parser.error("Work directory must be new or empty; existing runs are preserved")

    import torch
    from tools import make_mini_flood_dataset
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    device = torch.device(args.device)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(0.12)
    plan = make_plan()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        source_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip(),
        python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ("torch", "pennylane", "numpy", "scipy", "rasterio")},
        source_hashes={p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in (
            "scripts/run_parameter_budget_comparison.py", "utilsTorch/unetTorch.py", "utilsTorch/utilsTorch.py",
            "tools/training_common.py", "tools/make_mini_flood_dataset.py", "flood_data.py")},
        device=str(device), gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        torch_threads=2, torch_interop_threads=2, gpu_allocator_fraction=0.12 if args.device == "cuda" else None,
    )
    (args.work_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    data_root = args.work_dir / "mini_data"
    make_mini_flood_dataset.generate_mini_flood_dataset(
        str(data_root), n_train=args.n_train, n_test=args.n_test, image_size=args.image_size,
        seed=args.data_seed, seeds=args.seeds)
    hashes = {str(p.relative_to(data_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(data_root.rglob("*")) if p.is_file()}
    (args.work_dir / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    records = []
    for spec in plan["specs"]:
        for seed in args.seeds:
            print(f"Running {spec['id']} seed={seed}", flush=True)
            record = run_one(spec, seed, args, data_root, args.work_dir / "runs" / f"{spec['id']}_seed{seed}", device)
            records.append(record)
            summary = dict(plan=plan, seeds=args.seeds, data_seed=args.data_seed,
                           protocol={k: v for k, v in vars(args).items() if k != "work_dir"}, records=records)
            # Keep every completed fit if a later configuration fails.
            (args.work_dir / "partial_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    summary["analysis"] = summarize(records)
    (args.work_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = format_report(summary)
    (args.work_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

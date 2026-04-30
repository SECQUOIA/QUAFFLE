#!/usr/bin/env python3
"""
Shared utility to append run results to a central Excel summary file.
Each demo calls save_results_to_excel() at the end of training.
Results are split into per-dataset files: results_summary_optical.xlsx,
results_summary_sar.xlsx, results_summary_combined.xlsx.
"""

import os
from datetime import datetime


_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLUMNS = [
    "Timestamp",
    "Demo",
    "Framework",
    "Training",
    "Variant",
    "Dataset",
    # Validation metrics
    "Val_Loss",
    "Val_Accuracy",
    "Val_AUC",
    "Val_IoU",
    "Val_Dice",
    # Test metrics
    "Test_Loss",
    "Test_Accuracy",
    "Test_AUC",
    "Test_IoU",
    "Test_Dice",
    # Config
    "Partition",
    "Dirichlet_Alpha",
    "Epochs_Steps_Rounds",
    "Num_Clients",
    "Local_Epochs",
    "Base_Channels",
    "Quantum_Channels",
    "Batch_Size",
    "Learning_Rate",
    "Total_Params",
]


def save_results_to_excel(demo_name, framework, training, variant, val_metrics, test_metrics, config,
                          dataset='optical'):
    """
    Append one row of results to results/results_summary_{dataset}.xlsx.

    Args:
        demo_name:    Human-readable name, e.g. "Centralized Classical Torch"
        framework:    "torch" | "jax"
        training:     "centralized" | "federated"
        variant:      "classical" | "quantum"
        val_metrics:  dict with loss/accuracy/auc/iou/dice, or None
        test_metrics: dict with loss/accuracy/auc/iou/dice, or None
        config:       dict or ConfigDict with training hyperparameters
        dataset:      "optical" | "sar" | "combined"
    """
    try:
        import openpyxl
    except ImportError:
        print("Warning: openpyxl not installed — skipping Excel export. Install with: pip install openpyxl")
        return

    excel_path = os.path.join(_RESULTS_DIR, f"results_summary_{dataset}.xlsx")
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # File lock serializes concurrent writes from jobs finishing at the same time.
    # Without this, two jobs can load + modify + save simultaneously and produce
    # a corrupt .xlsx (happened on 2026-04-20: results_summary_optical.xlsx had to
    # be repaired after concurrent fed_*_optical jobs collided).
    import fcntl, time
    lock_path = excel_path + ".lock"
    lock_fd = open(lock_path, "w")
    for _ in range(60):  # up to ~60s of contention
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(1)
    else:
        print("Warning: could not acquire Excel lock after 60s — writing anyway")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # blocking as last resort

    try:
        _save_locked(excel_path, demo_name, framework, training, variant, dataset,
                     val_metrics, test_metrics, config)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _save_locked(excel_path, demo_name, framework, training, variant, dataset,
                 val_metrics, test_metrics, config):
    """Actual Excel write path — caller must hold the file lock."""
    import openpyxl

    def _get(d, *keys, default=None):
        """Try multiple keys, return first match."""
        for k in keys:
            try:
                v = d[k] if isinstance(d, dict) else getattr(d, k, None)
                if v is not None:
                    return v
            except (KeyError, TypeError):
                pass
        return default

    seed = _get(config, "seed")
    sheet_name = f"Seed_{seed}" if seed is not None else "Results"

    from openpyxl.styles import Font
    if os.path.exists(excel_path):
        wb = openpyxl.load_workbook(excel_path)
    else:
        wb = openpyxl.Workbook()
        # Remove default empty sheet
        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]

    # Get or create the target sheet
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(title=sheet_name)
        ws.append(COLUMNS)
        for cell in ws[1]:
            cell.font = Font(bold=True)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        demo_name,
        framework,
        training,
        variant,
        dataset,
        # Validation
        round(float(val_metrics["loss"]), 6) if val_metrics else None,
        round(float(val_metrics["accuracy"]), 6) if val_metrics else None,
        round(float(val_metrics["auc"]), 6) if val_metrics else None,
        round(float(val_metrics["iou"]), 6) if val_metrics else None,
        round(float(val_metrics["dice"]), 6) if val_metrics else None,
        # Test
        round(float(test_metrics["loss"]), 6) if test_metrics else None,
        round(float(test_metrics["accuracy"]), 6) if test_metrics else None,
        round(float(test_metrics["auc"]), 6) if test_metrics else None,
        round(float(test_metrics["iou"]), 6) if test_metrics else None,
        round(float(test_metrics["dice"]), 6) if test_metrics else None,
        # Config
        _get(config, "partition", default="centralized"),
        _get(config, "dirichlet_alpha"),
        _get(config, "num_epochs", "num_train_steps", "fed_rounds", "num_train_steps"),
        _get(config, "num_clients"),
        _get(config, "local_epochs"),
        _get(config, "base_channels"),
        _get(config, "quantum_channels"),
        _get(config, "batch_size"),
        _get(config, "learning_rate"),
        None,  # Total_Params — optionally passed via config['total_params']
    ]

    # Allow callers to pass total_params via config
    if _get(config, "total_params") is not None:
        row[-1] = _get(config, "total_params")

    ws.append(row)
    wb.save(excel_path)
    print(f"Results saved to Excel: {excel_path}")

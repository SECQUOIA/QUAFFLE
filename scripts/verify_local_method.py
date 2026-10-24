#!/usr/bin/env python3
"""End-to-end local workstation verification harness for QUAFFLE.

Generates a temporary synthetic 12-band Optical + 6-band SAR mini-flood dataset
in scratch/tmp and verifies:
  1. Mini-dataset generation, split manifests (conflict_policy='drop'), and
     fit-only norm_stats.json (excluding validation holdout).
  2. Centralized training, interrupted checkpoint resume (--debug-stop-after-epoch),
     and --quantum-knockout-eval across classical, ablation (legacy & reupload),
     and quantum (legacy & reupload).
  3. Federated training (random & dirichlet), interrupted checkpoint resume
     (--debug-stop-after-round), and --quantum-knockout-eval across classical,
     ablation (legacy & reupload), and quantum (legacy & reupload).

Completes in < 60 s on a single workstation CPU/GPU (including module imports).
"""

import time

_PROCESS_START = time.perf_counter()

import argparse  # noqa: E402
from contextlib import contextmanager  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ablation_studies  # noqa: E402
import centralized_demoTorchClassical  # noqa: E402
import centralized_demoTorchQ  # noqa: E402
import federated_demoTorchClassical  # noqa: E402
import federated_demoTorchQ  # noqa: E402
import flood_data  # noqa: E402
from tools import dedup_splits, training_common as tc  # noqa: E402
from tools.make_mini_flood_dataset import generate_mini_flood_dataset  # noqa: E402

_IMPORT_ELAPSED = time.perf_counter() - _PROCESS_START


@contextmanager
def _patched_argv(argv):
    old_argv = list(sys.argv)
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def _run_entry(name, fn, argv, *, expect_exit_code=None):
    t0 = time.perf_counter()
    caught_code = None
    with _patched_argv(argv):
        try:
            fn()
        except SystemExit as exc:
            caught_code = exc.code
            if expect_exit_code is None or caught_code != expect_exit_code:
                raise
    if expect_exit_code is not None:
        assert caught_code == expect_exit_code, (
            f"{name}: expected SystemExit({expect_exit_code}), got {caught_code!r}"
        )
    dt = time.perf_counter() - t0
    print(f"[PASS] {name} ({dt:.2f}s)")
    return dt


def _assert_metrics_json(
    run_dir,
    *,
    expect_knockout=False,
    expect_resumed_round=None,
    expect_resumed_epoch=None,
    expect_epochs_trained=None,
):
    summary_path = os.path.join(run_dir, "results", "summary.txt")
    metrics_path = os.path.join(run_dir, "results", "metrics.json")
    assert os.path.isfile(summary_path), f"Missing summary.txt in {run_dir}"
    assert os.path.getsize(summary_path) > 0, f"Empty summary.txt in {run_dir}"
    assert os.path.isfile(metrics_path), f"Missing metrics.json in {run_dir}"
    with open(metrics_path) as f:
        m = json.load(f)
    assert m.get("schema") == "quaffle-metrics-v1", m.get("schema")
    for split_key in ("val", "test"):
        block = m.get(split_key)
        assert isinstance(block, dict), f"Missing {split_key} metrics in {metrics_path}"
        for metric_key in ("loss", "accuracy", "auc", "iou", "dice"):
            assert metric_key in block, f"Missing {split_key}.{metric_key} in {metrics_path}"
    if expect_knockout:
        ko = m.get("knockout")
        assert isinstance(ko, dict), f"Expected knockout metrics in {metrics_path}"
        for mode in ("zero", "random"):
            assert isinstance(ko.get(mode), dict), f"Missing knockout[{mode!r}] in {metrics_path}"
            assert "iou" in ko[mode] and "auc" in ko[mode]
        with open(summary_path) as f:
            stext = f.read()
        assert "Quantum Knockout (zero)" in stext
        assert "Quantum Knockout (random)" in stext
    if expect_resumed_round is not None:
        assert m.get("resumed_from_round") == expect_resumed_round, (
            f"Expected resumed_from_round={expect_resumed_round}, got {m.get('resumed_from_round')}"
        )
    if expect_resumed_epoch is not None:
        assert m.get("resumed_from_epoch") == expect_resumed_epoch, (
            f"Expected resumed_from_epoch={expect_resumed_epoch}, got {m.get('resumed_from_epoch')}"
        )
    if expect_epochs_trained is not None:
        assert m.get("epochs_trained_this_run") == expect_epochs_trained, (
            f"Expected epochs_trained_this_run={expect_epochs_trained}, got {m.get('epochs_trained_this_run')}"
        )
    return m


def _assert_centralized_ckpt(run_dir, expected_step_key, expected_step_val, *, expected_target=None):
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    assert os.path.isfile(ckpt_path), f"Missing centralized checkpoint {ckpt_path}"
    ckpt = tc.safe_torch_load(ckpt_path, map_location="cpu")
    assert isinstance(ckpt, dict), f"Unreadable checkpoint {ckpt_path}"
    assert ckpt.get("run_format") == tc.RUN_FORMAT
    assert ckpt.get(expected_step_key) == expected_step_val, (
        f"Expected {expected_step_key}={expected_step_val}, got {ckpt.get(expected_step_key)}"
    )
    fp = ckpt.get("fingerprint")
    assert isinstance(fp, dict), "Missing centralized checkpoint fingerprint"
    if expected_target is not None:
        assert fp.get("epochs_or_steps") == expected_target, (
            f"Expected fingerprint epochs_or_steps={expected_target}, got {fp.get('epochs_or_steps')}"
        )
    assert fp.get("data_root") is not None
    assert isinstance(fp.get("norm_provenance"), dict)
    assert fp["norm_provenance"].get("fit_only") is True
    assert fp["norm_provenance"].get("fit_names_sha256") is not None
    return ckpt


def _assert_federated_ckpt(run_dir, expected_round, *, expected_total_rounds=None):
    ckpt_path = os.path.join(run_dir, tc.FED_CKPT_NAME)
    assert os.path.isfile(ckpt_path), f"Missing federated checkpoint {ckpt_path}"
    ckpt = tc.safe_torch_load(ckpt_path, map_location="cpu")
    assert isinstance(ckpt, dict), f"Unreadable federated checkpoint {ckpt_path}"
    assert ckpt.get("run_format") == tc.RUN_FORMAT
    assert ckpt.get("completed_round") == expected_round, (
        f"Expected completed_round={expected_round}, got {ckpt.get('completed_round')}"
    )
    fp = ckpt.get("fingerprint")
    assert isinstance(fp, dict), "Missing federated checkpoint fingerprint"
    if expected_total_rounds is not None:
        assert fp.get("fed_rounds") == expected_total_rounds, (
            f"Expected fingerprint fed_rounds={expected_total_rounds}, got {fp.get('fed_rounds')}"
        )
    assert fp.get("data_root") is not None
    assert isinstance(fp.get("norm_provenance"), dict)
    assert fp["norm_provenance"].get("fit_only") is True
    return ckpt


def run_verification(work_dir=None, max_seconds=60.0, device="cpu", include_process_start=False):
    t_start = _PROCESS_START if include_process_start else time.perf_counter()
    timings = []
    if include_process_start:
        timings.append(("module_imports", _IMPORT_ELAPSED))

    ctx = (
        tempfile.TemporaryDirectory(prefix="quaffle_verify_")
        if work_dir is None
        else _null_tempdir(work_dir)
    )
    old_env = {
        k: os.environ.get(k)
        for k in (
            "QUAFFLE_INPROCESS_FED",
            "QUAFFLE_SKIP_EXCEL_TRACKER",
            "QUAFFLE_SKIP_CURVE_PLOTS",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    }
    old_num_threads = torch.get_num_threads()
    old_cwd = os.getcwd()
    os.environ["QUAFFLE_INPROCESS_FED"] = "1"
    os.environ["QUAFFLE_SKIP_EXCEL_TRACKER"] = "1"
    os.environ["QUAFFLE_SKIP_CURVE_PLOTS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(min(2, old_num_threads))
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        with ctx as tmp_root:
            os.chdir(tmp_root)
            data_root = os.path.join(tmp_root, "mini_data")
            results_dir = os.path.join(tmp_root, "results")
            os.makedirs(results_dir, exist_ok=True)

            # 1. Generate synthetic mini dataset (12-band Optical + 6-band dB SAR + DEM -9999 nodata)
            t0 = time.perf_counter()
            info = generate_mini_flood_dataset(
                data_root,
                n_train=12,
                n_test=4,
                image_size=32,
                seed=42,
                val_fraction=0.15,
                seeds=(42,),
            )
            for ds_name in ("flood_optical", "flood_SAR", "flood_combined"):
                man = info["manifests"][ds_name]
                assert man["conflict_policy"] == "drop"
                dedup_splits.validate_manifest(man, expected_policy="drop")
            sar_raw = flood_data.read_raw(
                os.path.join(data_root, "flood_SAR", "Training", "images", "000.tif")
            )
            assert float(sar_raw[0].max()) < 0.0 and float(sar_raw[1].max()) < 0.0, (
                "Expected synthetic SAR VV/VH bands in negative dB scale"
            )
            dt = time.perf_counter() - t0
            timings.append(("generate_and_validate_mini_dataset", dt))
            assert os.path.isfile(info["stats_path"])

            common_central = [
                "--data-root", data_root,
                "--results-dir", results_dir,
                "--base-channels", "4",
                "--image-size", "32",
                "--batch-size", "4",
                "--eval-every", "1",
                "--seed", "42",
                "--no-save-samples",
            ]
            common_fed = [
                "--data-root", data_root,
                "--results-dir", results_dir,
                "--base-channels", "4",
                "--image-size", "32",
                "--batch-size", "4",
                "--seed", "42",
                "--num-clients", "2",
                "--clients-per-round", "2",
                "--min-client-samples", "2",
                "--local-epochs", "1",
                "--ckpt-every", "1",
                "--no-save-samples",
            ]

            # 2. Centralized Classical: interrupt after epoch 1 of 2, then resume to epoch 2
            dt = _run_entry(
                "centralized_classical_interrupt_epoch1",
                centralized_demoTorchClassical.main,
                [
                    "centralized_demoTorchClassical.py",
                    "--dataset", "optical",
                    "--epochs", "2",
                    "--debug-stop-after-epoch", "1",
                    *common_central,
                ],
                expect_exit_code=75,
            )
            timings.append(("centralized_classical_interrupt_epoch1", dt))
            c_cls_dir = os.path.join(results_dir, "flood_central_optical_pytorch_classical_seed42")
            ck1 = _assert_centralized_ckpt(c_cls_dir, "epoch", 1, expected_target=2)
            assert len(ck1["train_metrics"]) == 1
            ep1_loss = ck1["train_metrics"][0]["loss"]

            dt = _run_entry(
                "centralized_classical_resume_epoch2",
                centralized_demoTorchClassical.main,
                [
                    "centralized_demoTorchClassical.py",
                    "--dataset", "optical",
                    "--epochs", "2",
                    *common_central,
                ],
            )
            timings.append(("centralized_classical_resume_epoch2", dt))
            ck2 = _assert_centralized_ckpt(c_cls_dir, "epoch", 2, expected_target=2)
            assert len(ck2["train_metrics"]) == 2
            assert ck2["train_metrics"][0]["loss"] == ep1_loss
            _assert_metrics_json(
                c_cls_dir,
                expect_knockout=False,
                expect_resumed_epoch=1,
                expect_epochs_trained=1,
            )

            # 3. Centralized Ablation (legacy & reupload + interrupt/resume + knockout)
            dt = _run_entry(
                "centralized_ablation_legacy_epoch1",
                ablation_studies.main,
                [
                    "ablation_studies.py",
                    "--framework", "torch",
                    "--mode", "centralized",
                    "--dataset", "optical",
                    "--epochs", "1",
                    "--q-ansatz", "legacy",
                    "--quantum-knockout-eval",
                    *common_central,
                ],
            )
            timings.append(("centralized_ablation_legacy_epoch1", dt))
            c_abl_leg_dir = os.path.join(
                results_dir,
                "ablation_centralized_torch_bottleneck_matched_pennylane_legacy_seed42",
            )
            _assert_centralized_ckpt(c_abl_leg_dir, "epoch", 1, expected_target=1)
            _assert_metrics_json(
                c_abl_leg_dir,
                expect_knockout=True,
                expect_resumed_epoch=0,
                expect_epochs_trained=1,
            )

            dt = _run_entry(
                "centralized_ablation_reupload_interrupt_epoch1",
                ablation_studies.main,
                [
                    "ablation_studies.py",
                    "--framework", "torch",
                    "--mode", "centralized",
                    "--dataset", "optical",
                    "--epochs", "2",
                    "--q-ansatz", "reupload",
                    "--q-layers", "1",
                    "--quantum-knockout-eval",
                    "--debug-stop-after-epoch", "1",
                    *common_central,
                ],
                expect_exit_code=75,
            )
            timings.append(("centralized_ablation_reupload_interrupt_epoch1", dt))
            c_abl_dir = os.path.join(
                results_dir,
                "ablation_centralized_torch_bottleneck_matched_pennylane_reupload_L1_seed42",
            )
            ck_abl1 = _assert_centralized_ckpt(c_abl_dir, "epoch", 1, expected_target=2)
            assert len(ck_abl1["train_metrics"]) == 1
            abl_ep1_loss = ck_abl1["train_metrics"][0]["loss"]

            dt = _run_entry(
                "centralized_ablation_reupload_resume_epoch2",
                ablation_studies.main,
                [
                    "ablation_studies.py",
                    "--framework", "torch",
                    "--mode", "centralized",
                    "--dataset", "optical",
                    "--epochs", "2",
                    "--q-ansatz", "reupload",
                    "--q-layers", "1",
                    "--quantum-knockout-eval",
                    *common_central,
                ],
            )
            timings.append(("centralized_ablation_reupload_resume_epoch2", dt))
            ck_abl2 = _assert_centralized_ckpt(c_abl_dir, "epoch", 2, expected_target=2)
            assert len(ck_abl2["train_metrics"]) == 2
            assert ck_abl2["train_metrics"][0]["loss"] == abl_ep1_loss
            _assert_metrics_json(
                c_abl_dir,
                expect_knockout=True,
                expect_resumed_epoch=1,
                expect_epochs_trained=1,
            )

            # 4. Centralized Quantum (legacy SAR & reupload optical + interrupt/resume + knockout)
            dt = _run_entry(
                "centralized_quantum_legacy_sar_step1",
                centralized_demoTorchQ.main,
                [
                    "centralized_demoTorchQ.py",
                    "--dataset", "sar",
                    "--epochs", "1",
                    "--q-ansatz", "legacy",
                    "--quantum-knockout-eval",
                    *common_central,
                ],
            )
            timings.append(("centralized_quantum_legacy_sar_step1", dt))
            c_q_leg_dir = os.path.join(
                results_dir,
                "flood_central_optical_pytorch_pennylane_sar_legacy_seed42",
            )
            _assert_centralized_ckpt(c_q_leg_dir, "step", 1, expected_target=1)
            _assert_metrics_json(
                c_q_leg_dir,
                expect_knockout=True,
                expect_resumed_epoch=0,
                expect_epochs_trained=1,
            )

            dt = _run_entry(
                "centralized_quantum_reupload_interrupt_step1",
                centralized_demoTorchQ.main,
                [
                    "centralized_demoTorchQ.py",
                    "--dataset", "optical",
                    "--epochs", "2",
                    "--q-ansatz", "reupload",
                    "--q-layers", "1",
                    "--quantum-knockout-eval",
                    "--debug-stop-after-epoch", "1",
                    *common_central,
                ],
                expect_exit_code=75,
            )
            timings.append(("centralized_quantum_reupload_interrupt_step1", dt))
            c_q_dir = os.path.join(
                results_dir,
                "flood_central_optical_pytorch_pennylane_reupload_L1_seed42",
            )
            ck_q1 = _assert_centralized_ckpt(c_q_dir, "step", 1, expected_target=2)
            assert len(ck_q1["train_metrics"]) == 1
            q_step1_loss = ck_q1["train_metrics"][0]["loss"]

            dt = _run_entry(
                "centralized_quantum_reupload_resume_step2",
                centralized_demoTorchQ.main,
                [
                    "centralized_demoTorchQ.py",
                    "--dataset", "optical",
                    "--epochs", "2",
                    "--q-ansatz", "reupload",
                    "--q-layers", "1",
                    "--quantum-knockout-eval",
                    *common_central,
                ],
            )
            timings.append(("centralized_quantum_reupload_resume_step2", dt))
            ck_q2 = _assert_centralized_ckpt(c_q_dir, "step", 2, expected_target=2)
            assert len(ck_q2["train_metrics"]) == 2
            assert ck_q2["train_metrics"][0]["loss"] == q_step1_loss
            _assert_metrics_json(
                c_q_dir,
                expect_knockout=True,
                expect_resumed_epoch=1,
                expect_epochs_trained=1,
            )

            # 5. Federated Classical (random + interrupt/resume, and dirichlet)
            dt = _run_entry(
                "federated_classical_random_interrupt_round1",
                federated_demoTorchClassical.main,
                [
                    "federated_demoTorchClassical.py",
                    "--partition", "random",
                    "--dataset", "optical",
                    "--rounds", "2",
                    "--debug-stop-after-round", "1",
                    *common_fed,
                ],
                expect_exit_code=75,
            )
            timings.append(("federated_classical_random_interrupt_round1", dt))
            f_cls_rand_dir = os.path.join(
                results_dir,
                "flood_federated_optical_pytorch_classical_random_seed42",
            )
            f_ck1 = _assert_federated_ckpt(f_cls_rand_dir, 1, expected_total_rounds=2)
            assert len(f_ck1["round_metrics"]) == 1
            f_r1_loss = f_ck1["round_metrics"][0]["loss"]

            dt = _run_entry(
                "federated_classical_random_resume_round2",
                federated_demoTorchClassical.main,
                [
                    "federated_demoTorchClassical.py",
                    "--partition", "random",
                    "--dataset", "optical",
                    "--rounds", "2",
                    *common_fed,
                ],
            )
            timings.append(("federated_classical_random_resume_round2", dt))
            f_ck2 = _assert_federated_ckpt(f_cls_rand_dir, 2, expected_total_rounds=2)
            assert len(f_ck2["round_metrics"]) == 2
            assert f_ck2["round_metrics"][0]["loss"] == f_r1_loss
            _assert_metrics_json(f_cls_rand_dir, expect_resumed_round=1)

            dt = _run_entry(
                "federated_classical_dirichlet_round1",
                federated_demoTorchClassical.main,
                [
                    "federated_demoTorchClassical.py",
                    "--partition", "dirichlet",
                    "--alpha", "0.5",
                    "--dataset", "sar",
                    "--rounds", "1",
                    *common_fed,
                ],
            )
            timings.append(("federated_classical_dirichlet_round1", dt))
            f_cls_dir_dir = os.path.join(
                results_dir,
                "flood_federated_optical_pytorch_classical_dirichlet0.5_sar_seed42",
            )
            _assert_federated_ckpt(f_cls_dir_dir, 1, expected_total_rounds=1)
            _assert_metrics_json(f_cls_dir_dir, expect_resumed_round=0)

            # 6. Federated Ablation (random/legacy & dirichlet/reupload + interrupt/resume + knockout)
            dt = _run_entry(
                "federated_ablation_random_legacy_round1",
                ablation_studies.main,
                [
                    "ablation_studies.py",
                    "--framework", "torch",
                    "--mode", "federated",
                    "--partition", "random",
                    "--dataset", "optical",
                    "--q-ansatz", "legacy",
                    "--rounds", "1",
                    "--quantum-knockout-eval",
                    *common_fed,
                ],
            )
            timings.append(("federated_ablation_random_legacy_round1", dt))
            f_abl_rand_dir = os.path.join(
                results_dir,
                "ablation_federated_torch_bottleneck_matched_pennylane_random_legacy_seed42",
            )
            _assert_federated_ckpt(f_abl_rand_dir, 1, expected_total_rounds=1)
            _assert_metrics_json(f_abl_rand_dir, expect_knockout=True, expect_resumed_round=0)

            dt = _run_entry(
                "federated_ablation_dirichlet_reupload_interrupt_round1",
                ablation_studies.main,
                [
                    "ablation_studies.py",
                    "--framework", "torch",
                    "--mode", "federated",
                    "--partition", "dirichlet",
                    "--alpha", "0.5",
                    "--dataset", "optical",
                    "--q-ansatz", "reupload",
                    "--q-layers", "1",
                    "--rounds", "2",
                    "--quantum-knockout-eval",
                    "--debug-stop-after-round", "1",
                    *common_fed,
                ],
                expect_exit_code=75,
            )
            timings.append(("federated_ablation_dirichlet_reupload_interrupt_round1", dt))
            f_abl_dir_dir = os.path.join(
                results_dir,
                "ablation_federated_torch_bottleneck_matched_pennylane_dirichlet0.5_reupload_L1_seed42",
            )
            f_abl_ck1 = _assert_federated_ckpt(f_abl_dir_dir, 1, expected_total_rounds=2)
            assert len(f_abl_ck1["round_metrics"]) == 1
            f_abl_r1_loss = f_abl_ck1["round_metrics"][0]["loss"]

            dt = _run_entry(
                "federated_ablation_dirichlet_reupload_resume_round2",
                ablation_studies.main,
                [
                    "ablation_studies.py",
                    "--framework", "torch",
                    "--mode", "federated",
                    "--partition", "dirichlet",
                    "--alpha", "0.5",
                    "--dataset", "optical",
                    "--q-ansatz", "reupload",
                    "--q-layers", "1",
                    "--rounds", "2",
                    "--quantum-knockout-eval",
                    *common_fed,
                ],
            )
            timings.append(("federated_ablation_dirichlet_reupload_resume_round2", dt))
            f_abl_ck2 = _assert_federated_ckpt(f_abl_dir_dir, 2, expected_total_rounds=2)
            assert len(f_abl_ck2["round_metrics"]) == 2
            assert f_abl_ck2["round_metrics"][0]["loss"] == f_abl_r1_loss
            _assert_metrics_json(f_abl_dir_dir, expect_knockout=True, expect_resumed_round=1)

            # 7. Federated Quantum (random/legacy & dirichlet/reupload + interrupt/resume + knockout)
            dt = _run_entry(
                "federated_quantum_random_legacy_round1",
                federated_demoTorchQ.main,
                [
                    "federated_demoTorchQ.py",
                    "--partition", "random",
                    "--dataset", "optical",
                    "--q-ansatz", "legacy",
                    "--rounds", "1",
                    "--quantum-knockout-eval",
                    *common_fed,
                ],
            )
            timings.append(("federated_quantum_random_legacy_round1", dt))
            f_q_rand_dir = os.path.join(
                results_dir,
                "flood_federated_optical_pytorch_pennylane_random_legacy_seed42",
            )
            _assert_federated_ckpt(f_q_rand_dir, 1, expected_total_rounds=1)
            _assert_metrics_json(f_q_rand_dir, expect_knockout=True, expect_resumed_round=0)

            dt = _run_entry(
                "federated_quantum_dirichlet_reupload_interrupt_round1",
                federated_demoTorchQ.main,
                [
                    "federated_demoTorchQ.py",
                    "--partition", "dirichlet",
                    "--alpha", "0.5",
                    "--dataset", "optical",
                    "--q-ansatz", "reupload",
                    "--q-layers", "1",
                    "--rounds", "2",
                    "--quantum-knockout-eval",
                    "--debug-stop-after-round", "1",
                    *common_fed,
                ],
                expect_exit_code=75,
            )
            timings.append(("federated_quantum_dirichlet_reupload_interrupt_round1", dt))
            f_q_dir_dir = os.path.join(
                results_dir,
                "flood_federated_optical_pytorch_pennylane_dirichlet0.5_reupload_L1_seed42",
            )
            f_q_ck1 = _assert_federated_ckpt(f_q_dir_dir, 1, expected_total_rounds=2)
            assert len(f_q_ck1["round_metrics"]) == 1
            f_q_r1_loss = f_q_ck1["round_metrics"][0]["loss"]

            dt = _run_entry(
                "federated_quantum_dirichlet_reupload_resume_round2",
                federated_demoTorchQ.main,
                [
                    "federated_demoTorchQ.py",
                    "--partition", "dirichlet",
                    "--alpha", "0.5",
                    "--dataset", "optical",
                    "--q-ansatz", "reupload",
                    "--q-layers", "1",
                    "--rounds", "2",
                    "--quantum-knockout-eval",
                    *common_fed,
                ],
            )
            timings.append(("federated_quantum_dirichlet_reupload_resume_round2", dt))
            f_q_ck2 = _assert_federated_ckpt(f_q_dir_dir, 2, expected_total_rounds=2)
            assert len(f_q_ck2["round_metrics"]) == 2
            assert f_q_ck2["round_metrics"][0]["loss"] == f_q_r1_loss
            _assert_metrics_json(f_q_dir_dir, expect_knockout=True, expect_resumed_round=1)

    finally:
        os.chdir(old_cwd)
        torch.set_num_threads(old_num_threads)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

    total_elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print("LOCAL WORKSTATION VERIFICATION SUMMARY")
    print("=" * 60)
    for step_name, step_dt in timings:
        print(f"  {step_name:52s} {step_dt:6.2f}s")
    print("-" * 60)
    print(f"  TOTAL ELAPSED: {total_elapsed:.2f}s (budget: {max_seconds:.1f}s)")
    print("=" * 60)
    assert total_elapsed < max_seconds, (
        f"Verification exceeded {max_seconds:.1f}s budget: {total_elapsed:.2f}s"
    )
    return {"total_seconds": total_elapsed, "steps": timings}


@contextmanager
def _null_tempdir(path):
    os.makedirs(path, exist_ok=True)
    yield path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default=None, help="Optional directory to keep generated artifacts")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="Maximum allowed runtime in seconds (default: 60)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for mini verification (default: cpu)")
    args = ap.parse_args()
    run_verification(
        work_dir=args.work_dir,
        max_seconds=args.max_seconds,
        device=args.device,
        include_process_start=True,
    )


if __name__ == "__main__":
    main()

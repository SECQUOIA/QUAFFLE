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
    require_canonical_meta=False,
):
    summary_path = os.path.join(run_dir, "results", "summary.txt")
    metrics_path = os.path.join(run_dir, "results", "metrics.json")
    assert os.path.isfile(summary_path), f"Missing summary.txt in {run_dir}"
    assert os.path.getsize(summary_path) > 0, f"Empty summary.txt in {run_dir}"
    assert os.path.isfile(metrics_path), f"Missing metrics.json in {run_dir}"
    with open(metrics_path) as f:
        m = json.load(f)
    assert m.get("schema") == "quaffle-metrics-v1", m.get("schema")
    required_splits = ["val", "test"]
    if "select_metric" in m or require_canonical_meta:
        assert m.get("select_metric") in ("iou", "dice", "auc", "loss"), m.get("select_metric")
        required_splits.extend([
            "best_val_metrics",
            "final_val_metrics",
            "best_val_test_metrics",
            "final_test_metrics",
        ])
    for split_key in required_splits:
        block = m.get(split_key)
        assert isinstance(block, dict), f"Missing {split_key} metrics in {metrics_path}"
        for metric_key in ("loss", "accuracy", "auc", "iou", "dice"):
            assert metric_key in block, f"Missing {split_key}.{metric_key} in {metrics_path}"
            val = block[metric_key]
            assert isinstance(val, (int, float)) and not isinstance(val, bool), (
                f"Non-numeric {split_key}.{metric_key}={val!r} in {metrics_path}"
            )
            fval = float(val)
            assert fval == fval and abs(fval) != float("inf"), (
                f"Non-finite {split_key}.{metric_key}={val!r} in {metrics_path}"
            )
            if metric_key == "loss":
                assert fval >= 0.0, f"Negative {split_key}.loss={fval} in {metrics_path}"
            else:
                assert 0.0 <= fval <= 1.0, (
                    f"Out-of-range {split_key}.{metric_key}={fval} in {metrics_path}"
                )
    if require_canonical_meta:
        for meta_key in ("base_channels", "image_size", "data_root", "norm_provenance"):
            assert m.get(meta_key) is not None, (
                f"Missing canonical metadata {meta_key!r} in {metrics_path}"
            )
        assert m.get("is_mini_run") is True, f"Expected is_mini_run=True in {metrics_path}"
        assert m["norm_provenance"].get("fit_only") is True, (
            f"Expected norm_provenance.fit_only=True in {metrics_path}"
        )
    if expect_knockout:
        ko = m.get("knockout")
        assert isinstance(ko, dict), f"Expected knockout metrics in {metrics_path}"
        for mode in ("zero", "random"):
            assert isinstance(ko.get(mode), dict), f"Missing knockout[{mode!r}] in {metrics_path}"
            assert "iou" in ko[mode] and "auc" in ko[mode]
            for k_metric in ("loss", "accuracy", "auc", "iou", "dice"):
                if k_metric in ko[mode]:
                    kv = float(ko[mode][k_metric])
                    assert kv == kv and abs(kv) != float("inf"), (
                        f"Non-finite knockout[{mode!r}].{k_metric}={kv} in {metrics_path}"
                    )
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
    assert isinstance(ckpt.get("best_state_dict"), dict), f"Missing best_state_dict in {ckpt_path}"
    assert ckpt.get("select_metric") in tc.SELECT_METRIC_CHOICES
    fp = ckpt.get("fingerprint")
    assert isinstance(fp, dict), "Missing centralized checkpoint fingerprint"
    assert fp.get("select_metric") in tc.SELECT_METRIC_CHOICES
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
    assert isinstance(ckpt.get("best_parameters"), list), f"Missing best_parameters in {ckpt_path}"
    assert isinstance(ckpt.get("final_parameters"), list), f"Missing final_parameters in {ckpt_path}"
    assert ckpt.get("select_metric") in tc.SELECT_METRIC_CHOICES
    fp = ckpt.get("fingerprint")
    assert isinstance(fp, dict), "Missing federated checkpoint fingerprint"
    assert fp.get("select_metric") in tc.SELECT_METRIC_CHOICES
    if expected_total_rounds is not None:
        assert fp.get("fed_rounds") == expected_total_rounds, (
            f"Expected fingerprint fed_rounds={expected_total_rounds}, got {fp.get('fed_rounds')}"
        )
    assert fp.get("data_root") is not None
    assert isinstance(fp.get("norm_provenance"), dict)
    assert fp["norm_provenance"].get("fit_only") is True
    return ckpt


def run_verification(
    work_dir=None,
    max_seconds=60.0,
    device="cpu",
    include_process_start=False,
    keep=False,
):
    t_start = _PROCESS_START if include_process_start else time.perf_counter()
    timings = []
    if include_process_start:
        timings.append(("module_imports", _IMPORT_ELAPSED))

    if work_dir is None and keep:
        work_dir = tempfile.mkdtemp(prefix="quaffle_verify_")
        print(f"Keeping verification work_dir: {work_dir}")
    ctx = (
        tempfile.TemporaryDirectory(prefix="quaffle_verify_")
        if work_dir is None
        else _null_tempdir(work_dir)
    )
    import results_tracker

    old_env = {
        k: os.environ.get(k)
        for k in (
            "QUAFFLE_INPROCESS_FED",
            "QUAFFLE_SKIP_EXCEL_TRACKER",
            "QUAFFLE_SKIP_CURVE_PLOTS",
            "QUAFFLE_TRACKER_DIR",
            "QUAFFLE_RESULTS_DIR",
            "QUAFFLE_ANALYSIS_OUT_DIR",
            "QUAFFLE_INCLUDE_MINI",
            "QUAFFLE_PRIMARY_VARIANT",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    }
    old_tracker_results_dir = getattr(results_tracker, "_RESULTS_DIR", results_tracker._DEFAULT_RESULTS_DIR)
    old_num_threads = torch.get_num_threads()
    old_cwd = os.getcwd()
    os.environ["QUAFFLE_INPROCESS_FED"] = "1"
    os.environ["QUAFFLE_SKIP_EXCEL_TRACKER"] = "1"
    os.environ["QUAFFLE_SKIP_CURVE_PLOTS"] = "1"
    os.environ.pop("QUAFFLE_TRACKER_DIR", None)
    results_tracker._RESULTS_DIR = results_tracker._DEFAULT_RESULTS_DIR
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
            assert float(sar_raw[0].std()) > 1.0 and float(sar_raw[1].std()) > 1.0, (
                f"Expected positive SAR VV/VH backscatter variance, got std={sar_raw[0].std():.4f}, {sar_raw[1].std():.4f}"
            )
            dt = time.perf_counter() - t0
            timings.append(("generate_and_validate_mini_dataset", dt))
            assert os.path.isfile(info["stats_path"])

            mini_cfg = {
                "seed": 42,
                "base_channels": 4,
                "image_size": 32,
                "data_root": data_root,
                "results_dir": results_dir,
            }

            def _rdir(rel_name):
                return tc.resolve_output_dir(f"results/{rel_name}", mini_cfg)

            # Evaluate untrained seed=42 random-initialization baseline on optical val/test splits
            init_cfg = {
                "seed": 42,
                "dataset": "optical",
                "base_channels": 4,
                "init_dim": 4,
                "resnet_block_groups": 4,
                "image_size": 32,
                "batch_size": 4,
                "val_fraction": 0.15,
                "max_samples": None,
                "data_root": data_root,
                "results_dir": results_dir,
            }
            tc.seed_everything(42)
            _, tr_img_dir, tr_lbl_dir, te_img_dir, te_lbl_dir = tc.resolve_data_dirs("optical", data_root=data_root)
            opt_pool = tc.load_dataset(tr_img_dir, tr_lbl_dir, init_cfg)
            _, val_indices, _ = tc.holdout_validation(opt_pool, 42, 0.15)
            opt_val_sub = torch.utils.data.Subset(opt_pool, val_indices)
            val_loader_init = tc.make_loader(opt_val_sub, 4, False, 42)
            opt_test = tc.load_dataset(te_img_dir, te_lbl_dir, init_cfg)
            test_loader_init = tc.make_loader(opt_test, 4, False, 42)
            init_Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            init_model = tc.build_model(centralized_demoTorchClassical.UNetSegmentation, init_cfg).to(init_Device)
            init_val_m = centralized_demoTorchClassical.evaluate_model(init_model, val_loader_init, init_Device)
            init_test_m = centralized_demoTorchClassical.evaluate_model(init_model, test_loader_init, init_Device)
            init_val_iou = float(init_val_m["iou"])
            init_test_iou = float(init_test_m["iou"])

            common_central = [
                "--data-root", data_root,
                "--results-dir", results_dir,
                "--base-channels", "4",
                "--image-size", "32",
                "--batch-size", "4",
                "--learning-rate", "5e-3",
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
                "--learning-rate", "5e-3",
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
            c_cls_dir = _rdir("flood_central_optical_pytorch_classical_seed42")
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
            assert ck2["train_metrics"][1]["loss"] < ep1_loss, (
                f"Expected centralized classical training loss to decrease: {ep1_loss:.4f} -> {ck2['train_metrics'][1]['loss']:.4f}"
            )
            m_c_cls = _assert_metrics_json(
                c_cls_dir,
                expect_knockout=False,
                expect_resumed_epoch=1,
                expect_epochs_trained=1,
                require_canonical_meta=True,
            )
            assert m_c_cls["val"]["iou"] > init_val_iou and m_c_cls["test"]["iou"] > init_test_iou, (
                f"Expected centralized classical IoU to improve over initialization "
                f"(val {init_val_iou:.4f}->{m_c_cls['val']['iou']:.4f}, test {init_test_iou:.4f}->{m_c_cls['test']['iou']:.4f})"
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
            c_abl_leg_dir = _rdir(
                "ablation_centralized_torch_bottleneck_matched_pennylane_legacy_seed42"
            )
            _assert_centralized_ckpt(c_abl_leg_dir, "epoch", 1, expected_target=1)
            _assert_metrics_json(
                c_abl_leg_dir,
                expect_knockout=True,
                expect_resumed_epoch=0,
                expect_epochs_trained=1,
                require_canonical_meta=True,
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
            c_abl_dir = _rdir(
                "ablation_centralized_torch_bottleneck_matched_pennylane_reupload_L1_seed42"
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
            assert ck_abl2["train_metrics"][1]["loss"] < abl_ep1_loss, (
                f"Expected centralized ablation training loss to decrease: {abl_ep1_loss:.4f} -> {ck_abl2['train_metrics'][1]['loss']:.4f}"
            )
            m_c_abl = _assert_metrics_json(
                c_abl_dir,
                expect_knockout=True,
                expect_resumed_epoch=1,
                expect_epochs_trained=1,
                require_canonical_meta=True,
            )
            assert m_c_abl["val"]["iou"] > init_val_iou and m_c_abl["test"]["iou"] > init_test_iou

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
            c_q_leg_dir = _rdir(
                "flood_central_optical_pytorch_pennylane_sar_legacy_seed42"
            )
            _assert_centralized_ckpt(c_q_leg_dir, "step", 1, expected_target=1)
            _assert_metrics_json(
                c_q_leg_dir,
                expect_knockout=True,
                expect_resumed_epoch=0,
                expect_epochs_trained=1,
                require_canonical_meta=True,
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
            c_q_dir = _rdir(
                "flood_central_optical_pytorch_pennylane_reupload_L1_seed42"
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
            assert ck_q2["train_metrics"][1]["loss"] < q_step1_loss, (
                f"Expected centralized quantum training loss to decrease: {q_step1_loss:.4f} -> {ck_q2['train_metrics'][1]['loss']:.4f}"
            )
            m_c_q = _assert_metrics_json(
                c_q_dir,
                expect_knockout=True,
                expect_resumed_epoch=1,
                expect_epochs_trained=1,
                require_canonical_meta=True,
            )
            assert m_c_q["val"]["iou"] > init_val_iou and m_c_q["test"]["iou"] > init_test_iou

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
            f_cls_rand_dir = _rdir(
                "flood_federated_optical_pytorch_classical_random_K2C2_seed42"
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
            assert f_ck2["round_metrics"][1]["loss"] < f_r1_loss, (
                f"Expected federated classical round loss to decrease: {f_r1_loss:.4f} -> {f_ck2['round_metrics'][1]['loss']:.4f}"
            )
            m_f_cls = _assert_metrics_json(
                f_cls_rand_dir,
                expect_resumed_round=1,
                require_canonical_meta=True,
            )
            assert m_f_cls["val"]["iou"] > init_val_iou and m_f_cls["test"]["iou"] > init_test_iou

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
            f_cls_dir_dir = _rdir(
                "flood_federated_optical_pytorch_classical_dirichlet0.5_sar_K2C2_seed42"
            )
            _assert_federated_ckpt(f_cls_dir_dir, 1, expected_total_rounds=1)
            _assert_metrics_json(f_cls_dir_dir, expect_resumed_round=0, require_canonical_meta=True)

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
            f_abl_rand_dir = _rdir(
                "ablation_federated_torch_bottleneck_matched_pennylane_random_legacy_K2C2_seed42"
            )
            _assert_federated_ckpt(f_abl_rand_dir, 1, expected_total_rounds=1)
            _assert_metrics_json(
                f_abl_rand_dir,
                expect_knockout=True,
                expect_resumed_round=0,
                require_canonical_meta=True,
            )

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
            f_abl_dir_dir = _rdir(
                "ablation_federated_torch_bottleneck_matched_pennylane_dirichlet0.5_reupload_L1_K2C2_seed42"
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
            assert f_abl_ck2["round_metrics"][1]["loss"] < f_abl_r1_loss, (
                f"Expected federated ablation round loss to decrease: {f_abl_r1_loss:.4f} -> {f_abl_ck2['round_metrics'][1]['loss']:.4f}"
            )
            m_f_abl = _assert_metrics_json(
                f_abl_dir_dir,
                expect_knockout=True,
                expect_resumed_round=1,
                require_canonical_meta=True,
            )
            assert m_f_abl["val"]["iou"] > init_val_iou and m_f_abl["test"]["iou"] > init_test_iou

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
            f_q_rand_dir = _rdir(
                "flood_federated_optical_pytorch_pennylane_random_legacy_K2C2_seed42"
            )
            _assert_federated_ckpt(f_q_rand_dir, 1, expected_total_rounds=1)
            _assert_metrics_json(
                f_q_rand_dir,
                expect_knockout=True,
                expect_resumed_round=0,
                require_canonical_meta=True,
            )

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
            f_q_dir_dir = _rdir(
                "flood_federated_optical_pytorch_pennylane_dirichlet0.5_reupload_L1_K2C2_seed42"
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
            assert f_q_ck2["round_metrics"][1]["loss"] < f_q_r1_loss, (
                f"Expected federated quantum round loss to decrease: {f_q_r1_loss:.4f} -> {f_q_ck2['round_metrics'][1]['loss']:.4f}"
            )
            m_f_q = _assert_metrics_json(
                f_q_dir_dir,
                expect_knockout=True,
                expect_resumed_round=1,
                require_canonical_meta=True,
            )
            assert m_f_q["val"]["iou"] > init_val_iou and m_f_q["test"]["iou"] > init_test_iou

            # 8. Wave 2 features, combined modality, tracker isolation, plots, and tools
            # 8a. Centralized Quantum HQ-UNet on combined modality (pennylane & torch backends)
            #     with curve plots, sample visualizations, and Excel tracker enabled
            def _file_sig(p):
                if not os.path.exists(p):
                    return None
                st = os.stat(p)
                return (st.st_mtime_ns, st.st_size)

            repo_tracker_paths = [
                os.path.join(REPO_ROOT, "results", "results_summary_combined.xlsx"),
                os.path.join(REPO_ROOT, "results", "results_summary_combined_mini.xlsx"),
            ]
            repo_tracker_before = {p: _file_sig(p) for p in repo_tracker_paths}
            os.environ["QUAFFLE_SKIP_EXCEL_TRACKER"] = "0"
            os.environ["QUAFFLE_SKIP_CURVE_PLOTS"] = "0"
            wave2_args = [
                "--dataset", "combined",
                "--epochs", "1",
                "--q-ansatz", "hqunet",
                "--hqunet-grid", "2",
                "--q-layers", "1",
                "--backbone-conv-mode", "depthwise_sep",
                "--loss", "dice_ce",
                "--no-time-emb",
                "--augment",
                "--data-root", data_root,
                "--results-dir", results_dir,
                "--base-channels", "4",
                "--image-size", "32",
                "--batch-size", "4",
                "--learning-rate", "5e-3",
                "--eval-every", "1",
                "--seed", "42",
            ]
            dt = _run_entry(
                "centralized_quantum_hqunet_combined_pennylane",
                centralized_demoTorchQ.main,
                ["centralized_demoTorchQ.py", "--q-backend", "pennylane", *wave2_args],
            )
            timings.append(("centralized_quantum_hqunet_combined_pennylane", dt))

            dt = _run_entry(
                "centralized_quantum_hqunet_combined_torch",
                centralized_demoTorchQ.main,
                ["centralized_demoTorchQ.py", "--q-backend", "torch", *wave2_args, "--no-save-samples"],
            )
            timings.append(("centralized_quantum_hqunet_combined_torch", dt))

            dt = _run_entry(
                "centralized_ablation_hqunet_spatial_combined",
                ablation_studies.main,
                [
                    "ablation_studies.py",
                    "--framework", "torch",
                    "--mode", "centralized",
                    "--ablation-mode", "hqunet_spatial",
                    *wave2_args,
                    "--no-save-samples",
                ],
            )
            timings.append(("centralized_ablation_hqunet_spatial_combined", dt))
            os.environ["QUAFFLE_SKIP_EXCEL_TRACKER"] = "1"
            os.environ["QUAFFLE_SKIP_CURVE_PLOTS"] = "1"

            hq_pl_dir = _rdir(
                "flood_central_optical_pytorch_pennylane_combined_hqunet_G2_L1_depthwise_sep_dice_ce_notime_aug_seed42"
            )
            _assert_metrics_json(hq_pl_dir, expect_resumed_epoch=0, expect_epochs_trained=1, require_canonical_meta=True)
            assert os.path.isfile(os.path.join(hq_pl_dir, "results", "loss_curve.png"))
            assert os.path.isfile(os.path.join(hq_pl_dir, "results", "test_sample_1.png"))
            assert os.path.isfile(os.path.join(results_dir, "results_summary_combined_mini.xlsx")), (
                "Expected mini-run Excel tracker in results_dir/results_summary_combined_mini.xlsx"
            )
            for p, sig_before in repo_tracker_before.items():
                assert _file_sig(p) == sig_before, (
                    f"Mini-run Excel tracker must not create or modify {p}"
                )

            # 8b. Wave 3 verification (#14 dual_stem + modality/geo_cluster partitioning; #15 circular angle agg + FedProx)
            t0 = time.perf_counter()
            import numpy as np
            from utilsTorch import QVUNetSegmentation, ClassicalAblationUNetSegmentation, fedprox_proximal_loss
            w3_cfg = {
                "seed": 42,
                "dataset": "combined",
                "combined_input_mode": "dual_stem",
                "base_channels": 4,
                "init_dim": 4,
                "out_dim": 2,
                "dim_mults": (1, 2),
                "resnet_block_groups": 4,
                "quantum_channels": 4,
                "image_size": 32,
                "batch_size": 4,
                "val_fraction": 0.15,
                "num_clients": 4,
                "clients_per_round": 2,
                "min_client_samples": 1,
                "fl_strategy": "fedprox",
                "fedprox_mu": 0.05,
                "q_angle_agg": "circular",
                "data_root": data_root,
                "results_dir": results_dir,
            }
            _, comb_train_img, comb_train_msk, _, _ = tc.resolve_data_dirs("combined", data_root)
            comb_ds = tc.load_dataset(comb_train_img, comb_train_msk, w3_cfg)
            sample_img0, _ = comb_ds[0]
            assert tuple(sample_img0.shape) == (6, 32, 32), f"Expected 6-ch dual_stem sample, got {sample_img0.shape}"
            assert set(comb_ds.modalities) == {"optical", "sar"}, f"Expected both optical and sar modalities, got {set(comb_ds.modalities)}"
            for p_mode in ("modality", "geo_cluster"):
                w3_cfg["partition"] = p_mode
                c_lists, v_idxs, _ = tc.build_federated_split(comb_ds, w3_cfg)
                assert len(c_lists) == 4 and all(len(c) >= 1 for c in c_lists)
            w3_q = tc.build_model(QVUNetSegmentation, w3_cfg)
            w3_a = tc.build_model(ClassicalAblationUNetSegmentation, w3_cfg)
            assert w3_q.model.init_conv_sar is not None and w3_a.model.init_conv_sar is not None
            assert torch.equal(w3_q.model.init_conv.weight, w3_a.model.init_conv.weight)
            assert torch.equal(w3_q.model.init_conv_sar.weight, w3_a.model.init_conv_sar.weight)
            eps = 0.04
            circ = tc.circular_mean_angles(
                [np.array([np.pi - eps], dtype=np.float32), np.array([-np.pi + eps], dtype=np.float32)],
                [1.0, 1.0],
            )
            assert abs(abs(float(circ[0])) - np.pi) < 1e-5
            prox_l = fedprox_proximal_loss(w3_q, [p.detach().clone() + 0.5 for p in w3_q.parameters()], 0.1)
            assert float(prox_l.item()) > 0.0
            dt = time.perf_counter() - t0
            timings.append(("wave3_dual_stem_non_iid_fedprox_and_circular_agg", dt))
            print(f"[PASS] wave3_dual_stem_non_iid_fedprox_and_circular_agg ({dt:.2f}s)")

            # 9. Downstream analysis pipeline, exemplar figure, circuit diagram, and backend comparison
            t0 = time.perf_counter()
            import analyze_results
            import visualize_circuit
            from scripts import build_exemplar_figure, build_forest_plot, build_table2_data
            from tools import compare_backends

            os.environ["QUAFFLE_RESULTS_DIR"] = results_dir
            os.environ["QUAFFLE_ANALYSIS_OUT_DIR"] = results_dir
            os.environ["QUAFFLE_PRIMARY_VARIANT"] = "reupload_L1"

            # Verify default paper analysis excludes mini runs unless --include-mini / QUAFFLE_INCLUDE_MINI=1
            os.environ["QUAFFLE_INCLUDE_MINI"] = "0"
            df_paper_only = analyze_results.load_from_disk(include_mini=False, results_dir=results_dir)
            assert df_paper_only.empty, "Expected default paper analysis to exclude mini runs"

            os.environ["QUAFFLE_INCLUDE_MINI"] = "1"
            analyze_results.main(
                [
                    "--results-dir", results_dir,
                    "--out-dir", results_dir,
                    "--include-mini",
                    "--skip-plots",
                ]
            )
            xlsx_path = os.path.join(results_dir, "analysis_summary.xlsx")
            assert os.path.isfile(xlsx_path) and os.path.getsize(xlsx_path) > 0

            build_table2_data.main(["--out-dir", results_dir])
            t2_path = os.path.join(results_dir, "headline_pooled_table2_data.json")
            assert os.path.isfile(t2_path) and os.path.getsize(t2_path) > 0

            forest_dir = os.path.join(results_dir, "forest")
            build_forest_plot.main(["--out-dir", forest_dir])
            assert os.path.isfile(os.path.join(forest_dir, "iou_delta_forest.png"))
            assert os.path.isfile(os.path.join(forest_dir, "iou_delta_forest_data.json"))

            exemplar_dir = os.path.join(results_dir, "exemplars")
            build_exemplar_figure.main(
                [
                    "--results-dir", results_dir,
                    "--data-root", data_root,
                    "--out-dir", exemplar_dir,
                    "--seed", "42",
                ]
            )
            assert os.path.isfile(os.path.join(exemplar_dir, "exemplars_grid.png"))
            assert os.path.isfile(os.path.join(exemplar_dir, "meta.json"))

            circuit_dir = os.path.join(results_dir, "circuits")
            visualize_circuit.main(
                ["--q-ansatz", "hqunet", "--hqunet-grid", "2", "--q-layers", "1", "--out-dir", circuit_dir]
            )
            assert os.path.isfile(os.path.join(circuit_dir, "quantum_circuit_hqunet_G2_L1.txt"))
            assert os.path.isfile(os.path.join(circuit_dir, "quantum_circuit_hqunet_G2_L1.png"))
            assert os.path.isfile(os.path.join(circuit_dir, "quantum_circuit_hqunet_G2_L1_expvals.png"))

            cmp_csv = os.path.join(results_dir, "backend_compare.csv")
            compare_backends.main([results_dir, "--csv", cmp_csv])
            assert os.path.isfile(cmp_csv) and os.path.getsize(cmp_csv) > 0

            dt = time.perf_counter() - t0
            timings.append(("downstream_analysis_exemplar_circuit_and_compare", dt))
            print(f"[PASS] downstream_analysis_exemplar_circuit_and_compare ({dt:.2f}s)")

    finally:
        os.chdir(old_cwd)
        torch.set_num_threads(old_num_threads)
        results_tracker._RESULTS_DIR = old_tracker_results_dir
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
    ap.add_argument("--keep", action="store_true", help="Keep temporary verification directory on disk")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="Maximum allowed runtime in seconds (default: 60)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for mini verification (default: cpu)")
    args = ap.parse_args()
    run_verification(
        work_dir=args.work_dir,
        max_seconds=args.max_seconds,
        device=args.device,
        include_process_start=True,
        keep=args.keep,
    )


if __name__ == "__main__":
    main()

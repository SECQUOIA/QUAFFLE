#!/usr/bin/env python3
"""End-to-end local workstation verification harness for QUAFFLE.

Generates a temporary synthetic 12-band Optical + 6-band SAR mini-flood dataset
in scratch/tmp and verifies:
  1. Mini-dataset generation, split manifests (conflict_policy='drop'), and
     fit-only norm_stats.json (excluding validation holdout).
  2. Centralized training, checkpoint resume, and --quantum-knockout-eval
     across classical, ablation (legacy & reupload), and quantum (legacy & reupload).
  3. Federated training (random & dirichlet), checkpoint resume, and
     --quantum-knockout-eval across classical, ablation (legacy & reupload),
     and quantum (legacy & reupload).

Completes in < 60 s on a single workstation CPU/GPU.
"""

import argparse
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import tempfile
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ablation_studies  # noqa: E402
import centralized_demoTorchClassical  # noqa: E402
import centralized_demoTorchQ  # noqa: E402
import federated_demoTorchClassical  # noqa: E402
import federated_demoTorchQ  # noqa: E402
from tools import training_common as tc  # noqa: E402
from tools.make_mini_flood_dataset import generate_mini_flood_dataset  # noqa: E402


@contextmanager
def _patched_argv(argv):
    old_argv = list(sys.argv)
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def _run_entry(name, fn, argv):
    t0 = time.perf_counter()
    with _patched_argv(argv):
        fn()
    dt = time.perf_counter() - t0
    print(f"[PASS] {name} ({dt:.2f}s)")
    return dt


def _assert_metrics_json(run_dir, *, expect_knockout=False, expect_resumed_round=None):
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
    return m


def _assert_centralized_ckpt(run_dir, expected_step_key, expected_step_val):
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    assert os.path.isfile(ckpt_path), f"Missing centralized checkpoint {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt.get("run_format") == tc.RUN_FORMAT
    assert ckpt.get(expected_step_key) == expected_step_val, (
        f"Expected {expected_step_key}={expected_step_val}, got {ckpt.get(expected_step_key)}"
    )
    fp = ckpt.get("fingerprint")
    assert isinstance(fp, dict), "Missing centralized checkpoint fingerprint"
    assert fp.get("data_root") is not None
    assert isinstance(fp.get("norm_provenance"), dict)
    assert fp["norm_provenance"].get("fit_only") is True
    return ckpt


def _assert_federated_ckpt(run_dir, expected_round):
    ckpt_path = os.path.join(run_dir, tc.FED_CKPT_NAME)
    assert os.path.isfile(ckpt_path), f"Missing federated checkpoint {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt.get("run_format") == tc.RUN_FORMAT
    assert ckpt.get("completed_round") == expected_round, (
        f"Expected completed_round={expected_round}, got {ckpt.get('completed_round')}"
    )
    fp = ckpt.get("fingerprint")
    assert isinstance(fp, dict), "Missing federated checkpoint fingerprint"
    assert fp.get("data_root") is not None
    assert isinstance(fp.get("norm_provenance"), dict)
    return ckpt


def run_verification(work_dir=None, max_seconds=60.0, device="cpu"):
    t_start = time.perf_counter()
    timings = []

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

            # 1. Generate synthetic mini dataset (12-band Optical + 6-band SAR + DEM -9999 nodata)
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
            dt = time.perf_counter() - t0
            timings.append(("generate_mini_flood_dataset", dt))
            assert os.path.isfile(info["stats_path"])
            for ds_name in ("flood_optical", "flood_SAR", "flood_combined"):
                assert info["manifests"][ds_name]["conflict_policy"] == "drop"

            common_central = [
                "--data-root", data_root,
                "--results-dir", results_dir,
                "--base-channels", "4",
                "--image-size", "32",
                "--batch-size", "4",
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

            # 2. Centralized Classical: train 1 epoch (via CLI subprocess) + resume to epoch 2
            t0 = time.perf_counter()
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "centralized_demoTorchClassical.py"),
                    "--dataset", "optical",
                    "--epochs", "1",
                    *common_central,
                ],
                check=True,
                cwd=tmp_root,
            )
            dt = time.perf_counter() - t0
            timings.append(("centralized_classical_epoch1_cli", dt))
            c_cls_dir = os.path.join(results_dir, "flood_central_optical_pytorch_classical_seed42")
            _assert_centralized_ckpt(c_cls_dir, "epoch", 1)

            # Mutate checkpoint fingerprint epochs_or_steps to 2 so epoch 1->2 resumes seamlessly
            ckpt_p = os.path.join(c_cls_dir, "checkpoint.pt")
            ck = torch.load(ckpt_p, map_location="cpu")
            ck["fingerprint"]["epochs_or_steps"] = 2
            torch.save(ck, ckpt_p)
            dt = _run_entry(
                "centralized_classical_resume_epoch2",
                centralized_demoTorchClassical.main,
                ["centralized_demoTorchClassical.py", "--dataset", "optical", "--epochs", "2", *common_central],
            )
            timings.append(("centralized_classical_resume_epoch2", dt))
            _assert_centralized_ckpt(c_cls_dir, "epoch", 2)
            _assert_metrics_json(c_cls_dir, expect_knockout=False)

            # 3. Centralized Ablation (legacy & reupload) + resume + knockout
            for ansatz in ("legacy", "reupload"):
                extra_q = ["--q-ansatz", ansatz] + (["--q-layers", "1"] if ansatz == "reupload" else [])
                dt = _run_entry(
                    f"centralized_ablation_{ansatz}_epoch1",
                    ablation_studies.main,
                    [
                        "ablation_studies.py",
                        "--framework", "torch",
                        "--mode", "centralized",
                        "--dataset", "optical",
                        "--epochs", "1",
                        "--quantum-knockout-eval",
                        *extra_q,
                        *common_central,
                    ],
                )
                timings.append((f"centralized_ablation_{ansatz}_epoch1", dt))
                vtag = "_legacy" if ansatz == "legacy" else "_reupload_L1"
                c_abl_dir = os.path.join(
                    results_dir,
                    f"ablation_centralized_torch_bottleneck_matched_pennylane{vtag}_seed42",
                )
                _assert_centralized_ckpt(c_abl_dir, "epoch", 1)
                _assert_metrics_json(c_abl_dir, expect_knockout=True)

            # Resume centralized ablation (reupload) from epoch 1 to epoch 2
            ck_abl_p = os.path.join(c_abl_dir, "checkpoint.pt")
            ck_abl = torch.load(ck_abl_p, map_location="cpu")
            ck_abl["fingerprint"]["epochs_or_steps"] = 2
            torch.save(ck_abl, ck_abl_p)
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
            _assert_centralized_ckpt(c_abl_dir, "epoch", 2)

            # 4. Centralized Quantum (legacy & reupload, SAR & optical) + resume + knockout
            for ansatz, ds in (("legacy", "sar"), ("reupload", "optical")):
                extra_q = ["--q-ansatz", ansatz] + (["--q-layers", "1"] if ansatz == "reupload" else [])
                dt = _run_entry(
                    f"centralized_quantum_{ansatz}_{ds}_step1",
                    centralized_demoTorchQ.main,
                    [
                        "centralized_demoTorchQ.py",
                        "--dataset", ds,
                        "--epochs", "1",
                        "--quantum-knockout-eval",
                        *extra_q,
                        *common_central,
                    ],
                )
                timings.append((f"centralized_quantum_{ansatz}_{ds}_step1", dt))
                ds_sfx = "" if ds == "optical" else f"_{ds}"
                vtag = "_legacy" if ansatz == "legacy" else "_reupload_L1"
                c_q_dir = os.path.join(
                    results_dir,
                    f"flood_central_optical_pytorch_pennylane{ds_sfx}{vtag}_seed42",
                )
                _assert_centralized_ckpt(c_q_dir, "step", 1)
                _assert_metrics_json(c_q_dir, expect_knockout=True)

            # Resume centralized quantum (reupload) from step 1 to step 2
            ck_q_p = os.path.join(c_q_dir, "checkpoint.pt")
            ck_q = torch.load(ck_q_p, map_location="cpu")
            ck_q["fingerprint"]["epochs_or_steps"] = 2
            torch.save(ck_q, ck_q_p)
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
            _assert_centralized_ckpt(c_q_dir, "step", 2)

            # 5. Federated Classical (random + resume, and dirichlet)
            dt = _run_entry(
                "federated_classical_random_round1",
                federated_demoTorchClassical.main,
                [
                    "federated_demoTorchClassical.py",
                    "--partition", "random",
                    "--dataset", "optical",
                    "--rounds", "1",
                    *common_fed,
                ],
            )
            timings.append(("federated_classical_random_round1", dt))
            f_cls_rand_dir = os.path.join(
                results_dir,
                "flood_federated_optical_pytorch_classical_random_seed42",
            )
            _assert_federated_ckpt(f_cls_rand_dir, 1)
            f_ckpt_p = os.path.join(f_cls_rand_dir, tc.FED_CKPT_NAME)
            f_ck = torch.load(f_ckpt_p, map_location="cpu")
            f_ck["fingerprint"]["fed_rounds"] = 2
            torch.save(f_ck, f_ckpt_p)

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
            _assert_federated_ckpt(f_cls_rand_dir, 2)
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
            _assert_federated_ckpt(f_cls_dir_dir, 1)
            _assert_metrics_json(f_cls_dir_dir)

            # 6. Federated Ablation (random/legacy & dirichlet/reupload + resume + knockout)
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
            _assert_federated_ckpt(f_abl_rand_dir, 1)
            _assert_metrics_json(f_abl_rand_dir, expect_knockout=True)

            dt = _run_entry(
                "federated_ablation_dirichlet_reupload_round1",
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
                    "--rounds", "1",
                    "--quantum-knockout-eval",
                    *common_fed,
                ],
            )
            timings.append(("federated_ablation_dirichlet_reupload_round1", dt))
            f_abl_dir_dir = os.path.join(
                results_dir,
                "ablation_federated_torch_bottleneck_matched_pennylane_dirichlet0.5_reupload_L1_seed42",
            )
            _assert_federated_ckpt(f_abl_dir_dir, 1)
            f_abl_ck_p = os.path.join(f_abl_dir_dir, tc.FED_CKPT_NAME)
            f_abl_ck = torch.load(f_abl_ck_p, map_location="cpu")
            f_abl_ck["fingerprint"]["fed_rounds"] = 2
            torch.save(f_abl_ck, f_abl_ck_p)

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
            _assert_federated_ckpt(f_abl_dir_dir, 2)
            _assert_metrics_json(f_abl_dir_dir, expect_knockout=True, expect_resumed_round=1)

            # 7. Federated Quantum (random/legacy & dirichlet/reupload + resume + knockout)
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
            _assert_federated_ckpt(f_q_rand_dir, 1)
            _assert_metrics_json(f_q_rand_dir, expect_knockout=True)

            dt = _run_entry(
                "federated_quantum_dirichlet_reupload_round1",
                federated_demoTorchQ.main,
                [
                    "federated_demoTorchQ.py",
                    "--partition", "dirichlet",
                    "--alpha", "0.5",
                    "--dataset", "optical",
                    "--q-ansatz", "reupload",
                    "--q-layers", "1",
                    "--rounds", "1",
                    "--quantum-knockout-eval",
                    *common_fed,
                ],
            )
            timings.append(("federated_quantum_dirichlet_reupload_round1", dt))
            f_q_dir_dir = os.path.join(
                results_dir,
                "flood_federated_optical_pytorch_pennylane_dirichlet0.5_reupload_L1_seed42",
            )
            _assert_federated_ckpt(f_q_dir_dir, 1)
            f_q_ck_p = os.path.join(f_q_dir_dir, tc.FED_CKPT_NAME)
            f_q_ck = torch.load(f_q_ck_p, map_location="cpu")
            f_q_ck["fingerprint"]["fed_rounds"] = 2
            torch.save(f_q_ck, f_q_ck_p)

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
            _assert_federated_ckpt(f_q_dir_dir, 2)
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
    run_verification(work_dir=args.work_dir, max_seconds=args.max_seconds, device=args.device)


if __name__ == "__main__":
    main()

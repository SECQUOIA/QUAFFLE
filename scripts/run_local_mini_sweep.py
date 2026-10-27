#!/usr/bin/env python3
"""Multi-seed local mini-benchmark exploration harness for QUAFFLE.

Runs a structured, multi-seed comparative sweep on a synthetic mini-flood
dataset (`--base-channels 8 --image-size 32` by default) to preview comparative
trends locally on a workstation before launching full-scale SLURM campaigns on
Purdue Anvil:

  * Panel 1 (Ansatz Expressivity & Bottleneck Architecture — Centralized `optical`):
      - Classical U-Net
      - Classical Ablation (Linear, matched to Legacy L=1)
      - Quantum QUAFFLE (Legacy L=1, BasicEntanglerLayers)
      - Classical Ablation (MLP, matched to Reupload L=2)
      - Quantum QUAFFLE (Reupload L=2, StronglyEntanglingLayers)
      - Classical Ablation (HQ-UNet Spatial 2x2)
      - Quantum HQ-UNet (2x2 QCNN, L=1)
  * Panel 2 (Wave 3 Multimodal Input Stems — Centralized `combined`, Quantum `Reupload L=2`):
      - `shared3` (3-channel shared `init_conv`)
      - `concat6` (6-channel `[Optical(3), SAR(3)]` `init_conv`)
      - `dual_stem` (separate 3-channel optical `init_conv` and SAR `init_conv_sar` stems)
  * Panel 3 (Wave 3 Federated Heterogeneity — `combined` Non-IID Modality Silos, `dual_stem`):
      - Classical Fed (Modality Silos, FedAvg)
      - Ablation Fed (Modality Silos, FedAvg)
      - Quantum Fed (Modality Silos, FedAvg + Linear Angle Agg)
      - Quantum Fed (Modality Silos, FedProx mu=0.05 + Circular Angle Agg)
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ablation_studies  # noqa: E402
import analyze_results  # noqa: E402
import centralized_demoTorchClassical  # noqa: E402
import centralized_demoTorchQ  # noqa: E402
import federated_demoTorchClassical  # noqa: E402
import federated_demoTorchQ  # noqa: E402
from tools import make_mini_flood_dataset  # noqa: E402
from tools import training_common as tc  # noqa: E402


def _invoke_entry(entry_fn: Callable[[], Any], argv: List[str]) -> float:
    old_argv = sys.argv[:]
    t0 = time.perf_counter()
    try:
        sys.argv = list(argv)
        entry_fn()
    finally:
        sys.argv = old_argv
    return time.perf_counter() - t0


_SCHEMA_DEFAULTS: Dict[str, Any] = {
    "training": "centralized",
    "dataset": "optical",
    "base_channels": 32,
    "image_size": 128,
    "combined_input_mode": "shared3",
    "backbone_conv_mode": "ws_conv",
    "loss": "ce",
    "use_time_emb": True,
    "augment": False,
    "q_ansatz": "reupload",
    "q_layers": 2,
    "n_qubits": 4,
    "quantum_channels": 32,
    "q_spatial_mode": "global",
    "hqunet_grid": 2,
    "skip_modulation": "affine",
    "ablation_mode": "auto",
    "ablation_mlp_mode": "param_matched",
    "ablation_mlp_hidden": None,
}


def _resolve_expected_quantum_channels(
    requested_qc: int,
    base_channels: int,
    n_qubits: int = 4,
    q_ansatz: str = "reupload",
    ablation_mode: str = "none",
    quantum_channels_explicit: bool = False,
) -> int:
    if quantum_channels_explicit:
        return int(requested_qc)
    mid_dim = int(base_channels) * 8
    group_w = 4 if (q_ansatz == "hqunet" or ablation_mode == "hqunet_spatial") else int(n_qubits)
    if int(requested_qc) > mid_dim:
        if mid_dim >= group_w:
            return max(group_w, (mid_dim // group_w) * group_w)
        return mid_dim
    return int(requested_qc)


def _expected_metadata_from_argv(argv: List[str]) -> Tuple[Dict[str, Any], set]:
    """Resolve complete expected run configuration (including defaults) and explicit keys from argv."""
    meta: Dict[str, Any] = {}
    explicit: set = set()
    if not argv:
        return meta, explicit
    script = os.path.basename(str(argv[0]))
    if "Classical" in script:
        model_type = "classical"
    elif "ablation" in script:
        model_type = "ablation"
    elif "TorchQ" in script:
        model_type = "quantum"
    else:
        model_type = None
    if model_type is not None:
        meta["model_type"] = model_type
        explicit.add("model_type")
    meta["training"] = "federated" if "federated" in script.lower() else "centralized"
    for k in ("dataset", "base_channels", "image_size", "combined_input_mode", "backbone_conv_mode", "loss", "use_time_emb", "augment"):
        meta[k] = _SCHEMA_DEFAULTS[k]
    if model_type in ("quantum", "ablation"):
        for k in ("q_ansatz", "q_layers", "n_qubits", "quantum_channels", "q_spatial_mode", "hqunet_grid", "skip_modulation"):
            meta[k] = _SCHEMA_DEFAULTS[k]
    if model_type == "ablation":
        meta["ablation_mode"] = _SCHEMA_DEFAULTS["ablation_mode"]
        meta["ablation_mlp_mode"] = _SCHEMA_DEFAULTS["ablation_mlp_mode"]
        meta["ablation_mlp_hidden"] = _SCHEMA_DEFAULTS["ablation_mlp_hidden"]

    flag_map = {
        "--mode": ("training", str),
        "--seed": ("seed", int),
        "--dataset": ("dataset", str),
        "--base-channels": ("base_channels", int),
        "--base_channels": ("base_channels", int),
        "--image-size": ("image_size", int),
        "--image_size": ("image_size", int),
        "--combined-input-mode": ("combined_input_mode", str),
        "--combined_input_mode": ("combined_input_mode", str),
        "--backbone-conv-mode": ("backbone_conv_mode", str),
        "--backbone_conv_mode": ("backbone_conv_mode", str),
        "--loss": ("loss", str),
        "--q-ansatz": ("q_ansatz", str),
        "--q_ansatz": ("q_ansatz", str),
        "--q-layers": ("q_layers", int),
        "--q_layers": ("q_layers", int),
        "--n-qubits": ("n_qubits", int),
        "--n_qubits": ("n_qubits", int),
        "--quantum-channels": ("quantum_channels", int),
        "--quantum_channels": ("quantum_channels", int),
        "--ablation-mode": ("ablation_mode", str),
        "--ablation_mode": ("ablation_mode", str),
        "--ablation-mlp-mode": ("ablation_mlp_mode", str),
        "--ablation_mlp_mode": ("ablation_mlp_mode", str),
        "--ablation-mlp-hidden": ("ablation_mlp_hidden", int),
        "--ablation_mlp_hidden": ("ablation_mlp_hidden", int),
        "--hqunet-grid": ("hqunet_grid", int),
        "--hqunet_grid": ("hqunet_grid", int),
        "--q-spatial-mode": ("q_spatial_mode", str),
        "--q_spatial_mode": ("q_spatial_mode", str),
        "--skip-modulation": ("skip_modulation", str),
        "--skip_modulation": ("skip_modulation", str),
        "--partition": ("partition", str),
        "--alpha": ("dirichlet_alpha", float),
        "--fl-strategy": ("fl_strategy", str),
        "--fl_strategy": ("fl_strategy", str),
        "--fedprox-mu": ("fedprox_mu", float),
        "--fedprox_mu": ("fedprox_mu", float),
        "--q-angle-agg": ("q_angle_agg", str),
        "--q_angle_agg": ("q_angle_agg", str),
        "--num-clients": ("num_clients", int),
        "--num_clients": ("num_clients", int),
        "--clients-per-round": ("clients_per_round", int),
        "--clients_per_round": ("clients_per_round", int),
        "--fraction-fit": ("fraction_fit", float),
        "--fraction_fit": ("fraction_fit", float),
    }
    i = 1
    while i < len(argv):
        tok = str(argv[i])
        if tok in ("--no-time-emb", "--no_time_emb"):
            meta["use_time_emb"] = False
            explicit.add("use_time_emb")
            i += 1
        elif tok == "--augment":
            meta["augment"] = True
            explicit.add("augment")
            i += 1
        elif "=" in tok and tok.split("=", 1)[0] in flag_map:
            flag_tok, val_str = tok.split("=", 1)
            key, cast = flag_map[flag_tok]
            meta[key] = cast(val_str)
            explicit.add(key)
            i += 1
        elif tok in flag_map and i + 1 < len(argv):
            key, cast = flag_map[tok]
            meta[key] = cast(argv[i + 1])
            explicit.add(key)
            i += 2
        else:
            i += 1

    # Normalize expected configuration from the requested options themselves
    if model_type in ("quantum", "ablation"):
        meta["quantum_channels"] = _resolve_expected_quantum_channels(
            int(meta["quantum_channels"]),
            int(meta["base_channels"]),
            n_qubits=int(meta.get("n_qubits", 4)),
            q_ansatz=str(meta.get("q_ansatz", "reupload")),
            ablation_mode=str(meta.get("ablation_mode", "auto")) if model_type == "ablation" else "none",
            quantum_channels_explicit=("quantum_channels" in explicit),
        )
        if meta.get("q_ansatz") == "hqunet" or meta.get("ablation_mode") == "hqunet_spatial":
            meta.pop("n_qubits", None)
            explicit.discard("n_qubits")
        else:
            meta.pop("hqunet_grid", None)
            explicit.discard("hqunet_grid")
        if meta.get("q_ansatz") == "legacy" and meta.get("ablation_mode") != "hqunet_spatial":
            meta["q_layers"] = None
    return meta, explicit


def _matches_expected_metrics(
    d: Dict[str, Any],
    expected_seed: Optional[int],
    expected_meta: Dict[str, Any],
    explicit_keys: Optional[set] = None,
) -> bool:
    explicit_set = set(explicit_keys or ())
    if expected_seed is not None and int(d.get("seed", -1)) != int(expected_seed):
        return False
    cfg = d.get("config") if isinstance(d.get("config"), dict) else {}
    exp_bc = int(expected_meta.get("base_channels", _SCHEMA_DEFAULTS["base_channels"]))
    for k, exp_v in expected_meta.items():
        has_key = (k in d and d.get(k) is not None) or (k in cfg and cfg.get(k) is not None)
        actual_v = d.get(k) if d.get(k) is not None else cfg.get(k)
        if k == "model_type" and actual_v is None:
            actual_v = d.get("variant") or cfg.get("variant")
            has_key = actual_v is not None
        if k == "q_layers" and exp_v is None:
            if actual_v not in (None, 1):
                return False
            continue
        if exp_v is None:
            if actual_v is not None:
                return False
            continue
        if not has_key:
            if k in explicit_set:
                return False
            # Candidate omitted an unrequested key (e.g., minimal test mock); evaluate against schema default
            if k == "quantum_channels":
                actual_v = _resolve_expected_quantum_channels(
                    int(_SCHEMA_DEFAULTS["quantum_channels"]),
                    exp_bc,
                    n_qubits=int(expected_meta.get("n_qubits", 4)),
                    q_ansatz=str(expected_meta.get("q_ansatz", "reupload")),
                    ablation_mode=(
                        str(expected_meta.get("ablation_mode", "auto"))
                        if expected_meta.get("model_type") == "ablation"
                        else "none"
                    ),
                    quantum_channels_explicit=False,
                )
            else:
                actual_v = _SCHEMA_DEFAULTS.get(k)
        if actual_v is None:
            return False
        if isinstance(exp_v, bool):
            if bool(actual_v) != bool(exp_v):
                return False
        elif isinstance(exp_v, int):
            if int(actual_v) != int(exp_v):
                return False
        elif isinstance(exp_v, float):
            if abs(float(actual_v) - float(exp_v)) > 1e-9:
                return False
        else:
            if str(actual_v) != str(exp_v):
                return False
    return True


def _snapshot_metrics_files(results_dir: str) -> Dict[str, Tuple[int, int]]:
    pattern = os.path.join(results_dir, "*", "results", "metrics.json")
    snap: Dict[str, Tuple[int, int]] = {}
    for p in glob.glob(pattern):
        try:
            st = os.stat(p)
            snap[p] = (int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            continue
    return snap


def _find_metrics_json_since(
    results_dir: str,
    before_paths: Any,
    expected_seed: Optional[int] = None,
    expected_argv: Optional[List[str]] = None,
) -> str:
    pattern = os.path.join(results_dir, "*", "results", "metrics.json")
    now_paths = sorted(glob.glob(pattern))
    if isinstance(before_paths, dict):
        candidates = []
        for p in now_paths:
            try:
                st = os.stat(p)
                cur_sig = (int(st.st_mtime_ns), int(st.st_size))
            except OSError:
                continue
            if p not in before_paths or cur_sig != before_paths[p]:
                candidates.append(p)
    else:
        before_set = set(before_paths or ())
        candidates = [p for p in now_paths if p not in before_set]
    if not candidates:
        raise FileNotFoundError(f"No newly written metrics.json found under {results_dir}")
    expected_meta, explicit_keys = _expected_metadata_from_argv(expected_argv or [])
    if expected_seed is None and not expected_meta:
        if len(candidates) > 1:
            raise RuntimeError(f"Ambiguous metrics.json candidates (no expected_seed/argv provided): {candidates}")
        return candidates[0]
    matched = []
    for p in candidates:
        try:
            with open(p, "r") as f:
                d = json.load(f)
            if _matches_expected_metrics(d, expected_seed, expected_meta, explicit_keys):
                matched.append(p)
        except Exception:
            continue
    if not matched:
        raise FileNotFoundError(
            f"No metrics.json under {results_dir} matched seed={expected_seed}, meta={expected_meta} "
            f"(candidates={candidates})"
        )
    if len(matched) > 1:
        raise RuntimeError(
            f"Ambiguous metrics.json matches for seed={expected_seed}, meta={expected_meta}: {matched}"
        )
    return matched[0]


def _summarize_seed_runs(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(key_fn):
        out = []
        for r in records:
            v = key_fn(r)
            if v is not None and not np.isnan(float(v)):
                out.append(float(v))
        return out

    def _stat(arr):
        if not arr:
            return {"mean": None, "std": None, "n": 0}
        a = np.asarray(arr, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": int(len(a)),
        }

    test_iou = _vals(lambda r: r["test"].get("iou"))
    test_dice = _vals(lambda r: r["test"].get("dice"))
    test_auc = _vals(lambda r: r["test"].get("auc"))
    test_loss = _vals(lambda r: r["test"].get("loss"))
    val_iou = _vals(lambda r: r["val"].get("iou"))
    ko_delta_iou = _vals(
        lambda r: (
            float(r["test"]["iou"]) - float(r["knockout"]["zero"]["iou"])
            if isinstance(r.get("knockout"), dict)
            and isinstance(r["knockout"].get("zero"), dict)
            and r["knockout"]["zero"].get("iou") is not None
            else None
        )
    )
    ko_delta_loss = _vals(
        lambda r: (
            float(r["knockout"]["zero"]["loss"]) - float(r["test"]["loss"])
            if isinstance(r.get("knockout"), dict)
            and isinstance(r["knockout"].get("zero"), dict)
            and r["knockout"]["zero"].get("loss") is not None
            else None
        )
    )
    params = [int(r["total_params"]) for r in records if r.get("total_params") is not None]
    elapsed = [float(r["_elapsed_s"]) for r in records if r.get("_elapsed_s") is not None]
    return {
        "n_seeds": len(records),
        "total_params": params[0] if params else None,
        "val_iou": _stat(val_iou),
        "test_iou": _stat(test_iou),
        "test_dice": _stat(test_dice),
        "test_auc": _stat(test_auc),
        "test_loss": _stat(test_loss),
        "ko_zero_delta_iou": _stat(ko_delta_iou),
        "ko_zero_delta_loss": _stat(ko_delta_loss),
        "mean_elapsed_s": float(np.mean(elapsed)) if elapsed else 0.0,
    }


def _fmt_ms(stat_dict: Dict[str, Any], signed: bool = False) -> str:
    m = stat_dict.get("mean")
    s = stat_dict.get("std")
    if m is None:
        return "—"
    if signed:
        return f"{m:+.4f} ± {s:.4f}"
    return f"{m:.4f} ± {s:.4f}"


def _format_markdown_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Local Multi-Seed Mini-Benchmark Exploration Report",
        "",
        f"- **Seeds**: `{summary['seeds']}` (`n={len(summary['seeds'])}`)",
        f"- **Scale**: `base_channels={summary['base_channels']}`, `image_size={summary['image_size']}`, "
        f"`epochs={summary['epochs']}`, `fed_rounds={summary['rounds']}`, `local_epochs={summary['local_epochs']}`",
        f"- **Mini Dataset**: `{summary['n_train']}` training + `{summary['n_test']}` test tiles per sensor "
        f"(`{2 * summary['n_train']}` training + `{2 * summary['n_test']}` test in `flood_combined`)",
        f"- **Device**: `{summary['device']}` | **Total Wall Time**: `{summary['total_elapsed_s']:.1f}s`",
        "",
    ]
    for panel in summary["panels"]:
        lines.append(f"## {panel['title']}")
        lines.append("")
        lines.append(
            "| Spec / Variant | Params | Val IoU | Test IoU | Test Dice | Test AUC | Test Loss | Knockout $\\Delta$IoU (`on - zero`) | Time/seed |"
        )
        lines.append("| --- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |")
        for row in panel["rows"]:
            st = row["stats"]
            p_str = f"{st['total_params']:,}" if st.get("total_params") is not None else "—"
            lines.append(
                f"| {row['label']} | {p_str} | {_fmt_ms(st['val_iou'])} | "
                f"**{_fmt_ms(st['test_iou'])}** | {_fmt_ms(st['test_dice'])} | "
                f"{_fmt_ms(st['test_auc'])} | {_fmt_ms(st['test_loss'])} | "
                f"{_fmt_ms(st['ko_zero_delta_iou'], signed=True)} | {st['mean_elapsed_s']:.2f}s |"
            )
        lines.append("")
    return "\n".join(lines)


def run_mini_sweep(
    *,
    seeds: Sequence[int] = (42, 77, 123),
    base_channels: int = 8,
    image_size: int = 32,
    epochs: int = 4,
    rounds: int = 3,
    local_epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 5e-3,
    n_train: int = 16,
    n_test: int = 6,
    num_clients: int = 4,
    min_client_samples: int = 2,
    device: str = "auto",
    work_dir: Optional[str] = None,
    keep_work_dir: bool = False,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
    panels_filter: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("seeds must not be empty.")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError(f"Duplicate seeds are not allowed: {seed_list}")

    valid_panel_ids = {1, 2, 3, 4}
    default_panel_ids = {1, 2, 3}
    if panels_filter is not None:
        panel_list = [int(p) for p in panels_filter]
        if not panel_list or not set(panel_list).issubset(valid_panel_ids):
            raise ValueError(
                f"panels_filter must be a non-empty subset of {sorted(valid_panel_ids)}, got {panel_list}"
            )
        if len(set(panel_list)) != len(panel_list):
            raise ValueError(f"Duplicate panel IDs are not allowed: {panel_list}")
        active_panels = set(panel_list)
    else:
        active_panels = set(default_panel_ids)

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be 'auto', 'cpu', or 'cuda', got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")

    _env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "QUAFFLE_SKIP_EXCEL_TRACKER",
        "QUAFFLE_SKIP_CURVE_PLOTS",
    )
    _saved_env = {k: os.environ.get(k) for k in _env_keys}

    created_tmp = work_dir is None
    root_dir = work_dir or tempfile.mkdtemp(prefix="quaffle_mini_sweep_")
    data_root = os.path.join(root_dir, "mini_data")
    results_dir = os.path.join(root_dir, "results")

    try:
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            eff_device = "cpu"
        elif device == "cuda" or torch.cuda.is_available():
            eff_device = "cuda"
        else:
            eff_device = "cpu"

        os.environ["QUAFFLE_SKIP_EXCEL_TRACKER"] = "1"
        os.environ["QUAFFLE_SKIP_CURVE_PLOTS"] = "1"

        if os.path.exists(data_root):
            shutil.rmtree(data_root)
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        make_mini_flood_dataset.generate_mini_flood_dataset(
            data_root,
            n_train=int(n_train),
            n_test=int(n_test),
            image_size=int(image_size),
            seed=seed_list[0],
            seeds=tuple(seed_list),
        )

        common_c = [
            "--data-root", data_root,
            "--results-dir", results_dir,
            "--base-channels", str(base_channels),
            "--image-size", str(image_size),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--epochs", str(epochs),
            "--eval-every", "1",
            "--loss", "dice_ce",
            "--no-time-emb",
            "--no-save-samples",
        ]
        common_f = [
            "--data-root", data_root,
            "--results-dir", results_dir,
            "--base-channels", str(base_channels),
            "--image-size", str(image_size),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--rounds", str(rounds),
            "--local-epochs", str(local_epochs),
            "--num-clients", str(num_clients),
            "--clients-per-round", str(num_clients),
            "--min-client-samples", str(min_client_samples),
            "--ckpt-every", str(max(1, rounds)),
            "--loss", "dice_ce",
            "--no-time-emb",
            "--no-save-samples",
        ]

        panel_defs: List[Tuple[int, str, List[Tuple[str, Callable[[], Any], List[str]]]]] = [
            (
                1,
                "Panel 1: Ansatz Expressivity & Bottleneck Architecture (Centralized `optical`)",
                [
                    (
                        "Classical U-Net",
                        centralized_demoTorchClassical.main,
                        ["centralized_demoTorchClassical.py", "--dataset", "optical", *common_c],
                    ),
                    (
                        "Ablation (Linear, matched to Legacy L=1)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "centralized",
                            "--dataset", "optical", "--q-ansatz", "legacy", "--ablation-mode", "linear",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum QUAFFLE (Legacy L=1, Clifford-reducible)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "legacy",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Ablation (MLP, matched to Reupload L=2)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "centralized",
                            "--dataset", "optical", "--q-ansatz", "reupload", "--q-layers", "2",
                            "--ablation-mode", "mlp", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum QUAFFLE (Reupload L=2, StronglyEntangling)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Ablation (HQ-UNet Spatial 2x2)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "centralized",
                            "--dataset", "optical", "--q-ansatz", "hqunet", "--hqunet-grid", "2",
                            "--q-layers", "1", "--ablation-mode", "hqunet_spatial",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum HQ-UNet (2x2 QCNN, RX-RY-RZ + <Z>,<X>)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "hqunet",
                            "--hqunet-grid", "2", "--q-layers", "1", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                ],
            ),
            (
                2,
                "Panel 2: Wave 3 Multimodal Input Stems (Centralized `combined`, Quantum `Reupload L=2`)",
                [
                    (
                        "Quantum Reupload L=2 (`shared3` stem)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "combined", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--combined-input-mode", "shared3",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum Reupload L=2 (`concat6` stem)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "combined", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--combined-input-mode", "concat6",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum Reupload L=2 (`dual_stem` Optical/SAR stems)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "combined", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--combined-input-mode", "dual_stem",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                ],
            ),
            (
                3,
                "Panel 3: Wave 3 Federated Heterogeneity (`combined` Non-IID Modality Silos, `dual_stem`)",
                [
                    (
                        "Classical Fed (Modality Silos, FedAvg)",
                        federated_demoTorchClassical.main,
                        [
                            "federated_demoTorchClassical.py", "--dataset", "combined",
                            "--partition", "modality", "--combined-input-mode", "dual_stem",
                            "--fl-strategy", "fedavg", *common_f,
                        ],
                    ),
                    (
                        "Ablation Fed (Modality Silos, FedAvg)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "federated",
                            "--dataset", "combined", "--partition", "modality",
                            "--combined-input-mode", "dual_stem", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--fl-strategy", "fedavg",
                            "--quantum-knockout-eval", *common_f,
                        ],
                    ),
                    (
                        "Quantum Fed (Modality Silos, FedAvg + Linear Angle Agg)",
                        federated_demoTorchQ.main,
                        [
                            "federated_demoTorchQ.py", "--dataset", "combined",
                            "--partition", "modality", "--combined-input-mode", "dual_stem",
                            "--q-ansatz", "reupload", "--q-layers", "2",
                            "--fl-strategy", "fedavg", "--q-angle-agg", "linear",
                            "--quantum-knockout-eval", *common_f,
                        ],
                    ),
                    (
                        "Quantum Fed (Modality Silos, FedProx mu=0.05 + Circular S^1 Agg)",
                        federated_demoTorchQ.main,
                        [
                            "federated_demoTorchQ.py", "--dataset", "combined",
                            "--partition", "modality", "--combined-input-mode", "dual_stem",
                            "--q-ansatz", "reupload", "--q-layers", "2",
                            "--fl-strategy", "fedprox", "--fedprox-mu", "0.05",
                            "--q-angle-agg", "circular", "--quantum-knockout-eval", *common_f,
                        ],
                    ),
                ],
            ),
            (
                4,
                "Panel 4: Qubit-Width & Quantum-Channel Scaling (Centralized `optical`, `n_qubits` in `{4, 8, 16}`, `qc` in `{16, 32}`)",
                [
                    (
                        "Ablation (MLP, matched to Reupload L=2, n_qubits=4, qc=16)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "centralized",
                            "--dataset", "optical", "--q-ansatz", "reupload", "--q-layers", "2",
                            "--n-qubits", "4", "--quantum-channels", "16",
                            "--ablation-mode", "mlp", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum QUAFFLE (Reupload L=2, n_qubits=4, qc=16, 4x4q)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--n-qubits", "4", "--quantum-channels", "16",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Ablation (MLP, matched to Reupload L=2, n_qubits=8, qc=16)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "centralized",
                            "--dataset", "optical", "--q-ansatz", "reupload", "--q-layers", "2",
                            "--n-qubits", "8", "--quantum-channels", "16",
                            "--ablation-mode", "mlp", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum QUAFFLE (Reupload L=2, n_qubits=8, qc=16, 2x8q)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--n-qubits", "8", "--quantum-channels", "16",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Ablation (MLP, matched to Reupload L=2, n_qubits=16, qc=16)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "centralized",
                            "--dataset", "optical", "--q-ansatz", "reupload", "--q-layers", "2",
                            "--n-qubits", "16", "--quantum-channels", "16",
                            "--ablation-mode", "mlp", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum QUAFFLE (Reupload L=2, n_qubits=16, qc=16, 1x16q)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--n-qubits", "16", "--quantum-channels", "16",
                            "--q-backend", "torch", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Ablation (MLP, matched to Reupload L=2, n_qubits=4, qc=32)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "centralized",
                            "--dataset", "optical", "--q-ansatz", "reupload", "--q-layers", "2",
                            "--n-qubits", "4", "--quantum-channels", "32",
                            "--ablation-mode", "mlp", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum QUAFFLE (Reupload L=2, n_qubits=4, qc=32, 8x4q)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--n-qubits", "4", "--quantum-channels", "32",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Ablation (MLP, matched to Reupload L=2, n_qubits=8, qc=32)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "centralized",
                            "--dataset", "optical", "--q-ansatz", "reupload", "--q-layers", "2",
                            "--n-qubits", "8", "--quantum-channels", "32",
                            "--ablation-mode", "mlp", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum QUAFFLE (Reupload L=2, n_qubits=8, qc=32, 4x8q)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--n-qubits", "8", "--quantum-channels", "32",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Ablation (MLP, matched to Reupload L=2, n_qubits=16, qc=32)",
                        ablation_studies.main,
                        [
                            "ablation_studies.py", "--framework", "torch", "--mode", "centralized",
                            "--dataset", "optical", "--q-ansatz", "reupload", "--q-layers", "2",
                            "--n-qubits", "16", "--quantum-channels", "32",
                            "--ablation-mode", "mlp", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum QUAFFLE (Reupload L=2, n_qubits=16, qc=32, 2x16q)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "reupload",
                            "--q-layers", "2", "--n-qubits", "16", "--quantum-channels", "32",
                            "--q-backend", "torch", "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                    (
                        "Quantum HQ-UNet (4x4 QCNN, 16 qubits, L=1)",
                        centralized_demoTorchQ.main,
                        [
                            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "hqunet",
                            "--hqunet-grid", "4", "--q-layers", "1", "--quantum-channels", "32",
                            "--quantum-knockout-eval", *common_c,
                        ],
                    ),
                ],
            ),
        ]

        panels_out = []

        for panel_id, panel_title, specs in panel_defs:
            if panel_id not in active_panels:
                continue
            rows_out = []
            for label, entry_fn, base_argv in specs:
                seed_records = []
                for s in seed_list:
                    before = _snapshot_metrics_files(results_dir)
                    argv = [*base_argv, "--seed", str(s)]
                    dt = _invoke_entry(entry_fn, argv)
                    mpath = _find_metrics_json_since(
                        results_dir, before, expected_seed=int(s), expected_argv=argv
                    )
                    with open(mpath, "r") as f:
                        mdata = json.load(f)
                    mdata["_elapsed_s"] = dt
                    mdata["_metrics_path"] = mpath
                    seed_records.append(mdata)
                stats = _summarize_seed_runs(seed_records)
                rows_out.append({
                    "label": label,
                    "stats": stats,
                    "seeds": {int(r["seed"]): r["test"] for r in seed_records},
                })
            panels_out.append({
                "panel_id": panel_id,
                "title": panel_title,
                "rows": rows_out,
            })

        def _write_outputs(summary_dict: Dict[str, Any], md_text: str) -> None:
            if out_json:
                out_json_dir = os.path.dirname(os.path.abspath(out_json))
                if out_json_dir:
                    os.makedirs(out_json_dir, exist_ok=True)
                with open(out_json, "w") as f:
                    json.dump(summary_dict, f, indent=2)
            if out_md:
                out_md_dir = os.path.dirname(os.path.abspath(out_md))
                if out_md_dir:
                    os.makedirs(out_md_dir, exist_ok=True)
                with open(out_md, "w") as f:
                    f.write(md_text)

        total_elapsed = time.perf_counter() - t_start
        summary = {
            "seeds": seed_list,
            "base_channels": int(base_channels),
            "image_size": int(image_size),
            "epochs": int(epochs),
            "rounds": int(rounds),
            "local_epochs": int(local_epochs),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "device": eff_device,
            "total_elapsed_s": total_elapsed,
            "panels": panels_out,
        }
        md_report = _format_markdown_report(summary)
        _write_outputs(summary, md_report)

        analysis_dir = os.path.join(results_dir, "analysis")
        _invoke_entry(
            analyze_results.main,
            [
                "analyze_results.py",
                "--results-dir", results_dir,
                "--out-dir", analysis_dir,
                "--include-mini",
                "--seeds", *[str(s) for s in seed_list],
            ],
        )

        summary["total_elapsed_s"] = time.perf_counter() - t_start
        md_report = _format_markdown_report(summary)
        print("\n" + md_report)
        _write_outputs(summary, md_report)
        return summary
    finally:
        for k, val in _saved_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        if created_tmp and not keep_work_dir and os.path.exists(root_dir):
            shutil.rmtree(root_dir, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 77, 123])
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--n-train", type=int, default=16)
    parser.add_argument("--n-test", type=int, default=6)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--min-client-samples", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-md", type=str, default=None)
    parser.add_argument("--panels", type=int, nargs="+", choices=[1, 2, 3, 4], default=None)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error(f"Duplicate seeds are not allowed: {args.seeds}")
    if args.panels is not None and len(set(args.panels)) != len(args.panels):
        parser.error(f"Duplicate panel IDs are not allowed: {args.panels}")
    run_mini_sweep(
        seeds=args.seeds,
        base_channels=args.base_channels,
        image_size=args.image_size,
        epochs=args.epochs,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_train=args.n_train,
        n_test=args.n_test,
        num_clients=args.num_clients,
        min_client_samples=args.min_client_samples,
        device=args.device,
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
        out_json=args.out_json,
        out_md=args.out_md,
        panels_filter=args.panels,
    )


if __name__ == "__main__":
    main()

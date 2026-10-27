"""Coverage of the opt-in Wave 4 comparisons and retained paired-seed evidence."""

from scripts import run_local_mini_sweep as sweep


def test_wave4_panels_cover_independent_architecture_and_compression_comparisons():
    panels = sweep._wave4_panel_defs(["--no-time-emb"], ["--no-time-emb"], 8)
    architecture = panels[0][2]
    compression = panels[1][2]
    assert len(architecture) == 14
    settings = [sweep._expected_metadata_from_argv(argv)[0] for _, _, argv in architecture]
    assert sum(s["model_type"] == "classical" for s in settings) == 2
    for qc in (32, 64):
        for gate in ("none", "affine"):
            group = [s for s in settings if s.get("quantum_channels") == qc and s["decoder_skip_gate"] == gate]
            assert {s["model_type"] for s in group} == {"quantum", "ablation"}
            assert {s["ablation_mlp_mode"] for s in group if s["model_type"] == "ablation"} == {"param_matched", "full_rank"}
    assert all(s["use_time_emb"] is False for s in settings)
    assert all("--quantum-knockout-eval" in argv for _, _, argv in architecture if "Classical" not in argv[0])

    assert len(compression) == 12
    settings = [sweep._expected_metadata_from_argv(argv)[0] for _, _, argv in compression]
    assert {(s["model_type"], s["comm_compression"]) for s in settings} == {
        (model, mode) for model in ("classical", "ablation", "quantum")
        for mode in ("none", "fp16", "int8", "topk10")
    }
    assert all(s["training"] == "federated" and s["partition"] == "modality" for s in settings)
    small = sweep._wave4_panel_defs([], [], 4)[0][2]
    assert {sweep._expected_metadata_from_argv(argv)[0].get("quantum_channels") for _, _, argv in small} == {None, 16, 32}


def test_retained_evidence_preserves_knockouts_and_payloads_without_private_paths():
    record = {
        "seed": 42, "val": {"iou": 0.6}, "test": {"iou": 0.7, "loss": 0.4},
        "knockout": {"zero": {"iou": 0.5, "loss": 0.6}, "random": {"iou": 0.8}},
        "total_params": 1000, "upload_mib_per_client_round": 0.002,
        "compression_ratio": 0.5, "_elapsed_s": 2.0,
        "config": {"data_root": "/private/data"}, "_metrics_path": "/private/results/metrics.json",
    }
    argv = ["federated_demoTorchQ.py", "--seed", "42", "--comm-compression", "fp16"]
    retained = sweep._retain_seed_record(record, argv)
    assert retained["knockout"] == record["knockout"]
    assert retained["settings"]["comm_compression"] == "fp16"
    assert "/private" not in str(retained)
    stats = sweep._summarize_seed_runs([record])
    assert abs(stats["ko_zero_delta_iou"]["mean"] - 0.2) < 1e-12
    assert abs(stats["ko_random_delta_iou"]["mean"] + 0.1) < 1e-12
    assert stats["upload_mib_per_client_round"]["mean"] == 0.002
    assert stats["compression_ratio"]["mean"] == 0.5


def test_missing_timings_are_reported_as_unavailable():
    record = {"seed": 42, "val": {"iou": 0.6}, "test": {"iou": 0.7}}
    stats = sweep._summarize_seed_runs([record])
    assert stats["mean_elapsed_s"] is None
    summary = {
        "seeds": [42], "base_channels": 8, "image_size": 32, "epochs": 4,
        "rounds": 3, "local_epochs": 2, "n_train": 16, "n_test": 6,
        "device": "cuda", "total_elapsed_s": None,
        "panels": [{"title": "Recovered metrics", "rows": [{"label": "model", "stats": stats}]}],
    }
    report = sweep._format_markdown_report(summary)
    assert "unavailable" in report
    assert "0.00s" not in report

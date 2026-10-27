"""Regression tests for Issue #43: Full-bottleneck replacement (--quantum-channels 256)
and bottleneck-conditioned encoder-decoder skip gating (--decoder-skip-gate {none,se,affine})."""

import argparse
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import training_common as tc
from utilsTorch import unetTorch
from utilsTorch.utilsTorch import (
    ClassicalAblationUNetSegmentation,
    QVUNetSegmentation,
    UNetSegmentation,
    evaluate_model,
)


def _base_config(**overrides):
    cfg = {
        "image_size": 32,
        "base_channels": 8,
        "init_dim": 8,
        "out_dim": 2,
        "dim_mults": (1, 2, 4, 8),
        "resnet_block_groups": 4,
        "quantum_channels": 64,
        "quantum_backend": "pennylane",
        "q_ansatz": "reupload",
        "q_layers": 2,
        "q_backend": "torch",
        "q_sim_device": "cpu",
        "batch_size": 2,
        "learning_rate": 1e-3,
        "num_epochs": 2,
        "num_clients": 2,
        "fed_rounds": 2,
        "local_epochs": 1,
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


def test_full_bottleneck_parameter_counts_at_base_channels_32():
    """Verify exact parameter counts at base_channels=32, dim_mults=(1,2,4,8), quantum_channels=256,
    q_ansatz='reupload', q_layers=2 with and without --no-time-emb (#43)."""
    cfg_cls_time = _base_config(base_channels=32, init_dim=32, use_time_emb=True, seed=None)
    cfg_cls_notime = _base_config(base_channels=32, init_dim=32, use_time_emb=False, seed=None)
    m_cls_time = tc.build_model(UNetSegmentation, cfg_cls_time)
    m_cls_notime = tc.build_model(UNetSegmentation, cfg_cls_notime)
    assert sum(p.numel() for p in m_cls_time.parameters()) == 9_327_970
    assert sum(p.numel() for p in m_cls_notime.parameters()) == 8_787_106

    cfg_q_time = _base_config(
        base_channels=32,
        init_dim=32,
        quantum_channels=256,
        q_ansatz="reupload",
        q_layers=2,
        use_time_emb=True,
        seed=None,
    )
    cfg_q_notime = _base_config(
        base_channels=32,
        init_dim=32,
        quantum_channels=256,
        q_ansatz="reupload",
        q_layers=2,
        use_time_emb=False,
        seed=None,
    )
    m_q_time = tc.build_model(QVUNetSegmentation, cfg_q_time)
    m_q_notime = tc.build_model(QVUNetSegmentation, cfg_q_notime)
    assert sum(p.numel() for p in m_q_time.parameters()) == 6_835_554
    assert sum(p.numel() for p in m_q_notime.parameters()) == 6_426_786
    assert m_q_notime.model.vertex.qresblock1.classical_channels == 0
    assert m_q_notime.model.vertex.qresblock1.conv0 is None


def test_full_bottleneck_runs_across_all_ansatze_and_ablation_modes():
    """Verify full-bottleneck replacement (quantum_channels == mid_dim) forward/backward and
    knockout ('on', 'zero', 'random') across legacy, reupload, hqunet, and all 6 ablation modes."""
    x = torch.randn(2, 3, 32, 32)

    for ansatz in ("legacy", "reupload", "hqunet"):
        cfg = _base_config(
            base_channels=4,
            init_dim=4,
            quantum_channels=32,
            q_ansatz=ansatz,
            q_layers=2,
            use_time_emb=False,
        )
        qm = tc.build_model(QVUNetSegmentation, cfg)
        assert qm.model.vertex.qresblock1.effective_qc == 32
        assert qm.model.vertex.qresblock1.classical_channels == 0
        out_on = qm(x)
        assert out_on.shape == (2, 2, 32, 32)
        loss = out_on.pow(2).mean()
        loss.backward()

        qm.eval()
        with torch.no_grad():
            y_on = qm(x).clone()
            qm.set_quantum_mode("zero")
            y_zero = qm(x).clone()
            qm.set_quantum_mode("random", knockout_seed=7)
            y_rand = qm(x).clone()
            qm.set_quantum_mode("on")
            y_restored = qm(x).clone()
        assert torch.allclose(y_on, y_restored)
        assert not torch.allclose(y_on, y_zero)
        assert not torch.allclose(y_zero, y_rand)

    for abl_mode in ("linear", "mlp", "conv1x1", "depthwise3x3", "lowrank", "hqunet_spatial"):
        cfg_abl = _base_config(
            base_channels=4,
            init_dim=4,
            quantum_channels=32,
            q_ansatz="reupload",
            q_layers=2,
            ablation_mode=abl_mode,
            use_time_emb=False,
        )
        am = tc.build_model(ClassicalAblationUNetSegmentation, cfg_abl)
        assert am.model.vertex.qresblock1.effective_qc == 32
        assert am.model.vertex.qresblock1.classical_channels == 0
        out_a = am(x)
        assert out_a.shape == (2, 2, 32, 32)
        out_a.pow(2).mean().backward()

        am.eval()
        with torch.no_grad():
            ya_on = am(x).clone()
            am.set_quantum_mode("zero")
            ya_zero = am(x).clone()
            am.set_quantum_mode("random", knockout_seed=7)
            ya_rand = am(x).clone()
            am.set_quantum_mode("on")
            ya_restored = am(x).clone()
        assert torch.allclose(ya_on, ya_restored)
        assert not torch.allclose(ya_on, ya_zero)
        assert not torch.allclose(ya_zero, ya_rand)


def test_decoder_skip_gate_modes_and_paired_seed_invariance():
    """Verify --decoder-skip-gate {none,se,affine} construction, forward/backward gradients,
    and paired-seed weight equivalence between QVUNetSegmentation and ClassicalAblationUNetSegmentation."""
    x = torch.randn(2, 3, 32, 32)
    cfg_none = _base_config(quantum_channels=8, decoder_skip_gate="none")
    m_none = tc.build_model(QVUNetSegmentation, cfg_none)
    assert m_none.model.up.skip_gates is None
    assert m_none.model.up.final_skip_gate is None

    for gate_mode in ("se", "affine"):
        cfg_g = _base_config(quantum_channels=8, decoder_skip_gate=gate_mode, use_time_emb=False)
        qm = tc.build_model(QVUNetSegmentation, cfg_g)
        am = tc.build_model(ClassicalAblationUNetSegmentation, cfg_g)
        cm = tc.build_model(UNetSegmentation, cfg_g)

        assert qm.model.up.decoder_skip_gate == gate_mode
        assert len(qm.model.up.skip_gates) == 4
        assert qm.model.up.final_skip_gate is not None

        # Paired-seed non-bottleneck weights (including skip_gates) match 100% between Quantum and Ablation
        q_sd = qm.state_dict()
        a_sd = am.state_dict()
        shared_keys = [k for k in q_sd if k in a_sd and "quantum_block" not in k]
        for k in shared_keys:
            assert torch.equal(q_sd[k], a_sd[k]), f"Mismatch at {k} under decoder_skip_gate={gate_mode}"

        # Forward + backward through all skip gates and bottleneck parameters
        logits = qm(x)
        assert logits.shape == (2, 2, 32, 32)
        logits.pow(2).mean().backward()
        for p in qm.model.up.skip_gates.parameters():
            assert p.grad is not None and torch.any(p.grad != 0)
        for p in qm.model.up.final_skip_gate.parameters():
            assert p.grad is not None and torch.any(p.grad != 0)
        for p in qm.model.vertex.qresblock1.quantum_block.parameters():
            assert p.grad is not None and torch.any(p.grad != 0)

        # Classical U-Net also supports decoder_skip_gate
        logits_c = cm(x)
        assert logits_c.shape == (2, 2, 32, 32)


def test_knockout_attribution_nonzero_delta_with_full_bottleneck_and_skip_gate():
    """Verify post-hoc knockout ('on', 'zero', 'random') produces a non-trivial IoU delta
    when full-bottleneck replacement and decoder skip gating are enabled and trained briefly."""
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    imgs = torch.randn(4, 3, 32, 32)
    masks = (imgs[:, 0] > 0.0).long()
    loader = DataLoader(TensorDataset(imgs, masks), batch_size=4, shuffle=False)

    cfg = _base_config(
        base_channels=4,
        init_dim=4,
        quantum_channels=32,
        q_ansatz="reupload",
        q_layers=2,
        decoder_skip_gate="affine",
        use_time_emb=False,
        seed=42,
    )
    model = tc.build_model(QVUNetSegmentation, cfg)
    # Train the quantum bottleneck and decoder skip gates while freezing encoder skip convs
    # so the model relies directly on bottleneck + decoder skip gate modulation
    trainable_params = (
        list(model.model.vertex.parameters())
        + list(model.model.up.skip_gates.parameters())
        + list(model.model.up.final_skip_gate.parameters())
        + list(model.model.up.final_conv2.parameters())
    )
    opt = torch.optim.Adam(trainable_params, lr=5e-2)
    model.train()
    for _ in range(15):
        opt.zero_grad()
        logits = model(imgs)
        loss = torch.nn.functional.cross_entropy(logits, masks)
        loss.backward()
        opt.step()

    on_metrics = evaluate_model(model, loader, torch.device("cpu"))
    res = tc.knockout_eval(model, loader, torch.device("cpu"), evaluate_model, knockout_seed=42)
    assert set(res.keys()) == {"zero", "random"}
    assert abs(on_metrics["iou"] - res["zero"]["iou"]) > 1e-3
    assert abs(on_metrics["loss"] - res["zero"]["loss"]) > 1e-3


def test_decoder_skip_gate_cli_fingerprints_and_downstream_parsers():
    """Verify --decoder-skip-gate through CLI, variant_tag, fingerprints, SLURM script generators,
    analyze_results.py, and scripts/run_local_mini_sweep.py."""
    import analyze_results
    from misc import generate_grid
    from scripts import run_local_mini_sweep as rlms

    with pytest.raises(ValueError, match="decoder_skip_gate"):
        unetTorch.normalize_decoder_skip_gate("invalid_gate")

    parser = argparse.ArgumentParser()
    tc.add_common_args(parser, quantum=True, federated=True)
    args = parser.parse_args(
        ["--quantum-channels", "256", "--no-time-emb", "--decoder-skip-gate", "se"]
    )
    cfg = _base_config()
    tc.apply_common_args(cfg, args)
    assert cfg["quantum_channels"] == 256
    assert cfg["use_time_emb"] is False
    assert cfg["decoder_skip_gate"] == "se"
    assert tc.variant_tag(cfg) == "_reupload_L2_skipgate-se_notime"

    # Fingerprint & backward compatibility check
    fp_default = tc.centralized_fingerprint(
        _base_config(decoder_skip_gate="none"), "quantum"
    )
    fp_se = tc.centralized_fingerprint(
        _base_config(decoder_skip_gate="se"), "quantum"
    )
    assert fp_default["decoder_skip_gate"] == "none"
    assert fp_se["decoder_skip_gate"] == "se"
    assert fp_default != fp_se
    legacy_saved = {k: v for k, v in fp_default.items() if k != "decoder_skip_gate"}
    assert tc._normalize_saved_fingerprint(legacy_saved, "quantum") == fp_default

    # analyze_results
    v_str = analyze_results._variant_from_fields(
        "reupload", 2, data={"decoder_skip_gate": "se", "use_time_emb": False}
    )
    assert v_str == "reupload_L2_skipgate-se_notime"
    parsed = analyze_results._parse_run_dir_full(
        "flood_central_optical_pytorch_pennylane_qc256_reupload_L2_skipgate-se_notime_seed42"
    )
    assert parsed is not None
    assert parsed[3] == 256
    assert parsed[-1] == "reupload_L2_skipgate-se_notime"

    inferred = analyze_results._infer_variant_from_summary(
        "cli_args: --q-ansatz reupload --q-layers 2 --decoder-skip-gate affine --no-time-emb"
    )
    assert inferred == "reupload_L2_skipgate-affine_notime"

    # generate_grid
    job = generate_grid.build_job(
        "quantum",
        "centralized",
        "optical",
        42,
        REPO_ROOT,
        quantum_channels=256,
        no_time_emb=True,
        decoder_skip_gate="se",
    )
    assert "--decoder-skip-gate se" in job["command"]
    assert "--no-time-emb" in job["command"]
    assert "skipgate-se" in job["name"]

    # run_local_mini_sweep
    meta, explicit = rlms._expected_metadata_from_argv(
        [
            "centralized_demoTorchQ.py",
            "--seed",
            "42",
            "--quantum-channels",
            "256",
            "--no-time-emb",
            "--decoder-skip-gate",
            "se",
        ]
    )
    assert meta["decoder_skip_gate"] == "se"
    assert meta["quantum_channels"] == 256
    assert meta["use_time_emb"] is False
    assert "decoder_skip_gate" in explicit

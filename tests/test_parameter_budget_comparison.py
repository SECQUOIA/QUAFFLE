"""Checks for budget selection, normalization controls, and paired reporting."""

from copy import deepcopy

import pytest
import torch

from scripts import run_parameter_budget_comparison as budget
from tools import training_common as tc
from utilsTorch.utilsTorch import UNetSegmentation, QVUNetSegmentation


def test_budget_selection_uses_counts_and_normalization_controls():
    assert budget.choose_width({6: 400, 7: 510, 8: 640}, 500) == 7
    assert budget.choose_width({6: 490, 7: 510}, 500) == 6
    plan = budget.make_plan()
    assert plan["target_params"] == 500394
    assert plan["selected_width"] == 7
    assert plan["selected_params"] == 510631
    assert len(plan["specs"]) == 7
    for kind in ("classical", "quantum", "ablation"):
        assert {s["config"]["resnet_block_groups"] for s in plan["specs"]
                if s["model_type"] == kind and s["config"]["base_channels"] == 8} == {1, 4}
    slim = tc.build_model(UNetSegmentation, dict(budget.base_config(7, 1), seed=42))
    assert all(m.num_groups == 1 for m in slim.modules() if isinstance(m, torch.nn.GroupNorm))
    assert hasattr(slim.model.vertex, "resblock1")
    assert not hasattr(slim.model.vertex, "qresblock1")
    logits = slim(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 2, 32, 32)
    logits.square().mean().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in slim.parameters())
    q = tc.build_model(QVUNetSegmentation, dict(budget.base_config(8, 1), seed=42))
    assert sum(p.numel() for p in q.parameters()) == plan["target_params"]


def _records():
    records = []
    names = ["classical_bc7_g1"] + [f"{kind}_bc8_g{g}" for g in (1, 4)
                                    for kind in ("classical", "quantum", "ablation")]
    for i, name in enumerate(names):
        for seed in (42, 77):
            value = 0.5 + i * 0.01 + (0.1 if seed == 77 else 0)
            records.append(dict(id=name, seed=seed, total_params=500000,
                                test={"iou": value, "dice": 0.8}, knockout=None,
                                train_indices=[0, 1], val_indices=[2],
                                train_names=["a", "b", "c"], test_names=["d"]))
    return records


def test_paired_comparison_rejects_unpaired_or_duplicate_evidence():
    records = _records()
    result = budget.summarize(records)
    contrast = result["paired_comparisons"][0]
    assert contrast["left"] == "classical_bc8_g1"
    assert contrast["right"] == "classical_bc7_g1"
    assert contrast["mean"] == pytest.approx(0.01)
    assert contrast["n"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        budget.summarize(records + [records[0]])
    with pytest.raises(ValueError, match="seed coverage"):
        budget.summarize(records[:-1])
    changed = deepcopy(records)
    changed[-1]["val_indices"] = [0]
    with pytest.raises(ValueError, match="Unpaired val_indices"):
        budget.summarize(changed)

"""DEIMv2 criterion, matcher epoch switch and Copy-Blend unit contracts."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from detrs.data.transform.batch_operators import DEIMDenseO2OCollate
from detrs.modeling.losses.deimv2_loss import DEIMv2Criterion
from detrs.modeling.transformers.dfine_support import (
    DEIMv2HungarianMatcher,
    DFINEHungarianMatcher,
)


def _matcher(**kwargs):
    return DEIMv2HungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
        change_matcher=kwargs.get("change_matcher", True),
        iou_order_alpha=kwargs.get("iou_order_alpha", 4.0),
        matcher_change_epoch=kwargs.get("matcher_change_epoch", 10),
    )


def _criterion(gamma=1.5, **matcher_kwargs):
    return DEIMv2Criterion(
        matcher=_matcher(**matcher_kwargs),
        weight_dict={
            "loss_mal": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
        },
        losses=["mal", "boxes"],
        alpha=0.75,
        gamma=gamma,
        num_classes=5,
        use_uni_set=False,
    )


def _outputs_and_targets(seed=0):
    generator = torch.Generator().manual_seed(seed)
    outputs = {
        "pred_logits": torch.randn(2, 30, 5, generator=generator) * 2,
        "pred_boxes": torch.rand(2, 30, 4, generator=generator) * 0.3 + 0.2,
        "aux_outputs": [
            {
                "pred_logits": torch.randn(2, 30, 5, generator=generator) * 2,
                "pred_boxes": torch.rand(2, 30, 4, generator=generator) * 0.3 + 0.2,
            }
            for _ in range(2)
        ],
        "enc_aux_outputs": [
            {
                "pred_logits": torch.randn(2, 30, 5, generator=generator) * 2,
                "pred_boxes": torch.rand(2, 30, 4, generator=generator) * 0.3 + 0.2,
            }
        ],
        "enc_meta": {"class_agnostic": False},
        "dn_outputs": [],
    }
    del outputs["dn_outputs"]
    targets = [
        {
            "labels": torch.tensor([0, 2]),
            "boxes": torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]]),
        },
        {
            "labels": torch.tensor([4]),
            "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        },
    ]
    return outputs, targets


def test_deimv2_criterion_accepts_configured_gamma_values():
    assert _criterion(gamma=1.5).gamma == 1.5
    assert _criterion(gamma=2.0).gamma == 2.0
    with pytest.raises(ValueError):
        _criterion(gamma=0)


def test_deimv2_matcher_uses_cost_matching_before_switch_epoch():
    matcher = _matcher(matcher_change_epoch=10)
    outputs, targets = _outputs_and_targets()
    prediction = {
        "pred_logits": outputs["pred_logits"],
        "pred_boxes": outputs["pred_boxes"],
    }
    baseline = DFINEHungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}, use_focal_loss=True
    )
    before = matcher(prediction, targets, epoch=9)
    expected = baseline(prediction, targets)
    assert len(before["indices"]) == len(expected["indices"])
    for actual, wanted in zip(before["indices"], expected["indices"]):
        assert torch.equal(actual[0], wanted[0])
        assert torch.equal(actual[1], wanted[1])


def test_deimv2_matcher_switches_to_iou_order_at_epoch():
    matcher = _matcher(matcher_change_epoch=10)
    outputs, targets = _outputs_and_targets()
    prediction = {
        "pred_logits": outputs["pred_logits"],
        "pred_boxes": outputs["pred_boxes"],
    }
    switched = matcher(prediction, targets, epoch=10)
    assert "indices" in switched
    assert len(switched["indices"]) == len(targets)
    for source, target in switched["indices"]:
        assert len(source) == len(target)
    assert [len(pair[0]) for pair in switched["indices"]] == [2, 1]


def test_deimv2_criterion_receives_epoch_and_produces_losses():
    criterion = _criterion(matcher_change_epoch=1)
    outputs, targets = _outputs_and_targets()
    outputs = copy.deepcopy(outputs)
    for index in range(len(outputs["aux_outputs"])):
        outputs["aux_outputs"][index].setdefault(
            "pred_logits", outputs["aux_outputs"][index]["pred_logits"]
        )
    losses_epoch0 = criterion(outputs, targets, epoch=0)
    assert losses_epoch0
    assert all(torch.isfinite(value) for value in losses_epoch0.values())

    outputs2, targets2 = _outputs_and_targets()
    losses_epoch5 = criterion(outputs2, targets2, epoch=5)
    assert set(losses_epoch5) == set(losses_epoch0)


def _collate_samples(height=16, width=16, count=3):
    return [
        {
            "image": np.full((height, width, 3), 16 * (index + 1), dtype=np.float32),
            "gt_bbox": np.array([[2, 2, width - 2, height - 2]], dtype=np.float32),
            "gt_class": np.array([[index % 80]], dtype=np.int64),
            "gt_score": np.array([[1.0]], dtype=np.float32),
            "curr_epoch": 1,
        }
        for index in range(count)
    ]


def test_copyblend_appends_blended_objects_with_ratios():
    operator = DEIMDenseO2OCollate(
        mixup_prob=0.0,
        mixup_epochs=[0, 10],
        multiscale_stop_epoch=10,
        copyblend_prob=1.0,
        copyblend_epochs=[0, 10],
        area_threshold=4,
        num_objects=1,
        with_expand=False,
    )
    operator.set_epoch(1)
    samples = _collate_samples()
    output = operator(samples)
    assert all(len(item["gt_bbox"]) == 2 for item in output)
    first = output[0]
    assert first["gt_score"].shape == (2, 1)
    assert abs(float(first["gt_score"][0]) - 1.0) < 1e-6
    assert 0.0 < float(first["gt_score"][1]) <= 1.0
    assert not np.allclose(first["image"], 16.0)


def test_copyblend_inactive_outside_epoch_window():
    operator = DEIMDenseO2OCollate(
        mixup_prob=0.0,
        mixup_epochs=[0, 2],
        multiscale_stop_epoch=10,
        copyblend_prob=1.0,
        copyblend_epochs=[2, 10],
    )
    operator.set_epoch(1)
    samples = _collate_samples()
    output = operator(samples)
    assert all(len(item["gt_bbox"]) == 1 for item in output)
    assert np.allclose(output[0]["image"], 16.0)


def test_copyblend_skips_when_mixup_fires():
    operator = DEIMDenseO2OCollate(
        mixup_prob=1.0,
        mixup_epochs=[0, 10],
        multiscale_stop_epoch=10,
        multiscale_sizes=None,
        copyblend_prob=1.0,
        copyblend_epochs=[0, 10],
    )
    operator.set_epoch(1)
    samples = _collate_samples()
    output = operator(samples)
    # mixup doubles every image's targets instead of appending blends
    assert all(len(item["gt_bbox"]) == 2 for item in output)

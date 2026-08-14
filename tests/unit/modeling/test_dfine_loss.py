import pytest
import torch

from detrs.modeling.losses.dfine_loss import DFINECriterion
from detrs.modeling.transformers.dfine_support import DFINEHungarianMatcher


def _criterion():
    return DFINECriterion(
        matcher=DFINEHungarianMatcher(
            {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            use_focal_loss=True,
        ),
        weight_dict={
            "loss_vfl": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        losses=["vfl", "boxes", "local"],
        alpha=0.75,
        gamma=2,
        num_classes=3,
        reg_max=8,
    )


def _prediction(batch_size=2):
    logits = torch.zeros(batch_size, 3, 3, requires_grad=True)
    boxes = torch.full((batch_size, 3, 4), 0.25, requires_grad=True)
    base = {"pred_logits": logits, "pred_boxes": boxes}
    return {
        **base,
        "aux_outputs": [],
        "pre_outputs": {key: value.clone() for key, value in base.items()},
        "enc_aux_outputs": [],
        "enc_meta": {"class_agnostic": False},
    }


def _targets(empty=False):
    if empty:
        return [
            {
                "labels": torch.empty(0, dtype=torch.int64),
                "boxes": torch.empty(0, 4),
            }
            for _ in range(2)
        ]
    return [
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]])},
        {"labels": torch.tensor([2]), "boxes": torch.tensor([[0.4, 0.4, 0.3, 0.2]])},
    ]


def test_train_empty_targets_produces_finite_classification_loss():
    losses = _criterion()(_prediction(), _targets(empty=True))
    assert set(losses) == {
        "loss_vfl",
        "loss_bbox",
        "loss_giou",
        "loss_vfl_pre",
        "loss_bbox_pre",
        "loss_giou_pre",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()


def test_num_boxes_uses_distributed_average(monkeypatch):
    reduced = []

    def all_reduce(value):
        reduced.append(value.item())
        value.mul_(4)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    assert DFINECriterion._normalizer(3, torch.device("cpu")) == 6
    assert reduced == [3]


def test_rejects_aux_metadata_missing_before_matching():
    outputs = _prediction()
    del outputs["enc_meta"]
    with pytest.raises(ValueError, match="enc_meta"):
        _criterion()(outputs, _targets())


def test_rejects_missing_target_before_matching():
    targets = _targets()
    del targets[0]["boxes"]
    with pytest.raises(ValueError, match="labels and boxes"):
        _criterion()(_prediction(), targets)


def test_rejects_nonfinite_loss_input():
    outputs = _prediction()
    with torch.no_grad():
        outputs["pred_logits"][0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="nonfinite predictions"):
        _criterion()(outputs, _targets())

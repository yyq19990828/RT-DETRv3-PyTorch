from types import SimpleNamespace

import pytest
import torch
from torch import nn

from detrs.modeling.architectures import dfine as dfine_module
from detrs.modeling.architectures.dfine import DFINE
from detrs.modeling.post_process import DETRPostProcess


class _Backbone(nn.Module):
    out_shape = [SimpleNamespace(channels=4, stride=8)]

    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, inputs):
        self.called = True
        return [inputs["image"]]


class _Encoder(nn.Module):
    out_shape = [SimpleNamespace(channels=4, stride=8)]

    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, features):
        return (features, features[-1]) if self.tuple_output else features


class _Decoder(nn.Module):
    def __init__(self, malformed=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.malformed = malformed
        self.targets = None

    def forward(self, features, targets=None):
        self.targets = targets
        logits = self.weight.expand(features[0].shape[0], 2, 3)
        boxes = torch.full(
            (features[0].shape[0], 2, 4),
            0.5,
            device=features[0].device,
        )
        if self.malformed == "missing":
            return {"pred_logits": logits}
        if self.malformed == "nonfinite":
            boxes[0, 0, 0] = torch.inf
        return {"pred_logits": logits, "pred_boxes": boxes}


class _Criterion(nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.nonfinite = nonfinite

    def forward(self, outputs, targets, epoch=None):
        loss = outputs["pred_logits"].square().mean()
        if self.nonfinite:
            loss = loss * torch.tensor(float("inf"))
        return {"loss_vfl": loss, "loss_bbox": outputs["pred_boxes"].sum() * 0}


def _batch():
    return {
        "image": torch.randn(2, 4, 2, 2),
        "gt_class": [torch.tensor([[1]]), torch.empty(0, 1, dtype=torch.int64)],
        "gt_bbox": [torch.tensor([[0.5, 0.5, 0.2, 0.3]]), torch.empty(0, 4)],
        "im_shape": torch.tensor([[16.0, 16.0], [16.0, 16.0]]),
        "scale_factor": torch.ones(2, 2),
    }


def _model(**overrides):
    values = {
        "backbone": _Backbone(),
        "encoder": _Encoder(),
        "decoder": _Decoder(),
        "criterion": _Criterion(),
        "post_process": DETRPostProcess(
            num_classes=3, num_top_queries=3, use_focal_loss=True
        ),
    }
    values.update(overrides)
    return DFINE(**values)


def test_dfine_train_bridges_targets_sums_losses_and_backpropagates():
    model = _model().train()
    result = model(_batch())

    assert set(result) == {"loss_vfl", "loss_bbox", "loss"}
    assert torch.equal(result["loss"], result["loss_vfl"] + result["loss_bbox"])
    assert model.decoder.targets[0]["labels"].tolist() == [1]
    assert model.decoder.targets[1]["boxes"].shape == (0, 4)
    result["loss"].backward()
    assert torch.isfinite(model.decoder.weight.grad)


def test_dfine_eval_returns_repository_bbox_contract_and_raw_option():
    batch = _batch()
    model = _model().eval()
    with torch.inference_mode():
        result = model(batch)
    assert set(result) == {"bbox", "bbox_num"}
    assert result["bbox"].shape == (6, 6)
    assert torch.equal(result["bbox_num"], torch.tensor([3, 3], dtype=torch.int32))

    model.exclude_post_process = True
    with torch.inference_mode():
        raw = model(batch)
    assert set(raw) == {"pred_logits", "pred_boxes"}


def test_dfine_from_config_propagates_component_shapes(monkeypatch):
    calls = []
    backbone = SimpleNamespace(out_shape=["backbone-shape"])
    encoder = SimpleNamespace(out_shape=["encoder-shape"])
    components = {
        "backbone-config": backbone,
        "encoder-config": encoder,
        "decoder-config": "decoder",
        "criterion-config": "criterion",
        "post-config": "post",
    }

    def create(config, **kwargs):
        calls.append((config, kwargs))
        return components[config]

    monkeypatch.setattr(dfine_module, "create", create)
    result = DFINE.from_config(
        {
            "backbone": "backbone-config",
            "encoder": "encoder-config",
            "decoder": "decoder-config",
            "criterion": "criterion-config",
            "post_process": "post-config",
        }
    )

    assert calls == [
        ("backbone-config", {}),
        ("encoder-config", {"input_shape": backbone.out_shape}),
        ("decoder-config", {"input_shape": encoder.out_shape}),
        ("criterion-config", {}),
        ("post-config", {}),
    ]
    assert result == {
        "backbone": backbone,
        "encoder": encoder,
        "decoder": "decoder",
        "criterion": "criterion",
        "post_process": "post",
    }


def test_dfine_rejects_missing_targets_before_decoder():
    model = _model().train()
    batch = _batch()
    del batch["gt_bbox"]
    with pytest.raises(ValueError, match="gt_class and gt_bbox"):
        model(batch)
    assert not model.backbone.called


class _InvalidCriterion(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, outputs, targets, epoch=None):
        return self.value(outputs)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (lambda outputs: {"loss": outputs["pred_logits"].sum()}, "aggregate"),
        (lambda outputs: {"loss_vfl": outputs["pred_logits"][0]}, "non-scalar"),
    ],
)
def test_dfine_rejects_invalid_criterion_contract(value, message):
    model = _model(criterion=_InvalidCriterion(value)).train()
    with pytest.raises((ValueError, FloatingPointError), match=message):
        model(_batch())


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"encoder": _Encoder(tuple_output=True)}, TypeError, "feature list"),
        ({"decoder": _Decoder("missing")}, ValueError, "missing"),
        ({"decoder": _Decoder("nonfinite")}, FloatingPointError, "finite tensor"),
        ({"criterion": _Criterion(nonfinite=True)}, FloatingPointError, "non-finite"),
    ],
)
def test_dfine_rejects_invalid_family_boundaries(overrides, exception, message):
    model = _model(**overrides).train()
    with pytest.raises(exception, match=message):
        model(_batch())

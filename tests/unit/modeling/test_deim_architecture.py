from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ppdet_pytorch.core.workspace import get_registered_modules
from ppdet_pytorch.modeling.architectures import DEIM, DFINE
from ppdet_pytorch.modeling.architectures import dfine as dfine_module
from ppdet_pytorch.modeling.post_process import DETRPostProcess


class _Backbone(nn.Module):
    out_shape = [SimpleNamespace(channels=4, stride=8)]

    def forward(self, inputs):
        return [inputs["image"]]


class _Encoder(nn.Module):
    out_shape = [SimpleNamespace(channels=4, stride=8)]

    def forward(self, features):
        return features


class _Decoder(nn.Module):
    def __init__(self, family="rtdetrv2"):
        super().__init__()
        self.family = family
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.targets = None
        self.outputs = None
        self.deploy_calls = 0

    def forward(self, features, targets=None):
        self.targets = targets
        batch_size = features[0].shape[0]
        prediction = {
            "pred_logits": self.weight.expand(batch_size, 2, 3),
            "pred_boxes": torch.full(
                (batch_size, 2, 4), 0.5, device=features[0].device
            ),
        }
        outputs = dict(prediction)
        if self.training:
            outputs.update(
                aux_outputs=[dict(prediction)],
                enc_aux_outputs=[],
                enc_meta={"class_agnostic": False},
            )
        if self.training and self.family == "dfine":
            outputs.update(
                pred_corners=torch.zeros(batch_size, 2, 36),
                ref_points=prediction["pred_boxes"],
                up=torch.tensor([0.5]),
                reg_scale=torch.tensor([4.0]),
                pre_outputs=dict(prediction),
            )
        self.outputs = outputs
        return outputs

    def convert_to_deploy(self):
        self.deploy_calls += 1


class _Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.outputs = None
        self.targets = None

    def forward(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets
        return {"loss_mal": outputs["pred_logits"].square().mean()}


def _batch():
    return {
        "image": torch.randn(2, 4, 2, 2),
        "gt_class": [torch.tensor([[1]]), torch.empty(0, 1, dtype=torch.int64)],
        "gt_bbox": [torch.tensor([[0.5, 0.5, 0.2, 0.3]]), torch.empty(0, 4)],
        "im_shape": torch.full((2, 2), 16.0),
        "scale_factor": torch.ones(2, 2),
    }


def _model(exclude_post_process=False, family="rtdetrv2", architecture=DEIM):
    return architecture(
        backbone=_Backbone(),
        encoder=_Encoder(),
        decoder=_Decoder(family),
        criterion=_Criterion(),
        post_process=DETRPostProcess(
            num_classes=3, num_top_queries=3, use_focal_loss=True
        ),
        exclude_post_process=exclude_post_process,
    )


@pytest.mark.parametrize("family", ["dfine", "rtdetrv2"])
def test_decoder_graph_training_preserves_outputs_and_backpropagates(family):
    model = _model(family=family).train()
    losses = model(_batch())

    assert set(losses) == {"loss_mal", "loss"}
    assert model.decoder.targets[0]["labels"].tolist() == [1]
    assert model.criterion.targets is model.decoder.targets
    assert model.criterion.outputs is model.decoder.outputs
    if family == "dfine":
        assert {"pred_corners", "ref_points", "pre_outputs", "up", "reg_scale"} <= (
            model.criterion.outputs.keys()
        )
    else:
        assert "pre_outputs" not in model.criterion.outputs
    losses["loss"].backward()
    assert torch.isfinite(model.decoder.weight.grad)


def test_rtdetrv2_graph_uses_same_adapter_output_contract():
    model = _model(exclude_post_process=True).eval()
    with torch.inference_mode():
        raw = model(_batch())

    assert set(raw) == {"pred_logits", "pred_boxes"}
    assert raw["pred_logits"].shape == (2, 2, 3)


def test_eval_predictions_equal_shared_non_deim_graph():
    deim = _model().eval()
    shared = _model(architecture=DFINE).eval()
    shared.load_state_dict(deim.state_dict())
    batch = _batch()

    with torch.inference_mode():
        expected = shared(batch)
        actual = deim(batch)

    assert torch.equal(actual["bbox_num"], expected["bbox_num"])
    torch.testing.assert_close(actual["bbox"], expected["bbox"])


def test_deim_is_registered_as_one_shared_architecture():
    assert get_registered_modules()["DEIM"].cls is DEIM


def test_from_config_propagates_component_shapes(monkeypatch):
    calls = []
    backbone = SimpleNamespace(out_shape=[SimpleNamespace(channels=4, stride=8)])
    encoder = SimpleNamespace(out_shape=[SimpleNamespace(channels=4, stride=8)])
    decoder = SimpleNamespace(feat_channels=[4], feat_strides=[8])
    components = {
        "backbone": backbone,
        "encoder": encoder,
        "decoder": decoder,
        "criterion": "criterion",
        "post": "post",
    }

    def create(config, **kwargs):
        calls.append((config, kwargs))
        return components[config]

    monkeypatch.setattr(dfine_module, "create", create)
    result = DEIM.from_config(
        {
            "backbone": "backbone",
            "encoder": "encoder",
            "decoder": "decoder",
            "criterion": "criterion",
            "post_process": "post",
        }
    )

    assert calls == [
        ("backbone", {}),
        ("encoder", {"input_shape": backbone.out_shape}),
        ("decoder", {"input_shape": encoder.out_shape}),
        ("criterion", {}),
        ("post", {}),
    ]
    assert result["decoder"] is decoder


def test_deploy_is_inherited_and_idempotent():
    model = _model().eval()

    assert model.deploy() is model
    assert model.deploy() is model
    assert model.decoder.deploy_calls == 1

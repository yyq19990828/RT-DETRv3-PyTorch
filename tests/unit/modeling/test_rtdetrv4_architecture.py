from types import SimpleNamespace

import pytest
import torch
from torch import nn

from detrs.core.workspace import get_registered_modules
from detrs.modeling.architectures import RTDETRV4
from detrs.modeling.post_process import DETRPostProcess


class _Backbone(nn.Module):
    out_shape = [SimpleNamespace(channels=4, stride=8)]

    def forward(self, inputs):
        return [inputs["image"]]


class _Encoder(nn.Module):
    out_shape = [SimpleNamespace(channels=4, stride=8)]

    def forward(self, features):
        if self.training:
            return features, features[0] * 2
        return features


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.features = None

    def forward(self, features, targets=None):
        self.features = features
        batch_size = features[0].shape[0]
        prediction = {
            "pred_logits": self.weight.expand(batch_size, 2, 3),
            "pred_boxes": torch.full((batch_size, 2, 4), 0.5),
        }
        if self.training:
            prediction.update(
                aux_outputs=[],
                enc_aux_outputs=[],
                enc_meta={"class_agnostic": False},
            )
        return prediction


class _Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.outputs = None

    def forward(self, outputs, targets):
        self.outputs = outputs
        return {
            "loss_mal": outputs["pred_logits"].square().mean(),
            "loss_distill": (
                outputs["student_distill_output"] - outputs["teacher_encoder_output"]
            )
            .square()
            .mean(),
        }


def _batch(*, teacher=True):
    batch = {
        "image": torch.randn(2, 4, 2, 2),
        "gt_class": [torch.tensor([[1]]), torch.empty(0, 1, dtype=torch.int64)],
        "gt_bbox": [torch.tensor([[0.5, 0.5, 0.2, 0.3]]), torch.empty(0, 4)],
        "im_shape": torch.full((2, 2), 16.0),
        "scale_factor": torch.ones(2, 2),
    }
    if teacher:
        batch["teacher_encoder_output"] = torch.randn(2, 4, 2, 2, requires_grad=True)
    return batch


def _model(*, exclude_post_process=False):
    return RTDETRV4(
        backbone=_Backbone(),
        encoder=_Encoder(),
        decoder=_Decoder(),
        criterion=_Criterion(),
        post_process=DETRPostProcess(
            num_classes=3, num_top_queries=3, use_focal_loss=True
        ),
        exclude_post_process=exclude_post_process,
    )


def test_dsi_training_boundary_detaches_teacher_and_backpropagates_student():
    model = _model().train()
    batch = _batch()

    losses = model(batch)

    assert set(losses) == {"loss_mal", "loss_distill", "loss"}
    assert isinstance(model.decoder.features, list)
    assert model.criterion.outputs["student_distill_output"].shape == (2, 4, 2, 2)
    assert not model.criterion.outputs["teacher_encoder_output"].requires_grad
    losses["loss"].backward()
    assert torch.isfinite(model.decoder.weight.grad)
    assert batch["teacher_encoder_output"].grad is None


def test_rejects_absent_teacher_before_criterion():
    model = _model().train()

    with pytest.raises(ValueError, match="requires teacher_encoder_output"):
        model(_batch(teacher=False))


def test_student_eval_ignores_teacher_and_has_no_distillation_keys():
    model = _model(exclude_post_process=True).eval()

    with torch.inference_mode():
        outputs = model(_batch(teacher=False))

    assert set(outputs) == {"pred_logits", "pred_boxes"}
    assert not any("teacher" in key or "distill" in key for key in model.state_dict())


def test_rtdetrv4_is_registered():
    assert get_registered_modules()["RTDETRV4"].cls is RTDETRV4

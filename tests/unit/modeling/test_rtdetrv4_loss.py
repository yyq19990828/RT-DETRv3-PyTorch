import pytest
import torch
import torch.nn.functional as F

from detrs.modeling.losses.rtdetrv4_loss import RTDETRV4Criterion
from detrs.modeling.transformers.dfine_support import DFINEHungarianMatcher


def _criterion(weight=5.0):
    return RTDETRV4Criterion(
        matcher=DFINEHungarianMatcher(
            {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            use_focal_loss=True,
        ),
        weight_dict={
            "loss_mal": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_distill": weight,
        },
        losses=["mal", "boxes", "distill"],
        gamma=1.5,
        num_classes=3,
        use_uni_set=False,
    )


def _outputs(student, teacher):
    prediction = {
        "pred_logits": torch.zeros(1, 2, 3, requires_grad=True),
        "pred_boxes": torch.full((1, 2, 4), 0.5, requires_grad=True),
    }
    return {
        **prediction,
        "aux_outputs": [],
        "enc_aux_outputs": [],
        "enc_meta": {"class_agnostic": False},
        "student_distill_output": student,
        "teacher_encoder_output": teacher,
    }


def _targets():
    return [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5] * 4])}]


def test_dsi_matches_upstream_normalized_cosine_and_weight_once():
    student = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
    teacher = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]], requires_grad=True)
    expected_student = F.normalize(student.flatten(2).permute(0, 2, 1), dim=-1)
    expected_teacher = F.normalize(teacher.flatten(2).permute(0, 2, 1), dim=-1)
    expected = (
        1 - F.cosine_similarity(expected_student, expected_teacher, dim=-1)
    ).mean()

    losses = _criterion(weight=5)(_outputs(student, teacher), _targets())

    torch.testing.assert_close(losses["loss_distill"], expected * 5)
    losses["loss_distill"].backward()
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_dsi_resizes_teacher_with_bilinear_align_corners_false():
    student = torch.randn(1, 3, 2, 2)
    teacher = torch.randn(1, 3, 3, 3)
    resized = F.interpolate(teacher, (2, 2), mode="bilinear", align_corners=False)
    expected = (
        1
        - F.cosine_similarity(
            F.normalize(student.flatten(2).permute(0, 2, 1), dim=-1),
            F.normalize(resized.flatten(2).permute(0, 2, 1), dim=-1),
            dim=-1,
        )
    ).mean()

    actual = RTDETRV4Criterion.loss_distillation(_outputs(student, teacher))

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("student", "teacher", "message"),
    [
        (torch.randn(1, 3, 2, 2), None, "requires student and teacher"),
        (torch.randn(1, 3, 2, 2), torch.randn(1, 4, 2, 2), "channel mismatch"),
        (torch.randn(1, 3, 2, 2), torch.randn(2, 3, 2, 2), "batch size mismatch"),
    ],
)
def test_rejects_invalid_distillation_boundary(student, teacher, message):
    with pytest.raises(ValueError, match=message):
        RTDETRV4Criterion.loss_distillation(_outputs(student, teacher))


def test_identical_features_have_zero_dsi():
    feature = torch.randn(2, 8, 3, 3)
    loss = RTDETRV4Criterion.loss_distillation(_outputs(feature, feature.clone()))
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)


def test_set_distillation_weight_validates_runtime_gam_value():
    criterion = _criterion()
    criterion.set_distillation_weight(7.5)
    assert criterion.weight_dict["loss_distill"] == 7.5
    with pytest.raises(ValueError, match="finite and non-negative"):
        criterion.set_distillation_weight(float("nan"))

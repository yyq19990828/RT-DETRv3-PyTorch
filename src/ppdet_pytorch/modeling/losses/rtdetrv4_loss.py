"""RT-DETRv4 criterion with DINOv3 spatial feature distillation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from ppdet_pytorch.core.workspace import register, serializable

from .deim_loss import DEIMCriterion

__all__ = ["RTDETRV4Criterion"]


@register
@serializable
class RTDETRV4Criterion(DEIMCriterion):
    """Add the official DSI cosine loss to the shared DEIM criterion."""

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=1.5,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        distill_adaptive_params=None,
    ):
        if "distill" not in losses:
            raise ValueError("RT-DETRv4 losses must include distill")
        if "loss_distill" not in weight_dict:
            raise ValueError("RT-DETRv4 weight_dict must include loss_distill")
        distill_weight = float(weight_dict["loss_distill"])
        if not torch.isfinite(torch.tensor(distill_weight)) or distill_weight < 0:
            raise ValueError("loss_distill weight must be finite and non-negative")
        detection_losses = [loss for loss in losses if loss != "distill"]
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=detection_losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
            mal_alpha=mal_alpha,
            use_uni_set=use_uni_set,
        )
        self.distill_adaptive_params = (
            dict(distill_adaptive_params)
            if isinstance(distill_adaptive_params, Mapping)
            else distill_adaptive_params
        )

    @staticmethod
    def loss_distillation(outputs) -> torch.Tensor:
        student = outputs.get("student_distill_output")
        teacher = outputs.get("teacher_encoder_output")
        if not isinstance(student, torch.Tensor) or not isinstance(
            teacher, torch.Tensor
        ):
            raise ValueError(
                "RT-DETRv4 distillation requires student and teacher features"
            )
        if student.ndim != 4 or teacher.ndim != 4:
            raise ValueError("RT-DETRv4 distillation features must be BCHW tensors")
        if student.shape[0] != teacher.shape[0]:
            raise ValueError("RT-DETRv4 distillation batch size mismatch")
        if student.shape[1] != teacher.shape[1]:
            raise ValueError("RT-DETRv4 distillation channel mismatch")
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise FloatingPointError("RT-DETRv4 distillation features must be finite")

        teacher = teacher.detach()
        if student.shape[2:] != teacher.shape[2:]:
            teacher = F.interpolate(
                teacher,
                size=student.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        student = F.normalize(student.flatten(2).permute(0, 2, 1), p=2, dim=-1)
        teacher = F.normalize(teacher.flatten(2).permute(0, 2, 1), p=2, dim=-1)
        return (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()

    def set_distillation_weight(self, weight: float) -> None:
        value = float(weight)
        if not torch.isfinite(torch.tensor(value)) or value < 0:
            raise ValueError("distillation weight must be finite and non-negative")
        self.weight_dict["loss_distill"] = value

    def forward(self, outputs, targets, epoch=None, **kwargs):
        losses = super().forward(outputs, targets, epoch=epoch, **kwargs)
        losses["loss_distill"] = (
            self.loss_distillation(outputs) * self.weight_dict["loss_distill"]
        )
        if not torch.isfinite(losses["loss_distill"]):
            raise FloatingPointError("nonfinite RT-DETRv4 distillation loss")
        return losses

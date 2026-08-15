"""RT-DETRv4 adapter over the shared D-FINE student graph."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from detrs.core.workspace import register
from detrs.modeling.transformers.dfine_support import (
    repository_batch_to_dfine_targets,
)

from .dfine import DFINE

__all__ = ["RTDETRV4"]


@register
class RTDETRV4(DFINE):
    """Attach training-only DSI features without changing student inference.

    Wiring and constructor arguments are shared with `DFINE`; DSI
    (teacher feature distillation) only participates in training.
    """

    def _forward(self):
        if not self.training:
            return super()._forward()

        targets = repository_batch_to_dfine_targets(self.inputs)
        encoder_output = self.encoder(self.backbone(self.inputs))
        if not (
            isinstance(encoder_output, tuple)
            and len(encoder_output) == 2
            and isinstance(encoder_output[0], list)
            and isinstance(encoder_output[1], torch.Tensor)
        ):
            raise ValueError(
                "RT-DETRv4 training requires encoder features and projected F5"
            )
        features, student_feature = encoder_output
        teacher_feature = self.inputs.get("teacher_encoder_output")
        if not isinstance(teacher_feature, torch.Tensor):
            raise ValueError("RT-DETRv4 training requires teacher_encoder_output")

        outputs = self.decoder(features, targets)
        self._validate_predictions(outputs)
        outputs["student_distill_output"] = student_feature
        outputs["teacher_encoder_output"] = teacher_feature.detach()
        losses = self.criterion(outputs, targets)
        if not isinstance(losses, Mapping) or not losses:
            raise ValueError("RTDETRV4Criterion must return a non-empty loss mapping")
        if "loss" in losses:
            raise ValueError("RTDETRV4Criterion must not return the aggregate loss key")
        for name, value in losses.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.numel() != 1
                or not torch.isfinite(value).all()
            ):
                raise FloatingPointError(
                    "RTDETRV4Criterion returned a non-scalar or non-finite loss {}".format(
                        name
                    )
                )
        result = dict(losses)
        result["loss"] = sum(result.values())
        return result

    def deploy(self):
        super().deploy()
        self.encoder.feature_projector = None
        return self

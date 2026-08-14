"""D-FINE detection architecture adapter for repository batch contracts."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from detrs.core.workspace import create, register
from detrs.modeling.transformers.dfine_support import (
    repository_batch_to_dfine_targets,
)

from .meta_arch import BaseArch

__all__ = ["DFINE"]


@register
class DFINE(BaseArch):
    """Compose shared D-FINE components without changing their upstream semantics."""

    __category__ = "architecture"
    __shared__ = ["exclude_post_process"]

    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        criterion,
        post_process,
        exclude_post_process=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.post_process = post_process
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        del args, kwargs
        backbone = create(cfg["backbone"])
        encoder = create(cfg["encoder"], input_shape=backbone.out_shape)
        decoder = create(cfg["decoder"], input_shape=encoder.out_shape)
        encoder_channels = [
            getattr(shape, "channels", None) for shape in encoder.out_shape
        ]
        encoder_strides = [
            getattr(shape, "stride", None) for shape in encoder.out_shape
        ]
        if (
            all(value is not None for value in encoder_channels)
            and getattr(decoder, "feat_channels", encoder_channels) != encoder_channels
        ):
            raise ValueError(
                "D-FINE decoder feat_channels must match encoder output channels"
            )
        if (
            all(value is not None for value in encoder_strides)
            and getattr(decoder, "feat_strides", encoder_strides)[
                : len(encoder_strides)
            ]
            != encoder_strides
        ):
            raise ValueError(
                "D-FINE decoder feat_strides must match encoder output strides"
            )
        return {
            "backbone": backbone,
            "encoder": encoder,
            "decoder": decoder,
            "criterion": create(cfg["criterion"]),
            "post_process": create(cfg["post_process"]),
        }

    @staticmethod
    def _validate_predictions(outputs):
        if not isinstance(outputs, Mapping):
            raise TypeError("D-FINE decoder output must be a mapping")
        missing = {"pred_logits", "pred_boxes"} - set(outputs)
        if missing:
            raise ValueError(
                "D-FINE decoder output is missing: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        for name in ("pred_logits", "pred_boxes"):
            value = outputs[name]
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise FloatingPointError(
                    "D-FINE decoder output {} must be a finite tensor".format(name)
                )

    def _forward(self):
        targets = (
            repository_batch_to_dfine_targets(self.inputs) if self.training else None
        )
        features = self.backbone(self.inputs)
        features = self.encoder(features)
        if not isinstance(features, list):
            raise TypeError("D-FINE encoder must return a feature list")

        outputs = self.decoder(features, targets)
        self._validate_predictions(outputs)

        if self.training:
            epoch = (
                int(self.inputs["curr_epoch"][0])
                if "curr_epoch" in self.inputs
                else None
            )
            losses = self.criterion(outputs, targets, epoch=epoch)
            if not isinstance(losses, Mapping) or not losses:
                raise ValueError("DFINECriterion must return a non-empty loss mapping")
            if "loss" in losses:
                raise ValueError(
                    "DFINECriterion must not return the aggregate loss key"
                )
            for name, value in losses.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.numel() != 1
                    or not torch.isfinite(value).all()
                ):
                    raise FloatingPointError(
                        "DFINECriterion returned a non-scalar or non-finite loss {}".format(
                            name
                        )
                    )
            losses = dict(losses)
            losses["loss"] = sum(losses.values())
            return losses

        if self.exclude_post_process:
            return outputs
        bbox, bbox_num, mask = self.post_process(
            (outputs["pred_boxes"], outputs["pred_logits"], None),
            self.inputs["im_shape"],
            self.inputs["scale_factor"],
        )
        result = {"bbox": bbox, "bbox_num": bbox_num}
        if mask is not None:
            result["mask"] = mask
        return result

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def deploy(self):
        if getattr(self, "_deployed", False):
            return self
        self.eval()
        for module in self.modules():
            if module is not self and hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        self._deployed = True
        return self

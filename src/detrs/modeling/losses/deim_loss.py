"""DEIM matching-aware criterion built on the verified D-FINE loss math."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from detrs.core.workspace import register, serializable

from ..transformers.dfine_support import box_cxcywh_to_xyxy, box_iou
from .dfine_loss import DFINECriterion

__all__ = ["DEIMCriterion"]


@register
@serializable
class DEIMCriterion(DFINECriterion):
    """Compute DEIM MAL and optional D-FINE localization losses.

    Args:
        matcher: Injected matcher (e.g. `DFINEHungarianMatcher`).
        weight_dict (dict): Weights of the individual loss terms.
        losses (list): Enabled loss names, e.g. `['mal', 'boxes', 'local']`.
        alpha (float): Alpha of the focal/VFL classification loss.
        gamma (float): Gamma of the focal/VFL classification loss.
        num_classes (int): Number of foreground classes.
        reg_max (int): Discrete bins of the box distribution used by the
            `local` (FDR) loss.
        boxes_weight_format (str|None): How matched box losses are weighted.
        share_matched_indices (bool): Reuse one set of matching indices for
            all loss terms.
        mal_alpha (float|None): Separate alpha of the MAL loss; `None`
            reuses `alpha`.
        use_uni_set (bool): Use the unified set matching shared across
            Dense O2O branches.
    """

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
    ):
        if gamma != 1.5:
            raise ValueError("DEIM MAL gamma must be 1.5")
        invalid = set(losses) - {"mal", "boxes", "local"}
        if invalid:
            raise ValueError(f"unsupported DEIM losses: {sorted(invalid)}")
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=[],
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
        )
        self.losses = list(losses)
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = self._matched_target_boxes(targets, indices)
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
            ious = torch.diag(ious).detach()
        else:
            ious = values
        ious = torch.nan_to_num(ious, nan=0.0).clamp(0.0, 1.0)

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [
                target["labels"][target_idx]
                for target, (_, target_idx) in zip(targets, indices)
            ]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = (target_score_o.unsqueeze(-1) * target).pow(self.gamma)

        negative_weight = src_logits.sigmoid().detach().pow(self.gamma) * (1 - target)
        if self.mal_alpha is not None:
            negative_weight = self.mal_alpha * negative_weight
        weight = negative_weight + target
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == "mal":
            return self.loss_labels_mal(outputs, targets, indices, num_boxes, **kwargs)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        metadata = super().get_loss_meta_info(loss, outputs, targets, indices)
        if loss != "mal" or self.boxes_weight_format is None:
            return metadata
        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = self._matched_target_boxes(targets, indices)
        if self.boxes_weight_format == "iou":
            values = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
        else:
            from ..transformers.dfine_support import generalized_box_iou

            values = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            )
        return {"values": values}

    def _validate_outputs(self, outputs):
        if not isinstance(outputs, dict):
            raise TypeError("outputs must be a mapping")
        batch_size = self._validate_prediction("outputs", outputs)
        if "aux_outputs" not in outputs or not isinstance(outputs["aux_outputs"], list):
            raise ValueError("outputs must contain aux_outputs")
        for field in ("aux_outputs", "enc_aux_outputs", "dn_outputs"):
            if field in outputs:
                if not isinstance(outputs[field], list):
                    raise ValueError(f"{field} must be a list")
                for index, output in enumerate(outputs[field]):
                    self._validate_prediction(f"{field}[{index}]", output, batch_size)
        if "pre_outputs" in outputs:
            self._validate_prediction("pre_outputs", outputs["pre_outputs"], batch_size)
        if "enc_aux_outputs" in outputs and "enc_meta" not in outputs:
            raise ValueError("enc_aux_outputs requires enc_meta")
        if "dn_outputs" in outputs and "dn_meta" not in outputs:
            raise ValueError("dn_outputs requires dn_meta")
        if "dn_outputs" in outputs:
            metadata = outputs["dn_meta"]
            if (
                not isinstance(metadata, Mapping)
                or not {"dn_positive_idx", "dn_num_group"} <= metadata.keys()
            ):
                raise ValueError("dn_meta is malformed")
            if "dn_pre_outputs" in outputs:
                self._validate_prediction(
                    "dn_pre_outputs", outputs["dn_pre_outputs"], batch_size
                )
        if "local" in self.losses and not {"up", "reg_scale"} <= outputs.keys():
            raise ValueError("DEIM local loss requires D-FINE distribution outputs")
        return batch_size

    def _weighted_deim_loss(self, loss, outputs, targets, indices, num_boxes):
        metadata = self.get_loss_meta_info(loss, outputs, targets, indices)
        values = self.get_loss(loss, outputs, targets, indices, num_boxes, **metadata)
        return {
            key: value * self.weight_dict[key]
            for key, value in values.items()
            if key in self.weight_dict
        }

    def _match_indices(self, output, targets, epoch=None):
        """Match one prediction dict; epoch hooks are family-specific."""
        del epoch
        return self.matcher(output, targets)["indices"]

    def forward(self, outputs, targets, epoch=None, **kwargs):
        del kwargs
        batch_size = self._validate_outputs(outputs)
        self._validate_targets(targets, batch_size)
        indices = self._match_indices(
            {key: value for key, value in outputs.items() if "aux" not in key},
            targets,
            epoch,
        )
        self._clear_cache()

        cached_aux = [
            self._match_indices(output, targets, epoch)
            for output in outputs["aux_outputs"]
        ]
        cached_pre = (
            self._match_indices(outputs["pre_outputs"], targets, epoch)
            if "pre_outputs" in outputs
            else None
        )
        class_agnostic = outputs.get("enc_meta", {}).get("class_agnostic", False)
        enc_targets = copy.deepcopy(targets) if class_agnostic else targets
        if class_agnostic:
            for target in enc_targets:
                target["labels"] = torch.zeros_like(target["labels"])
        cached_enc = [
            self._match_indices(output, enc_targets, epoch)
            for output in outputs.get("enc_aux_outputs", [])
        ]
        all_indices = [*cached_aux]
        if cached_pre is not None:
            all_indices.append(cached_pre)
        all_indices.extend(cached_enc)
        indices_go = self._get_go_indices(indices, all_indices)
        device = outputs["pred_logits"].device
        num_boxes = self._normalizer(
            sum(len(target["labels"]) for target in targets), device
        )
        num_boxes_go = self._normalizer(
            sum(len(pair[0]) for pair in indices_go), device
        )

        def selection(loss, regular_indices):
            union = self.use_uni_set and loss in {"boxes", "local"}
            return (indices_go, num_boxes_go) if union else (regular_indices, num_boxes)

        losses = {}
        for loss in self.losses:
            selected, normalizer = selection(loss, indices)
            losses.update(
                self._weighted_deim_loss(loss, outputs, targets, selected, normalizer)
            )

        for index, output in enumerate(outputs["aux_outputs"]):
            if "local" in self.losses:
                output["up"], output["reg_scale"] = outputs["up"], outputs["reg_scale"]
            for loss in self.losses:
                selected, normalizer = selection(loss, cached_aux[index])
                values = self._weighted_deim_loss(
                    loss, output, targets, selected, normalizer
                )
                losses.update(
                    {f"{key}_aux_{index}": value for key, value in values.items()}
                )

        if cached_pre is not None:
            for loss in self.losses:
                selected, normalizer = selection(loss, cached_pre)
                values = self._weighted_deim_loss(
                    loss, outputs["pre_outputs"], targets, selected, normalizer
                )
                losses.update({f"{key}_pre": value for key, value in values.items()})

        if cached_enc:
            original_num_classes = self.num_classes
            if class_agnostic:
                self.num_classes = 1
            try:
                for index, output in enumerate(outputs["enc_aux_outputs"]):
                    for loss in self.losses:
                        union = self.use_uni_set and loss == "boxes"
                        selected = indices_go if union else cached_enc[index]
                        normalizer = num_boxes_go if union else num_boxes
                        values = self._weighted_deim_loss(
                            loss, output, enc_targets, selected, normalizer
                        )
                        losses.update(
                            {
                                f"{key}_enc_{index}": value
                                for key, value in values.items()
                            }
                        )
            finally:
                self.num_classes = original_num_classes

        if "dn_outputs" in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = max(num_boxes * outputs["dn_meta"]["dn_num_group"], 1)
            for index, output in enumerate(outputs["dn_outputs"]):
                if "local" in self.losses:
                    output["is_dn"] = True
                    output["up"], output["reg_scale"] = (
                        outputs["up"],
                        outputs["reg_scale"],
                    )
                for loss in self.losses:
                    values = self._weighted_deim_loss(
                        loss, output, targets, indices_dn, dn_num_boxes
                    )
                    losses.update(
                        {f"{key}_dn_{index}": value for key, value in values.items()}
                    )
            if "dn_pre_outputs" in outputs:
                for loss in self.losses:
                    values = self._weighted_deim_loss(
                        loss,
                        outputs["dn_pre_outputs"],
                        targets,
                        indices_dn,
                        dn_num_boxes,
                    )
                    losses.update(
                        {f"{key}_dn_pre": value for key, value in values.items()}
                    )

        nonfinite = [
            name for name, value in losses.items() if not torch.isfinite(value).all()
        ]
        if nonfinite:
            raise FloatingPointError(f"nonfinite DEIM losses: {nonfinite}")
        return losses

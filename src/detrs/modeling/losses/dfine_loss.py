"""D-FINE criterion ported from the pinned official PyTorch implementation."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Callable, Optional, cast

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from detrs.core.workspace import register, serializable

from ..transformers.dfine_support import (
    box_cxcywh_to_xyxy,
    box_iou,
    generalized_box_iou,
)
from ..transformers.dfine_utils import bbox2distance

__all__ = ["DFINECriterion"]


@register
@serializable
class DFINECriterion(nn.Module):
    """Compute the official D-FINE matching and distribution losses.

    Args:
        matcher: Injected matcher (e.g. `DFINEHungarianMatcher`).
        weight_dict (dict): Weights of the individual loss terms.
        losses (list): Enabled loss names, e.g. `['vfl', 'boxes', 'local']`.
        alpha (float): Alpha of the focal/VFL classification loss.
        gamma (float): Gamma of the focal/VFL classification loss.
        num_classes (int): Number of foreground classes.
        reg_max (int): Discrete bins of the box distribution used by the
            `local` (FDR) loss.
        boxes_weight_format (str|None): How matched box losses are weighted,
            e.g. `giou`; `None` keeps them unweighted.
        share_matched_indices (bool): Reuse one set of matching indices for
            all loss terms.
    """

    __shared__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
    ):
        super().__init__()
        if boxes_weight_format not in (None, "iou", "giou"):
            raise ValueError("boxes_weight_format must be None, 'iou', or 'giou'")
        valid_losses = {"boxes", "focal", "vfl", "local"}
        unknown = set(losses) - valid_losses
        if unknown:
            raise ValueError(f"unsupported D-FINE losses: {sorted(unknown)}")
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.losses = list(losses)
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        self.fgl_targets: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            None
        )
        self.fgl_targets_dn: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self.own_targets = None
        self.own_targets_dn = None
        self.num_pos: Optional[torch.Tensor] = None
        self.num_neg: Optional[torch.Tensor] = None

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
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
        target = F.one_hot(target_classes, self.num_classes + 1)[..., :-1].to(
            src_logits.dtype
        )
        loss = sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
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
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = src_logits.sigmoid().detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = self._matched_target_boxes(targets, indices)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight
        return {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
        }

    def loss_local(self, outputs, targets, indices, num_boxes, temperature=5):
        if "pred_corners" not in outputs:
            return {}
        idx = self._get_src_permutation_idx(indices)
        target_boxes = self._matched_target_boxes(targets, indices)
        pred_corners = outputs["pred_corners"][idx].reshape(-1, self.reg_max + 1)
        ref_points = outputs["ref_points"][idx].detach()
        with torch.no_grad():
            if self.fgl_targets_dn is None and "is_dn" in outputs:
                self.fgl_targets_dn = bbox2distance(
                    ref_points,
                    box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max,
                    outputs["reg_scale"],
                    outputs["up"],
                )
            if self.fgl_targets is None and "is_dn" not in outputs:
                self.fgl_targets = bbox2distance(
                    ref_points,
                    box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max,
                    outputs["reg_scale"],
                    outputs["up"],
                )
        target_corners, weight_right, weight_left = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets,
        )
        ious = torch.diag(
            box_iou(
                box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]),
                box_cxcywh_to_xyxy(target_boxes),
            )[0]
        )
        weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
        losses = {
            "loss_fgl": self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )
        }

        if "teacher_corners" not in outputs:
            return losses
        pred_corners = outputs["pred_corners"].reshape(-1, self.reg_max + 1)
        target_corners = outputs["teacher_corners"].reshape(-1, self.reg_max + 1)
        if torch.equal(pred_corners, target_corners):
            losses["loss_ddf"] = pred_corners.sum() * 0
            return losses

        weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
        mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
        mask[idx] = True
        mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)
        weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
            weight_targets_local.dtype
        )
        weight_targets_local = (
            weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
        )
        loss_match_local = (
            weight_targets_local
            * (temperature**2)
            * (
                F.kl_div(
                    F.log_softmax(pred_corners / temperature, dim=1),
                    F.softmax(target_corners.detach() / temperature, dim=1),
                    reduction="none",
                ).sum(-1)
            )
        )
        if "is_dn" not in outputs:
            batch_scale = 8 / outputs["pred_boxes"].shape[0]
            self.num_pos = (mask.sum() * batch_scale) ** 0.5
            self.num_neg = ((~mask).sum() * batch_scale) ** 0.5
        loss_pos = loss_match_local[mask].mean() if mask.any() else 0
        loss_neg = loss_match_local[~mask].mean() if (~mask).any() else 0
        num_pos = cast(torch.Tensor, self.num_pos)
        num_neg = cast(torch.Tensor, self.num_neg)
        losses["loss_ddf"] = (loss_pos * num_pos + loss_neg * num_neg) / (
            num_pos + num_neg
        )
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat(
            [torch.full_like(source, i) for i, (source, _) in enumerate(indices)]
        )
        source_idx = torch.cat([source for source, _ in indices])
        return batch_idx, source_idx

    @staticmethod
    def _matched_target_boxes(targets, indices):
        return torch.cat(
            [
                target["boxes"][target_idx]
                for target, (_, target_idx) in zip(targets, indices)
            ],
            dim=0,
        )

    @staticmethod
    def _get_go_indices(indices, indices_aux_list):
        combined = indices.copy()
        for indices_aux in indices_aux_list:
            combined = [
                (torch.cat([left[0], right[0]]), torch.cat([left[1], right[1]]))
                for left, right in zip(combined.copy(), indices_aux.copy())
            ]
        results = []
        for pairs in [
            torch.cat([pair[0][:, None], pair[1][:, None]], 1) for pair in combined
        ]:
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            unique = unique[torch.argsort(counts, descending=True)]
            row_to_column = {}
            for pair in unique:
                row, column = pair[0].item(), pair[1].item()
                if row not in row_to_column:
                    row_to_column[row] = column
            rows = torch.tensor(list(row_to_column), device=pairs.device)
            columns = torch.tensor(list(row_to_column.values()), device=pairs.device)
            results.append((rows.long(), columns.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.own_targets = None
        self.own_targets_dn = None
        self.num_pos = None
        self.num_neg = None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map: dict[str, Callable[..., dict[str, torch.Tensor]]] = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "local": self.loss_local,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @staticmethod
    def _normalizer(count, device):
        value = torch.as_tensor([count], dtype=torch.float, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(value)
            value /= torch.distributed.get_world_size()
        return torch.clamp(value, min=1).item()

    @staticmethod
    def _validate_targets(targets, batch_size):
        if not isinstance(targets, Sequence) or len(targets) != batch_size:
            raise ValueError("targets must match the prediction batch size")
        for index, target in enumerate(targets):
            if (
                not isinstance(target, Mapping)
                or not {"labels", "boxes"} <= target.keys()
            ):
                raise ValueError(f"target {index} must contain labels and boxes")
            if target["labels"].ndim != 1 or target["boxes"].shape != (
                len(target["labels"]),
                4,
            ):
                raise ValueError(f"target {index} has malformed labels or boxes")
            if not torch.isfinite(target["boxes"]).all():
                raise ValueError(f"target {index} contains nonfinite boxes")

    @staticmethod
    def _validate_prediction(name, output, batch_size=None):
        if (
            not isinstance(output, Mapping)
            or not {"pred_logits", "pred_boxes"} <= output.keys()
        ):
            raise ValueError(f"{name} must contain pred_logits and pred_boxes")
        logits, boxes = output["pred_logits"], output["pred_boxes"]
        if logits.ndim != 3 or boxes.shape != (*logits.shape[:2], 4):
            raise ValueError(f"{name} has malformed prediction shapes")
        if batch_size is not None and logits.shape[0] != batch_size:
            raise ValueError(f"{name} has inconsistent batch size")
        if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
            raise ValueError(f"{name} contains nonfinite predictions")
        return logits.shape[0]

    def _validate_outputs(self, outputs):
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        batch_size = self._validate_prediction("outputs", outputs)
        required = {"aux_outputs", "pre_outputs", "enc_aux_outputs"}
        missing = required - outputs.keys()
        if missing:
            raise ValueError(
                f"outputs missing required auxiliary fields: {sorted(missing)}"
            )
        for field in ("aux_outputs", "enc_aux_outputs", "dn_outputs"):
            if field in outputs:
                if not isinstance(outputs[field], list):
                    raise ValueError(f"{field} must be a list")
                for index, output in enumerate(outputs[field]):
                    self._validate_prediction(f"{field}[{index}]", output, batch_size)
        self._validate_prediction("pre_outputs", outputs["pre_outputs"], batch_size)
        if "dn_outputs" in outputs:
            if "dn_meta" not in outputs:
                raise ValueError("dn_outputs requires dn_meta")
            metadata = outputs["dn_meta"]
            if (
                not isinstance(metadata, Mapping)
                or not {
                    "dn_positive_idx",
                    "dn_num_group",
                }
                <= metadata.keys()
            ):
                raise ValueError("dn_meta is malformed")
            if "dn_pre_outputs" in outputs:
                self._validate_prediction(
                    "dn_pre_outputs", outputs["dn_pre_outputs"], batch_size
                )
        if "enc_meta" not in outputs:
            raise ValueError("enc_aux_outputs requires enc_meta")
        return batch_size

    def _weighted_loss(self, loss, outputs, targets, indices, num_boxes):
        metadata = self.get_loss_meta_info(loss, outputs, targets, indices)
        values = self.get_loss(loss, outputs, targets, indices, num_boxes, **metadata)
        return {
            key: value * self.weight_dict[key]
            for key, value in values.items()
            if key in self.weight_dict
        }

    def forward(self, outputs, targets, **kwargs):
        del kwargs
        batch_size = self._validate_outputs(outputs)
        self._validate_targets(targets, batch_size)
        outputs_without_aux = {
            key: value for key, value in outputs.items() if "aux" not in key
        }
        indices = self.matcher(outputs_without_aux, targets)["indices"]
        self._clear_cache()

        cached_indices = []
        cached_indices_enc = []
        indices_aux_list = []
        for aux_output in outputs["aux_outputs"] + [outputs["pre_outputs"]]:
            matched = self.matcher(aux_output, targets)["indices"]
            cached_indices.append(matched)
            indices_aux_list.append(matched)
        for enc_output in outputs["enc_aux_outputs"]:
            matched = self.matcher(enc_output, targets)["indices"]
            cached_indices_enc.append(matched)
            indices_aux_list.append(matched)
        indices_go = self._get_go_indices(indices, indices_aux_list)
        device = outputs["pred_logits"].device
        num_boxes_go = self._normalizer(
            sum(len(pair[0]) for pair in indices_go), device
        )
        num_boxes = self._normalizer(
            sum(len(target["labels"]) for target in targets), device
        )

        losses = {}
        for loss in self.losses:
            local = loss in ("boxes", "local")
            losses.update(
                self._weighted_loss(
                    loss,
                    outputs,
                    targets,
                    indices_go if local else indices,
                    num_boxes_go if local else num_boxes,
                )
            )

        for index, aux_output in enumerate(outputs["aux_outputs"]):
            aux_output["up"] = outputs["up"]
            aux_output["reg_scale"] = outputs["reg_scale"]
            for loss in self.losses:
                local = loss in ("boxes", "local")
                values = self._weighted_loss(
                    loss,
                    aux_output,
                    targets,
                    indices_go if local else cached_indices[index],
                    num_boxes_go if local else num_boxes,
                )
                losses.update(
                    {f"{key}_aux_{index}": value for key, value in values.items()}
                )

        for loss in self.losses:
            local = loss in ("boxes", "local")
            values = self._weighted_loss(
                loss,
                outputs["pre_outputs"],
                targets,
                indices_go if local else cached_indices[-1],
                num_boxes_go if local else num_boxes,
            )
            losses.update({f"{key}_pre": value for key, value in values.items()})

        class_agnostic = outputs["enc_meta"]["class_agnostic"]
        original_num_classes = self.num_classes
        enc_targets = targets
        if class_agnostic:
            self.num_classes = 1
            enc_targets = copy.deepcopy(targets)
            for target in enc_targets:
                target["labels"] = torch.zeros_like(target["labels"])
        try:
            for index, enc_output in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    local = loss == "boxes"
                    values = self._weighted_loss(
                        loss,
                        enc_output,
                        enc_targets,
                        indices_go if local else cached_indices_enc[index],
                        num_boxes_go if local else num_boxes,
                    )
                    losses.update(
                        {f"{key}_enc_{index}": value for key, value in values.items()}
                    )
        finally:
            self.num_classes = original_num_classes

        if "dn_outputs" in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = max(num_boxes * outputs["dn_meta"]["dn_num_group"], 1)
            for index, dn_output in enumerate(outputs["dn_outputs"]):
                dn_output["is_dn"] = True
                dn_output["up"] = outputs["up"]
                dn_output["reg_scale"] = outputs["reg_scale"]
                for loss in self.losses:
                    values = self._weighted_loss(
                        loss, dn_output, targets, indices_dn, dn_num_boxes
                    )
                    losses.update(
                        {f"{key}_dn_{index}": value for key, value in values.items()}
                    )
            if "dn_pre_outputs" in outputs:
                for loss in self.losses:
                    values = self._weighted_loss(
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
            raise FloatingPointError(f"nonfinite D-FINE losses: {nonfinite}")
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}
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
            values = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            )
        if loss == "boxes":
            return {"boxes_weight": values}
        if loss == "vfl":
            return {"values": values}
        return {}

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        positive_indices = dn_meta["dn_positive_idx"]
        num_groups = dn_meta["dn_num_group"]
        if len(positive_indices) != len(targets) or num_groups < 1:
            raise ValueError("dn_meta does not match targets")
        device = targets[0]["labels"].device
        results = []
        for index, target in enumerate(targets):
            num_gt = len(target["labels"])
            if num_gt:
                target_indices = torch.arange(
                    num_gt, dtype=torch.int64, device=device
                ).tile(num_groups)
                if len(positive_indices[index]) != len(target_indices):
                    raise ValueError(
                        "dn_positive_idx length does not match dn_num_group"
                    )
                results.append((positive_indices[index], target_indices))
            else:
                empty = torch.zeros(0, dtype=torch.int64, device=device)
                results.append((empty, empty))
        return results

    @staticmethod
    def unimodal_distribution_focal_loss(
        pred,
        label,
        weight_right,
        weight_left,
        weight=None,
        reduction="sum",
        avg_factor=None,
    ):
        left = label.long()
        right = left + 1
        loss = F.cross_entropy(pred, left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, right, reduction="none") * weight_right.reshape(-1)
        if weight is not None:
            loss = loss * weight.float()
        if avg_factor is not None:
            return loss.sum() / avg_factor
        if reduction == "mean":
            return loss.mean()
        return loss.sum()

    @staticmethod
    def feature_loss_function(features, target_features):
        loss = (features - target_features) ** 2
        loss = loss * ((features > 0) | (target_features > 0)).float()
        return loss.abs()

    @staticmethod
    def get_gradual_steps(outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1) if num_layers > 1 else 0
        return (
            [0.5 + step * index for index in range(num_layers)]
            if num_layers > 1
            else [1]
        )

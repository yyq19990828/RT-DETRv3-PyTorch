"""Shared target, matching, box, and denoising support for D-FINE families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from ppdet_pytorch.core.workspace import register, serializable

from .utils import inverse_sigmoid

__all__ = [
    "DFINEHungarianMatcher",
    "box_cxcywh_to_xyxy",
    "box_iou",
    "box_xyxy_to_cxcywh",
    "generalized_box_iou",
    "get_contrastive_denoising_training_group",
    "repository_batch_to_dfine_targets",
]


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    width = width.clamp(min=0.0)
    height = height.clamp(min=0.0)
    return torch.stack(
        [
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ],
        dim=-1,
    )


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    size = (right_bottom - left_top).clamp(min=0)
    intersection = size[..., 0] * size[..., 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError("boxes1 contains malformed xyxy boxes")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError("boxes2 contains malformed xyxy boxes")
    iou, union = box_iou(boxes1, boxes2)
    left_top = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    size = (right_bottom - left_top).clamp(min=0)
    area = size[..., 0] * size[..., 1]
    return iou - (area - union) / area


def _validate_targets(targets: Sequence[Mapping[str, torch.Tensor]]) -> None:
    if not targets:
        raise ValueError("targets must contain at least one image")
    for index, target in enumerate(targets):
        if "labels" not in target or "boxes" not in target:
            raise ValueError(f"target {index} must contain labels and boxes")
        labels, boxes = target["labels"], target["boxes"]
        if labels.ndim != 1 or boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError(f"target {index} has malformed labels or boxes")
        if len(labels) != len(boxes):
            raise ValueError(f"target {index} has mismatched label and box lengths")
        if not torch.isfinite(boxes).all():
            raise ValueError(f"target {index} contains nonfinite boxes")


def repository_batch_to_dfine_targets(
    batch: Mapping[str, object],
) -> list[dict[str, torch.Tensor]]:
    """Convert repository list-based targets to the official D-FINE target form."""
    if "gt_class" not in batch or "gt_bbox" not in batch:
        raise ValueError("batch must contain gt_class and gt_bbox")
    labels = batch["gt_class"]
    boxes = batch["gt_bbox"]
    if not isinstance(labels, Sequence) or not isinstance(boxes, Sequence):
        raise ValueError("gt_class and gt_bbox must be per-image sequences")
    if len(labels) != len(boxes):
        raise ValueError("batch has mismatched gt_class and gt_bbox lengths")

    targets = []
    for index, (image_labels, image_boxes) in enumerate(zip(labels, boxes)):
        if not isinstance(image_labels, torch.Tensor) or not isinstance(
            image_boxes, torch.Tensor
        ):
            raise TypeError(f"target {index} labels and boxes must be tensors")
        target = {
            "labels": image_labels.reshape(-1).to(dtype=torch.int64),
            "boxes": image_boxes,
        }
        targets.append(target)

    _validate_targets(targets)
    if "im_shape" in batch and "scale_factor" in batch:
        im_shape = batch["im_shape"]
        scale_factor = batch["scale_factor"]
        if not isinstance(im_shape, torch.Tensor) or not isinstance(
            scale_factor, torch.Tensor
        ):
            raise TypeError("im_shape and scale_factor must be tensors")
        if im_shape.shape != scale_factor.shape or im_shape.shape != (len(targets), 2):
            raise ValueError("im_shape and scale_factor must have shape [batch, 2]")
        original_sizes = torch.floor(im_shape / scale_factor + 0.5).flip(-1)
        for target, original_size in zip(targets, original_sizes):
            target["orig_size"] = original_size
    return targets


@register
@serializable
class DFINEHungarianMatcher(nn.Module):
    """Pinned D-FINE matcher contract backed by the official cost equations."""

    __shared__ = ["use_focal_loss"]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        if self.cost_class == self.cost_bbox == self.cost_giou == 0:
            raise ValueError("all matcher costs cannot be zero")

    @torch.no_grad()
    def forward(self, outputs, targets):
        _validate_targets(targets)
        logits = outputs.get("pred_logits")
        boxes = outputs.get("pred_boxes")
        if logits is None or boxes is None:
            raise ValueError("outputs must contain pred_logits and pred_boxes")
        if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
            raise ValueError("matcher predictions must be finite")

        batch_size, num_queries = logits.shape[:2]
        sizes = [len(target["boxes"]) for target in targets]
        if batch_size != len(targets):
            raise ValueError("prediction and target batch lengths differ")
        if sum(sizes) == 0:
            empty = torch.empty(0, dtype=torch.int64, device=boxes.device)
            return {"indices": [(empty, empty) for _ in targets]}

        probabilities = logits.flatten(0, 1)
        probabilities = (
            probabilities.sigmoid()
            if self.use_focal_loss
            else probabilities.softmax(-1)
        )
        flat_boxes = boxes.flatten(0, 1)
        target_labels = torch.cat([target["labels"] for target in targets])
        target_boxes = torch.cat([target["boxes"] for target in targets])
        selected_probabilities = probabilities[:, target_labels]
        if self.use_focal_loss:
            negative_cost = (
                (1 - self.alpha)
                * selected_probabilities.pow(self.gamma)
                * -(1 - selected_probabilities + 1e-8).log()
            )
            positive_cost = (
                self.alpha
                * (1 - selected_probabilities).pow(self.gamma)
                * -(selected_probabilities + 1e-8).log()
            )
            class_cost = positive_cost - negative_cost
        else:
            class_cost = -selected_probabilities

        bbox_cost = torch.cdist(flat_boxes, target_boxes, p=1)
        giou_cost = -generalized_box_iou(
            box_cxcywh_to_xyxy(flat_boxes), box_cxcywh_to_xyxy(target_boxes)
        )
        cost = (
            self.cost_bbox * bbox_cost
            + self.cost_class * class_cost
            + self.cost_giou * giou_cost
        ).view(batch_size, num_queries, -1)
        cost = torch.nan_to_num(cost, nan=1.0)
        split_costs = cost.split(sizes, dim=-1)
        indices = [
            linear_sum_assignment(image_cost[index].cpu().numpy())
            for index, image_cost in enumerate(split_costs)
        ]
        return {
            "indices": [
                (
                    torch.as_tensor(source, dtype=torch.int64, device=boxes.device),
                    torch.as_tensor(target, dtype=torch.int64, device=boxes.device),
                )
                for source, target in indices
            ]
        }


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
):
    """Create D-FINE contrastive denoising queries with official mask semantics."""
    if num_denoising <= 0:
        return None, None, None, None
    _validate_targets(targets)
    num_gts = [len(target["labels"]) for target in targets]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        metadata: dict[str, object] = {
            "dn_positive_idx": None,
            "dn_num_group": 0,
            "dn_num_split": [0, num_queries],
        }
        return None, None, None, metadata

    device = targets[0]["labels"].device
    num_group = max(1, num_denoising // max_gt_num)
    batch_size = len(targets)
    query_class = torch.full(
        (batch_size, max_gt_num), num_classes, dtype=torch.int32, device=device
    )
    query_bbox = torch.zeros((batch_size, max_gt_num, 4), device=device)
    padding_mask = torch.zeros(
        (batch_size, max_gt_num), dtype=torch.bool, device=device
    )
    for index, target in enumerate(targets):
        count = num_gts[index]
        query_class[index, :count] = target["labels"]
        query_bbox[index, :count] = target["boxes"]
        padding_mask[index, :count] = True

    query_class = query_class.tile((1, 2 * num_group))
    query_bbox = query_bbox.tile((1, 2 * num_group, 1))
    padding_mask = padding_mask.tile((1, 2 * num_group))
    negative_mask = torch.zeros((batch_size, max_gt_num * 2, 1), device=device)
    negative_mask[:, max_gt_num:] = 1
    negative_mask = negative_mask.tile((1, num_group, 1))
    positive_mask = (1 - negative_mask).squeeze(-1) * padding_mask
    positive_indices = torch.split(
        torch.nonzero(positive_mask)[:, 1],
        [count * num_group for count in num_gts],
    )
    total_denoising = max_gt_num * 2 * num_group

    if label_noise_ratio > 0:
        noise_mask = torch.rand_like(query_class, dtype=torch.float) < (
            label_noise_ratio * 0.5
        )
        new_labels = torch.randint_like(
            noise_mask, 0, num_classes, dtype=query_class.dtype
        )
        query_class = torch.where(noise_mask & padding_mask, new_labels, query_class)
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(query_bbox)
        difference = query_bbox[..., 2:].tile((1, 1, 2)) * 0.5 * box_noise_scale
        random_sign = torch.randint_like(query_bbox, 0, 2) * 2.0 - 1.0
        random_part = torch.rand_like(query_bbox)
        random_part = (random_part + 1.0) * negative_mask + random_part * (
            1 - negative_mask
        )
        known_bbox = (known_bbox + random_sign * random_part * difference).clamp(0, 1)
        query_bbox = box_xyxy_to_cxcywh(known_bbox)
        query_bbox[query_bbox < 0] *= -1
        query_bbox = inverse_sigmoid(query_bbox)

    query_logits = class_embed(query_class)
    target_size = total_denoising + num_queries
    attention_mask = torch.zeros(
        (target_size, target_size), dtype=torch.bool, device=device
    )
    attention_mask[total_denoising:, :total_denoising] = True
    group_size = max_gt_num * 2
    for group_index in range(num_group):
        start = group_size * group_index
        end = group_size * (group_index + 1)
        attention_mask[start:end, :start] = True
        attention_mask[start:end, end:total_denoising] = True

    metadata = {
        "dn_positive_idx": positive_indices,
        "dn_num_group": num_group,
        "dn_num_split": [total_denoising, num_queries],
    }
    return query_logits, query_bbox, attention_mask, metadata

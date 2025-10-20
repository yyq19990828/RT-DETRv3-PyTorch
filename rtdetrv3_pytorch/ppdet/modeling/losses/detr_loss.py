# Copyright (c) 2025 RT-DETRv3 PyTorch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from PaddlePaddle RT-DETRv3
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

"""
DINOv3Loss implementation for RT-DETRv3 PyTorch.

This module implements the loss function for DINOv3, which includes:
- Varifocal Loss for classification
- GIoU Loss and L1 Loss for bounding box regression
- Hungarian Matching for one-to-one assignment
- Support for one-to-many supervision
- Support for denoising queries
"""

from typing import Dict, List, Optional, Tuple
from ppdet.core.workspace import register
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def bbox_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

    Args:
        x: Tensor of shape (..., 4) containing boxes in cxcywh format

    Returns:
        Tensor of shape (..., 4) containing boxes in xyxy format
    """
    cxcy, wh = x.split(2, dim=-1)
    return torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1)


def bbox_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.

    Args:
        x: Tensor of shape (..., 4) containing boxes in xyxy format

    Returns:
        Tensor of shape (..., 4) containing boxes in cxcywh format
    """
    x1, y1, x2, y2 = x.split(1, dim=-1)
    return torch.cat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    normalizer: float = 1.0,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Sigmoid focal loss for classification.

    Args:
        logits: Predicted logits, shape (N, num_classes)
        labels: Target labels (one-hot), shape (N, num_classes)
        normalizer: Normalization factor
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t) ^ gamma

    Returns:
        Scalar loss value
    """
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t = prob * labels + (1 - prob) * (1 - labels)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    return loss.mean(1).sum() / normalizer


def varifocal_loss_with_logits(
    pred_logits: torch.Tensor,
    gt_score: torch.Tensor,
    label: torch.Tensor,
    normalizer: float = 1.0,
    alpha: float = 0.75,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Varifocal loss for classification with quality estimation.

    Args:
        pred_logits: Predicted logits, shape (N, num_classes)
        gt_score: Target quality scores (e.g., IoU), shape (N, num_classes)
        label: Target labels (one-hot), shape (N, num_classes)
        normalizer: Normalization factor
        alpha: Weighting factor for negative examples
        gamma: Focusing parameter

    Returns:
        Scalar loss value
    """
    pred_score = torch.sigmoid(pred_logits)
    # Detach weight to prevent gradient flow through weight parameter
    weight = (alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label).detach()
    loss = F.binary_cross_entropy_with_logits(
        pred_logits, gt_score, weight=weight, reduction='none'
    )
    return loss.mean(1).sum() / normalizer


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.

    Args:
        box1: Boxes in xyxy format, shape (..., 4)
        box2: Boxes in xyxy format, shape (..., 4)
        eps: Small value to avoid division by zero

    Returns:
        IoU values, shape matching input shapes
    """
    x1, y1, x2, y2 = box1.unbind(-1)
    x1g, y1g, x2g, y2g = box2.unbind(-1)

    # Intersection area
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    w_inter = (xkis2 - xkis1).clamp(min=0)
    h_inter = (ykis2 - ykis1).clamp(min=0)
    overlap = w_inter * h_inter

    # Union area
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - overlap + eps

    iou = overlap / union
    return iou


class GIoULoss(nn.Module):
    """
    Generalized Intersection over Union Loss.

    Reference: https://arxiv.org/abs/1902.09630
    """

    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-10, reduction: str = 'none'):
        """
        Args:
            loss_weight: Weight for the loss
            eps: Small value to avoid division by zero
            reduction: Reduction method: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        iou_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate GIoU loss.

        Args:
            pred_boxes: Predicted boxes in xyxy format, shape (..., 4)
            target_boxes: Target boxes in xyxy format, shape (..., 4)
            iou_weight: Optional per-box weight, shape (...)

        Returns:
            GIoU loss value
        """
        x1, y1, x2, y2 = pred_boxes.unbind(-1)
        x1g, y1g, x2g, y2g = target_boxes.unbind(-1)

        # IoU calculation
        xkis1 = torch.max(x1, x1g)
        ykis1 = torch.max(y1, y1g)
        xkis2 = torch.min(x2, x2g)
        ykis2 = torch.min(y2, y2g)

        w_inter = (xkis2 - xkis1).clamp(min=0)
        h_inter = (ykis2 - ykis1).clamp(min=0)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + self.eps
        iou = overlap / union

        # GIoU calculation
        xc1 = torch.min(x1, x1g)
        yc1 = torch.min(y1, y1g)
        xc2 = torch.max(x2, x2g)
        yc2 = torch.max(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        giou = iou - ((area_c - union) / area_c)
        loss = 1 - giou

        if iou_weight is not None:
            loss = loss * iou_weight

        if self.reduction == 'none':
            pass
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:  # mean
            loss = loss.mean()

        return loss * self.loss_weight


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for bipartite matching between predictions and ground truth.

    This module computes an assignment between predictions and targets using
    the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        use_focal_loss: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        """
        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for L1 bbox cost
            cost_giou: Weight for GIoU cost
            use_focal_loss: Whether to use focal loss for classification cost
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.giou_loss = GIoULoss()

    @torch.no_grad()
    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.

        Args:
            pred_boxes: Predicted boxes, shape (batch_size, num_queries, 4) in cxcywh format
            pred_logits: Predicted logits, shape (batch_size, num_queries, num_classes)
            gt_boxes: List of ground truth boxes for each image, each shape (num_gt, 4) in cxcywh format
            gt_labels: List of ground truth labels for each image, each shape (num_gt,)

        Returns:
            List of tuples (pred_indices, gt_indices) for each image in the batch
        """
        bs, num_queries = pred_boxes.shape[:2]

        # Ensure gt_boxes and gt_labels have same length as batch size
        assert len(gt_boxes) == bs, f"gt_boxes length {len(gt_boxes)} != batch size {bs}"
        assert len(gt_labels) == bs, f"gt_labels length {len(gt_labels)} != batch size {bs}"

        # Check if there are any ground truth boxes
        num_gts = [len(labels) for labels in gt_labels]
        if sum(num_gts) == 0:
            return [(torch.tensor([], dtype=torch.int64, device=pred_boxes.device),
                     torch.tensor([], dtype=torch.int64, device=pred_boxes.device))
                    for _ in range(bs)]

        # Flatten predictions for batch processing
        # [batch_size * num_queries, num_classes]
        out_prob = pred_logits.flatten(0, 1).sigmoid() if self.use_focal_loss else pred_logits.flatten(0, 1).softmax(-1)
        # [batch_size * num_queries, 4]
        out_bbox = pred_boxes.flatten(0, 1)

        # Concatenate all targets
        tgt_ids = torch.cat(gt_labels).flatten()
        tgt_bbox = torch.cat(gt_boxes)

        # Compute classification cost
        out_prob = out_prob[:, tgt_ids]
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (
                -(1 - out_prob + 1e-8).log()
            )
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (
                -(out_prob + 1e-8).log()
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        # Compute L1 bbox cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute GIoU cost
        cost_giou = -bbox_iou(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))
        ).squeeze(-1)

        # Final cost matrix
        C = (
            self.cost_class * cost_class +
            self.cost_bbox * cost_bbox +
            self.cost_giou * cost_giou
        )

        # Reshape to [batch_size, num_queries, num_total_gt]
        C = C.view(bs, num_queries, -1).cpu()

        # Split by batch and perform Hungarian matching
        sizes = [len(gt) for gt in gt_boxes]
        indices = []

        # Split cost matrix by batch
        start_idx = 0
        for i in range(bs):
            size_i = sizes[i]
            if size_i > 0:
                # Extract cost matrix for current batch
                c_i = C[i, :, start_idx:start_idx + size_i]  # [num_queries, num_gt_i]
                row_ind, col_ind = linear_sum_assignment(c_i.numpy())
                indices.append((
                    torch.as_tensor(row_ind, dtype=torch.int64, device=pred_boxes.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=pred_boxes.device)
                ))
                start_idx += size_i
            else:
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=pred_boxes.device),
                    torch.tensor([], dtype=torch.int64, device=pred_boxes.device)
                ))

        return indices


@register
class DINOv3Loss(nn.Module):
    """

    __category__ = 'loss'
    DINOv3 Loss for RT-DETRv3.

    This loss function implements:
    - Varifocal Loss for classification
    - L1 Loss for bbox regression
    - GIoU Loss for bbox regression
    - Hungarian matching for one-to-one assignment
    - Support for one-to-many supervision
    - Support for denoising queries
    """

    def __init__(
        self,
        num_classes: int = 80,
        loss_coeff: Optional[Dict[str, float]] = None,
        aux_loss: bool = True,
        use_focal_loss: bool = True,
        use_vfl: bool = True,
        matcher: Optional[HungarianMatcher] = None
    ):
        """
        Args:
            num_classes: Number of object classes
            loss_coeff: Dictionary of loss coefficients with keys:
                'class', 'bbox', 'giou', 'no_object'
            aux_loss: Whether to compute auxiliary losses for intermediate decoder layers
            use_focal_loss: Whether to use focal loss for classification
            use_vfl: Whether to use varifocal loss
            matcher: Hungarian matcher instance (will be created if None)
        """
        super().__init__()
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl

        # Default loss coefficients
        if loss_coeff is None:
            loss_coeff = {
                'class': 1.0,
                'bbox': 5.0,
                'giou': 2.0,
                'no_object': 0.1
            }
        self.loss_coeff = loss_coeff

        # Create matcher if not provided
        if matcher is None:
            self.matcher = HungarianMatcher(
                cost_class=loss_coeff.get('class', 1.0),
                cost_bbox=loss_coeff.get('bbox', 5.0),
                cost_giou=loss_coeff.get('giou', 2.0),
                use_focal_loss=use_focal_loss
            )
        else:
            self.matcher = matcher

        self.giou_loss = GIoULoss()

    def _get_num_gts(self, gt_labels: List[torch.Tensor]) -> torch.Tensor:
        """
        Get the number of ground truth boxes across all samples and GPUs.

        Args:
            gt_labels: List of ground truth labels for each image

        Returns:
            Scalar tensor with the averaged number of GTs
        """
        num_gts = sum(len(labels) for labels in gt_labels)
        num_gts = torch.as_tensor([num_gts], dtype=torch.float32, device=gt_labels[0].device if gt_labels else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Synchronize across GPUs in distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_gts)
            num_gts = num_gts / torch.distributed.get_world_size()

        # Clamp to at least 1 to avoid division by zero
        num_gts = torch.clamp(num_gts, min=1.0)
        return num_gts

    def _get_src_target_assign(
        self,
        src: torch.Tensor,
        target: List[torch.Tensor],
        match_indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign source and target according to match indices.

        Args:
            src: Source tensor, shape (batch_size, num_queries, dim)
            target: List of target tensors, each shape (num_gt, dim)
            match_indices: List of (src_idx, tgt_idx) tuples from Hungarian matching

        Returns:
            Tuple of (assigned_src, assigned_target), both shape (total_matched, dim)
        """
        batch_idx = torch.cat([
            torch.full_like(src_idx, i) for i, (src_idx, _) in enumerate(match_indices)
        ])
        src_idx = torch.cat([src_idx for (src_idx, _) in match_indices])

        src_assign = src[batch_idx, src_idx]

        target_assign = torch.cat([
            tgt[tgt_idx] if len(tgt_idx) > 0 else torch.zeros((0, tgt.shape[-1]), device=tgt.device)
            for tgt, (_, tgt_idx) in zip(target, match_indices)
        ])

        return src_assign, target_assign

    def _get_loss_class(
        self,
        pred_logits: torch.Tensor,
        gt_labels: List[torch.Tensor],
        match_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_gts: torch.Tensor,
        iou_score: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss.

        Args:
            pred_logits: Predicted logits, shape (batch_size, num_queries, num_classes)
            gt_labels: List of ground truth labels for each image
            match_indices: List of matching indices from Hungarian matcher
            num_gts: Number of ground truth boxes
            iou_score: Optional IoU scores for varifocal loss

        Returns:
            Dictionary with 'loss_class' key
        """
        bs, num_queries = pred_logits.shape[:2]

        # Create target labels (all background initially)
        target_label = torch.full(
            (bs, num_queries),
            self.num_classes,
            dtype=torch.int64,
            device=pred_logits.device
        )

        # Assign matched labels
        num_gt = sum(len(labels) for labels in gt_labels)
        if num_gt > 0:
            batch_idx = torch.cat([
                torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)
            ])
            src_idx = torch.cat([src for (src, _) in match_indices])
            tgt_labels = torch.cat([
                labels[tgt] for labels, (_, tgt) in zip(gt_labels, match_indices)
            ])

            target_label[batch_idx, src_idx] = tgt_labels

        if self.use_focal_loss:
            # One-hot encoding (excluding background class for focal loss)
            target_label_onehot = F.one_hot(target_label, self.num_classes + 1)[..., :-1].float()

            if iou_score is not None and self.use_vfl:
                # Varifocal loss with IoU quality score
                target_score = torch.zeros(
                    (bs, num_queries, 1),
                    dtype=torch.float32,
                    device=pred_logits.device
                )
                if num_gt > 0:
                    target_score[batch_idx, src_idx] = iou_score.unsqueeze(-1)

                target_score = target_score * target_label_onehot

                loss_class = varifocal_loss_with_logits(
                    pred_logits,
                    target_score,
                    target_label_onehot,
                    num_gts / num_queries
                )
            else:
                # Standard focal loss
                loss_class = sigmoid_focal_loss(
                    pred_logits,
                    target_label_onehot,
                    num_gts / num_queries
                )

            loss_class = self.loss_coeff['class'] * loss_class
        else:
            # Cross entropy loss
            loss_weight = torch.ones(self.num_classes + 1, device=pred_logits.device) * self.loss_coeff['class']
            loss_weight[-1] = self.loss_coeff.get('no_object', 0.1)
            loss_class = F.cross_entropy(
                pred_logits.flatten(0, 1),
                target_label.flatten(0, 1),
                weight=loss_weight,
                reduction='mean'
            )

        return {'loss_class': loss_class.squeeze() if loss_class.dim() > 0 else loss_class}

    def _get_loss_bbox(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        match_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_gts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute bounding box losses (L1 and GIoU).

        Args:
            pred_boxes: Predicted boxes, shape (batch_size, num_queries, 4) in cxcywh format
            gt_boxes: List of ground truth boxes for each image in cxcywh format
            match_indices: List of matching indices from Hungarian matcher
            num_gts: Number of ground truth boxes

        Returns:
            Dictionary with 'loss_bbox' and 'loss_giou' keys
        """
        if sum(len(boxes) for boxes in gt_boxes) == 0:
            return {
                'loss_bbox': torch.tensor(0.0, device=pred_boxes.device),
                'loss_giou': torch.tensor(0.0, device=pred_boxes.device)
            }

        src_boxes, target_boxes = self._get_src_target_assign(
            pred_boxes, gt_boxes, match_indices
        )

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='sum') / num_gts
        loss_bbox = self.loss_coeff['bbox'] * loss_bbox

        # GIoU loss
        loss_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_boxes),
            bbox_cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = loss_giou.sum() / num_gts
        loss_giou = self.loss_coeff['giou'] * loss_giou

        # Squeeze to ensure scalar outputs
        return {
            'loss_bbox': loss_bbox.squeeze() if loss_bbox.dim() > 0 else loss_bbox,
            'loss_giou': loss_giou.squeeze() if loss_giou.dim() > 0 else loss_giou
        }

    def _get_loss_aux(
        self,
        pred_boxes_list: List[torch.Tensor],
        pred_logits_list: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        num_gts: torch.Tensor,
        postfix: str = "_aux"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for intermediate decoder layers.

        Args:
            pred_boxes_list: List of predicted boxes from intermediate layers
            pred_logits_list: List of predicted logits from intermediate layers
            gt_boxes: List of ground truth boxes
            gt_labels: List of ground truth labels
            num_gts: Number of ground truth boxes
            postfix: Suffix for loss names

        Returns:
            Dictionary with auxiliary loss values
        """
        loss_class_aux = []
        loss_bbox_aux = []
        loss_giou_aux = []

        for aux_boxes, aux_logits in zip(pred_boxes_list, pred_logits_list):
            # Perform matching for this layer
            match_indices = self.matcher(aux_boxes, aux_logits, gt_boxes, gt_labels)

            # Compute IoU score for varifocal loss if enabled
            iou_score = None
            if self.use_vfl and sum(len(boxes) for boxes in gt_boxes) > 0:
                src_boxes, target_boxes = self._get_src_target_assign(
                    aux_boxes.detach(), gt_boxes, match_indices
                )
                iou_score = bbox_iou(
                    bbox_cxcywh_to_xyxy(src_boxes.unsqueeze(1)),
                    bbox_cxcywh_to_xyxy(target_boxes.unsqueeze(1))
                ).squeeze(-1)

            # Classification loss
            loss_class = self._get_loss_class(
                aux_logits, gt_labels, match_indices, num_gts, iou_score
            )
            loss_class_aux.append(loss_class['loss_class'])

            # Bbox losses
            loss_bbox = self._get_loss_bbox(
                aux_boxes, gt_boxes, match_indices, num_gts
            )
            loss_bbox_aux.append(loss_bbox['loss_bbox'])
            loss_giou_aux.append(loss_bbox['loss_giou'])

        return {
            f'loss_class{postfix}': sum(loss_class_aux),
            f'loss_bbox{postfix}': sum(loss_bbox_aux),
            f'loss_giou{postfix}': sum(loss_giou_aux)
        }

    @staticmethod
    def get_dn_match_indices(
        gt_labels: List[torch.Tensor],
        dn_positive_idx: List[torch.Tensor],
        dn_num_group: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get match indices for denoising queries.

        Args:
            gt_labels: List of ground truth labels for each image
            dn_positive_idx: List of positive query indices for denoising
            dn_num_group: Number of denoising groups

        Returns:
            List of (query_idx, gt_idx) tuples for each image
        """
        dn_match_indices = []
        for i in range(len(gt_labels)):
            num_gt = len(gt_labels[i])
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=gt_labels[i].device)
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                device = gt_labels[i].device if isinstance(gt_labels[i], torch.Tensor) else torch.device('cpu')
                dn_match_indices.append((
                    torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device)
                ))
        return dn_match_indices

    def forward(
        self,
        pred_boxes: List[torch.Tensor],
        pred_logits: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        dn_meta: Optional[Dict] = None,
        o2m: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DINOv3 loss.

        Args:
            pred_boxes: List of predicted boxes from each decoder layer,
                        each shape (batch_size, num_queries, 4) in cxcywh format
            pred_logits: List of predicted logits from each decoder layer,
                         each shape (batch_size, num_queries, num_classes)
            gt_boxes: List of ground truth boxes for each image, each shape (num_gt, 4)
            gt_labels: List of ground truth labels for each image, each shape (num_gt,)
            dn_meta: Optional dictionary with denoising metadata containing:
                     'dn_positive_idx', 'dn_num_group', 'dn_out_bboxes', 'dn_out_logits'
            o2m: One-to-many ratio (number of times to replicate GT for o2m supervision)

        Returns:
            Dictionary of loss values
        """
        # Handle one-to-many supervision
        if o2m != 1:
            gt_boxes_copy = [box.repeat(o2m, 1) for box in gt_boxes]
            gt_labels_copy = [label.repeat(o2m) for label in gt_labels]
        else:
            gt_boxes_copy = gt_boxes
            gt_labels_copy = gt_labels

        num_gts = self._get_num_gts(gt_labels_copy)

        # Main prediction loss (last decoder layer)
        match_indices = self.matcher(pred_boxes[-1], pred_logits[-1], gt_boxes_copy, gt_labels_copy)

        # Compute IoU score for varifocal loss
        iou_score = None
        if self.use_vfl and sum(len(boxes) for boxes in gt_boxes_copy) > 0:
            src_boxes, target_boxes = self._get_src_target_assign(
                pred_boxes[-1].detach(), gt_boxes_copy, match_indices
            )
            iou_score = bbox_iou(
                bbox_cxcywh_to_xyxy(src_boxes.unsqueeze(1)),
                bbox_cxcywh_to_xyxy(target_boxes.unsqueeze(1))
            ).squeeze(-1)

        total_loss = {}
        total_loss.update(self._get_loss_class(
            pred_logits[-1], gt_labels_copy, match_indices, num_gts, iou_score
        ))
        total_loss.update(self._get_loss_bbox(
            pred_boxes[-1], gt_boxes_copy, match_indices, num_gts
        ))

        # Auxiliary losses for intermediate decoder layers
        if self.aux_loss and len(pred_boxes) > 1:
            total_loss.update(self._get_loss_aux(
                pred_boxes[:-1], pred_logits[:-1],
                gt_boxes_copy, gt_labels_copy, num_gts
            ))

        # Denoising losses
        if dn_meta is not None:
            num_gts_dn = self._get_num_gts(gt_labels)
            dn_positive_idx = dn_meta['dn_positive_idx']
            dn_num_group = dn_meta['dn_num_group']

            # Get denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_labels, dn_positive_idx, dn_num_group
            )

            # Multiply num_gts by dn_num_group for denoising
            num_gts_dn = num_gts_dn * dn_num_group

            # Get denoising predictions
            dn_pred_boxes = dn_meta['dn_out_bboxes']
            dn_pred_logits = dn_meta['dn_out_logits']

            # Compute denoising losses for all layers
            dn_loss = {}
            for layer_idx, (dn_boxes, dn_logits) in enumerate(zip(dn_pred_boxes, dn_pred_logits)):
                # Classification loss
                dn_loss_class = self._get_loss_class(
                    dn_logits, gt_labels, dn_match_indices, num_gts_dn, None
                )
                # Bbox losses
                dn_loss_bbox = self._get_loss_bbox(
                    dn_boxes, gt_boxes, dn_match_indices, num_gts_dn
                )

                if layer_idx == 0:
                    dn_loss['loss_class_dn'] = dn_loss_class['loss_class']
                    dn_loss['loss_bbox_dn'] = dn_loss_bbox['loss_bbox']
                    dn_loss['loss_giou_dn'] = dn_loss_bbox['loss_giou']
                else:
                    dn_loss['loss_class_dn'] += dn_loss_class['loss_class']
                    dn_loss['loss_bbox_dn'] += dn_loss_bbox['loss_bbox']
                    dn_loss['loss_giou_dn'] += dn_loss_bbox['loss_giou']

            total_loss.update(dn_loss)
        else:
            # Add zero denoising losses if not present
            total_loss.update({
                'loss_class_dn': torch.tensor(0.0, device=pred_boxes[-1].device),
                'loss_bbox_dn': torch.tensor(0.0, device=pred_boxes[-1].device),
                'loss_giou_dn': torch.tensor(0.0, device=pred_boxes[-1].device)
            })

        return total_loss

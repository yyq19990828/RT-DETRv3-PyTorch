"""
DETR Post-processing for RT-DETRv3

This module implements post-processing for DETR-based models, converting model
outputs to final detection results.

Components:
- DETRPostProcessor: Main post-processing class for object detection
- bbox_cxcywh_to_xyxy: Coordinate conversion utility

Following PaddlePaddle's implementation for consistency.

Reference:
- PaddlePaddle RT-DETR: ppdet/modeling/post_process.py:450-586
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def bbox_cxcywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format

    Args:
        bbox: Bounding boxes in (cx, cy, w, h) format, shape (..., 4)

    Returns:
        Bounding boxes in (x1, y1, x2, y2) format, shape (..., 4)

    Example:
        >>> bbox = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        >>> bbox_cxcywh_to_xyxy(bbox)
        tensor([[0.4, 0.35, 0.6, 0.65]])
    """
    cxcy, wh = bbox.split(2, dim=-1)
    return torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1)


def bbox_xyxy_to_cxcywh(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format

    Args:
        bbox: Bounding boxes in (x1, y1, x2, y2) format, shape (..., 4)

    Returns:
        Bounding boxes in (cx, cy, w, h) format, shape (..., 4)
    """
    x1y1, x2y2 = bbox.split(2, dim=-1)
    return torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1], dim=-1)


class DETRPostProcessor(nn.Module):
    """
    Post-processor for DETR-based detection models

    This class handles the conversion of model outputs (normalized coordinates and logits)
    to final detection results (pixel coordinates, scores, and labels).

    Key features:
    - Multi-group query support (O2O, O2M)
    - Flexible coordinate decoding (original or padded)
    - Top-K selection
    - Support for both Softmax and Sigmoid classification

    Following PaddlePaddle's implementation for consistency.

    Reference:
    - PaddlePaddle: ppdet/modeling/post_process.py:450-586

    Args:
        num_classes: Number of object classes (default: 80 for COCO)
        num_top_queries: Number of top queries to keep (default: 100)
        dual_queries: Whether using dual query mechanism (O2O + O2M) (default: False)
        dual_groups: Number of dual query groups (default: 0)
                    0 = O2O + O2M (2 groups)
                    1 = O2O + Noise + O2M (3 groups)
        use_focal_loss: Whether to use sigmoid (True) or softmax (False) for classification
        bbox_decode_type: Coordinate decoding type ('origin' or 'pad')
                         'origin': decode to original image size
                         'pad': decode to padded image size
    """

    def __init__(
        self,
        num_classes: int = 80,
        num_top_queries: int = 100,
        dual_queries: bool = False,
        dual_groups: int = 0,
        use_focal_loss: bool = False,
        bbox_decode_type: str = 'origin'
    ):
        super().__init__()
        assert bbox_decode_type in ['origin', 'pad'], \
            f"bbox_decode_type must be 'origin' or 'pad', got {bbox_decode_type}"

        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.dual_queries = dual_queries
        self.dual_groups = dual_groups
        self.use_focal_loss = use_focal_loss
        self.bbox_decode_type = bbox_decode_type

    def forward(
        self,
        bboxes: torch.Tensor,
        logits: torch.Tensor,
        im_shape: torch.Tensor,
        scale_factor: torch.Tensor,
        pad_shape: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Post-process model outputs to final detection results

        Args:
            bboxes: Predicted bounding boxes in (cx, cy, w, h) format, normalized to [0, 1]
                   Shape: (batch_size, num_queries, 4)
            logits: Classification logits
                   Shape: (batch_size, num_queries, num_classes) if use_focal_loss
                          (batch_size, num_queries, num_classes + 1) if not use_focal_loss
            im_shape: Original image shape before padding (batch_size, 2) [height, width]
            scale_factor: Scale factor used for resizing (batch_size, 2) [scale_h, scale_w]
            pad_shape: Padded image shape (batch_size, 2) [height, width]
                      Required if bbox_decode_type == 'pad'

        Returns:
            bbox_pred: Detection results (N, 6) where N = batch_size * num_top_queries
                      Each row: [class_id, confidence_score, x1, y1, x2, y2]
            bbox_num: Number of detections per batch (batch_size,)

        Processing flow:
            1. Extract O2O queries if using dual_queries
            2. Convert bbox from (cx, cy, w, h) to (x1, y1, x2, y2)
            3. Calculate original image size (remove padding)
            4. Scale bbox to pixel coordinates
            5. Apply sigmoid/softmax to logits
            6. Select top-K detections
            7. Assemble final output
        """
        # Step 1: Extract O2O queries if using dual_queries
        # Following Paddle: ppdet/modeling/post_process.py:491-494
        if self.dual_queries:
            num_queries = logits.shape[1]
            # Keep only first 1/(dual_groups+1) queries (O2O group)
            # dual_groups=0 (O2O+O2M): keep 50%
            # dual_groups=1 (O2O+Noise+O2M): keep 33.3%
            keep_queries = int(num_queries // (self.dual_groups + 1))
            logits = logits[:, :keep_queries, :]
            bboxes = bboxes[:, :keep_queries, :]

        # Step 2: Convert bbox from (cx, cy, w, h) to (x1, y1, x2, y2)
        # Following Paddle: ppdet/modeling/post_process.py:496
        bbox_pred = bbox_cxcywh_to_xyxy(bboxes)

        # Step 3: Calculate original image size (remove padding and scaling)
        # Following Paddle: ppdet/modeling/post_process.py:497-499
        origin_shape = torch.floor(im_shape / scale_factor + 0.5)

        # Step 4: Calculate output shape for coordinate scaling
        # Following Paddle: ppdet/modeling/post_process.py:500-510
        if self.bbox_decode_type == 'pad':
            # Decode to padded image coordinates
            assert pad_shape is not None, "pad_shape is required when bbox_decode_type='pad'"
            out_shape = pad_shape / im_shape * origin_shape
            out_shape = out_shape.flip(-1).tile(1, 2).unsqueeze(1)
        elif self.bbox_decode_type == 'origin':
            # Decode to original image coordinates
            out_shape = origin_shape.flip(-1).tile(1, 2).unsqueeze(1)
            # out_shape: (B, 1, 4) = [[w, h, w, h]]
        else:
            raise ValueError(f"Invalid bbox_decode_type: {self.bbox_decode_type}")

        # Scale bbox from normalized [0, 1] to pixel coordinates
        # Following Paddle: ppdet/modeling/post_process.py:511
        bbox_pred = bbox_pred * out_shape

        # Step 5: Apply activation to classification logits
        # Following Paddle: ppdet/modeling/post_process.py:513-514
        if self.use_focal_loss:
            # Sigmoid for multi-label classification (Focal Loss)
            scores = torch.sigmoid(logits)
        else:
            # Softmax for single-label classification, remove background class
            scores = F.softmax(logits, dim=-1)[:, :, :-1]

        # Step 6: Select top-K detections
        # Following Paddle: ppdet/modeling/post_process.py:516-537
        if not self.use_focal_loss:
            # Softmax mode: each query selects one class
            # Following Paddle: ppdet/modeling/post_process.py:517-527
            scores, labels = scores.max(dim=-1)  # (B, Q)

            if scores.shape[1] > self.num_top_queries:
                # Select top-K queries by score
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)

                # Gather corresponding labels and bboxes
                batch_size = scores.shape[0]
                batch_ind = torch.arange(batch_size, device=scores.device).unsqueeze(-1).expand(
                    -1, self.num_top_queries
                )

                # Use advanced indexing to gather
                labels = labels[batch_ind, index]
                bbox_pred = bbox_pred[batch_ind, index]
        else:
            # Sigmoid mode: flatten and select top-K across all classes
            # Following Paddle: ppdet/modeling/post_process.py:529-537
            scores_flat = scores.flatten(1)  # (B, Q*C)
            scores, index = torch.topk(scores_flat, self.num_top_queries, dim=-1)

            # Recover class ID from flattened index
            labels = index % self.num_classes
            # Recover query index
            query_index = index // self.num_classes

            # Gather bboxes
            batch_size = scores.shape[0]
            batch_ind = torch.arange(batch_size, device=scores.device).unsqueeze(-1).expand(
                -1, self.num_top_queries
            )
            bbox_pred = bbox_pred[batch_ind, query_index]

        # Step 7: Assemble final output
        # Following Paddle: ppdet/modeling/post_process.py:575-580
        bbox_pred = torch.cat([
            labels.unsqueeze(-1).float(),  # class_id (float32)
            scores.unsqueeze(-1),          # confidence_score (0-1)
            bbox_pred                       # x1, y1, x2, y2 (pixel coordinates)
        ], dim=-1)

        # Reshape to (N, 6) where N = batch_size * num_top_queries
        # Following Paddle: ppdet/modeling/post_process.py:582-584
        bbox_num = torch.full(
            (bbox_pred.shape[0],),
            self.num_top_queries,
            dtype=torch.int32,
            device=bbox_pred.device
        )
        bbox_pred = bbox_pred.reshape(-1, 6)

        return bbox_pred, bbox_num


def build_detr_post_processor(
    num_classes: int = 80,
    num_top_queries: int = 100,
    dual_queries: bool = False,
    dual_groups: int = 0,
    use_focal_loss: bool = True,
    bbox_decode_type: str = 'origin'
) -> DETRPostProcessor:
    """
    Build DETRPostProcessor from config

    Args:
        num_classes: Number of object classes
        num_top_queries: Number of top detections to keep
        dual_queries: Enable dual query mechanism (O2O + O2M)
        dual_groups: Number of query groups
        use_focal_loss: Use sigmoid (True) or softmax (False)
        bbox_decode_type: Coordinate decoding type

    Returns:
        DETRPostProcessor instance

    Example:
        >>> # For RT-DETRv3 with O2M branch
        >>> post_processor = build_detr_post_processor(
        ...     num_classes=80,
        ...     num_top_queries=300,
        ...     dual_queries=True,
        ...     dual_groups=1,  # O2O + Noise + O2M
        ...     use_focal_loss=True
        ... )
    """
    return DETRPostProcessor(
        num_classes=num_classes,
        num_top_queries=num_top_queries,
        dual_queries=dual_queries,
        dual_groups=dual_groups,
        use_focal_loss=use_focal_loss,
        bbox_decode_type=bbox_decode_type
    )

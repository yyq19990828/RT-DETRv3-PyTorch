# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025 PyTorch Migration. All Rights Reserved.
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

"""
Layers Module - PyTorch Migration from PaddlePaddle

Reference: ppdet/modeling/layers.py
"""

from __future__ import absolute_import, division, print_function

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from detrs.core.workspace import register, serializable

__all__ = ["MultiClassNMS", "MultiHeadAttention"]


@register
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module - Aligned with PaddlePaddle implementation

    Uses fused in_proj_weight and in_proj_bias for Q, K, V projections to match
    PaddlePaddle's parameter naming convention for weight conversion.

    Args:
        embed_dim (int): Embedding dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout rate. Default: 0.0
        kdim (int, optional): Key dimension. If None, use embed_dim. Default: None
        vdim (int, optional): Value dimension. If None, use embed_dim. Default: None
        need_weights (bool): Whether to return attention weights. Default: False
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        need_weights: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_prob = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        # Use fused in_proj_weight and in_proj_bias to match Paddle's parameter naming
        if self._qkv_same_embed_dim:
            # Fused QKV projection: shape [embed_dim, 3 * embed_dim]
            # This matches Paddle's in_proj_weight parameter naming
            self.in_proj_weight = nn.Parameter(torch.empty(embed_dim, 3 * embed_dim))
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            # Separate Q, K, V projections (fallback, not commonly used in RT-DETRv3)
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
            nn.init.constant_(self.in_proj_bias, 0.0)
        else:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N, C)
            key: (B, M, C)
            value: (B, M, C)
            attn_mask: (B, N, M) or (N, M), additive mask (0 for valid, -inf for invalid)

        Returns:
            output: (B, N, C)
        """
        B, N, C = query.shape
        M = key.shape[1]

        if self._qkv_same_embed_dim:
            # Use fused in_proj for Q, K, V
            # in_proj_weight: [embed_dim, 3 * embed_dim]
            # Split into Q, K, V weights: each [embed_dim, embed_dim]
            q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=1)
            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)

            # Project Q, K, V
            q = F.linear(query, q_weight.t(), q_bias)
            k = F.linear(key, k_weight.t(), k_bias)
            v = F.linear(value, v_weight.t(), v_bias)
        else:
            # Use separate projections
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        # Reshape to (B, num_heads, N/M, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # (1, N, M)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, M) for broadcasting
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        output = output.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        output = self.out_proj(output)

        return output


@register
@serializable
class MultiClassNMS(object):
    """Multi-class Non-Maximum Suppression (NMS)

    Args:
        score_threshold (float): Threshold to filter out bounding boxes with
            low confidence score. Default: 0.05
        nms_top_k (int): Maximum number of detections to be kept according to
            the confidences after filtering detections based on score_threshold.
            Default: -1 (keep all)
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
            step. -1 means keeping all bboxes after NMS step. Default: 100
        nms_threshold (float): The threshold to be used in NMS. Default: 0.5
        normalized (bool): Whether detections are normalized. Default: True
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        return_index (bool): Whether return selected index. Default: False
        return_rois_num (bool): Whether return rois_num. Default: True
        trt (bool): Whether use TensorRT mode. Default: False
        cpu (bool): Whether force to use CPU. Default: False
    """

    def __init__(
        self,
        score_threshold=0.05,
        nms_top_k=-1,
        keep_top_k=100,
        nms_threshold=0.5,
        normalized=True,
        nms_eta=1.0,
        return_index=False,
        return_rois_num=True,
        trt=False,
        cpu=False,
    ):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_index = return_index
        self.return_rois_num = return_rois_num
        self.trt = trt
        self.cpu = cpu

    def __call__(self, bboxes, score, background_label=-1):
        """Perform multi-class NMS

        Args:
            bboxes (Tensor): Predicted bboxes with shape [N, M, 4], N is the
                batch size and M is the number of bboxes
            score (Tensor): Predicted scores with shape [N, C, M], C is the
                number of classes
            background_label (int): Ignore the background label; For example,
                RCNN is num_classes and YOLO is -1.

        Returns:
            bbox (Tensor): Detected bboxes with shape [K, 6], each row is
                [label, confidence, xmin, ymin, xmax, ymax]
            bbox_num (Tensor): Number of detected bboxes for each image in batch,
                shape [N]
            index (Tensor|None): Selected index if return_index is True
        """
        # Move to CPU if required
        original_device = bboxes.device
        if self.cpu:
            bboxes = bboxes.cpu()
            score = score.cpu()

        batch_size = bboxes.shape[0]
        num_classes = score.shape[1]

        # Filter by score threshold and apply NMS
        all_outputs = []
        all_bbox_num = []

        for batch_idx in range(batch_size):
            batch_bboxes = bboxes[batch_idx]  # [M, 4]
            batch_scores = score[batch_idx]  # [C, M]

            batch_outputs = []

            for class_idx in range(num_classes):
                if class_idx == background_label:
                    continue

                class_scores = batch_scores[class_idx]  # [M]

                # Filter by score threshold
                keep_mask = class_scores > self.score_threshold
                if keep_mask.sum() == 0:
                    continue

                filtered_scores = class_scores[keep_mask]
                filtered_boxes = batch_bboxes[keep_mask]

                # Apply top-k filtering before NMS
                if self.nms_top_k > 0 and filtered_scores.shape[0] > self.nms_top_k:
                    _, topk_indices = torch.topk(filtered_scores, self.nms_top_k)
                    filtered_scores = filtered_scores[topk_indices]
                    filtered_boxes = filtered_boxes[topk_indices]

                # Apply NMS
                if filtered_boxes.shape[0] > 0:
                    keep_indices = ops.nms(
                        filtered_boxes, filtered_scores, self.nms_threshold
                    )
                    nms_boxes = filtered_boxes[keep_indices]
                    nms_scores = filtered_scores[keep_indices]

                    # Create output with [label, confidence, xmin, ymin, xmax, ymax]
                    labels = torch.full(
                        (nms_boxes.shape[0], 1),
                        class_idx,
                        dtype=torch.float32,
                        device=nms_boxes.device,
                    )
                    scores_col = nms_scores.unsqueeze(1)
                    class_outputs = torch.cat([labels, scores_col, nms_boxes], dim=1)
                    batch_outputs.append(class_outputs)

            if len(batch_outputs) > 0:
                # Concatenate all classes for this batch
                batch_output = torch.cat(batch_outputs, dim=0)

                # Apply keep_top_k
                if self.keep_top_k > 0 and batch_output.shape[0] > self.keep_top_k:
                    # Sort by confidence (column 1)
                    _, topk_indices = torch.topk(batch_output[:, 1], self.keep_top_k)
                    batch_output = batch_output[topk_indices]

                all_outputs.append(batch_output)
                all_bbox_num.append(
                    torch.tensor(batch_output.shape[0], dtype=torch.int32)
                )
            else:
                # No detections for this batch
                all_bbox_num.append(torch.tensor(0, dtype=torch.int32))

        # Concatenate all batches
        if len(all_outputs) > 0:
            bbox = torch.cat(all_outputs, dim=0)
        else:
            bbox = torch.zeros((0, 6), dtype=torch.float32, device=original_device)

        bbox_num = (
            torch.stack(all_bbox_num)
            if all_bbox_num
            else torch.zeros(batch_size, dtype=torch.int32, device=original_device)
        )

        # Move back to original device
        if self.cpu:
            bbox = bbox.to(original_device)
            bbox_num = bbox_num.to(original_device)

        # Return based on flags
        if self.return_index:
            # Index not implemented in this simplified version
            index = None
            if self.return_rois_num:
                return bbox, bbox_num, index
            else:
                return bbox, index
        else:
            if self.return_rois_num:
                return bbox, bbox_num, None
            else:
                return bbox, None, None

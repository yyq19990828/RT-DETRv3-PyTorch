"""
DETR-style Detection Heads for RT-DETRv3

This module implements detection heads following PaddlePaddle's approach.

Components:
- DINOv3Head: Main detection head for RT-DETRv3 (wrapper for decoder outputs)

TODO (T028): Implement PPYOLOEHead as auxiliary branch
  - CNN-based detection head operating on neck features (body_feats)
  - Anchor-free design with Distribution Focal Loss (DFL)
  - Used for auxiliary supervision during training
  - Reference: PaddlePaddle ppdet/modeling/heads/ppyoloe_head.py

Reference:
- PaddlePaddle RT-DETR: ppdet/modeling/heads/detr_head.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from ppdet_pytorch.core.workspace import register

__all__ = ['DINOv3Head']


@register
class DINOv3Head(nn.Module):
    """
    DINOv3 Detection Head

    Following PaddlePaddle's implementation for consistency.
    Reference: ppdet/modeling/heads/detr_head.py:542-645

    Args:
        loss: Loss function (default: 'DINOLoss')
        eval_idx: Which decoder layer to use for evaluation (default: -1 = last layer)
        o2m: One-to-many multiplier for auxiliary loss (default: 4)
        o2m_branch: Whether to use one-to-many branch (default: False)
        num_queries_o2m: Number of one-to-many queries (default: 450)
    """

    __inject__ = ['loss']
    __shared__ = ['o2m_branch', 'num_queries_o2m']

    def __init__(
        self,
        loss='DINOLoss',
        eval_idx: int = -1,
        o2m: int = 4,
        o2m_branch: bool = False,
        num_queries_o2m: int = 450
    ):
        super().__init__()
        self.loss = loss
        self.eval_idx = eval_idx
        self.o2m = o2m
        self.o2m_branch = o2m_branch
        self.num_queries_o2m = num_queries_o2m

    def forward(
        self,
        out_transformer: Tuple[torch.Tensor, ...],
        body_feats: Optional[torch.Tensor] = None,
        inputs: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of DINOv3Head

        Following PaddlePaddle: ppdet/modeling/heads/detr_head.py:555-645

        Args:
            out_transformer: Tuple of (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)
                - dec_out_bboxes: Decoder bbox predictions (num_layers, B, N, 4)
                - dec_out_logits: Decoder class logits (num_layers, B, N, num_classes)
                - enc_topk_bboxes: Encoder top-k bboxes (B, N, 4)
                - enc_topk_logits: Encoder top-k logits (B, N, num_classes)
                - dn_meta: Denoising metadata (dict, list of dicts, or None)
            body_feats: Backbone features (optional, for auxiliary heads)
            inputs: Training inputs with 'gt_bbox', 'gt_class' (optional)

        Returns:
            In training mode: loss dict
            In eval mode: (pred_bboxes, pred_logits, None)
                - pred_bboxes: (B, N, 4) in [0, 1] range
                - pred_logits: (B, N, num_classes) raw logits
        """
        dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta = out_transformer

        if self.training:
            # Training mode: compute losses
            # Following Paddle: ppdet/modeling/heads/detr_head.py:558-642
            assert inputs is not None, "inputs must be provided in training mode"
            assert 'gt_bbox' in inputs and 'gt_class' in inputs, "inputs must contain 'gt_bbox' and 'gt_class'"
            assert self.loss is not None, "loss must be set for training mode"

            # Case 1: Multi-group denoising (dn_meta is a list of dicts)
            # Following Paddle: ppdet/modeling/heads/detr_head.py:562-625
            if dn_meta is not None and isinstance(dn_meta, list):
                num_groups = len(dn_meta)
                total_dec_queries = dec_out_bboxes.shape[2]
                total_enc_queries = enc_topk_bboxes.shape[1]
                loss = {}

                # Handle o2m_branch if enabled
                # Following Paddle: ppdet/modeling/heads/detr_head.py:567-588
                if self.o2m_branch:
                    # Split o2m queries from main queries
                    dec_out_bboxes, dec_out_bboxes_o2m = torch.split(
                        dec_out_bboxes,
                        [total_dec_queries - self.num_queries_o2m, self.num_queries_o2m],
                        dim=2
                    )
                    dec_out_logits, dec_out_logits_o2m = torch.split(
                        dec_out_logits,
                        [total_dec_queries - self.num_queries_o2m, self.num_queries_o2m],
                        dim=2
                    )
                    enc_topk_bboxes, enc_topk_bboxes_o2m = torch.split(
                        enc_topk_bboxes,
                        [total_enc_queries - self.num_queries_o2m, self.num_queries_o2m],
                        dim=1
                    )
                    enc_topk_logits, enc_topk_logits_o2m = torch.split(
                        enc_topk_logits,
                        [total_enc_queries - self.num_queries_o2m, self.num_queries_o2m],
                        dim=1
                    )

                    # Compute o2m branch loss
                    # Concatenate encoder + decoder outputs for o2m
                    # Match Paddle: paddle.concat([enc_topk_bboxes_o2m.unsqueeze(0), dec_out_bboxes_o2m])
                    out_bboxes_o2m = torch.cat([enc_topk_bboxes_o2m.unsqueeze(0), dec_out_bboxes_o2m])
                    out_logits_o2m = torch.cat([enc_topk_logits_o2m.unsqueeze(0), dec_out_logits_o2m])

                    loss_o2m = self.loss(
                        out_bboxes_o2m,
                        out_logits_o2m,
                        inputs['gt_bbox'],
                        inputs['gt_class'],
                        dn_out_bboxes=None,
                        dn_out_logits=None,
                        dn_meta=None,
                        o2m=self.o2m
                    )
                    # Add o2m_branch suffix to loss keys
                    # Match Paddle: loss.update({key: loss.get(key, paddle.zeros([1])) + value})
                    for key, value in loss_o2m.items():
                        loss_key = key + '_o2m_branch'
                        loss[loss_key] = loss.get(
                            loss_key, value.new_zeros(())) + value

                # Split queries by groups
                # Following Paddle: ppdet/modeling/heads/detr_head.py:590-595
                split_dec_num = [sum(dn['dn_num_split']) for dn in dn_meta]
                split_enc_num = [dn['dn_num_split'][1] for dn in dn_meta]

                dec_out_bboxes = torch.split(dec_out_bboxes, split_dec_num, dim=2)
                dec_out_logits = torch.split(dec_out_logits, split_dec_num, dim=2)
                enc_topk_bboxes = torch.split(enc_topk_bboxes, split_enc_num, dim=1)
                enc_topk_logits = torch.split(enc_topk_logits, split_enc_num, dim=1)

                # Compute loss for each group
                # Following Paddle: ppdet/modeling/heads/detr_head.py:597-619
                for g_id in range(num_groups):
                    # Split denoising and matching queries for this group
                    dn_out_bboxes_gid, dec_out_bboxes_gid = torch.split(
                        dec_out_bboxes[g_id],
                        dn_meta[g_id]['dn_num_split'],
                        dim=2
                    )
                    dn_out_logits_gid, dec_out_logits_gid = torch.split(
                        dec_out_logits[g_id],
                        dn_meta[g_id]['dn_num_split'],
                        dim=2
                    )

                    # Concatenate encoder + decoder outputs
                    # Match Paddle: paddle.concat([enc_topk_bboxes[g_id].unsqueeze(0), dec_out_bboxes_gid])
                    out_bboxes_gid = torch.cat([enc_topk_bboxes[g_id].unsqueeze(0), dec_out_bboxes_gid])
                    out_logits_gid = torch.cat([enc_topk_logits[g_id].unsqueeze(0), dec_out_logits_gid])

                    # Compute loss for this group
                    # Match Paddle: passes dn_out_bboxes_gid and dn_out_logits_gid directly (as tensors)
                    loss_gid = self.loss(
                        out_bboxes_gid,
                        out_logits_gid,
                        inputs['gt_bbox'],
                        inputs['gt_class'],
                        dn_out_bboxes=dn_out_bboxes_gid,
                        dn_out_logits=dn_out_logits_gid,
                        dn_meta=dn_meta[g_id]
                    )

                    # Accumulate losses across groups
                    # Match Paddle: loss.update({key: loss.get(key, paddle.zeros([1])) + value})
                    for key, value in loss_gid.items():
                        loss[key] = loss.get(key, value.new_zeros(())) + value

                # Average losses across groups (except o2m_branch losses)
                # Following Paddle: ppdet/modeling/heads/detr_head.py:622-624
                for key, value in loss.items():
                    if '_o2m_branch' not in key:
                        loss[key] = value / num_groups

                return loss

            # Case 2: No denoising or single-group denoising
            # Following Paddle: ppdet/modeling/heads/detr_head.py:626-642
            else:
                # Set dn outputs to None (loss function will handle dn_meta internally)
                # Match Paddle: ppdet/modeling/heads/detr_head.py:627
                dn_out_bboxes, dn_out_logits = None, None

                # Concatenate encoder + decoder outputs
                # Match Paddle: paddle.concat([enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
                out_bboxes = torch.cat([enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
                out_logits = torch.cat([enc_topk_logits.unsqueeze(0), dec_out_logits])

                # Compute loss (loss function handles dn_meta internally)
                # Following Paddle: ppdet/modeling/heads/detr_head.py:634-642
                return self.loss(
                    out_bboxes,
                    out_logits,
                    inputs['gt_bbox'],
                    inputs['gt_class'],
                    dn_out_bboxes=dn_out_bboxes,
                    dn_out_logits=dn_out_logits,
                    dn_meta=dn_meta,
                    gt_score=inputs.get('gt_score', None)
                )
        else:
            # Evaluation mode: return predictions from specified decoder layer
            # Following Paddle: ppdet/modeling/heads/detr_head.py:643-645
            # dec_out_bboxes shape: (num_layers, B, N, 4)
            # dec_out_logits shape: (num_layers, B, N, num_classes)
            return (
                dec_out_bboxes[self.eval_idx],  # (B, N, 4)
                dec_out_logits[self.eval_idx],  # (B, N, num_classes)
                None  # No auxiliary outputs in eval mode
            )

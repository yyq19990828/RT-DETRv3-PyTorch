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
from ppdet.core.workspace import register


@register
class DINOv3Head(nn.Module):
    """
    DINOv3 Detection Head

    This head processes the outputs from the RTDETRTransformerv3 decoder.
    In training mode, it handles loss computation.
    In eval mode, it simply returns the decoder predictions.

    Following PaddlePaddle's implementation for consistency.

    Reference:
    - PaddlePaddle: ppdet/modeling/heads/detr_head.py:542-646

    Args:
        loss_fn: Loss function module (DINOv3Loss instance)
        eval_idx: Which decoder layer to use for evaluation (-1 means last layer)
        o2m: One-to-many multiplier for auxiliary loss (default: 4)
        o2m_branch: Whether to use one-to-many branch (default: False)
        num_queries_o2m: Number of one-to-many queries (default: 450)
        num_classes: Number of object classes (default: 80)
        hidden_dim: Hidden dimension from transformer (default: 256)
    """

    __category__ = 'head'
    __inject__ = []  # No component dependencies
    __shared__ = ['num_classes', 'hidden_dim']  # Shared from global config

    def __init__(
        self,
        loss_fn: Optional[nn.Module] = None,
        eval_idx: int = -1,
        o2m: int = 4,
        o2m_branch: bool = False,
        num_queries_o2m: int = 450,
        num_classes: int = 80,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.eval_idx = eval_idx
        self.o2m = o2m
        self.o2m_branch = o2m_branch
        self.num_queries_o2m = num_queries_o2m
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

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
            assert self.loss_fn is not None, "loss_fn must be set for training mode"

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
                    out_bboxes_o2m = [enc_topk_bboxes_o2m] + [dec_out_bboxes_o2m[i] for i in range(dec_out_bboxes_o2m.shape[0])]
                    out_logits_o2m = [enc_topk_logits_o2m] + [dec_out_logits_o2m[i] for i in range(dec_out_logits_o2m.shape[0])]

                    loss_o2m = self.loss_fn(
                        out_bboxes_o2m,
                        out_logits_o2m,
                        inputs['gt_bbox'],
                        inputs['gt_class'],
                        dn_meta=None,
                        o2m=self.o2m
                    )
                    # Add o2m_branch suffix to loss keys
                    for key, value in loss_o2m.items():
                        loss_key = key + '_o2m_branch'
                        loss[loss_key] = loss.get(loss_key, torch.zeros_like(value)) + value

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
                    out_bboxes_gid = [enc_topk_bboxes[g_id]] + [dec_out_bboxes_gid[i] for i in range(dec_out_bboxes_gid.shape[0])]
                    out_logits_gid = [enc_topk_logits[g_id]] + [dec_out_logits_gid[i] for i in range(dec_out_logits_gid.shape[0])]

                    # Convert denoising outputs to list format
                    dn_out_bboxes_list = [dn_out_bboxes_gid[i] for i in range(dn_out_bboxes_gid.shape[0])]
                    dn_out_logits_list = [dn_out_logits_gid[i] for i in range(dn_out_logits_gid.shape[0])]

                    # Prepare dn_meta dict with denoising outputs
                    dn_meta_gid = {
                        'dn_positive_idx': dn_meta[g_id]['dn_positive_idx'],
                        'dn_num_group': dn_meta[g_id]['dn_num_group'],
                        'dn_out_bboxes': dn_out_bboxes_list,
                        'dn_out_logits': dn_out_logits_list
                    }

                    # Compute loss for this group
                    loss_gid = self.loss_fn(
                        out_bboxes_gid,
                        out_logits_gid,
                        inputs['gt_bbox'],
                        inputs['gt_class'],
                        dn_meta=dn_meta_gid
                    )

                    # Accumulate losses across groups
                    for key, value in loss_gid.items():
                        loss[key] = loss.get(key, torch.zeros_like(value)) + value

                # Average losses across groups (except o2m_branch losses)
                # Following Paddle: ppdet/modeling/heads/detr_head.py:622-624
                for key, value in loss.items():
                    if '_o2m_branch' not in key:
                        loss[key] = value / num_groups

                return loss

            # Case 2: No denoising or single-group denoising
            # Following Paddle: ppdet/modeling/heads/detr_head.py:626-642
            else:
                # Concatenate encoder + decoder outputs
                # Following Paddle: ppdet/modeling/heads/detr_head.py:629-632
                out_bboxes = [enc_topk_bboxes] + [dec_out_bboxes[i] for i in range(dec_out_bboxes.shape[0])]
                out_logits = [enc_topk_logits] + [dec_out_logits[i] for i in range(dec_out_logits.shape[0])]

                # Prepare dn_meta if present
                if dn_meta is not None:
                    # Convert denoising outputs to list format
                    # Note: In single-group case, dn_meta is a dict, not a list
                    # We need to extract dn outputs from dec_out using dn_num_split
                    dn_num_split = dn_meta['dn_num_split']

                    # Split each decoder layer's output into dn and matching parts
                    dn_out_bboxes_list = []
                    dn_out_logits_list = []
                    out_bboxes_clean = [enc_topk_bboxes]  # Start with encoder outputs
                    out_logits_clean = [enc_topk_logits]

                    for layer_idx in range(dec_out_bboxes.shape[0]):
                        dn_bbox, match_bbox = torch.split(
                            dec_out_bboxes[layer_idx],
                            dn_num_split,
                            dim=1
                        )
                        dn_logit, match_logit = torch.split(
                            dec_out_logits[layer_idx],
                            dn_num_split,
                            dim=1
                        )
                        dn_out_bboxes_list.append(dn_bbox)
                        dn_out_logits_list.append(dn_logit)
                        out_bboxes_clean.append(match_bbox)
                        out_logits_clean.append(match_logit)

                    # Update dn_meta with outputs
                    dn_meta_with_outputs = {
                        'dn_positive_idx': dn_meta['dn_positive_idx'],
                        'dn_num_group': dn_meta['dn_num_group'],
                        'dn_out_bboxes': dn_out_bboxes_list,
                        'dn_out_logits': dn_out_logits_list
                    }

                    # Use cleaned outputs (without dn queries)
                    out_bboxes = out_bboxes_clean
                    out_logits = out_logits_clean
                    dn_meta_arg = dn_meta_with_outputs
                else:
                    dn_meta_arg = None

                # Compute loss
                # Following Paddle: ppdet/modeling/heads/detr_head.py:634-642
                return self.loss_fn(
                    out_bboxes,
                    out_logits,
                    inputs['gt_bbox'],
                    inputs['gt_class'],
                    dn_meta=dn_meta_arg,
                    o2m=1  # No o2m multiplication in non-branch mode
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

    @classmethod
    def from_config(cls, cfg: Dict, global_config: Optional[Dict] = None) -> Dict:
        """
        Build DINOv3Head from config (PaddlePaddle-style).

        Args:
            cfg: Head configuration dict
            global_config: Global configuration for shared values

        Returns:
            Dict of kwargs for DINOv3Head.__init__

        Example config:
            {
                'eval_idx': -1,
                'o2m': 4,
                'o2m_branch': False,
                'num_queries_o2m': 450
            }
        """
        return {
            'loss_fn': cfg.get('loss_fn', None),
            'eval_idx': cfg.get('eval_idx', -1),
            'o2m': cfg.get('o2m', 4),
            'o2m_branch': cfg.get('o2m_branch', False),
            'num_queries_o2m': cfg.get('num_queries_o2m', 450),
            'num_classes': cfg.get('num_classes', 80),
            'hidden_dim': cfg.get('hidden_dim', 256)
        }

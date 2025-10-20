# Copyright (c) 2025 PyTorch Implementation
# Original PaddlePaddle implementation: Copyright (c) 2022 PaddlePaddle Authors
# Licensed under the Apache License, Version 2.0

"""PPYOLOEHead for RT-DETRv3 auxiliary branch.

This module implements the PPYOLOEHead detection head, which serves as the
auxiliary branch in RT-DETRv3 during training. It operates on multi-scale features
from the neck and provides CNN-based detections to assist the transformer branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from ppdet.core.workspace import register


class ESEAttn(nn.Module):
    """Effective Squeeze-and-Excitation Attention.

    Args:
        feat_channels: Number of input feature channels
        act: Activation function name (default: 'swish')
    """

    def __init__(self, feat_channels: int, act: str = 'swish'):
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.SiLU() if act == 'swish' else nn.ReLU()
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feat: torch.Tensor, avg_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            feat: Input features of shape (B, C, H, W)
            avg_feat: Averaged features of shape (B, C, 1, 1)

        Returns:
            Attention-weighted features of shape (B, C, H, W)
        """
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


@register
class PPYOLOEHead(nn.Module):
    """PPYOLOEHead detection head for auxiliary branch.

    This head operates on multi-scale features from the neck and produces
    CNN-based detections. During training, it provides auxiliary supervision
    through TaskAligned assignment and Varifocal/GIoU/DFL losses.

    Args:
        in_channels: List of input channel dimensions for each FPN level
        num_classes: Number of object classes (default: 80 for COCO)
        fpn_strides: Feature pyramid strides (default: [8, 16, 32])
        reg_max: Maximum regression value for DFL (default: 16)
        act: Activation function name (default: 'swish')
    """

    __category__ = 'head'
    __inject__ = []  # No component dependencies
    __shared__ = ['num_classes']  # Shared from global config

    def __init__(
        self,
        in_channels: List[int] = [256, 256, 256],
        num_classes: int = 80,
        fpn_strides: Tuple[int, ...] = (8, 16, 32),
        reg_max: int = 16,
        act: str = 'swish'
    ):
        super().__init__()
        assert len(in_channels) > 0, "in_channels must have at least one element"

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.reg_max = reg_max
        self.reg_channels = reg_max + 1  # 0 to reg_max inclusive

        # Attention stems for classification and regression
        self.stem_cls = nn.ModuleList([
            ESEAttn(in_c, act=act) for in_c in in_channels
        ])
        self.stem_reg = nn.ModuleList([
            ESEAttn(in_c, act=act) for in_c in in_channels
        ])

        # Prediction heads
        self.pred_cls = nn.ModuleList([
            nn.Conv2d(in_c, num_classes, 3, padding=1) for in_c in in_channels
        ])
        self.pred_reg = nn.ModuleList([
            nn.Conv2d(in_c, 4 * self.reg_channels, 3, padding=1) for in_c in in_channels
        ])

        # Distribution Focal Loss projection layer
        self.proj_conv = nn.Conv2d(self.reg_channels, 1, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize prediction head weights."""
        # Bias for classification (prior probability = 0.01)
        bias_cls = -torch.log(torch.tensor((1 - 0.01) / 0.01))

        for cls_layer, reg_layer in zip(self.pred_cls, self.pred_reg):
            # Initialize classification head
            nn.init.zeros_(cls_layer.weight)
            nn.init.constant_(cls_layer.bias, bias_cls)

            # Initialize regression head
            nn.init.zeros_(reg_layer.weight)
            nn.init.constant_(reg_layer.bias, 1.0)

        # Initialize projection layer with linear spacing
        proj_weight = torch.linspace(0, self.reg_max, self.reg_channels).view(1, -1, 1, 1)
        self.proj_conv.weight.data = proj_weight
        self.proj_conv.weight.requires_grad = False  # Fixed projection

    def forward(
        self,
        feats: List[torch.Tensor],
        targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            feats: List of multi-scale features from neck, each of shape (B, C, H, W)
            targets: Ground truth targets (only used in training, for compatibility)

        Returns:
            Tuple of:
                - cls_score_list: Concatenated classification scores (B, total_anchors, num_classes)
                - reg_dist_list: Concatenated regression distributions (B, total_anchors, 4)
        """
        assert len(feats) == len(self.fpn_strides), \
            f"Number of features ({len(feats)}) must match fpn_strides ({len(self.fpn_strides)})"

        if self.training:
            return self._forward_train(feats)
        else:
            return self._forward_eval(feats)

    def _forward_train(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Returns flattened outputs for loss computation.
        """
        cls_score_list = []
        reg_distri_list = []

        for i, feat in enumerate(feats):
            # Apply attention with global average pooling
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))

            # Classification branch with residual connection
            cls_feat = self.stem_cls[i](feat, avg_feat) + feat
            cls_logit = self.pred_cls[i](cls_feat)
            cls_score = torch.sigmoid(cls_logit)

            # Regression branch (no residual)
            reg_feat = self.stem_reg[i](feat, avg_feat)
            reg_distri = self.pred_reg[i](reg_feat)

            # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
            cls_score_list.append(cls_score.flatten(2).permute(0, 2, 1))
            reg_distri_list.append(reg_distri.flatten(2).permute(0, 2, 1))

        # Concatenate all scales
        cls_scores = torch.cat(cls_score_list, dim=1)  # (B, total_anchors, num_classes)
        reg_distris = torch.cat(reg_distri_list, dim=1)  # (B, total_anchors, 4*reg_channels)

        return cls_scores, reg_distris

    def _forward_eval(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluation forward pass.

        Applies DFL projection to convert distributions to distances.
        """
        cls_score_list = []
        reg_dist_list = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            num_anchors = h * w

            # Apply attention with global average pooling
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))

            # Classification branch with residual
            cls_feat = self.stem_cls[i](feat, avg_feat) + feat
            cls_logit = self.pred_cls[i](cls_feat)
            cls_score = torch.sigmoid(cls_logit)

            # Regression branch
            reg_feat = self.stem_reg[i](feat, avg_feat)
            reg_dist = self.pred_reg[i](reg_feat)

            # Reshape regression: (B, 4*reg_channels, H, W) -> (B, reg_channels, H*W, 4)
            reg_dist = reg_dist.reshape(b, 4, self.reg_channels, num_anchors).permute(0, 2, 3, 1)

            # Apply softmax and project to get distances
            reg_dist = F.softmax(reg_dist, dim=1)
            reg_dist = self.proj_conv(reg_dist).squeeze(1)  # (B, H*W, 4)

            # Collect outputs
            cls_score_list.append(cls_score.reshape(b, self.num_classes, num_anchors))
            reg_dist_list.append(reg_dist)

        # Concatenate all scales
        cls_scores = torch.cat(cls_score_list, dim=2)  # (B, num_classes, total_anchors)
        reg_dists = torch.cat(reg_dist_list, dim=1)    # (B, total_anchors, 4)

        return cls_scores, reg_dists

    @classmethod
    def from_config(cls, cfg: Dict, global_config: Optional[Dict] = None) -> Dict:
        """Build PPYOLOEHead from config (PaddlePaddle-style).

        Args:
            cfg: Head configuration dict
            global_config: Global configuration for shared values

        Returns:
            Dict of kwargs for PPYOLOEHead.__init__

        Example config:
            {
                'in_channels': [256, 256, 256],
                'num_classes': 80,
                'fpn_strides': [8, 16, 32],
                'reg_max': 16,
                'act': 'swish'
            }
        """
        return {
            'in_channels': cfg.get('in_channels', [256, 256, 256]),
            'num_classes': cfg.get('num_classes', 80),
            'fpn_strides': tuple(cfg.get('fpn_strides', [8, 16, 32])),
            'reg_max': cfg.get('reg_max', 16),
            'act': cfg.get('act', 'swish')
        }

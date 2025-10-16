"""
RT-DETRv3 Main Model Class

This module implements the complete RT-DETRv3 architecture integrating:
- Backbone (ResNet variants)
- Neck (HybridEncoder with FPN-PAN)
- Transformer (RTDETRTransformerv3 with multi-group queries)
- Detection Heads (DINOv3Head for main branch, PPYOLOEHead for auxiliary)

Following PaddlePaddle's implementation for consistency.

Reference:
- PaddlePaddle RT-DETRv3: ppdet/modeling/architectures/rtdetrv3.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from .backbones.resnet import build_resnet
from .necks.hybrid_encoder import build_hybrid_encoder
from .transformers.rtdetr_transformer import build_rtdetr_transformer
from .heads.detr_head import build_dinov3_head
from .heads.ppyoloe_head import build_ppyoloe_head


class RTDETRv3(nn.Module):
    """
    RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision

    Architecture:
        Input Image → Backbone → Neck → Transformer → Detection Heads → Outputs

    Training mode:
        - Outputs: loss dict with all supervision branches
        - Auxiliary branch (PPYOLOEHead) active
        - One-to-many supervision active

    Eval mode:
        - Outputs: (pred_bboxes, pred_logits)
        - Only main detection branch active
        - No auxiliary branches

    Following PaddlePaddle's implementation structure.

    Args:
        backbone: Backbone network config or instance
        neck: Neck network config or instance
        transformer: Transformer config or instance
        detr_head: Main detection head config or instance
        aux_head: Auxiliary detection head config or instance (optional)
        num_classes: Number of object classes (default: 80 for COCO)
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        transformer: Optional[nn.Module] = None,
        detr_head: Optional[nn.Module] = None,
        aux_head: Optional[nn.Module] = None,
        num_classes: int = 80,
        # Backbone config
        backbone_type: str = 'resnet',
        backbone_depth: int = 50,
        backbone_variant: str = 'd',
        backbone_frozen_stages: int = 1,
        backbone_return_idx: List[int] = [1, 2, 3],
        # Neck config
        neck_in_channels: List[int] = [512, 1024, 2048],
        neck_feat_strides: List[int] = [8, 16, 32],
        neck_hidden_dim: int = 256,
        neck_use_encoder_idx: List[int] = [2],
        neck_num_encoder_layers: int = 1,
        neck_expansion: float = 1.0,
        # Transformer config
        transformer_num_queries: int = 300,
        transformer_num_decoder_layers: int = 6,
        transformer_num_levels: int = 3,
        transformer_num_points: int = 4,
        transformer_hidden_dim: int = 256,
        # Head config
        head_eval_idx: int = -1,
        head_o2m: int = 4,
        head_o2m_branch: bool = False,
        head_num_queries_o2m: int = 450,
    ):
        super().__init__()

        # Build components if not provided
        if backbone is None:
            backbone = build_resnet({
                'depth': backbone_depth,
                'variant': backbone_variant,
                'frozen_stages': backbone_frozen_stages,
                'return_idx': backbone_return_idx
            })

        if neck is None:
            neck = build_hybrid_encoder({
                'in_channels': neck_in_channels,
                'feat_strides': neck_feat_strides,
                'hidden_dim': neck_hidden_dim,
                'use_encoder_idx': neck_use_encoder_idx,
                'num_encoder_layers': neck_num_encoder_layers,
                'expansion': neck_expansion
            })

        if transformer is None:
            transformer = build_rtdetr_transformer(
                num_queries=transformer_num_queries,
                num_decoder_layers=transformer_num_decoder_layers,
                num_levels=transformer_num_levels,
                num_decoder_points=transformer_num_points,
                hidden_dim=transformer_hidden_dim,
                eval_idx=head_eval_idx,
                o2m_branch=head_o2m_branch,
                num_queries_o2m=head_num_queries_o2m
            )

        if detr_head is None:
            detr_head = build_dinov3_head(
                eval_idx=head_eval_idx,
                o2m=head_o2m,
                o2m_branch=head_o2m_branch,
                num_queries_o2m=head_num_queries_o2m
            )

        self.backbone = backbone
        self.neck = neck
        self.transformer = transformer
        self.detr_head = detr_head
        self.aux_head = aux_head  # PPYOLOEHead (optional, for training)
        self.num_classes = num_classes

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of RTDETRv3

        Args:
            images: Input images (B, 3, H, W)
            targets: Training targets (optional, only used in training mode)
                List of dicts, each containing:
                - 'boxes': (N, 4) in [x, y, w, h] format, normalized to [0, 1]
                - 'labels': (N,) class labels

        Returns:
            In training mode:
                loss_dict: Dict with keys:
                    - 'loss_cls': Classification loss
                    - 'loss_bbox': Bbox L1 loss
                    - 'loss_giou': Bbox GIoU loss
                    - 'loss': Total loss
                    - (optional) 'loss_aux_*': Auxiliary branch losses

            In eval mode:
                output_dict: Dict with keys:
                    - 'pred_logits': (B, num_queries, num_classes) class logits
                    - 'pred_boxes': (B, num_queries, 4) bbox predictions in [0, 1]
        """
        # Extract multi-scale features from backbone
        # feats: List of [(B, C3, H/8, W/8), (B, C4, H/16, W/16), (B, C5, H/32, W/32)]
        feats = self.backbone(images)

        # Process features through neck (FPN-PAN fusion)
        # body_feats: List of [(B, 256, H/8, W/8), (B, 256, H/16, W/16), (B, 256, H/32, W/32)]
        body_feats = self.neck(feats)

        # Create padding mask for valid pixels (all True for now, can be extended)
        batch_size = images.shape[0]
        device = images.device
        # For now, no padding mask (all pixels are valid)
        # In production, this would handle variable image sizes
        pad_mask = None

        # Process through full RTDETRTransformerv3
        # Returns: (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)
        out_transformer = self.transformer(body_feats, targets)

        if self.training:
            # Training mode: compute losses
            # IMPORTANT: Loss computation will be implemented in T040 (DINOv3Loss)
            # For now, we delegate to the head which will raise NotImplementedError
            losses = self.detr_head(out_transformer, body_feats, targets)

            # Add auxiliary head losses if available
            if self.aux_head is not None:
                # PPYOLOEHead forward pass on neck features
                aux_cls_scores, aux_reg_distris = self.aux_head(body_feats, targets)
                # TODO (T040): Implement auxiliary loss computation
                # This will be done when DINOv3Loss is implemented
                # aux_losses = compute_ppyoloe_loss(aux_cls_scores, aux_reg_distris, targets)
                # for k, v in aux_losses.items():
                #     if k == 'loss':
                #         losses[k] += v
                #     losses[k + '_aux'] = v
                pass

            return losses
        else:
            # Evaluation mode: return predictions
            pred_bboxes, pred_logits, _ = self.detr_head(out_transformer, body_feats, targets)

            return {
                'pred_logits': pred_logits,  # (B, num_queries, num_classes)
                'pred_boxes': pred_bboxes,   # (B, num_queries, 4)
            }


def build_rtdetrv3(
    num_classes: int = 80,
    backbone: str = 'resnet50',
    variant: str = 'd',
    frozen_stages: int = 1,
    hidden_dim: int = 256,
    num_queries: int = 300,
    num_decoder_layers: int = 6,
    num_levels: int = 3,
    num_points: int = 4,
    eval_idx: int = -1,
    o2m: int = 4,
    o2m_branch: bool = False,
    num_queries_o2m: int = 450,
    use_aux_head: bool = False
) -> RTDETRv3:
    """
    Build RTDETRv3 model from config

    Args:
        num_classes: Number of object classes
        backbone: Backbone type ('resnet18', 'resnet50', 'resnet101')
        variant: ResNet variant ('d' for ResNet-vd)
        frozen_stages: Number of frozen backbone stages
        hidden_dim: Hidden dimension for transformer
        num_queries: Number of one-to-one queries
        num_decoder_layers: Number of decoder layers
        num_levels: Number of feature pyramid levels
        num_points: Number of sampling points in deformable attention
        eval_idx: Decoder layer index for evaluation (-1 for last layer)
        o2m: One-to-many multiplier
        o2m_branch: Enable one-to-many branch
        num_queries_o2m: Number of one-to-many queries
        use_aux_head: Enable auxiliary detection head (PPYOLOEHead)

    Returns:
        RTDETRv3 instance
    """
    # Parse backbone depth
    backbone_depths = {
        'resnet18': 18,
        'resnet34': 34,
        'resnet50': 50,
        'resnet101': 101
    }
    depth = backbone_depths.get(backbone, 50)

    # Determine backbone output channels
    if depth in [18, 34]:
        # ResNet-18/34 use BasicBlock
        in_channels = [128, 256, 512]
    else:
        # ResNet-50/101 use Bottleneck
        in_channels = [512, 1024, 2048]

    # Build auxiliary head if requested
    aux_head = None
    if use_aux_head:
        aux_head = build_ppyoloe_head({
            'in_channels': [hidden_dim, hidden_dim, hidden_dim],  # Unified by neck
            'num_classes': num_classes,
            'fpn_strides': [8, 16, 32],
            'reg_max': 16,
            'act': 'swish'
        })

    return RTDETRv3(
        num_classes=num_classes,
        # Backbone config
        backbone_depth=depth,
        backbone_variant=variant,
        backbone_frozen_stages=frozen_stages,
        # Neck config
        neck_in_channels=in_channels,
        neck_hidden_dim=hidden_dim,
        # Transformer config
        transformer_num_queries=num_queries,
        transformer_num_decoder_layers=num_decoder_layers,
        transformer_num_levels=num_levels,
        transformer_num_points=num_points,
        transformer_hidden_dim=hidden_dim,
        # Head config
        head_eval_idx=eval_idx,
        head_o2m=o2m,
        head_o2m_branch=o2m_branch,
        head_num_queries_o2m=num_queries_o2m,
        # Auxiliary head
        aux_head=aux_head
    )

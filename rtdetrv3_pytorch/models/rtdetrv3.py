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
from typing import Optional, Dict, Tuple, List, Any

from . import ARCHITECTURE_REGISTRY, create


@ARCHITECTURE_REGISTRY.register()
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

    Following PaddlePaddle's implementation structure with dependency injection.

    Args:
        backbone: Backbone network instance (can be auto-injected from config)
        neck: Neck network instance (can be auto-injected from config)
        transformer: Transformer instance (can be auto-injected from config)
        detr_head: Main detection head instance (can be auto-injected from config)
        aux_head: Auxiliary detection head instance (optional, can be auto-injected)
        num_classes: Number of object classes (default: 80 for COCO)
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'neck', 'transformer', 'detr_head', 'aux_head']  # Auto-inject from config
    __shared__ = ['num_classes']  # Shared from global config

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

        # Build components if not provided (using create() for PaddlePaddle-style)
        if backbone is None:
            backbone = create(
                'ResNet',
                depth=backbone_depth,
                variant=backbone_variant,
                frozen_stages=backbone_frozen_stages,
                return_idx=backbone_return_idx
            )

        if neck is None:
            neck = create(
                'HybridEncoder',
                in_channels=neck_in_channels,
                feat_strides=neck_feat_strides,
                hidden_dim=neck_hidden_dim,
                use_encoder_idx=neck_use_encoder_idx,
                num_encoder_layers=neck_num_encoder_layers,
                expansion=neck_expansion
            )

        if transformer is None:
            transformer = create(
                'RTDETRTransformerv3',
                num_queries=transformer_num_queries,
                num_decoder_layers=transformer_num_decoder_layers,
                num_levels=transformer_num_levels,
                num_decoder_points=transformer_num_points,
                hidden_dim=transformer_hidden_dim,
                eval_idx=head_eval_idx,
                o2m_branch=head_o2m_branch,
                num_queries_o2m=head_num_queries_o2m,
                num_classes=num_classes
            )

        if detr_head is None:
            detr_head = create(
                'DINOv3Head',
                eval_idx=head_eval_idx,
                o2m=head_o2m,
                o2m_branch=head_o2m_branch,
                num_queries_o2m=head_num_queries_o2m,
                num_classes=num_classes,
                hidden_dim=transformer_hidden_dim
            )

        self.backbone = backbone
        self.neck = neck
        self.transformer = transformer
        self.detr_head = detr_head
        self.aux_head = aux_head  # PPYOLOEHead (optional, for training)
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], global_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Build RT-DETRv3 components from config (PaddlePaddle-style).

        This method creates module instances from configuration dicts and returns
        them as kwargs dict for __init__. Supports dependency injection pattern.

        Args:
            cfg: RTDETRv3 configuration dict
            global_config: Global configuration for shared values

        Returns:
            Dict of kwargs for RTDETRv3.__init__

        Example config structure:
            {
                'backbone': {'type': 'ResNet', 'depth': 50, 'variant': 'd'},
                'neck': {'type': 'HybridEncoder', 'hidden_dim': 256},
                'transformer': {'type': 'RTDETRTransformerv3', 'num_queries': 300},
                'detr_head': {'type': 'DINOv3Head'},
                'aux_head': {'type': 'PPYOLOEHead'}  # optional
            }
        """
        kwargs = {}

        # Create backbone (only if it's a config dict, not an instance)
        if 'backbone' in cfg:
            if isinstance(cfg['backbone'], dict):
                backbone_cfg = cfg['backbone'].copy()
                backbone_type = backbone_cfg.pop('type', 'ResNet')  # Default to ResNet
                kwargs['backbone'] = create(backbone_type, global_config, **backbone_cfg)
            else:
                # Already an instance (from __inject__)
                kwargs['backbone'] = cfg['backbone']

        # Create neck with backbone output shape dependency
        if 'neck' in cfg:
            if isinstance(cfg['neck'], dict):
                neck_cfg = cfg['neck'].copy()
                neck_type = neck_cfg.pop('type', 'HybridEncoder')  # Default to HybridEncoder

                # Inject backbone output shape (convert to in_channels if not specified)
                if 'backbone' in kwargs and hasattr(kwargs['backbone'], 'out_shape') and 'in_channels' not in neck_cfg:
                    # Extract in_channels from backbone.out_shape
                    neck_cfg['in_channels'] = [s['channels'] for s in kwargs['backbone'].out_shape]

                kwargs['neck'] = create(neck_type, global_config, **neck_cfg)
            else:
                # Already an instance (from __inject__)
                kwargs['neck'] = cfg['neck']

        # Create transformer
        # Following PaddlePaddle: ppdet/modeling/architectures/rtdetrv3.py:62-64
        if 'transformer' in cfg:
            if isinstance(cfg['transformer'], dict):
                transformer_cfg = cfg['transformer'].copy()
                transformer_type = transformer_cfg.pop('type', 'RTDETRTransformerv3')  # Default type

                # PaddlePaddle passes input_shape from neck, but RTDETRTransformerv3 doesn't use it
                # Optionally pass input_shape for compatibility (will be ignored)
                if 'neck' in kwargs and hasattr(kwargs['neck'], 'out_shape'):
                    transformer_cfg['input_shape'] = kwargs['neck'].out_shape

                kwargs['transformer'] = create(transformer_type, global_config, **transformer_cfg)
            else:
                # Already an instance (from __inject__)
                kwargs['transformer'] = cfg['transformer']

        # Create detr_head with transformer dependencies
        if 'detr_head' in cfg:
            if isinstance(cfg['detr_head'], dict):
                head_cfg = cfg['detr_head'].copy()
                head_type = head_cfg.pop('type', 'DINOv3Head')  # Default to DINOv3Head

                # Inject transformer properties
                if 'transformer' in kwargs and hasattr(kwargs['transformer'], 'hidden_dim'):
                    head_cfg.setdefault('hidden_dim', kwargs['transformer'].hidden_dim)
                if 'transformer' in kwargs and hasattr(kwargs['transformer'], 'nhead'):
                    head_cfg.setdefault('nhead', kwargs['transformer'].nhead)

                kwargs['detr_head'] = create(head_type, global_config, **head_cfg)
            else:
                # Already an instance (from __inject__)
                kwargs['detr_head'] = cfg['detr_head']

        # Create aux_head (optional)
        if 'aux_head' in cfg:
            if isinstance(cfg['aux_head'], dict):
                aux_cfg = cfg['aux_head'].copy()
                aux_type = aux_cfg.pop('type', 'PPYOLOEHead')  # Default to PPYOLOEHead
                kwargs['aux_head'] = create(aux_type, global_config, **aux_cfg)
            else:
                # Already an instance (from __inject__)
                kwargs['aux_head'] = cfg['aux_head']

        # Remove 'type' key if present (it's config-only, not a __init__ parameter)
        kwargs.pop('type', None)

        return kwargs

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

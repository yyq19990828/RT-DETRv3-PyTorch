"""
Model Interface Contract: RT-DETRv3 PyTorch Migration

This file defines the interface contracts that all models must implement
to ensure compatibility with the training/evaluation pipeline.
"""

from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn


class BaseDetector(nn.Module):
    """
    Base interface for all detection models.

    All RT-DETRv3 models must inherit from this class and implement
    the required methods.
    """

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass of the detection model.

        Args:
            images: Batch of images, shape [B, 3, H, W]
            targets: List of target dicts (training mode), each containing:
                - 'boxes': Tensor [N, 4] in (x1, y1, x2, y2) format
                - 'labels': Tensor [N] class labels
                - 'image_id': int, image identifier

        Returns:
            - Training mode (targets is not None):
                Dict of losses:
                {
                    'loss': total loss (Tensor),
                    'loss_class': classification loss,
                    'loss_bbox': bbox regression loss,
                    'loss_giou': giou loss,
                    ... (additional losses)
                }
            - Inference mode (targets is None):
                List of predictions, one dict per image:
                [
                    {
                        'boxes': Tensor [N, 4],
                        'scores': Tensor [N],
                        'labels': Tensor [N]
                    },
                    ...
                ]
        """
        raise NotImplementedError

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file
        """
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        """Number of detection classes (excluding background)."""
        raise NotImplementedError


class RTDETRv3Interface(BaseDetector):
    """
    Interface contract for RT-DETRv3 model.

    Extends BaseDetector with RT-DETRv3 specific requirements.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        detr_head: nn.Module,
        aux_o2m_head: Optional[nn.Module] = None,
        num_classes: int = 80,
        num_queries: int = 300,
        num_queries_one2many: int = 1500
    ):
        """
        Initialize RT-DETRv3 model.

        Args:
            backbone: Feature extraction network (e.g., ResNet)
            neck: Feature fusion network (e.g., HybridEncoder)
            transformer: Query-key-value transformer
            detr_head: Main detection head (DINOv3Head)
            aux_o2m_head: Auxiliary one-to-many head (optional, training only)
            num_classes: Number of object classes
            num_queries: Number of one-to-one queries
            num_queries_one2many: Number of one-to-many queries
        """
        super().__init__()

    def extract_features(
        self,
        images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract multi-scale features and encoded features.

        Args:
            images: Batch of images [B, 3, H, W]

        Returns:
            - Multi-scale features: List of [B, C_i, H_i, W_i]
            - Encoded features: [B, C, H, W] from neck
        """
        raise NotImplementedError

    def get_queries(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate query embeddings.

        Args:
            batch_size: Batch size
            device: Target device

        Returns:
            - One-to-one queries: [B, num_queries, hidden_dim]
            - One-to-many queries: [B, num_queries_one2many, hidden_dim]
        """
        raise NotImplementedError


class BackboneInterface(nn.Module):
    """Interface contract for backbone networks."""

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            List of feature maps at different scales:
            [
                C2: [B, C2, H/4, W/4],
                C3: [B, C3, H/8, W/8],
                C4: [B, C4, H/16, W/16],
                C5: [B, C5, H/32, W/32]
            ]
        """
        raise NotImplementedError

    @property
    def out_channels(self) -> List[int]:
        """Output channels for each feature level."""
        raise NotImplementedError


class NeckInterface(nn.Module):
    """Interface contract for neck (feature fusion) networks."""

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse multi-scale features.

        Args:
            features: List of multi-scale features from backbone

        Returns:
            Fused features: [B, hidden_dim, H, W]
        """
        raise NotImplementedError


class TransformerInterface(nn.Module):
    """Interface contract for transformer modules."""

    def forward(
        self,
        memory: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transform queries using memory.

        Args:
            memory: Encoded features [B, C, H, W] or [B, HW, C]
            query_embed: Query embeddings [B, N, C]
            pos_embed: Positional embeddings (optional)

        Returns:
            Decoded queries: [B, N, C]
        """
        raise NotImplementedError


class HeadInterface(nn.Module):
    """Interface contract for detection heads."""

    def forward(
        self,
        queries: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate predictions or compute losses.

        Args:
            queries: Decoded queries [B, N, C]
            targets: Ground truth targets (training mode)

        Returns:
            - Training mode:
                Dict of losses
            - Inference mode:
                Tuple of (pred_logits, pred_boxes)
                - pred_logits: [B, N, num_classes]
                - pred_boxes: [B, N, 4] in (cx, cy, w, h) normalized format
        """
        raise NotImplementedError


class LossInterface(nn.Module):
    """Interface contract for loss functions."""

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.

        Args:
            predictions: Model predictions
                {
                    'pred_logits': [B, N, num_classes],
                    'pred_boxes': [B, N, 4],
                    'aux_outputs': List of dicts (for deep supervision)
                }
            targets: Ground truth targets
                [
                    {
                        'boxes': [N_i, 4],
                        'labels': [N_i]
                    },
                    ...
                ]

        Returns:
            Dict of losses:
            {
                'loss': total loss,
                'loss_class': classification loss,
                'loss_bbox': bbox loss,
                'loss_giou': giou loss
            }
        """
        raise NotImplementedError


# Type aliases for clarity
ImageTensor = torch.Tensor  # [B, 3, H, W]
FeatureTensor = torch.Tensor  # [B, C, H, W]
QueryTensor = torch.Tensor  # [B, N, C]
PredictionDict = Dict[str, torch.Tensor]
TargetDict = Dict[str, torch.Tensor]
LossDict = Dict[str, torch.Tensor]

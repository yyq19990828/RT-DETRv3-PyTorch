"""
Transformer Utility Components for RT-DETRv3

This module provides common building blocks for transformer models:
- MLP (Multi-Layer Perceptron)
- Helper functions for position embeddings, bbox conversion, denoising, etc.

Reference:
- PaddlePaddle RT-DETR: ppdet/modeling/transformers/
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _get_clones(module, N):
    """Clone a module N times

    Args:
        module (nn.Module): Module to clone
        N (int): Number of clones

    Returns:
        nn.ModuleList: List of N cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True
) -> torch.Tensor:
    """
    Generate sinusoidal position embeddings

    Args:
        pos_tensor: Position tensor of shape (B, N, 2) where N is number of positions
                   and last dim is (x, y) coordinates in [0, 1]
        num_pos_feats: Dimension of position embeddings (default: 128)
        temperature: Temperature for sinusoidal encoding (default: 10000)
        exchange_xy: Whether to exchange x and y coordinates (default: True)

    Returns:
        Position embeddings of shape (B, N, num_pos_feats * 2)
        First half is x encoding, second half is y encoding
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)

    def get_sine_embed(pos, dim_t):
        # pos: (B, N)
        # dim_t: (num_pos_feats,)
        # output: (B, N, num_pos_feats)
        pos = pos[:, :, None] / dim_t  # (B, N, num_pos_feats)
        pos = torch.stack([pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()], dim=3).flatten(2)
        return pos

    if exchange_xy:
        # Exchange x and y
        x_embed = get_sine_embed(pos_tensor[:, :, 1] * scale, dim_t)
        y_embed = get_sine_embed(pos_tensor[:, :, 0] * scale, dim_t)
    else:
        x_embed = get_sine_embed(pos_tensor[:, :, 0] * scale, dim_t)
        y_embed = get_sine_embed(pos_tensor[:, :, 1] * scale, dim_t)

    pos_embed = torch.cat([x_embed, y_embed], dim=-1)
    return pos_embed


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP)

    A simple feed-forward network with configurable number of layers,
    hidden dimensions, activation function, and dropout.

    This is used in various parts of the transformer (e.g., after attention layers).

    Example:
        >>> mlp = MLP(input_dim=256, hidden_dim=1024, output_dim=256, num_layers=2)
        >>> x = torch.randn(2, 100, 256)
        >>> y = mlp(x)
        >>> print(y.shape)  # (2, 100, 256)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers (minimum 2)
            activation: Activation function ('relu' or 'gelu')
            dropout: Dropout probability (default: 0.0)
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP

        Args:
            x: Input tensor of shape (B, N, input_dim) or (B, input_dim)

        Returns:
            Output tensor of shape (B, N, output_dim) or (B, output_dim)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        return x


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute inverse sigmoid (logit function)

    Args:
        x: Input tensor with values in (0, 1)
        eps: Small epsilon to avoid log(0)

    Returns:
        Inverse sigmoid of x
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def bbox_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2) format

    Args:
        x: Bounding boxes in cxcywh format, shape (..., 4)

    Returns:
        Bounding boxes in xyxy format, shape (..., 4)
    """
    cxcy, wh = x.split(2, dim=-1)
    return torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1)


def bbox_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (center_x, center_y, width, height) format

    Args:
        x: Bounding boxes in xyxy format, shape (..., 4)

    Returns:
        Bounding boxes in cxcywh format, shape (..., 4)
    """
    x1, y1, x2, y2 = x.split(1, dim=-1)
    return torch.cat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def sigmoid_focal_loss(logit: torch.Tensor, label: torch.Tensor, normalizer: float = 1.0, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Sigmoid focal loss for classification.

    Args:
        logit: Predicted logits, shape (N, num_classes)
        label: Target labels (one-hot), shape (N, num_classes)
        normalizer: Normalization factor
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t) ^ gamma

    Returns:
        Scalar loss value
    """
    prob = torch.sigmoid(logit)
    ce_loss = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    p_t = prob * label + (1 - prob) * (1 - label)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
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
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    loss = F.binary_cross_entropy_with_logits(
        pred_logits, gt_score, weight=weight, reduction='none'
    )
    return loss.mean(1).sum() / normalizer


def get_contrastive_denoising_training_group(
    targets: dict,
    num_classes: int,
    num_queries: int,
    class_embed: torch.Tensor,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict]]:
    """
    Generate contrastive denoising training groups for RT-DETR

    This function creates positive and negative query groups for contrastive denoising training.
    Each group contains both positive (matched) and negative (mismatched) queries.

    Args:
        targets: Dictionary containing:
            - 'gt_class': List of ground truth class tensors per batch
            - 'gt_bbox': List of ground truth bbox tensors per batch (in cxcywh format, normalized)
        num_classes: Number of object classes
        num_queries: Number of object queries
        class_embed: Class embedding tensor of shape (num_classes, embed_dim)
        num_denoising: Total number of denoising queries (default: 100)
        label_noise_ratio: Ratio of label noise to add (default: 0.5)
        box_noise_scale: Scale of box noise to add (default: 1.0)

    Returns:
        Tuple of (input_query_class, input_query_bbox, attn_mask, dn_meta):
            - input_query_class: Denoising query class embeddings (bs, num_denoising, embed_dim)
            - input_query_bbox: Denoising query bboxes (bs, num_denoising, 4)
            - attn_mask: Attention mask (tgt_size, tgt_size) where tgt_size = num_denoising + num_queries
            - dn_meta: Metadata dict with keys 'dn_positive_idx', 'dn_num_group', 'dn_num_split'
        Returns (None, None, None, None) if no denoising is needed
    """
    if num_denoising <= 0:
        return None, None, None, None

    # Get number of ground truths per batch
    num_gts = [len(t) for t in targets["gt_class"]]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group

    # Pad gt to max_num of a batch
    bs = len(targets["gt_class"])
    device = class_embed.device

    input_query_class = torch.full(
        (bs, max_gt_num), num_classes, dtype=torch.int64, device=device
    )
    input_query_bbox = torch.zeros((bs, max_gt_num, 4), device=device)
    pad_gt_mask = torch.zeros((bs, max_gt_num), device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets["gt_class"][i].squeeze(-1)
            input_query_bbox[i, :num_gt] = targets["gt_bbox"][i]
            pad_gt_mask[i, :num_gt] = 1

    # Each group has positive and negative queries
    input_query_class = input_query_class.repeat(1, 2 * num_group)
    input_query_bbox = input_query_bbox.repeat(1, 2 * num_group, 1)
    pad_gt_mask = pad_gt_mask.repeat(1, 2 * num_group)

    # Positive and negative mask
    negative_gt_mask = torch.zeros((bs, max_gt_num * 2, 1), device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.repeat(1, num_group, 1)
    positive_gt_mask = 1 - negative_gt_mask

    # Contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])

    # Total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    # Add label noise
    if label_noise_ratio > 0:
        input_query_class = input_query_class.flatten()
        pad_gt_mask_flat = pad_gt_mask.flatten()

        # Half of bbox prob
        mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
        chosen_idx = torch.nonzero(mask.float() * pad_gt_mask_flat).squeeze(-1)

        # Randomly put a new one here
        new_label = torch.randint(
            0, num_classes, chosen_idx.shape, dtype=input_query_class.dtype, device=device
        )
        input_query_class.scatter_(0, chosen_idx, new_label)
        input_query_class = input_query_class.reshape(bs, num_denoising)
        pad_gt_mask = pad_gt_mask_flat.reshape(bs, num_denoising)

    # Add box noise
    if box_noise_scale > 0:
        known_bbox = bbox_cxcywh_to_xyxy(input_query_bbox)

        diff = input_query_bbox[..., 2:].repeat(1, 1, 2) * 0.5 * box_noise_scale

        rand_sign = torch.randint(0, 2, input_query_bbox.shape, device=device).float() * 2.0 - 1.0
        rand_part = torch.rand(input_query_bbox.shape, device=device)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox = known_bbox.clamp(min=0.0, max=1.0)
        input_query_bbox = bbox_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # Get class embeddings
    class_embed = torch.cat([class_embed, torch.zeros(1, class_embed.shape[-1], device=device)])
    input_query_class = torch.gather(
        class_embed, 0, input_query_class.flatten().unsqueeze(-1).expand(-1, class_embed.shape[-1])
    ).reshape(bs, num_denoising, -1)

    # Create attention mask
    tgt_size = num_denoising + num_queries
    attn_mask = torch.ones((tgt_size, tgt_size), dtype=torch.bool, device=device)

    # Match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = False

    # Reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising
            ] = False
        if i == num_group - 1:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                : max_gt_num * 2 * i
            ] = False
        else:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising
            ] = False
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                : max_gt_num * 2 * i
            ] = False

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta


def get_encoder_memory_and_spatial_shapes(features):
    """
    Flatten multi-scale features into a single memory tensor

    Args:
        features: List of feature tensors [(B, C, H1, W1), (B, C, H2, W2), (B, C, H3, W3)]

    Returns:
        memory: Flattened features (B, H1*W1 + H2*W2 + H3*W3, C)
        spatial_shapes: Tensor of (num_levels, 2) containing (H, W) for each level
        level_start_index: Tensor of (num_levels,) containing start index of each level
    """
    memory_list = []
    spatial_shapes_list = []

    for feat in features:
        B, C, H, W = feat.shape
        # Flatten spatial dimensions and transpose to (B, H*W, C)
        memory_list.append(feat.flatten(2).permute(0, 2, 1))
        spatial_shapes_list.append((H, W))

    # Concatenate along sequence dimension
    memory = torch.cat(memory_list, dim=1)  # (B, sum(H*W), C)

    # Create spatial shapes tensor
    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long, device=memory.device)  # (num_levels, 2)

    # Create level start indices
    level_start_index = torch.cat([
        torch.zeros(1, dtype=torch.long, device=memory.device),
        torch.cumsum(spatial_shapes.prod(dim=1)[:-1], dim=0)
    ])  # (num_levels,)

    return memory, spatial_shapes, level_start_index

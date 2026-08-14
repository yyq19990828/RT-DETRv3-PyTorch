"""
Multi-Scale Deformable Attention for RT-DETRv3

This module implements Multi-Scale Deformable Attention following PaddlePaddle's approach
using PyTorch's grid_sample function for numerical equivalence.

Current Implementation:
    - Pure PyTorch using F.grid_sample
    - Matches PaddlePaddle's implementation
    - Good numerical equivalence
    - Slower than CUDA implementation (~2-3x)

TODO: Performance Optimization
    Add optional CUDA extension support for faster inference/training:
    1. Try MultiScaleDeformableAttention package: pip install MultiScaleDeformableAttention
    2. Try mmcv.ops if available
    3. Fallback to current grid_sample implementation

    Expected benefits:
    - 2-3x faster training
    - 1.5-2x faster inference
    - Lower memory usage

    Note: CUDA extension requires compilation with matching CUDA/PyTorch versions
    Current grid_sample approach is more portable and easier to debug

Reference:
- PaddlePaddle RT-DETR: ppdet/modeling/transformers/deformable_transformer.py
- Deformable DETR: https://github.com/fundamentalvision/Deformable-DETR
- CUDA Extension: https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def deformable_attention_core_func(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-Scale Deformable Attention core function using grid_sample

    This follows PaddlePaddle's implementation for numerical equivalence.

    Args:
        value: Value tensor of shape (bs, value_length, n_head, c)
        value_spatial_shapes: Spatial shapes tensor of shape (n_levels, 2) [(H_0, W_0), ...]
        value_level_start_index: Start index tensor of shape (n_levels,) [0, H_0*W_0, ...]
        sampling_locations: Sampling locations of shape (bs, query_length, n_head, n_levels, n_points, 2)
                          in range [0, 1]
        attention_weights: Attention weights of shape (bs, query_length, n_head, n_levels, n_points)

    Returns:
        Output tensor of shape (bs, query_length, n_head * c)
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    # Split value by levels
    split_shape = [int(h * w) for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)

    # Convert sampling locations from [0, 1] to [-1, 1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # Reshape value: (bs, H*W, n_head, c) -> (bs*n_head, c, H, W)
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * n_head, c, int(h), int(w))
        )

        # Reshape sampling grid: (bs, Len_q, n_head, n_points, 2) -> (bs*n_head, Len_q, n_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # Sample features using grid_sample
        # Output: (bs*n_head, c, Len_q, n_points)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # Reshape attention weights: (bs, Len_q, n_head, n_levels, n_points) -> (bs*n_head, 1, Len_q, n_levels*n_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points
    )

    # Weighted sum: (bs*n_head, c, Len_q, n_levels*n_points) * (bs*n_head, 1, Len_q, n_levels*n_points)
    # -> (bs*n_head, c, Len_q)
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1)

    # Reshape output: (bs*n_head, c, Len_q) -> (bs, Len_q, n_head*c)
    output = output.reshape(bs, n_head * c, Len_q).transpose(1, 2)

    return output


class MSDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module

    This module implements multi-scale deformable attention for efficient multi-scale
    feature aggregation in detection transformers.

    Current Implementation:
        Uses PyTorch's grid_sample (follows PaddlePaddle implementation)
        Provides good numerical equivalence but slower than CUDA implementation

    TODO: Add CUDA extension support for better performance
        - Try importing from MultiScaleDeformableAttention PyPI package
        - Try importing from mmcv.ops if available
        - Fallback to current grid_sample implementation
        - Expected speedup: 2-3x faster for training, 1.5-2x for inference

    Args:
        embed_dim: Embedding dimension (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_levels: Number of feature pyramid levels (default: 4)
        num_points: Number of sampling points per level (default: 4)
        lr_mult: Learning rate multiplier for offset parameters (default: 0.1)

    Example:
        >>> attn = MSDeformableAttention(embed_dim=256, num_heads=8, num_levels=3, num_points=4)
        >>> query = torch.randn(2, 100, 256)
        >>> reference_points = torch.rand(2, 100, 3, 2)  # (bs, Len_q, n_levels, 2)
        >>> value = torch.randn(2, 6400, 256)  # Multi-scale features concatenated
        >>> value_spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]])
        >>> value_level_start_index = torch.tensor([0, 6400, 8000])
        >>> output = attn(query, reference_points, value, value_spatial_shapes, value_level_start_index)
        >>> print(output.shape)  # (2, 100, 256)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        lr_mult: float = 0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points
        self.head_dim = embed_dim // num_heads

        # TODO: Add CUDA operator detection and usage
        # Try to import CUDA operators for better performance:
        # try:
        #     from MultiScaleDeformableAttention import ms_deform_attn_forward
        #     self.ms_deform_attn_core = ms_deform_attn_forward
        #     self.use_cuda_op = True
        # except ImportError:
        #     try:
        #         from mmcv.ops import ms_deform_attn
        #         self.ms_deform_attn_core = ms_deform_attn
        #         self.use_cuda_op = True
        #     except ImportError:
        #         self.ms_deform_attn_core = deformable_attention_core_func
        #         self.use_cuda_op = False
        #         import warnings
        #         warnings.warn(
        #             "CUDA operator for MultiScaleDeformableAttention not found. "
        #             "Using PyTorch grid_sample implementation (slower). "
        #             "Install with: pip install MultiScaleDeformableAttention"
        #         )

        # Currently using grid_sample implementation (matches PaddlePaddle)
        self.ms_deform_attn_core = deformable_attention_core_func
        self.use_cuda_op = False

        # Sampling offset prediction
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)

        # Attention weight prediction
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters following PaddlePaddle convention
        """
        # Initialize sampling_offsets
        nn.init.constant_(self.sampling_offsets.weight, 0.0)

        # Initialize offset biases with grid pattern
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1
        )

        # Scale by sampling point index
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).view(
            1, 1, -1, 1
        )
        grid_init = grid_init * scaling

        self.sampling_offsets.bias.data = grid_init.view(-1)

        # Initialize attention_weights
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        # Initialize projections
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-scale deformable attention

        Args:
            query: Query tensor of shape (bs, query_length, embed_dim)
            reference_points: Reference points of shape (bs, query_length, n_levels, 2)
                            in range [0, 1], (0,0) is top-left, (1,1) is bottom-right
            value: Value tensor of shape (bs, value_length, embed_dim)
                  Multi-scale features concatenated
            value_spatial_shapes: Spatial shapes of shape (n_levels, 2)
                                [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index: Start indices of shape (n_levels,)
                                    [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask: Optional mask of shape (bs, value_length)
                       True for valid positions, False for padding

        Returns:
            Output tensor of shape (bs, query_length, embed_dim)
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        # Verify value length matches spatial shapes
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v, (
            f"Value length {Len_v} does not match spatial shapes {value_spatial_shapes}"
        )

        # Project value
        value = self.value_proj(value)

        # Apply mask if provided
        if value_mask is not None:
            value = value * value_mask.unsqueeze(-1).float()

        # Reshape value: (bs, Len_v, embed_dim) -> (bs, Len_v, num_heads, head_dim)
        value = value.view(bs, Len_v, self.num_heads, self.head_dim)

        # Compute sampling offsets: (bs, Len_q, total_points * 2) -> (bs, Len_q, num_heads, num_levels, num_points, 2)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Compute attention weights: (bs, Len_q, total_points) -> (bs, Len_q, num_heads, num_levels, num_points)
        attention_weights = self.attention_weights(query).view(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points
        )

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            # Case 1: Reference points are 2D coordinates (x, y)
            # Normalize offsets by spatial shapes: (1, 1, 1, n_levels, 1, 2)
            offset_normalizer = (
                value_spatial_shapes.flip([1])
                .view(1, 1, 1, self.num_levels, 1, 2)
                .float()
            )

            # Compute sampling locations: reference_points + normalized_offsets
            sampling_locations = (
                reference_points.view(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )

        elif reference_points.shape[-1] == 4:
            # Case 2: Reference points are bounding boxes (x, y, w, h)
            # Offset is relative to box size
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )

        # Apply deformable attention
        output = deformable_attention_core_func(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )

        # Output projection
        output = self.output_proj(output)

        return output


def build_ms_deformable_attention(cfg: dict) -> MSDeformableAttention:
    """
    Build MSDeformableAttention from config

    Args:
        cfg: Configuration dict with keys:
            - embed_dim: Embedding dimension
            - num_heads: Number of attention heads
            - num_levels: Number of feature levels
            - num_points: Number of sampling points

    Returns:
        MSDeformableAttention instance
    """
    return MSDeformableAttention(
        embed_dim=cfg.get("embed_dim", 256),
        num_heads=cfg.get("num_heads", 8),
        num_levels=cfg.get("num_levels", 4),
        num_points=cfg.get("num_points", 4),
        lr_mult=cfg.get("lr_mult", 0.1),
    )

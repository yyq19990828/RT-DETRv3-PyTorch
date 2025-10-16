"""
Transformer Utility Components for RT-DETRv3

This module provides common building blocks for transformer models:
- Position embeddings (sinusoidal and learnable)
- MLP (Multi-Layer Perceptron)
- Helper functions

Reference:
- PaddlePaddle RT-DETR: ppdet/modeling/transformers/
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal Position Embedding for 2D feature maps

    This is commonly used in DETR-style models for spatial position encoding.

    Example:
        >>> pos_embed = PositionEmbeddingSine(num_pos_feats=128)
        >>> x = torch.randn(2, 256, 20, 20)  # (B, C, H, W)
        >>> mask = torch.zeros(2, 20, 20, dtype=torch.bool)  # (B, H, W)
        >>> pos = pos_embed(x, mask)
        >>> print(pos.shape)  # (2, 256, 20, 20)
    """

    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None
    ):
        """
        Args:
            num_pos_feats: Half of the embedding dimension (default: 128)
                          Final embedding dim is 2 * num_pos_feats
            temperature: Temperature for sinusoidal encoding (default: 10000)
            normalize: Whether to normalize coordinates to [0, 1] (default: True)
            scale: Scale factor for normalized coordinates (default: 2*pi if None)
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate position embeddings for feature map

        Args:
            x: Feature tensor of shape (B, C, H, W)
            mask: Optional mask tensor of shape (B, H, W) where True means invalid

        Returns:
            Position embeddings of shape (B, C, H, W) where C = 2 * num_pos_feats
        """
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=x.device)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)

        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Learnable Position Embedding

    Uses learned embeddings for x and y coordinates.
    """

    def __init__(self, num_pos_feats: int = 256):
        """
        Args:
            num_pos_feats: Half of the embedding dimension
        """
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, C, H, W)

        Returns:
            Position embeddings of shape (B, C*2, H, W)
        """
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


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


def build_position_encoding(
    hidden_dim: int,
    position_embedding: str = 'sine',
    **kwargs
) -> nn.Module:
    """
    Build position encoding module from config

    Args:
        hidden_dim: Hidden dimension for position embeddings
        position_embedding: Type of position embedding ('sine' or 'learned')
        **kwargs: Additional arguments for position embedding

    Returns:
        Position embedding module
    """
    num_pos_feats = hidden_dim // 2

    if position_embedding == 'sine':
        return PositionEmbeddingSine(
            num_pos_feats=num_pos_feats,
            temperature=kwargs.get('temperature', 10000),
            normalize=kwargs.get('normalize', True)
        )
    elif position_embedding == 'learned':
        return PositionEmbeddingLearned(num_pos_feats=num_pos_feats)
    else:
        raise ValueError(f"Unknown position embedding: {position_embedding}")


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

"""
HybridEncoder (FPN-PAN) Neck for RT-DETRv3

This module implements the Feature Pyramid Network (FPN) with Path Aggregation Network (PAN)
for multi-scale feature fusion. Faithfully mirrors PaddlePaddle's implementation for numerical equivalence.

Structure:
1. Input Projection: Project backbone features to hidden_dim
2. Transformer Encoder: Apply self-attention to high-level features (optional)
3. FPN (Top-down pathway): C5 -> P5, C4 -> P4, C3 -> P3
4. PAN (Bottom-up pathway): P3 -> N3, P4 -> N4, P5 -> N5
5. Output: Multi-scale features [N3, N4, N5] with feat_strides=[8, 16, 32]

Reference:
- PaddlePaddle RT-DETR: ppdet/modeling/transformers/hybrid_encoder.py
- FPN Paper: https://arxiv.org/abs/1612.03144
- PAN Paper: https://arxiv.org/abs/1803.01534
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class ConvNormAct(nn.Module):
    """
    Convolution + Normalization + Activation block (BaseConv equivalent)

    This is a common building block used throughout the neck.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        norm: str = 'bn',
        act: str = 'silu'
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding (auto-calculated if None)
            groups: Number of groups for grouped convolution
            norm: Normalization type ('bn' for BatchNorm2d)
            act: Activation type ('silu', 'relu', 'none')
        """
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()

        if act == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C', H', W')
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class RepVGGBlock(nn.Module):
    """
    RepVGG Block - Structural reparameterization block

    Uses 3x3 conv + 1x1 conv + identity branches during training,
    can be merged into single 3x3 conv for deployment.

    For migration equivalence, we use the training-time structure.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: str = 'silu'
    ):
        super().__init__()

        # Main 3x3 branch
        self.conv3x3 = ConvNormAct(in_channels, out_channels, 3, 1, 1, act=act)
        # 1x1 branch
        self.conv1x1 = ConvNormAct(in_channels, out_channels, 1, 1, 0, act=act)
        # Identity branch (only if same channels)
        self.identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels else None

        if act == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining all branches

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C', H, W)
        """
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)


class CSPRepLayer(nn.Module):
    """
    CSP (Cross Stage Partial) Repeat Layer

    Used in PAN pathway for feature processing.
    Splits input into two branches, processes one branch with repeated blocks,
    then adds them together (following PaddlePaddle implementation).

    PaddlePaddle Reference:
        Forward: conv3(bottlenecks(conv1(x)) + conv2(x))
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        act: str = 'silu'
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of repeated blocks
            expansion: Channel expansion ratio for intermediate layers
            act: Activation type
        """
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        # Branch 1: Process with repeated blocks
        self.conv1 = ConvNormAct(in_channels, hidden_channels, 1, 1, 0, act=act)

        # Branch 2: Direct path
        self.conv2 = ConvNormAct(in_channels, hidden_channels, 1, 1, 0, act=act)

        # Repeated RepVGG blocks
        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])

        # Output fusion (only if channels differ)
        if hidden_channels != out_channels:
            self.conv3 = ConvNormAct(hidden_channels, out_channels, 1, 1, 0, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass following PaddlePaddle convention:
        out = conv3(bottlenecks(conv1(x)) + conv2(x))

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C', H, W)
        """
        x1 = self.conv1(x)
        x1 = self.bottlenecks(x1)

        x2 = self.conv2(x)

        # Add (Paddle convention) instead of concatenate
        x = x1 + x2
        x = self.conv3(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Self-Attention and FFN

    Matches PaddlePaddle's TransformerLayer implementation.
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False
    ):
        super().__init__()

        self.normalize_before = normalize_before

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: Optional[torch.Tensor]):
        """Add positional embeddings to tensor"""
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            src: Source tensor of shape (L, B, C) where L=H*W
            src_mask: Attention mask (optional)
            pos_embed: Positional embeddings (optional), shape (1, L, C)

        Returns:
            Output tensor of shape (L, B, C)
        """
        # Self-attention with residual
        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        # Add positional embeddings to q and k
        q = k = self.with_pos_embed(src, pos_embed)
        src2, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src2)
        if not self.normalize_before:
            src = self.norm1(src)

        # Feedforward with residual
        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        if not self.normalize_before:
            src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple encoder layers
    """
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=encoder_layer.get('d_model', 256),
                nhead=encoder_layer.get('nhead', 8),
                dim_feedforward=encoder_layer.get('dim_feedforward', 1024),
                dropout=encoder_layer.get('dropout', 0.0),
                activation=encoder_layer.get('activation', 'relu'),
                normalize_before=encoder_layer.get('normalize_before', False)
            ) if isinstance(encoder_layer, dict) else
            TransformerEncoderLayer()  # Use defaults
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through all encoder layers

        Args:
            src: Source tensor of shape (L, B, C)
            src_mask: Attention mask (optional)
            pos_embed: Positional embeddings (optional)

        Returns:
            Output tensor of shape (L, B, C)
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask, pos_embed)
        return output


class HybridEncoder(nn.Module):
    """
    HybridEncoder: FPN + PAN neck with optional Transformer encoder

    This neck takes multi-scale features from the backbone and produces
    enhanced multi-scale features with both top-down and bottom-up information flow.

    Faithfully mirrors PaddlePaddle's implementation:
    1. Input projection layers (input_proj)
    2. Optional Transformer encoder on selected feature levels
    3. Top-down FPN with lateral connections
    4. Bottom-up PAN pathway

    Input: [C3, C4, C5] from backbone with strides [8, 16, 32]
    Output: [N3, N4, N5] with hidden_dim channels and strides [8, 16, 32]

    Example:
        >>> neck = HybridEncoder(in_channels=[512, 1024, 2048], hidden_dim=256)
        >>> c3 = torch.randn(2, 512, 80, 80)
        >>> c4 = torch.randn(2, 1024, 40, 40)
        >>> c5 = torch.randn(2, 2048, 20, 20)
        >>> n3, n4, n5 = neck([c3, c4, c5])
        >>> print(n3.shape, n4.shape, n5.shape)
        torch.Size([2, 256, 80, 80]) torch.Size([2, 256, 40, 40]) torch.Size([2, 256, 20, 20])
    """

    def __init__(
        self,
        in_channels: List[int] = [512, 1024, 2048],
        feat_strides: List[int] = [8, 16, 32],
        hidden_dim: int = 256,
        num_encoder_layers: int = 1,
        use_encoder_idx: List[int] = [2],  # Apply encoder layers only to highest level (C5)
        pe_temperature: int = 10000,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        act: str = 'silu',
        eval_size: Optional[List[int]] = None
    ):
        """
        Args:
            in_channels: List of input channel numbers [C3_channels, C4_channels, C5_channels]
            feat_strides: Feature map strides [8, 16, 32]
            hidden_dim: Hidden dimension for output features (typically 256)
            num_encoder_layers: Number of transformer encoder layers (typically 1)
            use_encoder_idx: Indices where to apply encoder layers (typically [2] for C5 only)
            pe_temperature: Temperature for positional embeddings (10000 in PaddlePaddle)
            expansion: Channel expansion ratio for CSP layers
            depth_mult: Depth multiplier for number of blocks
            act: Activation function type ('silu' for PaddlePaddle, 'relu' for RT-DETR)
            eval_size: Fixed evaluation size [height, width] for cached positional embeddings
        """
        super().__init__()

        assert len(in_channels) == len(feat_strides) == 3, \
            "HybridEncoder expects 3 input feature levels"

        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.use_encoder_idx = use_encoder_idx
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # ============ Input Projection (CRITICAL - was missing in old implementation) ============
        # These project backbone features to hidden_dim BEFORE any processing
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim)
            )
            for in_ch in in_channels
        ])

        # ============ Transformer Encoder (CRITICAL - was placeholder in old implementation) ============
        # Apply transformer encoder to selected feature levels (usually C5)
        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList([
                TransformerEncoder(
                    encoder_layer={'d_model': hidden_dim, 'nhead': 8, 'dim_feedforward': 1024},
                    num_layers=num_encoder_layers
                )
                for _ in range(len(use_encoder_idx))
            ])
        else:
            self.encoder = None

        # ============ FPN (Top-down pathway) ============

        # Lateral convolutions for FPN fusion (DIFFERENT from input_proj)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()

        for idx in range(len(in_channels) - 1, 0, -1):
            # Lateral 1x1 conv before fusion
            self.lateral_convs.append(
                ConvNormAct(hidden_dim, hidden_dim, 1, 1, 0, act=act)
            )
            # CSP block for fusion
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,  # Concatenation
                    hidden_dim,
                    num_blocks=round(3 * depth_mult),
                    expansion=expansion,
                    act=act
                )
            )

        # ============ PAN (Bottom-up pathway) ============

        # Downsampling convolutions
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()

        for idx in range(len(in_channels) - 1):
            # 3x3 conv with stride 2 for downsampling
            self.downsample_convs.append(
                ConvNormAct(hidden_dim, hidden_dim, 3, stride=2, act=act)
            )
            # CSP block for fusion
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,  # Concatenation
                    hidden_dim,
                    num_blocks=round(3 * depth_mult),
                    expansion=expansion,
                    act=act
                )
            )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights and cached positional embeddings"""
        # Initialize conv and bn layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Cache positional embeddings for eval mode if eval_size is specified
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                h = self.eval_size[0] // stride
                w = self.eval_size[1] // stride
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature
                )
                self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int,
        h: int,
        embed_dim: int = 256,
        temperature: float = 10000.
    ) -> torch.Tensor:
        """
        Build 2D sinusoidal positional embeddings

        Args:
            w: Width of feature map
            h: Height of feature map
            embed_dim: Embedding dimension (must be divisible by 4)
            temperature: Temperature for sinusoidal encoding

        Returns:
            Positional embeddings of shape (1, H*W, C)
        """
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'

        # Create grid coordinates
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

        # Calculate frequencies
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        # Apply sinusoidal encoding
        out_w = grid_w.flatten()[:, None] @ omega[None, :]
        out_h = grid_h.flatten()[:, None] @ omega[None, :]

        # Concatenate sin/cos for both dimensions
        pos_embed = torch.cat([
            torch.sin(out_w), torch.cos(out_w),
            torch.sin(out_h), torch.cos(out_h)
        ], dim=1)

        return pos_embed.unsqueeze(0)  # (1, H*W, C)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through HybridEncoder

        Args:
            feats: List of backbone features [C3, C4, C5]
                C3: (B, in_channels[0], H/8, W/8)
                C4: (B, in_channels[1], H/16, W/16)
                C5: (B, in_channels[2], H/32, W/32)

        Returns:
            List of output features [N3, N4, N5]
                All with shape (B, hidden_dim, H/stride, W/stride)
        """
        assert len(feats) == 3, f"Expected 3 feature levels, got {len(feats)}"

        # ============ Input Projection ============
        # Project all features to hidden_dim
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # ============ Transformer Encoder ============
        # Apply transformer encoder to selected feature levels
        if self.encoder is not None and self.num_encoder_layers > 0:
            for i, enc_idx in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_idx].shape[2:]

                # Flatten spatial dimensions: (B, C, H, W) -> (H*W, B, C)
                src_flatten = proj_feats[enc_idx].flatten(2).permute(2, 0, 1)

                # Generate or use cached positional embeddings
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    )
                    pos_embed = pos_embed.to(src_flatten.device)
                    pos_embed = pos_embed.permute(1, 0, 2)  # (1, H*W, C) -> (H*W, 1, C)
                else:
                    # Use cached positional embeddings
                    pos_embed = getattr(self, f'pos_embed{enc_idx}', None)
                    if pos_embed is not None:
                        pos_embed = pos_embed.permute(1, 0, 2)  # (1, H*W, C) -> (H*W, 1, C)

                # Apply transformer encoder
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)

                # Reshape back: (H*W, B, C) -> (B, C, H, W)
                proj_feats[enc_idx] = memory.permute(1, 2, 0).reshape(-1, self.hidden_dim, h, w)

        # ============ FPN: Top-down pathway ============
        # Build top-down features
        inner_outs = [proj_feats[-1]]  # Start with highest level (P5)

        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            # Apply lateral conv to high-level feature
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high

            # Upsample and concatenate
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        # ============ PAN: Bottom-up pathway ============
        outs = [inner_outs[0]]  # Start with lowest level (N3)

        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            # Downsample and concatenate
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.cat([downsample_feat, feat_high], dim=1)
            )
            outs.append(out)

        return outs


def build_hybrid_encoder(cfg: dict) -> HybridEncoder:
    """
    Build HybridEncoder from config

    Args:
        cfg: Configuration dict with keys:
            - in_channels: List of input channel numbers
            - feat_strides: Feature strides
            - hidden_dim: Hidden dimension
            - num_encoder_layers: Number of encoder layers
            - use_encoder_idx: Indices to apply encoder
            - expansion: Channel expansion ratio
            - depth_mult: Depth multiplier
            - act: Activation function

    Returns:
        HybridEncoder instance
    """
    return HybridEncoder(
        in_channels=cfg.get('in_channels', [512, 1024, 2048]),
        feat_strides=cfg.get('feat_strides', [8, 16, 32]),
        hidden_dim=cfg.get('hidden_dim', 256),
        num_encoder_layers=cfg.get('num_encoder_layers', 1),
        use_encoder_idx=cfg.get('use_encoder_idx', [2]),
        pe_temperature=cfg.get('pe_temperature', 10000),
        expansion=cfg.get('expansion', 1.0),
        depth_mult=cfg.get('depth_mult', 1.0),
        act=cfg.get('act', 'silu'),
        eval_size=cfg.get('eval_size', None)
    )

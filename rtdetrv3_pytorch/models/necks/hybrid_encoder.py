"""
HybridEncoder (FPN-PAN) Neck for RT-DETRv3

This module implements the Feature Pyramid Network (FPN) with Path Aggregation Network (PAN)
for multi-scale feature fusion. It follows PaddlePaddle's implementation for numerical equivalence.

Structure:
1. FPN (Top-down pathway): C5 -> P5, C4 -> P4, C3 -> P3
2. PAN (Bottom-up pathway): P3 -> N3, P4 -> N4, P5 -> N5
3. Output: Multi-scale features [N3, N4, N5] with feat_strides=[8, 16, 32]

Reference:
- PaddlePaddle RT-DETR: ppdet/modeling/necks/hybrid_encoder.py
- FPN Paper: https://arxiv.org/abs/1612.03144
- PAN Paper: https://arxiv.org/abs/1803.01534
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvNormAct(nn.Module):
    """
    Convolution + Normalization + Activation block

    This is a common building block used throughout the neck.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        norm: str = 'bn',
        act: str = 'relu'
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            groups: Number of groups for grouped convolution
            norm: Normalization type ('bn' for BatchNorm2d)
            act: Activation type ('relu', 'silu', 'none')
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'silu':
            self.act = nn.SiLU(inplace=True)
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
        act: str = 'relu'
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

        # Repeated blocks (RepVggBlock in Paddle)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvNormAct(hidden_channels, hidden_channels, 3, 1, 1, act=act),
                ConvNormAct(hidden_channels, hidden_channels, 3, 1, 1, act=act)
            )
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
        out = conv3(blocks(conv1(x)) + conv2(x))

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C', H, W)
        """
        x1 = self.conv1(x)
        x1 = self.blocks(x1)

        x2 = self.conv2(x)

        # Add (Paddle convention) instead of concatenate
        x = x1 + x2
        x = self.conv3(x)

        return x


class HybridEncoder(nn.Module):
    """
    HybridEncoder: FPN + PAN neck for multi-scale feature fusion

    This neck takes multi-scale features from the backbone and produces
    enhanced multi-scale features with both top-down and bottom-up information flow.

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
        use_encoder_idx: List[int] = [2],  # Apply encoder layers only to highest level
        num_csp_blocks: int = 3,
        expansion: float = 1.0,
        act: str = 'relu'
    ):
        """
        Args:
            in_channels: List of input channel numbers [C3_channels, C4_channels, C5_channels]
            feat_strides: Feature map strides [8, 16, 32]
            hidden_dim: Hidden dimension for output features (typically 256)
            num_encoder_layers: Number of encoder layers (typically 1)
            use_encoder_idx: Indices where to apply encoder layers (typically [2] for C5 only)
            num_csp_blocks: Number of CSP blocks in PAN pathway
            expansion: Channel expansion ratio for CSP layers
            act: Activation function type
        """
        super().__init__()

        assert len(in_channels) == len(feat_strides) == 3, \
            "HybridEncoder expects 3 input feature levels"

        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.use_encoder_idx = use_encoder_idx

        # ============ FPN (Top-down pathway) ============

        # Lateral convolutions: Reduce channels to hidden_dim
        self.lateral_convs = nn.ModuleList([
            ConvNormAct(in_ch, hidden_dim, 1, 1, 0, act=act)
            for in_ch in in_channels
        ])

        # Top-down fusion blocks (using CSPRepLayer like PaddlePaddle)
        self.fpn_blocks = nn.ModuleList([
            CSPRepLayer(
                hidden_dim * 2,  # Concatenation of upsampled + lateral feature
                hidden_dim,
                num_blocks=num_csp_blocks,
                expansion=expansion,
                act=act
            )
            for _ in range(len(in_channels) - 1)  # P4, P3 (no P5)
        ])

        # ============ PAN (Bottom-up pathway) ============

        # Downsampling convolutions
        self.downsample_convs = nn.ModuleList([
            ConvNormAct(hidden_dim, hidden_dim, 3, 2, 1, act=act)  # Stride 2
            for _ in range(len(in_channels) - 1)  # N3->N4, N4->N5
        ])

        # Bottom-up fusion blocks
        self.pan_blocks = nn.ModuleList([
            CSPRepLayer(
                hidden_dim * 2,  # Concatenation of downsampled + FPN feature
                hidden_dim,
                num_blocks=num_csp_blocks,
                expansion=expansion,
                act=act
            )
            for _ in range(len(in_channels) - 1)  # N4, N5
        ])

        # ============ Optional encoder layers ============
        # These can be transformer encoder layers applied to specific feature levels
        # For now, we use simple conv layers as placeholder
        if num_encoder_layers > 0:
            self.encoder_layers = nn.ModuleList([
                ConvNormAct(hidden_dim, hidden_dim, 3, 1, 1, act=act)
                for _ in range(num_encoder_layers)
            ])
        else:
            self.encoder_layers = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        c3, c4, c5 = feats

        # ============ FPN: Top-down pathway ============

        # Apply lateral convolutions to reduce channels
        lat_c3 = self.lateral_convs[0](c3)  # (B, hidden_dim, H/8, W/8)
        lat_c4 = self.lateral_convs[1](c4)  # (B, hidden_dim, H/16, W/16)
        lat_c5 = self.lateral_convs[2](c5)  # (B, hidden_dim, H/32, W/32)

        # Apply encoder layers to C5 if specified
        if self.encoder_layers is not None and 2 in self.use_encoder_idx:
            for layer in self.encoder_layers:
                lat_c5 = layer(lat_c5)

        # Top-down fusion: C5 -> C4 -> C3 (following PaddlePaddle convention)
        # P5 = lat_c5 (no fusion needed)
        p5 = lat_c5

        # P4 = CSP(concat(upsample(P5), lat_c4))
        p5_up = F.interpolate(p5, size=lat_c4.shape[2:], mode='nearest')
        p4 = torch.cat([p5_up, lat_c4], dim=1)  # Concatenate
        p4 = self.fpn_blocks[1](p4)  # CSP fusion

        # P3 = CSP(concat(upsample(P4), lat_c3))
        p4_up = F.interpolate(p4, size=lat_c3.shape[2:], mode='nearest')
        p3 = torch.cat([p4_up, lat_c3], dim=1)  # Concatenate
        p3 = self.fpn_blocks[0](p3)  # CSP fusion

        # ============ PAN: Bottom-up pathway ============

        # N3 = P3 (no fusion needed)
        n3 = p3

        # N4 = CSP( downsample(N3) + P4 )
        n3_down = self.downsample_convs[0](n3)  # Stride 2
        n4 = torch.cat([n3_down, p4], dim=1)  # Concatenate
        n4 = self.pan_blocks[0](n4)  # CSP fusion

        # N5 = CSP( downsample(N4) + P5 )
        n4_down = self.downsample_convs[1](n4)  # Stride 2
        n5 = torch.cat([n4_down, p5], dim=1)  # Concatenate
        n5 = self.pan_blocks[1](n5)  # CSP fusion

        return [n3, n4, n5]


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

    Returns:
        HybridEncoder instance
    """
    return HybridEncoder(
        in_channels=cfg.get('in_channels', [512, 1024, 2048]),
        feat_strides=cfg.get('feat_strides', [8, 16, 32]),
        hidden_dim=cfg.get('hidden_dim', 256),
        num_encoder_layers=cfg.get('num_encoder_layers', 1),
        use_encoder_idx=cfg.get('use_encoder_idx', [2]),
        num_csp_blocks=cfg.get('num_csp_blocks', 3),
        expansion=cfg.get('expansion', 1.0),
        act=cfg.get('act', 'relu')
    )

"""
ResNet Backbone Implementation for RT-DETRv3

This module implements ResNet variants (ResNet-18, 34, 50, 101) with ResNet-vd modifications
following PaddlePaddle's implementation for numerical equivalence.

ResNet-vd modifications:
- Use average pooling for stride downsampling instead of conv stride=2
- Use 3x3 stem convolutions instead of 7x7

Reference:
- PaddlePaddle RT-DETR: ppdet/modeling/backbones/resnet.py
- PyTorch torchvision: torchvision.models.resnet
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional

# Import registry for PaddlePaddle-style registration
from ppdet.core.workspace import register


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34

    Structure:
        x -> 3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> (+) -> ReLU
        |__________________________________________________|
    """
    expansion = 1  # Output channels = input channels * expansion

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
        use_dcn: bool = False
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (before expansion)
            stride: Stride for first conv layer (1 or 2)
            downsample: Downsample module for shortcut when stride=2 or channels mismatch
            use_dcn: Use deformable convolution (not implemented, for compatibility)
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C', H', W')
            where H' = H // stride, W' = W // stride
        """
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet-50/101/152

    Structure:
        x -> 1x1 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 1x1 conv -> BN -> (+) -> ReLU
        |___________________________________________________________________________|

    Output channels = input channels * expansion
    """
    expansion = 4  # Output channels = out_channels * 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
        use_dcn: bool = False,
        variant: str = 'd'
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of intermediate channels (before expansion)
            stride: Stride for 3x3 conv layer (1 or 2)
            downsample: Downsample module for shortcut when stride=2 or channels mismatch
            use_dcn: Use deformable convolution (not implemented, for compatibility)
            variant: ResNet variant ('a', 'b', 'c', 'd')
                - 'd': ResNet-vd, use avgpool for stride downsampling
        """
        super().__init__()
        self.variant = variant

        # 1x1 conv for channel reduction
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv (potentially with stride for downsampling)
        # For variant 'd', we apply avgpool before conv instead of stride in conv
        if variant == 'd' and stride != 1:
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            conv2_stride = 1
        else:
            self.avgpool = None
            conv2_stride = stride

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=conv2_stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv for channel expansion
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C*expansion, H', W')
            where H' = H // stride, W' = W // stride
        """
        identity = x

        # 1x1 conv (channel reduction)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply avgpool before 3x3 conv for variant 'd'
        if self.avgpool is not None:
            out = self.avgpool(out)

        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 conv (channel expansion)
        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@register
class ResNet(nn.Module):
    """
    ResNet Backbone with support for ResNet-vd variant

    Output: Multi-scale features [C3, C4, C5] at strides [8, 16, 32]

    PaddlePaddle-style registration with @register decorator.

    Example:
        >>> # Method 1: Direct instantiation
        >>> backbone = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])
        >>>
        >>> # Method 2: PaddlePaddle-style create
        >>> from models import create
        >>> backbone = create('ResNet', depth=50, variant='d')
        >>>
        >>> # Forward pass
        >>> x = torch.randn(2, 3, 640, 640)
        >>> c3, c4, c5 = backbone(x)
        >>> print(c3.shape, c4.shape, c5.shape)
        torch.Size([2, 512, 80, 80]) torch.Size([2, 1024, 40, 40]) torch.Size([2, 2048, 20, 20])
    """

    __category__ = 'backbone'
    __inject__ = []  # No dependencies (backbone is root component)
    __shared__ = []  # No shared config needed

    # ResNet architecture specifications
    arch_settings = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3])
    }

    def __init__(
        self,
        depth: int = 50,
        variant: str = 'd',
        frozen_stages: int = -1,
        return_idx: List[int] = [1, 2, 3],
        use_dcn: bool = False,
        num_stages: int = 4
    ):
        """
        Args:
            depth: ResNet depth (18, 34, 50, 101, 152)
            variant: ResNet variant ('a', 'b', 'c', 'd')
                - 'd': ResNet-vd with avgpool downsampling and 3x3 stem
            frozen_stages: Freeze the first N stages (0-4). -1 means no freezing.
                Stage 0 is the stem, stages 1-4 are the residual layers.
            return_idx: Indices of stages to return (0-indexed from layer1)
                [0] -> layer1 output (C2, stride 4)
                [1] -> layer2 output (C3, stride 8)
                [2] -> layer3 output (C4, stride 16)
                [3] -> layer4 output (C5, stride 32)
            use_dcn: Use deformable convolution (not implemented)
            num_stages: Number of residual stages (typically 4)
        """
        super().__init__()

        if depth not in self.arch_settings:
            raise ValueError(f"Unsupported depth {depth}. Choose from {list(self.arch_settings.keys())}")

        self.depth = depth
        self.variant = variant
        self.frozen_stages = frozen_stages
        self.return_idx = return_idx
        self.num_stages = num_stages

        block, layers = self.arch_settings[depth]
        self.block = block
        self.layers = layers

        self.in_channels = 64

        # Stem layers
        if variant == 'd':
            # ResNet-vd: Use three 3x3 convs instead of one 7x7 conv
            self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1_1 = nn.BatchNorm2d(32)
            self.relu1_1 = nn.ReLU(inplace=True)

            self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.relu1_2 = nn.ReLU(inplace=True)

            self.conv1_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1_3 = nn.BatchNorm2d(64)
            self.relu1_3 = nn.ReLU(inplace=True)
        else:
            # Standard ResNet: 7x7 conv
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Initialize weights
        self._init_weights()

        # Freeze stages if specified
        self._freeze_stages()

        # Set output shape for dependency injection (PaddlePaddle pattern)
        self._setup_out_shape()

    def _setup_out_shape(self):
        """Setup output shape info for dependency injection"""
        # Calculate output channels for each stage
        block = self.block
        if self.depth in [18, 34]:
            # BasicBlock expansion = 1
            stage_channels = [64, 128, 256, 512]
        else:
            # Bottleneck expansion = 4
            stage_channels = [256, 512, 1024, 2048]

        # Create output shape list for return_idx
        # Strides: layer1=4, layer2=8, layer3=16, layer4=32
        # Formula: stride = 2^(idx+2) where idx is layer index (0,1,2,3)
        self.out_shape = []
        for idx in self.return_idx:
            channels = stage_channels[idx]
            # Stride calculation:
            # idx=0 (layer1): 2^(0+2) = 4
            # idx=1 (layer2): 2^(1+2) = 8
            # idx=2 (layer3): 2^(2+2) = 16
            # idx=3 (layer4): 2^(3+2) = 32
            stride = 2 ** (idx + 2)
            self.out_shape.append({
                'channels': channels,
                'stride': stride
            })

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], global_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Build ResNet from config (PaddlePaddle-style).

        Args:
            cfg: ResNet configuration dict
            global_config: Global configuration (unused for backbone)

        Returns:
            Dict of kwargs for ResNet.__init__

        Example config:
            {
                'depth': 50,
                'variant': 'd',
                'frozen_stages': 1,
                'return_idx': [1, 2, 3]
            }
        """
        return {
            'depth': cfg.get('depth', 50),
            'variant': cfg.get('variant', 'd'),
            'frozen_stages': cfg.get('frozen_stages', -1),
            'return_idx': cfg.get('return_idx', [1, 2, 3]),
            'use_dcn': cfg.get('use_dcn', False),
            'num_stages': cfg.get('num_stages', 4)
        }

    def _make_layer(
        self,
        block: nn.Module,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a residual layer with multiple blocks

        Args:
            block: Block type (BasicBlock or Bottleneck)
            out_channels: Number of output channels (before expansion)
            num_blocks: Number of blocks in this layer
            stride: Stride for the first block (1 or 2)

        Returns:
            Sequential module containing all blocks
        """
        downsample = None

        # Create downsample module if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            if self.variant == 'd' and stride != 1:
                # ResNet-vd: avgpool + 1x1 conv
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(
                        self.in_channels, out_channels * block.expansion,
                        kernel_size=1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * block.expansion)
                )
            else:
                # Standard: 1x1 conv with stride
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels, out_channels * block.expansion,
                        kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * block.expansion)
                )

        layers = []
        # First block (with potential downsampling)
        layers.append(
            block(
                self.in_channels, out_channels, stride, downsample,
                variant=self.variant if hasattr(block, '__init__') and 'variant' in block.__init__.__code__.co_varnames else None
            ) if block == Bottleneck else block(self.in_channels, out_channels, stride, downsample)
        )

        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * block.expansion

        # Remaining blocks (no downsampling)
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.in_channels, out_channels,
                    variant=self.variant if block == Bottleneck else None
                ) if block == Bottleneck else block(self.in_channels, out_channels)
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights following PyTorch convention"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_stages(self):
        """Freeze specified stages"""
        if self.frozen_stages >= 0:
            # Freeze stem (stage 0)
            if self.variant == 'd':
                for module in [self.conv1_1, self.bn1_1, self.conv1_2, self.bn1_2, self.conv1_3, self.bn1_3]:
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                for module in [self.conv1, self.bn1]:
                    for param in module.parameters():
                        param.requires_grad = False

            # Freeze maxpool
            for param in self.maxpool.parameters():
                param.requires_grad = False

        # Freeze residual stages
        for i in range(1, self.frozen_stages + 1):
            if i <= 4:
                layer = getattr(self, f'layer{i}')
                layer.eval()  # Set to eval mode (freeze BN statistics)
                for param in layer.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train to keep frozen stages in eval mode"""
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through ResNet backbone

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Tuple of feature maps at specified return indices
            For return_idx=[1, 2, 3]:
                - C3: (B, 512, H/8, W/8) for ResNet-50
                - C4: (B, 1024, H/16, W/16)
                - C5: (B, 2048, H/32, W/32)
        """
        # Stem
        if self.variant == 'd':
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = self.relu1_1(x)

            x = self.conv1_2(x)
            x = self.bn1_2(x)
            x = self.relu1_2(x)

            x = self.conv1_3(x)
            x = self.bn1_3(x)
            x = self.relu1_3(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.maxpool(x)

        # Residual layers
        outputs = []
        layer_outputs = []

        x = self.layer1(x)
        layer_outputs.append(x)

        x = self.layer2(x)
        layer_outputs.append(x)

        x = self.layer3(x)
        layer_outputs.append(x)

        x = self.layer4(x)
        layer_outputs.append(x)

        # Return specified feature maps
        for idx in self.return_idx:
            outputs.append(layer_outputs[idx])

        return tuple(outputs)

"""
ResNet backbone for RT-DETRv3 PyTorch implementation
Migrated from PaddlePaddle RT-DETRv3 implementation
"""

import math
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from ...core import register

__all__ = ['ResNet', 'BasicBlock', 'Bottleneck', 'ResNetVD']

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class ConvNormLayer(nn.Module):
    """Convolution + Normalization layer"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 groups: int = 1,
                 activation: Optional[str] = None,
                 norm_type: str = 'bn',
                 norm_decay: float = 0.0,
                 freeze_norm: bool = False):
        super(ConvNormLayer, self).__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        
        # Normalization layer
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(out_channels)
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
        
        # Freeze normalization if needed
        if freeze_norm:
            for param in self.norm.parameters():
                param.requires_grad = False
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'swish':
            self.activation = nn.SiLU(inplace=True)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    
    expansion = 1
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_type: str = 'bn',
                 norm_decay: float = 0.0,
                 freeze_norm: bool = False):
        super(BasicBlock, self).__init__()
        
        self.conv1 = ConvNormLayer(
            in_channels, out_channels, 3, stride,
            activation='relu', norm_type=norm_type,
            norm_decay=norm_decay, freeze_norm=freeze_norm
        )
        
        self.conv2 = ConvNormLayer(
            out_channels, out_channels, 3, 1,
            activation=None, norm_type=norm_type,
            norm_decay=norm_decay, freeze_norm=freeze_norm
        )
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block"""
    
    expansion = 4
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_type: str = 'bn',
                 norm_decay: float = 0.0,
                 freeze_norm: bool = False):
        super(Bottleneck, self).__init__()
        
        self.conv1 = ConvNormLayer(
            in_channels, out_channels, 1, 1,
            activation='relu', norm_type=norm_type,
            norm_decay=norm_decay, freeze_norm=freeze_norm
        )
        
        self.conv2 = ConvNormLayer(
            out_channels, out_channels, 3, stride,
            activation='relu', norm_type=norm_type,
            norm_decay=norm_decay, freeze_norm=freeze_norm
        )
        
        self.conv3 = ConvNormLayer(
            out_channels, out_channels * self.expansion, 1, 1,
            activation=None, norm_type=norm_type,
            norm_decay=norm_decay, freeze_norm=freeze_norm
        )
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Blocks(nn.Module):
    """ResNet blocks"""
    
    def __init__(self,
                 block: nn.Module,
                 in_channels: int,
                 out_channels: int,
                 count: int,
                 stride: int = 1,
                 norm_type: str = 'bn',
                 norm_decay: float = 0.0,
                 freeze_norm: bool = False):
        super(Blocks, self).__init__()
        
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = ConvNormLayer(
                in_channels, out_channels * block.expansion, 1, stride,
                activation=None, norm_type=norm_type,
                norm_decay=norm_decay, freeze_norm=freeze_norm
            )
        
        layers = []
        layers.append(block(
            in_channels, out_channels, stride, downsample,
            norm_type, norm_decay, freeze_norm
        ))
        
        for _ in range(1, count):
            layers.append(block(
                out_channels * block.expansion, out_channels, 1, None,
                norm_type, norm_decay, freeze_norm
            ))
        
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


@register()
class ResNet(nn.Module):
    """ResNet backbone"""
    
    def __init__(self,
                 depth: int = 50,
                 num_stages: int = 4,
                 return_idx: List[int] = [1, 2, 3],
                 norm_type: str = 'bn',
                 norm_decay: float = 0.0,
                 freeze_norm: bool = False,
                 freeze_at: int = -1,
                 variant: str = 'b'):
        super(ResNet, self).__init__()
        
        assert depth in ResNet_cfg.keys(), f"Depth {depth} not supported"
        assert variant in ['a', 'b', 'c', 'd'], f"Variant {variant} not supported"
        assert num_stages >= 1 and num_stages <= 4
        
        self.depth = depth
        self.num_stages = num_stages
        self.return_idx = return_idx
        self.norm_type = norm_type
        self.variant = variant
        self.freeze_at = freeze_at
        
        # Choose block type
        if depth < 50:
            block = BasicBlock
        else:
            block = Bottleneck
        
        # Stage configurations
        stage_blocks = ResNet_cfg[depth][:num_stages]
        
        # Stem layers
        if variant == 'a':
            self.conv1 = ConvNormLayer(
                3, 64, 7, 2, activation='relu',
                norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm
            )
        elif variant == 'b':
            self.conv1 = ConvNormLayer(
                3, 64, 7, 2, activation='relu',
                norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm
            )
        elif variant == 'c':
            self.conv1 = nn.Sequential(
                ConvNormLayer(3, 32, 3, 2, activation='relu',
                            norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm),
                ConvNormLayer(32, 32, 3, 1, activation='relu',
                            norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm),
                ConvNormLayer(32, 64, 3, 1, activation='relu',
                            norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)
            )
        elif variant == 'd':
            self.conv1 = nn.Sequential(
                ConvNormLayer(3, 32, 3, 2, activation='relu',
                            norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm),
                ConvNormLayer(32, 32, 3, 1, activation='relu',
                            norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm),
                ConvNormLayer(32, 64, 3, 1, activation='relu',
                            norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)
            )
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build stages
        self.stages = nn.ModuleList()
        in_channels = 64
        out_channels = 64
        
        for i, num_blocks in enumerate(stage_blocks):
            stride = 1 if i == 0 else 2
            
            stage = Blocks(
                block=block,
                in_channels=in_channels,
                out_channels=out_channels,
                count=num_blocks,
                stride=stride,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm
            )
            
            self.stages.append(stage)
            in_channels = out_channels * block.expansion
            out_channels *= 2
        
        # Output channel specifications
        self.out_channels = []
        out_ch = 64
        for i in range(num_stages):
            if i in return_idx:
                self.out_channels.append(out_ch * block.expansion)
            out_ch *= 2
        
        self._freeze_parameters()
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _freeze_parameters(self):
        """Freeze parameters up to freeze_at stage"""
        if self.freeze_at < 0:
            return
        
        # Freeze stem
        if self.freeze_at >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        
        # Freeze stages
        for i, stage in enumerate(self.stages):
            if i < self.freeze_at:
                for param in stage.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass"""
        x = self.conv1(x)
        x = self.maxpool(x)
        
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.return_idx:
                outputs.append(x)
        
        return outputs
    
    @property
    def out_shape(self):
        """Output shape specification"""
        return [{'channels': ch} for ch in self.out_channels]


@register()
class ResNetVD(ResNet):
    """ResNet with variant D (ResNet-VD)"""
    
    def __init__(self, depth: int = 50, **kwargs):
        kwargs['variant'] = 'd'
        super(ResNetVD, self).__init__(depth, **kwargs)


# Factory functions for common ResNet variants
def resnet18(**kwargs):
    return ResNet(depth=18, **kwargs)

def resnet34(**kwargs):
    return ResNet(depth=34, **kwargs)

def resnet50(**kwargs):
    return ResNet(depth=50, **kwargs)

def resnet101(**kwargs):
    return ResNet(depth=101, **kwargs)

def resnet152(**kwargs):
    return ResNet(depth=152, **kwargs)

# ResNet-VD variants
def resnet18vd(**kwargs):
    return ResNetVD(depth=18, **kwargs)

def resnet34vd(**kwargs):
    return ResNetVD(depth=34, **kwargs)

def resnet50vd(**kwargs):
    return ResNetVD(depth=50, **kwargs)

def resnet101vd(**kwargs):
    return ResNetVD(depth=101, **kwargs)

def resnet152vd(**kwargs):
    return ResNetVD(depth=152, **kwargs)
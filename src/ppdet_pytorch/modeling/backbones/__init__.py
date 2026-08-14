"""
Backbone Networks for RT-DETRv3

Available backbones:
- ResNet-18, 34, 50, 101, 152 (with ResNet-vd variant support)
"""

from .deimv2_dinov3 import DINOv3STAs
from .hgnetv2 import HGNetv2
from .presnet import PResNet
from .resnet import ResNet

__all__ = ["DINOv3STAs", "HGNetv2", "PResNet", "ResNet"]

"""
Backbone Networks for RT-DETRv3

Available backbones:
- ResNet-18, 34, 50, 101, 152 (with ResNet-vd variant support)
"""

from .resnet import ResNet, build_resnet

__all__ = ['ResNet', 'build_resnet']

"""
Detection Heads for RT-DETRv3

Available heads:
- DINOv3Head: Main detection head for RT-DETRv3 (transformer-based)
- PPYOLOEHead: Auxiliary CNN-based detection branch for training
"""

from .detr_head import DINOv3Head, build_dinov3_head
from .ppyoloe_head import PPYOLOEHead, build_ppyoloe_head

__all__ = [
    'DINOv3Head',
    'build_dinov3_head',
    'PPYOLOEHead',
    'build_ppyoloe_head',
]

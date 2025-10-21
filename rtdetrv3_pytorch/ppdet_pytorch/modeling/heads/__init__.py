"""
Detection Heads for RT-DETRv3

Available heads:
- DINOv3Head: Main detection head for RT-DETRv3 (transformer-based)
- PPYOLOEHead: Auxiliary CNN-based detection branch for training
"""

from .detr_head import DINOv3Head
from .ppyoloe_head import PPYOLOEHead

__all__ = [
    'DINOv3Head',
    'PPYOLOEHead',
]

"""
RT-DETRv3 PyTorch Data Module

Contains data loading and transformation components:
- source: Dataset implementations (COCO, etc.)
- transform: Data augmentation operators (Mosaic, Mixup, etc.)
- reader: DataLoader construction utilities
"""

from . import source
from . import transform
from . import reader

__all__ = [
    'source',
    'transform',
    'reader',
]

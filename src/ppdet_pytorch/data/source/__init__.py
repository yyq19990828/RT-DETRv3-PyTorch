"""
Dataset source implementations for RT-DETRv3 PyTorch

Migrated from PaddlePaddle ppdet/data/source/ to maintain compatibility.
"""

from .dataset import DetDataset
from .coco import COCODataSet  # Renamed for Paddle compatibility
from .lvis import LVISDataSet
from .voc import VOCDataSet

__all__ = [
    'DetDataset',
    'COCODataSet',
    'LVISDataSet',
    'VOCDataSet',
]

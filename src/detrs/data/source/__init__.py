"""
Dataset source implementations for RT-DETRv3 PyTorch

Migrated from PaddlePaddle ppdet/data/source/ to maintain compatibility.
"""

from .coco import COCODataSet  # Renamed for Paddle compatibility
from .dataset import DetDataset
from .lvis import LVISDataSet
from .voc import VOCDataSet
from .yolo import YOLODataSet

__all__ = [
    "DetDataset",
    "COCODataSet",
    "LVISDataSet",
    "VOCDataSet",
    "YOLODataSet",
]

"""
COCO dataset loader for RT-DETRv3 PyTorch

Compatible with COCO 2017 detection dataset format.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class COCODetection(Dataset):
    """
    COCO detection dataset.
    
    Args:
        data_path: Path to COCO images directory (e.g., 'coco/train2017')
        ann_file: Path to COCO annotation JSON file
        transforms: Optional data transforms
        return_masks: Whether to return segmentation masks
    """
    
    def __init__(
        self,
        data_path: str,
        ann_file: str,
        transforms: Optional[Any] = None,
        return_masks: bool = False
    ):
        self.data_path = Path(data_path)
        self.ann_file = Path(ann_file)
        self.transforms = transforms
        self.return_masks = return_masks
        
        # Load COCO annotations
        logger.info(f"Loading COCO annotations from {self.ann_file}")
        self.coco = COCO(str(self.ann_file))
        
        # Get image IDs
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        logger.info(f"Loaded {len(self.ids)} images from COCO dataset")
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> Tuple[Any, Dict]:
        """
        Get dataset item.
        
        Args:
            index: Dataset index
            
        Returns:
            Tuple of (image, target) where target is a dict with:
                - boxes: (N, 4) bounding boxes in [x, y, w, h] format
                - labels: (N,) class labels
                - image_id: Image ID
                - orig_size: Original image size (H, W)
                - size: Current image size (H, W) after transforms
        """
        img_id = self.ids[index]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.data_path / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        # Parse annotations
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Skip invalid annotations
            if 'bbox' not in ann or ann['area'] <= 0:
                continue
            
            x, y, w, h = ann['bbox']
            
            # Skip degenerate boxes
            if w <= 0 or h <= 0:
                continue
            
            boxes.append([x, y, w, h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.tensor([img_info['height'], img_info['width']]),
            'size': torch.tensor([img_info['height'], img_info['width']]),
        }
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def get_img_info(self, index: int) -> Dict:
        """Get image metadata"""
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        return img_info
    
    def get_cat_ids(self) -> List[int]:
        """Get category IDs"""
        return list(sorted(self.coco.getCatIds()))
    
    def get_categories(self) -> List[Dict]:
        """Get category information"""
        return self.coco.loadCats(self.coco.getCatIds())


def build_coco_dataset(
    anno_file: str,
    image_dir: str,
    input_size: int = 640,
    is_train: bool = False
) -> COCODetection:
    """
    Build COCO dataset with transforms

    Args:
        anno_file: Path to COCO annotation JSON file
        image_dir: Path to COCO images directory
        input_size: Input image size (square)
        is_train: Whether this is training dataset

    Returns:
        COCODetection instance
    """
    from .transforms import Compose, Resize, ToTensor, Normalize

    # Build transforms
    transforms = Compose([
        Resize([input_size, input_size]),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = COCODetection(
        data_path=image_dir,
        ann_file=anno_file,
        transforms=transforms,
        return_masks=False
    )

    return dataset

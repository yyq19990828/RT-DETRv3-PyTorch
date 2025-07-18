"""Dataset implementation for RT-DETRv3 PyTorch."""

import os
import json
import torch
import torch.utils.data as data
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import numpy as np

from ..core.workspace import register


@register()
class CocoDetection(data.Dataset):
    """COCO Detection Dataset for RT-DETRv3.
    
    Compatible with PaddlePaddle RT-DETRv3 data format.
    """
    
    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        return_masks: bool = False,
        remap_mscoco_category: bool = False,
        allow_empty: bool = False,
        data_fields: List[str] = None,
        sample_num: int = -1,
        **kwargs
    ):
        """Initialize COCO dataset.
        
        Args:
            image_dir: Directory containing images
            ann_file: Path to annotation file
            transforms: Transform pipeline
            return_masks: Whether to return segmentation masks
            remap_mscoco_category: Whether to remap COCO categories
            allow_empty: Whether to allow empty annotations
            data_fields: List of data fields to load
            sample_num: Number of samples to load (-1 for all)
        """
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.allow_empty = allow_empty
        self.data_fields = data_fields or ['image', 'gt_bbox', 'gt_class']
        
        # Load COCO annotations
        self.coco = COCO(ann_file)
        
        # Get image ids
        self.image_ids = list(self.coco.imgs.keys())
        if sample_num > 0:
            self.image_ids = self.image_ids[:sample_num]
            
        # Filter out images without annotations if not allow_empty
        if not allow_empty:
            self.image_ids = [
                img_id for img_id in self.image_ids
                if len(self.coco.getAnnIds(imgIds=img_id)) > 0
            ]
        
        # Category mapping
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label2cat = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
        
        # MS COCO category remapping
        if remap_mscoco_category:
            self.mscoco_category2label = self._get_mscoco_category_mapping()
    
    def _get_mscoco_category_mapping(self) -> Dict[int, int]:
        """Get MS COCO category to label mapping."""
        # MS COCO has 80 categories but category ids are not continuous
        # Map them to continuous labels 0-79
        category_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return {cat_id: i for i, cat_id in enumerate(category_ids)}
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index."""
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Parse annotations
        target = self._parse_annotations(anns, img_info)
        
        # Create sample
        sample = {
            'image': image,
            'image_id': img_id,
            'orig_size': torch.tensor([img_info['height'], img_info['width']], dtype=torch.int64),
            'size': torch.tensor([img_info['height'], img_info['width']], dtype=torch.int64),
        }
        
        # Add target fields
        for key, value in target.items():
            sample[key] = value
        
        # Apply transforms
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample
    
    def _parse_annotations(self, anns: List[Dict], img_info: Dict) -> Dict[str, torch.Tensor]:
        """Parse COCO annotations."""
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        masks = []
        
        for ann in anns:
            if ann.get('ignore', False):
                continue
                
            # Bounding box
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2] format
            box = [x, y, x + w, y + h]
            boxes.append(box)
            
            # Category label
            cat_id = ann['category_id']
            if self.remap_mscoco_category:
                label = self.mscoco_category2label[cat_id]
            else:
                label = self.cat2label[cat_id]
            labels.append(label)
            
            # Area
            areas.append(ann['area'])
            
            # Is crowd
            iscrowd.append(ann.get('iscrowd', 0))
            
            # Segmentation mask
            if self.return_masks and 'segmentation' in ann:
                mask = self._parse_mask(ann['segmentation'], img_info)
                masks.append(mask)
        
        # Convert to tensors
        target = {}
        if boxes:
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)
            target['areas'] = torch.tensor(areas, dtype=torch.float32)
            target['iscrowd'] = torch.tensor(iscrowd, dtype=torch.int64)
            
            if self.return_masks and masks:
                target['masks'] = torch.stack(masks)
        else:
            # Empty annotations
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
            target['areas'] = torch.zeros(0, dtype=torch.float32)
            target['iscrowd'] = torch.zeros(0, dtype=torch.int64)
            
            if self.return_masks:
                target['masks'] = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
        
        return target
    
    def _parse_mask(self, segmentation: Union[List, Dict], img_info: Dict) -> torch.Tensor:
        """Parse segmentation mask."""
        if isinstance(segmentation, list):
            # Polygon format
            mask = coco_mask.frPyObjects(segmentation, img_info['height'], img_info['width'])
            mask = coco_mask.decode(mask)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = mask.any(axis=2)
        else:
            # RLE format
            mask = coco_mask.decode(segmentation)
        
        return torch.tensor(mask, dtype=torch.uint8)


@register()
class LVISDetection(data.Dataset):
    """LVIS Detection Dataset for RT-DETRv3.
    
    Compatible with PaddlePaddle RT-DETRv3 data format.
    """
    
    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        return_masks: bool = False,
        allow_empty: bool = False,
        data_fields: List[str] = None,
        sample_num: int = -1,
        **kwargs
    ):
        """Initialize LVIS dataset.
        
        Args:
            image_dir: Directory containing images
            ann_file: Path to annotation file
            transforms: Transform pipeline
            return_masks: Whether to return segmentation masks
            allow_empty: Whether to allow empty annotations
            data_fields: List of data fields to load
            sample_num: Number of samples to load (-1 for all)
        """
        try:
            from lvis import LVIS
        except ImportError:
            raise ImportError("Please install lvis-api: pip install lvis")
        
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks
        self.allow_empty = allow_empty
        self.data_fields = data_fields or ['image', 'gt_bbox', 'gt_class']
        
        # Load LVIS annotations
        self.lvis = LVIS(ann_file)
        
        # Get image ids
        self.image_ids = list(self.lvis.imgs.keys())
        if sample_num > 0:
            self.image_ids = self.image_ids[:sample_num]
            
        # Filter out images without annotations if not allow_empty
        if not allow_empty:
            self.image_ids = [
                img_id for img_id in self.image_ids
                if len(self.lvis.get_ann_ids(img_ids=[img_id])) > 0
            ]
        
        # Category mapping
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label2cat = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index."""
        img_id = self.image_ids[idx]
        img_info = self.lvis.load_imgs([img_id])[0]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        anns = self.lvis.load_anns(ann_ids)
        
        # Parse annotations
        target = self._parse_annotations(anns, img_info)
        
        # Create sample
        sample = {
            'image': image,
            'image_id': img_id,
            'orig_size': torch.tensor([img_info['height'], img_info['width']], dtype=torch.int64),
            'size': torch.tensor([img_info['height'], img_info['width']], dtype=torch.int64),
        }
        
        # Add target fields
        for key, value in target.items():
            sample[key] = value
        
        # Apply transforms
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample
    
    def _parse_annotations(self, anns: List[Dict], img_info: Dict) -> Dict[str, torch.Tensor]:
        """Parse LVIS annotations."""
        boxes = []
        labels = []
        areas = []
        masks = []
        
        for ann in anns:
            # Bounding box
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2] format
            box = [x, y, x + w, y + h]
            boxes.append(box)
            
            # Category label
            cat_id = ann['category_id']
            label = self.cat2label[cat_id]
            labels.append(label)
            
            # Area
            areas.append(ann['area'])
            
            # Segmentation mask
            if self.return_masks and 'segmentation' in ann:
                mask = self._parse_mask(ann['segmentation'], img_info)
                masks.append(mask)
        
        # Convert to tensors
        target = {}
        if boxes:
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)
            target['areas'] = torch.tensor(areas, dtype=torch.float32)
            
            if self.return_masks and masks:
                target['masks'] = torch.stack(masks)
        else:
            # Empty annotations
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
            target['areas'] = torch.zeros(0, dtype=torch.float32)
            
            if self.return_masks:
                target['masks'] = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
        
        return target
    
    def _parse_mask(self, segmentation: Dict, img_info: Dict) -> torch.Tensor:
        """Parse segmentation mask."""
        mask = coco_mask.decode(segmentation)
        return torch.tensor(mask, dtype=torch.uint8)


def build_dataset(dataset_cfg: Dict[str, Any]) -> data.Dataset:
    """Build dataset from configuration."""
    dataset_type = dataset_cfg.get('type', 'CocoDetection')
    
    if dataset_type == 'CocoDetection':
        return CocoDetection(**dataset_cfg)
    elif dataset_type == 'LVISDetection':
        return LVISDetection(**dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
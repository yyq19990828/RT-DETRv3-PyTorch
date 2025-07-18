"""DataLoader implementation for RT-DETRv3 PyTorch."""

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, DistributedSampler
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from ..core.workspace import register


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Default collate function for RT-DETRv3."""
    batch_dict = {}
    
    # Handle images
    images = []
    for sample in batch:
        image = sample['image']
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        images.append(image)
    
    # Stack images
    batch_dict['images'] = torch.stack(images)
    
    # Handle targets
    targets = []
    for sample in batch:
        target = {}
        for key in ['boxes', 'labels', 'areas', 'iscrowd']:
            if key in sample:
                target[key] = sample[key]
        
        # Add additional fields
        for key in ['image_id', 'orig_size', 'size']:
            if key in sample:
                target[key] = sample[key]
        
        targets.append(target)
    
    batch_dict['targets'] = targets
    
    return batch_dict


@register()
class BatchImageCollateFuncion:
    """Batch image collate function with multi-scale support."""
    
    def __init__(self, scales: Optional[List[int]] = None, pad_to_max: bool = True):
        """Initialize batch collate function.
        
        Args:
            scales: List of scales for multi-scale training
            pad_to_max: Whether to pad images to max size in batch
        """
        self.scales = scales
        self.pad_to_max = pad_to_max
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch with multi-scale support."""
        batch_dict = {}
        
        # Handle images
        images = []
        max_height = 0
        max_width = 0
        
        for sample in batch:
            image = sample['image']
            if isinstance(image, Image.Image):
                image = TF.to_tensor(image)
            
            # Multi-scale training
            if self.scales is not None and self.training:
                scale = np.random.choice(self.scales)
                h, w = image.shape[-2:]
                scale_factor = scale / min(h, w)
                new_h = int(h * scale_factor)
                new_w = int(w * scale_factor)
                
                image = TF.resize(image, (new_h, new_w))
            
            images.append(image)
            
            # Track max dimensions
            if self.pad_to_max:
                max_height = max(max_height, image.shape[-2])
                max_width = max(max_width, image.shape[-1])
        
        # Pad images to same size
        if self.pad_to_max:
            padded_images = []
            for image in images:
                h, w = image.shape[-2:]
                pad_h = max_height - h
                pad_w = max_width - w
                
                if pad_h > 0 or pad_w > 0:
                    image = torch.nn.functional.pad(
                        image, 
                        (0, pad_w, 0, pad_h), 
                        mode='constant', 
                        value=0
                    )
                padded_images.append(image)
            
            batch_dict['images'] = torch.stack(padded_images)
        else:
            batch_dict['images'] = images
        
        # Handle targets
        targets = []
        for sample in batch:
            target = {}
            for key in ['boxes', 'labels', 'areas', 'iscrowd']:
                if key in sample:
                    target[key] = sample[key]
            
            # Add additional fields
            for key in ['image_id', 'orig_size', 'size']:
                if key in sample:
                    target[key] = sample[key]
            
            targets.append(target)
        
        batch_dict['targets'] = targets
        
        return batch_dict


@register()
class AspectRatioGroupedDataset(data.Dataset):
    """Dataset wrapper that groups images by aspect ratio."""
    
    def __init__(self, dataset: data.Dataset, group_size: int = 8):
        """Initialize aspect ratio grouped dataset.
        
        Args:
            dataset: Base dataset
            group_size: Size of each group
        """
        self.dataset = dataset
        self.group_size = group_size
        
        # Group images by aspect ratio
        self.groups = self._group_images()
    
    def _group_images(self) -> List[List[int]]:
        """Group images by aspect ratio."""
        aspect_ratios = []
        
        # Calculate aspect ratios
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            if 'orig_size' in sample:
                h, w = sample['orig_size']
                aspect_ratio = w / h
            else:
                image = sample['image']
                if isinstance(image, Image.Image):
                    w, h = image.size
                else:
                    h, w = image.shape[-2:]
                aspect_ratio = w / h
            
            aspect_ratios.append(aspect_ratio)
        
        # Sort by aspect ratio
        indices = sorted(range(len(aspect_ratios)), key=lambda i: aspect_ratios[i])
        
        # Group into batches
        groups = []
        for i in range(0, len(indices), self.group_size):
            groups.append(indices[i:i + self.group_size])
        
        return groups
    
    def __len__(self) -> int:
        return len(self.groups)
    
    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        """Get group of samples."""
        group_indices = self.groups[idx]
        return [self.dataset[i] for i in group_indices]


@register()
class RTDETRDataLoader(DataLoader):
    """Custom DataLoader for RT-DETRv3."""
    
    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        distributed: bool = False,
        aspect_ratio_group_size: Optional[int] = None,
        **kwargs
    ):
        """Initialize RT-DETR DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            collate_fn: Collate function
            distributed: Whether to use distributed training
            aspect_ratio_group_size: Size of aspect ratio groups
        """
        # Handle aspect ratio grouping
        if aspect_ratio_group_size is not None:
            dataset = AspectRatioGroupedDataset(dataset, aspect_ratio_group_size)
            batch_size = 1  # Each "sample" is already a batch
        
        # Handle distributed training
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # Sampler handles shuffling
        
        # Default collate function
        if collate_fn is None:
            if aspect_ratio_group_size is not None:
                collate_fn = self._grouped_collate_fn
            else:
                collate_fn = collate_fn
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs
        )
    
    def _grouped_collate_fn(self, batch: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Collate function for aspect ratio grouped batches."""
        # Flatten the batch
        flat_batch = []
        for group in batch:
            flat_batch.extend(group)
        
        return collate_fn(flat_batch)


def build_dataloader(
    dataset: data.Dataset,
    dataloader_cfg: Dict[str, Any],
    distributed: bool = False,
    training: bool = True
) -> DataLoader:
    """Build dataloader from configuration.
    
    Args:
        dataset: Dataset to load from
        dataloader_cfg: DataLoader configuration
        distributed: Whether to use distributed training
        training: Whether this is for training
    
    Returns:
        Configured DataLoader
    """
    dataloader_type = dataloader_cfg.get('type', 'DataLoader')
    
    # Extract configuration
    batch_size = dataloader_cfg.get('batch_size', 1)
    shuffle = dataloader_cfg.get('shuffle', training)
    num_workers = dataloader_cfg.get('num_workers', 0)
    pin_memory = dataloader_cfg.get('pin_memory', False)
    drop_last = dataloader_cfg.get('drop_last', training)
    
    # Handle collate function
    collate_fn_cfg = dataloader_cfg.get('collate_fn', None)
    if collate_fn_cfg is not None:
        collate_fn_type = collate_fn_cfg.get('type', 'BatchImageCollateFuncion')
        if collate_fn_type == 'BatchImageCollateFuncion':
            collate_fn = BatchImageCollateFuncion(**collate_fn_cfg)
        else:
            collate_fn = globals().get(collate_fn_type)(**collate_fn_cfg)
    else:
        collate_fn = None
    
    # Handle aspect ratio grouping
    aspect_ratio_group_size = dataloader_cfg.get('aspect_ratio_group_size', None)
    
    if dataloader_type == 'RTDETRDataLoader':
        return RTDETRDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            distributed=distributed,
            aspect_ratio_group_size=aspect_ratio_group_size
        )
    else:
        # Handle distributed training
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn
        )


class DatasetMixin:
    """Mixin class for dataset utilities."""
    
    @staticmethod
    def convert_to_coco_format(annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Convert annotations to COCO format."""
        # This is a placeholder for annotation format conversion
        return annotations
    
    @staticmethod
    def filter_empty_annotations(annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out empty annotations."""
        if 'boxes' in annotations:
            boxes = annotations['boxes']
            # Remove boxes with zero area
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = areas > 0
            
            filtered_annotations = {}
            for key, value in annotations.items():
                if isinstance(value, torch.Tensor) and len(value) == len(boxes):
                    filtered_annotations[key] = value[keep]
                else:
                    filtered_annotations[key] = value
            
            return filtered_annotations
        
        return annotations
    
    @staticmethod
    def clip_boxes_to_image(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Clip boxes to image boundaries."""
        height, width = image_size
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, height)
        return boxes
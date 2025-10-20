"""
Data transforms for RT-DETRv3 PyTorch

Augmentation pipeline compatible with detection tasks.
"""

import random
from typing import Any, Dict, List, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Any, Dict]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image to Tensor"""
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image with mean and std"""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    """Resize image to fixed size"""
    
    def __init__(self, size: List[int]):
        self.size = size  # [H, W]
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        orig_h, orig_w = image.height, image.width
        new_h, new_w = self.size
        
        # Resize image
        image = F.resize(image, self.size)
        
        # Update target
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            # Scale boxes: [x, y, w, h]
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes[:, 0] *= scale_x  # x
            boxes[:, 1] *= scale_y  # y
            boxes[:, 2] *= scale_x  # w
            boxes[:, 3] *= scale_y  # h
            target['boxes'] = boxes
        
        target['size'] = torch.tensor([new_h, new_w])
        
        return image, target


class RandomResize:
    """Randomly resize image to one of the given scales"""
    
    def __init__(self, scales: List[int]):
        self.scales = scales
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        size = random.choice(self.scales)
        resize = Resize([size, size])
        return resize(image, target)


class RandomHorizontalFlip:
    """Randomly flip image horizontally"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        if random.random() < self.p:
            image = F.hflip(image)
            
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                w = image.width
                # Flip boxes: [x, y, w, h] -> [w - x - w, y, w, h]
                boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]
                target['boxes'] = boxes
        
        return image, target


class RandomCrop:
    """Randomly crop image"""
    
    def __init__(self, crop_size: List[int]):
        self.crop_size = crop_size  # [H, W]
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        orig_w, orig_h = image.width, image.height
        crop_h, crop_w = self.crop_size
        
        # Skip if image is smaller than crop size
        if orig_h < crop_h or orig_w < crop_w:
            resize = Resize(self.crop_size)
            return resize(image, target)
        
        # Random crop coordinates
        top = random.randint(0, orig_h - crop_h)
        left = random.randint(0, orig_w - crop_w)
        
        # Crop image
        image = F.crop(image, top, left, crop_h, crop_w)
        
        # Adjust boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            
            # Adjust coordinates
            boxes[:, 0] -= left  # x
            boxes[:, 1] -= top   # y
            
            # Filter boxes that are outside crop
            keep = (
                (boxes[:, 0] + boxes[:, 2] > 0) &
                (boxes[:, 1] + boxes[:, 3] > 0) &
                (boxes[:, 0] < crop_w) &
                (boxes[:, 1] < crop_h)
            )
            
            boxes = boxes[keep]
            
            # Clip boxes to crop boundaries
            boxes[:, 0] = torch.clamp(boxes[:, 0], 0, crop_w)
            boxes[:, 1] = torch.clamp(boxes[:, 1], 0, crop_h)
            boxes[:, 2] = torch.clamp(boxes[:, 2], 0, crop_w - boxes[:, 0])
            boxes[:, 3] = torch.clamp(boxes[:, 3], 0, crop_h - boxes[:, 1])
            
            target['boxes'] = boxes
            target['labels'] = target['labels'][keep]
            
            if 'area' in target:
                target['area'] = target['area'][keep]
            if 'iscrowd' in target:
                target['iscrowd'] = target['iscrowd'][keep]
        
        target['size'] = torch.tensor([crop_h, crop_w])
        
        return image, target


def build_transforms(cfg: Dict, is_train: bool = True) -> Compose:
    """
    Build transforms from config.
    
    Args:
        cfg: Config dict with 'transforms_train' or 'transforms_val'
        is_train: Whether to build training or validation transforms
        
    Returns:
        Composed transforms
    """
    transforms_cfg = cfg.get('transforms_train' if is_train else 'transforms_val', [])
    
    transforms = []
    for t_cfg in transforms_cfg:
        t_type = t_cfg['type']
        t_params = {k: v for k, v in t_cfg.items() if k != 'type'}
        
        if t_type == 'Resize':
            transforms.append(Resize(**t_params))
        elif t_type == 'RandomResize':
            transforms.append(RandomResize(**t_params))
        elif t_type == 'RandomHorizontalFlip':
            transforms.append(RandomHorizontalFlip(**t_params))
        elif t_type == 'RandomCrop':
            transforms.append(RandomCrop(**t_params))
        elif t_type == 'Normalize':
            transforms.append(Normalize(**t_params))
        elif t_type == 'ToTensor':
            transforms.append(ToTensor())
        else:
            raise ValueError(f"Unknown transform type: {t_type}")
    
    return Compose(transforms)

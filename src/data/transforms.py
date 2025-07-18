"""Data transforms for RT-DETRv3 PyTorch."""

import random
import math
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..core.workspace import register


@register()
class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[Dict[str, Any]]):
        """Initialize compose transform.
        
        Args:
            transforms: List of transform configurations
        """
        self.transforms = []
        for transform_cfg in transforms:
            transform_type = transform_cfg.pop('type')
            transform_cls = globals().get(transform_type)
            if transform_cls is None:
                raise ValueError(f"Unknown transform type: {transform_type}")
            self.transforms.append(transform_cls(**transform_cfg))
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms to sample."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample


@register()
class ToTensor:
    """Convert PIL Image to tensor."""
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert image to tensor."""
        image = sample['image']
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        sample['image'] = image
        return sample


@register()
class Normalize:
    """Normalize image with mean and std."""
    
    def __init__(self, mean: List[float], std: List[float]):
        """Initialize normalize transform.
        
        Args:
            mean: Mean values for each channel
            std: Standard deviation values for each channel
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize image."""
        image = sample['image']
        if isinstance(image, torch.Tensor):
            image = TF.normalize(image, self.mean, self.std)
        sample['image'] = image
        return sample


@register()
class Resize:
    """Resize image and adjust bounding boxes."""
    
    def __init__(self, size: Union[int, Tuple[int, int]], max_size: Optional[int] = None):
        """Initialize resize transform.
        
        Args:
            size: Target size (height, width) or single size
            max_size: Maximum size for the longer edge
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.max_size = max_size
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Resize image and adjust annotations."""
        image = sample['image']
        orig_size = sample.get('orig_size', torch.tensor([image.height, image.width]))
        
        # Calculate new size
        if self.max_size is not None:
            new_size = self._get_size_with_aspect_ratio(image.size, self.size, self.max_size)
        else:
            new_size = self.size
        
        # Resize image
        if isinstance(image, Image.Image):
            image = image.resize(new_size, Image.BILINEAR)
        elif isinstance(image, torch.Tensor):
            image = F.interpolate(image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False).squeeze(0)
        
        # Update sample
        sample['image'] = image
        sample['size'] = torch.tensor([new_size[1], new_size[0]], dtype=torch.int64)
        
        # Scale bounding boxes
        if 'boxes' in sample:
            boxes = sample['boxes']
            scale_x = new_size[0] / orig_size[1].item()
            scale_y = new_size[1] / orig_size[0].item()
            
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            sample['boxes'] = boxes
        
        return sample
    
    def _get_size_with_aspect_ratio(self, image_size: Tuple[int, int], 
                                   target_size: Tuple[int, int], 
                                   max_size: int) -> Tuple[int, int]:
        """Calculate new size while maintaining aspect ratio."""
        w, h = image_size
        target_h, target_w = target_size
        
        # Calculate scale factors
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)
        
        # Apply max_size constraint
        if max_size is not None:
            scale = min(scale, max_size / max(w, h))
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return (new_w, new_h)


@register()
class RandomResize:
    """Random resize with multiple size options."""
    
    def __init__(self, sizes: List[int], max_size: Optional[int] = None):
        """Initialize random resize transform.
        
        Args:
            sizes: List of possible sizes
            max_size: Maximum size for the longer edge
        """
        self.sizes = sizes
        self.max_size = max_size
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Random resize image."""
        size = random.choice(self.sizes)
        resize_transform = Resize(size, self.max_size)
        return resize_transform(sample)


@register()
class RandomHorizontalFlip:
    """Random horizontal flip."""
    
    def __init__(self, prob: float = 0.5):
        """Initialize random horizontal flip.
        
        Args:
            prob: Probability of applying flip
        """
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Random horizontal flip."""
        if random.random() < self.prob:
            image = sample['image']
            
            # Flip image
            if isinstance(image, Image.Image):
                image = TF.hflip(image)
            elif isinstance(image, torch.Tensor):
                image = TF.hflip(image)
            
            sample['image'] = image
            
            # Flip bounding boxes
            if 'boxes' in sample:
                boxes = sample['boxes']
                width = sample['size'][1].item()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                sample['boxes'] = boxes
        
        return sample


@register()
class RandomVerticalFlip:
    """Random vertical flip."""
    
    def __init__(self, prob: float = 0.5):
        """Initialize random vertical flip.
        
        Args:
            prob: Probability of applying flip
        """
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Random vertical flip."""
        if random.random() < self.prob:
            image = sample['image']
            
            # Flip image
            if isinstance(image, Image.Image):
                image = TF.vflip(image)
            elif isinstance(image, torch.Tensor):
                image = TF.vflip(image)
            
            sample['image'] = image
            
            # Flip bounding boxes
            if 'boxes' in sample:
                boxes = sample['boxes']
                height = sample['size'][0].item()
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                sample['boxes'] = boxes
        
        return sample


@register()
class RandomCrop:
    """Random crop with bbox adjustment."""
    
    def __init__(self, size: Tuple[int, int], pad_if_needed: bool = True):
        """Initialize random crop.
        
        Args:
            size: Target crop size (height, width)
            pad_if_needed: Whether to pad if image is smaller than crop size
        """
        self.size = size
        self.pad_if_needed = pad_if_needed
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Random crop image."""
        image = sample['image']
        
        # Get image size
        if isinstance(image, Image.Image):
            img_width, img_height = image.size
        elif isinstance(image, torch.Tensor):
            img_height, img_width = image.shape[-2:]
        
        target_height, target_width = self.size
        
        # Pad if needed
        if self.pad_if_needed:
            if img_width < target_width or img_height < target_height:
                pad_left = max(0, (target_width - img_width) // 2)
                pad_top = max(0, (target_height - img_height) // 2)
                pad_right = max(0, target_width - img_width - pad_left)
                pad_bottom = max(0, target_height - img_height - pad_top)
                
                if isinstance(image, Image.Image):
                    image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
                elif isinstance(image, torch.Tensor):
                    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0)
                
                # Update image size
                if isinstance(image, Image.Image):
                    img_width, img_height = image.size
                elif isinstance(image, torch.Tensor):
                    img_height, img_width = image.shape[-2:]
        
        # Random crop
        if img_width > target_width or img_height > target_height:
            left = random.randint(0, max(0, img_width - target_width))
            top = random.randint(0, max(0, img_height - target_height))
            
            if isinstance(image, Image.Image):
                image = image.crop((left, top, left + target_width, top + target_height))
            elif isinstance(image, torch.Tensor):
                image = image[..., top:top + target_height, left:left + target_width]
            
            sample['image'] = image
            sample['size'] = torch.tensor([target_height, target_width], dtype=torch.int64)
            
            # Adjust bounding boxes
            if 'boxes' in sample:
                boxes = sample['boxes']
                boxes[:, [0, 2]] -= left
                boxes[:, [1, 3]] -= top
                
                # Clip boxes to image boundaries
                boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, target_width)
                boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, target_height)
                
                # Remove boxes with zero area
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                keep = area > 0
                
                sample['boxes'] = boxes[keep]
                if 'labels' in sample:
                    sample['labels'] = sample['labels'][keep]
                if 'areas' in sample:
                    sample['areas'] = sample['areas'][keep]
        
        return sample


@register()
class ColorJitter:
    """Color jitter transform."""
    
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, 
                 saturation: float = 0.2, hue: float = 0.1):
        """Initialize color jitter.
        
        Args:
            brightness: Brightness factor
            contrast: Contrast factor
            saturation: Saturation factor
            hue: Hue factor
        """
        self.transform = T.ColorJitter(
            brightness=brightness, 
            contrast=contrast, 
            saturation=saturation, 
            hue=hue
        )
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply color jitter."""
        image = sample['image']
        if isinstance(image, Image.Image):
            image = self.transform(image)
        sample['image'] = image
        return sample


@register()
class GaussianBlur:
    """Gaussian blur transform."""
    
    def __init__(self, radius: float = 1.0, prob: float = 0.5):
        """Initialize Gaussian blur.
        
        Args:
            radius: Blur radius
            prob: Probability of applying blur
        """
        self.radius = radius
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Gaussian blur."""
        if random.random() < self.prob:
            image = sample['image']
            if isinstance(image, Image.Image):
                image = image.filter(ImageFilter.GaussianBlur(radius=self.radius))
            sample['image'] = image
        return sample


@register()
class RandomPhotometricDistort:
    """Random photometric distortion."""
    
    def __init__(self, brightness_delta: float = 32, contrast_range: Tuple[float, float] = (0.5, 1.5),
                 saturation_range: Tuple[float, float] = (0.5, 1.5), hue_delta: float = 18):
        """Initialize random photometric distort.
        
        Args:
            brightness_delta: Max brightness delta
            contrast_range: Contrast range
            saturation_range: Saturation range
            hue_delta: Max hue delta
        """
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random photometric distortion."""
        image = sample['image']
        
        if isinstance(image, torch.Tensor):
            # Convert to PIL for easier manipulation
            image = TF.to_pil_image(image)
        
        if isinstance(image, Image.Image):
            # Random brightness
            if random.random() < 0.5:
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                image = TF.adjust_brightness(image, 1 + delta / 255.0)
            
            # Random contrast
            if random.random() < 0.5:
                contrast = random.uniform(*self.contrast_range)
                image = TF.adjust_contrast(image, contrast)
            
            # Random saturation
            if random.random() < 0.5:
                saturation = random.uniform(*self.saturation_range)
                image = TF.adjust_saturation(image, saturation)
            
            # Random hue
            if random.random() < 0.5:
                hue = random.uniform(-self.hue_delta, self.hue_delta) / 360.0
                image = TF.adjust_hue(image, hue)
        
        sample['image'] = image
        return sample


@register()
class Mixup:
    """Mixup augmentation."""
    
    def __init__(self, alpha: float = 0.2):
        """Initialize mixup.
        
        Args:
            alpha: Mixup alpha parameter
        """
        self.alpha = alpha
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mixup (requires batch processing)."""
        # This is typically applied at batch level
        return sample


@register()
class PadToSize:
    """Pad image to target size."""
    
    def __init__(self, size: Tuple[int, int], fill: int = 0):
        """Initialize pad to size.
        
        Args:
            size: Target size (height, width)
            fill: Fill value for padding
        """
        self.size = size
        self.fill = fill
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Pad image to target size."""
        image = sample['image']
        target_height, target_width = self.size
        
        if isinstance(image, Image.Image):
            img_width, img_height = image.size
            if img_width < target_width or img_height < target_height:
                pad_left = (target_width - img_width) // 2
                pad_top = (target_height - img_height) // 2
                pad_right = target_width - img_width - pad_left
                pad_bottom = target_height - img_height - pad_top
                
                image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        elif isinstance(image, torch.Tensor):
            _, img_height, img_width = image.shape
            if img_width < target_width or img_height < target_height:
                pad_left = (target_width - img_width) // 2
                pad_top = (target_height - img_height) // 2
                pad_right = target_width - img_width - pad_left
                pad_bottom = target_height - img_height - pad_top
                
                image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=self.fill)
        
        sample['image'] = image
        sample['size'] = torch.tensor([target_height, target_width], dtype=torch.int64)
        return sample


def build_transforms(transform_cfg: Dict[str, Any]) -> Compose:
    """Build transforms from configuration."""
    if isinstance(transform_cfg, dict) and 'type' in transform_cfg:
        # Single transform
        return Compose([transform_cfg])
    elif isinstance(transform_cfg, list):
        # List of transforms
        return Compose(transform_cfg)
    else:
        raise ValueError(f"Invalid transform configuration: {transform_cfg}")
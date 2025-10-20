"""
Batch collation for RT-DETRv3 PyTorch

Handle variable-length annotations and create properly batched tensors.
"""

from typing import Any, Dict, List, Tuple

import torch


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Collate batch of images and targets.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        Tuple of (batched_images, targets) where:
            - batched_images: (B, C, H, W) tensor
            - targets: List of target dicts (one per image)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, targets


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Create batched tensor from list of tensors with potentially different sizes.
    Pads tensors to max size in batch.
    
    Args:
        tensor_list: List of tensors
        
    Returns:
        Batched tensor with padding
    """
    if len(tensor_list) == 0:
        return torch.empty(0)
    
    # Get max size
    max_size = [max(s) for s in zip(*[img.shape for img in tensor_list])]
    
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    
    # Create padded tensor
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    
    # Copy images into tensor
    for img, pad_img in zip(tensor_list, tensor):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    
    return tensor

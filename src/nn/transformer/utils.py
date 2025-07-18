"""
Utility functions for RT-DETRv3 Transformer
"""

import math
import copy
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse sigmoid function"""
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_sine_pos_embed(pos_tensor: torch.Tensor, 
                      num_pos_feats: int = 128, 
                      temperature: float = 10000) -> torch.Tensor:
    """Generate sine positional embedding"""
    scale = 2 * math.pi
    pos_tensor = pos_tensor * scale
    
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    
    pos_x = pos_tensor[..., 0, None] / dim_t
    pos_y = pos_tensor[..., 1, None] / dim_t
    
    pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
    pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
    
    return torch.cat([pos_y, pos_x], dim=-1)


def get_contrastive_denoising_training_group(
    gt_meta: Dict[str, Any],
    num_classes: int,
    num_queries: int,
    class_embed: torch.Tensor,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Get contrastive denoising training group"""
    device = class_embed.device
    
    if 'gt_bbox' not in gt_meta or 'gt_class' not in gt_meta:
        # Return empty tensors if no GT available
        batch_size = len(gt_meta.get('gt_bbox', [{}]))
        return (
            torch.zeros(batch_size, num_queries, class_embed.shape[-1], device=device),
            torch.zeros(batch_size, num_queries, 4, device=device),
            torch.zeros(num_queries, num_queries, dtype=torch.bool, device=device),
            {'dn_num_split': [0, num_queries]}
        )
    
    gt_bboxes = gt_meta['gt_bbox']
    gt_labels = gt_meta['gt_class']
    batch_size = len(gt_bboxes)
    
    # Initialize output tensors
    denoising_classes = []
    denoising_bboxes = []
    attn_masks = []
    
    max_gt_num = max(len(bbox) for bbox in gt_bboxes) if gt_bboxes else 0
    
    for batch_idx in range(batch_size):
        gt_bbox = gt_bboxes[batch_idx]
        gt_label = gt_labels[batch_idx]
        
        if len(gt_bbox) == 0:
            # No ground truth objects
            denoising_classes.append(torch.zeros(num_queries, class_embed.shape[-1], device=device))
            denoising_bboxes.append(torch.zeros(num_queries, 4, device=device))
            continue
        
        # Convert to tensors
        gt_bbox = torch.tensor(gt_bbox, device=device, dtype=torch.float32)
        gt_label = torch.tensor(gt_label, device=device, dtype=torch.long)
        
        # Generate positive and negative samples
        positive_idx = torch.randint(0, len(gt_bbox), (num_denoising // 2,), device=device)
        negative_idx = torch.randint(0, len(gt_bbox), (num_denoising // 2,), device=device)
        
        # Positive samples (with small noise)
        pos_bboxes = gt_bbox[positive_idx]
        pos_labels = gt_label[positive_idx]
        
        # Add noise to positive bboxes
        pos_bboxes = pos_bboxes + torch.randn_like(pos_bboxes) * box_noise_scale * 0.1
        pos_bboxes = pos_bboxes.clamp(0, 1)
        
        # Negative samples (with large noise)
        neg_bboxes = gt_bbox[negative_idx]
        neg_labels = gt_label[negative_idx]
        
        # Add large noise to negative bboxes
        neg_bboxes = neg_bboxes + torch.randn_like(neg_bboxes) * box_noise_scale
        neg_bboxes = neg_bboxes.clamp(0, 1)
        
        # Randomly change some negative labels
        label_noise_mask = torch.rand(len(neg_labels), device=device) < label_noise_ratio
        neg_labels[label_noise_mask] = torch.randint(0, num_classes, (label_noise_mask.sum(),), device=device)
        
        # Combine positive and negative samples
        dn_bboxes = torch.cat([pos_bboxes, neg_bboxes], dim=0)
        dn_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # Pad to num_queries
        if len(dn_bboxes) < num_queries:
            pad_size = num_queries - len(dn_bboxes)
            dn_bboxes = torch.cat([dn_bboxes, torch.zeros(pad_size, 4, device=device)], dim=0)
            dn_labels = torch.cat([dn_labels, torch.zeros(pad_size, dtype=torch.long, device=device)], dim=0)
        elif len(dn_bboxes) > num_queries:
            dn_bboxes = dn_bboxes[:num_queries]
            dn_labels = dn_labels[:num_queries]
        
        # Convert labels to embeddings
        dn_class_embed = class_embed[dn_labels]
        
        denoising_classes.append(dn_class_embed)
        denoising_bboxes.append(dn_bboxes)
    
    # Stack batch
    denoising_classes = torch.stack(denoising_classes, dim=0)
    denoising_bboxes = torch.stack(denoising_bboxes, dim=0)
    
    # Create attention mask
    attn_mask = torch.zeros(num_queries, num_queries, dtype=torch.bool, device=device)
    
    # Metadata
    dn_meta = {
        'dn_num_split': [num_denoising, num_queries - num_denoising]
    }
    
    return denoising_classes, denoising_bboxes, attn_mask, dn_meta


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Get N clones of a module"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str) -> nn.Module:
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    elif activation == 'glu':
        return F.glu
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'selu':
        return F.selu
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_position_encoding(hidden_dim: int, position_embedding: str = 'sine'):
    """Build position encoding"""
    if position_embedding in ('v2', 'sine'):
        # TODO: the v2 is not implemented yet
        N_steps = hidden_dim // 2
        if position_embedding == 'v2':
            # TODO: implement v2
            pass
        else:
            # Default sine position encoding
            pass
    else:
        raise ValueError(f"not supported {position_embedding}")


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """Create nested tensor from tensor list"""
    # Simple implementation - just stack if same size
    if len(tensor_list) == 1:
        return tensor_list[0]
    
    # Check if all tensors have same shape
    first_shape = tensor_list[0].shape
    if all(tensor.shape == first_shape for tensor in tensor_list):
        return torch.stack(tensor_list, dim=0)
    
    # Different shapes - need padding
    max_size = tuple(max(tensor.shape[i] for tensor in tensor_list) for i in range(len(first_shape)))
    batch_size = len(tensor_list)
    
    # Create padded tensor
    padded_tensor = torch.zeros(batch_size, *max_size, dtype=tensor_list[0].dtype, device=tensor_list[0].device)
    
    for i, tensor in enumerate(tensor_list):
        # Copy tensor to padded location
        slices = tuple(slice(0, s) for s in tensor.shape)
        padded_tensor[i][slices] = tensor
    
    return padded_tensor


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Compute accuracy for specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert box format from (cx, cy, w, h) to (x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert box format from (x1, y1, x2, y2) to (cx, cy, w, h)"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU between two sets of boxes"""
    # Input boxes should be in (x1, y1, x2, y2) format
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    # Compute IoU
    iou, union = box_iou(boxes1, boxes2)
    
    # Compute the area of the smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    
    return iou - (area - union) / area


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute IoU between two sets of boxes"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou, union


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute area of boxes"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def is_dist_avail_and_initialized() -> bool:
    """Check if distributed training is available and initialized"""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Get world size for distributed training"""
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    """Get rank for distributed training"""
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    """Check if current process is main process"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Save only on master process"""
    if is_main_process():
        torch.save(*args, **kwargs)


def interpolate(input: torch.Tensor, size: Optional[int] = None, scale_factor: Optional[float] = None, 
                mode: str = 'nearest', align_corners: Optional[bool] = None) -> torch.Tensor:
    """Wrapper for F.interpolate"""
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class NestedTensor:
    """Nested tensor class for handling batches of different sized tensors"""
    
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor):
        self.tensors = tensors
        self.mask = mask
    
    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)
    
    def decompose(self):
        return self.tensors, self.mask
    
    def __repr__(self):
        return str(self.tensors)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for data loader"""
    # This is a simplified version - you might need to adapt based on your data format
    batch_size = len(batch)
    
    # Handle images
    images = [item['image'] for item in batch]
    images = nested_tensor_from_tensor_list(images)
    
    # Handle targets
    targets = [item.get('target', {}) for item in batch]
    
    return {
        'images': images,
        'targets': targets
    }
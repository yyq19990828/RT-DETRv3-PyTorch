"""
RT-DETR Head for detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

from ...core import register
from ..transformer.layers import MLP
from ..transformer.utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


@register()
class RTDETRHead(nn.Module):
    """RT-DETR Detection Head"""
    
    def __init__(self,
                 num_classes: int = 80,
                 hidden_dim: int = 256,
                 num_queries: int = 300,
                 num_levels: int = 3,
                 num_decoder_layers: int = 6,
                 aux_loss: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_levels = num_levels
        self.num_decoder_layers = num_decoder_layers
        self.aux_loss = aux_loss
        
        # Classification heads for each decoder layer
        self.class_embed = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)
        ])
        
        # Bbox regression heads for each decoder layer
        self.bbox_embed = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers)
        ])
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        # Initialize classification heads
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        
        for class_head in self.class_embed:
            nn.init.constant_(class_head.bias, bias_value)
        
        # Initialize bbox regression heads
        for bbox_head in self.bbox_embed:
            nn.init.constant_(bbox_head.layers[-1].weight, 0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                reference_points: torch.Tensor,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                valid_ratios: Optional[torch.Tensor] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            hidden_states: Output from transformer decoder [num_layers, bs, num_queries, hidden_dim]
            reference_points: Reference points [bs, num_queries, 2] or [bs, num_queries, 4]
            spatial_shapes: Spatial shapes of feature maps
            level_start_index: Level start indices
            valid_ratios: Valid ratios for each level
            
        Returns:
            List of prediction dictionaries for each decoder layer
        """
        outputs = []
        
        for layer_idx in range(self.num_decoder_layers):
            layer_hidden_states = hidden_states[layer_idx]  # [bs, num_queries, hidden_dim]
            layer_reference_points = reference_points[layer_idx]  # [bs, num_queries, 2/4]
            
            # Classification prediction
            outputs_class = self.class_embed[layer_idx](layer_hidden_states)  # [bs, num_queries, num_classes]
            
            # Bbox regression
            bbox_delta = self.bbox_embed[layer_idx](layer_hidden_states)  # [bs, num_queries, 4]
            
            # Apply sigmoid to bbox delta and add to reference points
            if layer_reference_points.shape[-1] == 4:
                # Reference points are in (cx, cy, w, h) format
                outputs_coord = layer_reference_points + bbox_delta
            else:
                # Reference points are in (cx, cy) format
                # Convert to (cx, cy, w, h) format
                assert layer_reference_points.shape[-1] == 2
                bbox_delta = bbox_delta.sigmoid()
                outputs_coord = torch.cat([layer_reference_points, bbox_delta[..., 2:]], dim=-1)
            
            outputs_coord = outputs_coord.sigmoid()
            
            outputs.append({
                'pred_logits': outputs_class,
                'pred_boxes': outputs_coord
            })
        
        return outputs


@register()
class RTDETRPostProcessor(nn.Module):
    """Post-processor for RT-DETR predictions"""
    
    def __init__(self,
                 num_classes: int = 80,
                 num_queries: int = 300,
                 use_focal_loss: bool = True,
                 use_nms: bool = False,
                 nms_threshold: float = 0.5,
                 score_threshold: float = 0.0,
                 max_detections: int = 300):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.use_focal_loss = use_focal_loss
        self.use_nms = use_nms
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
    
    def forward(self, 
                predictions: List[Dict[str, torch.Tensor]],
                target_sizes: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process predictions
        
        Args:
            predictions: List of prediction dictionaries
            target_sizes: Target image sizes [batch_size, 2] (height, width)
            
        Returns:
            List of processed predictions for each image
        """
        # Use the last layer predictions
        pred_logits = predictions[-1]['pred_logits']  # [bs, num_queries, num_classes]
        pred_boxes = predictions[-1]['pred_boxes']    # [bs, num_queries, 4]
        
        batch_size = pred_logits.shape[0]
        
        # Convert logits to probabilities
        if self.use_focal_loss:
            # For focal loss, use sigmoid
            prob = pred_logits.sigmoid()
        else:
            # For standard cross-entropy, use softmax
            prob = F.softmax(pred_logits, dim=-1)
        
        # Get scores and labels
        scores, labels = prob.max(dim=-1)
        
        # Convert boxes to (x1, y1, x2, y2) format
        boxes = box_cxcywh_to_xyxy(pred_boxes)
        
        # Scale boxes to target sizes
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        results = []
        
        for i in range(batch_size):
            # Filter by score threshold
            keep = scores[i] > self.score_threshold
            
            batch_scores = scores[i][keep]
            batch_labels = labels[i][keep]
            batch_boxes = boxes[i][keep]
            
            # Apply NMS if needed
            if self.use_nms and len(batch_boxes) > 0:
                from torchvision.ops import nms
                
                # Apply NMS for each class
                final_boxes = []
                final_scores = []
                final_labels = []
                
                for class_id in range(self.num_classes):
                    class_mask = batch_labels == class_id
                    if class_mask.sum() == 0:
                        continue
                    
                    class_boxes = batch_boxes[class_mask]
                    class_scores = batch_scores[class_mask]
                    
                    # Apply NMS
                    nms_indices = nms(class_boxes, class_scores, self.nms_threshold)
                    
                    final_boxes.append(class_boxes[nms_indices])
                    final_scores.append(class_scores[nms_indices])
                    final_labels.append(torch.full_like(class_scores[nms_indices], class_id, dtype=torch.long))
                
                if len(final_boxes) > 0:
                    batch_boxes = torch.cat(final_boxes, dim=0)
                    batch_scores = torch.cat(final_scores, dim=0)
                    batch_labels = torch.cat(final_labels, dim=0)
                else:
                    batch_boxes = torch.zeros(0, 4, device=boxes.device)
                    batch_scores = torch.zeros(0, device=scores.device)
                    batch_labels = torch.zeros(0, dtype=torch.long, device=labels.device)
            
            # Limit number of detections
            if len(batch_boxes) > self.max_detections:
                top_indices = batch_scores.topk(self.max_detections)[1]
                batch_boxes = batch_boxes[top_indices]
                batch_scores = batch_scores[top_indices]
                batch_labels = batch_labels[top_indices]
            
            results.append({
                'scores': batch_scores,
                'labels': batch_labels,
                'boxes': batch_boxes
            })
        
        return results
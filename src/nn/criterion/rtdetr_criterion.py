"""
RT-DETR Criterion (Loss Function)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

from ...core import register
from ..transformer.utils import box_cxcywh_to_xyxy, generalized_box_iou, box_iou


class HungarianMatcher(nn.Module):
    """Hungarian matcher for bipartite matching"""
    
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform the matching
        
        Args:
            outputs: Dict with 'pred_logits' and 'pred_boxes'
            targets: List of target dictionaries
            
        Returns:
            List of (index_i, index_j) tuples
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        
        # Flatten predictions
        out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()  # [bs * num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [bs * num_queries, 4]
        
        # Concatenate all targets
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])
        
        if len(tgt_ids) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]
        
        # Compute classification cost
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        
        # Compute L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute GIoU cost
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # Hungarian matching
        indices = []
        sizes = [len(v['boxes']) for v in targets]
        
        for i, c in enumerate(C.split(sizes, -1)):
            # Use scipy.optimize.linear_sum_assignment if available
            try:
                from scipy.optimize import linear_sum_assignment
                indices_i, indices_j = linear_sum_assignment(c[i])
                indices.append((torch.tensor(indices_i, dtype=torch.long), torch.tensor(indices_j, dtype=torch.long)))
            except ImportError:
                # Fallback to greedy matching
                indices.append(self._greedy_matching(c[i]))
        
        return indices
    
    def _greedy_matching(self, cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy matching as fallback"""
        num_queries, num_targets = cost_matrix.shape
        
        if num_targets == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        
        # Simple greedy matching
        indices_i = []
        indices_j = []
        
        for j in range(num_targets):
            best_i = cost_matrix[:, j].argmin()
            indices_i.append(best_i)
            indices_j.append(j)
            # Set this row to inf to avoid reuse
            cost_matrix[best_i, :] = float('inf')
        
        return torch.tensor(indices_i, dtype=torch.long), torch.tensor(indices_j, dtype=torch.long)


@register()
class RTDETRCriterion(nn.Module):
    """RT-DETR Loss Criterion"""
    
    def __init__(self,
                 num_classes: int = 80,
                 matcher: Optional[HungarianMatcher] = None,
                 weight_dict: Optional[Dict[str, float]] = None,
                 losses: List[str] = ['labels', 'boxes', 'cardinality'],
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 bbox_loss_coef: float = 5.0,
                 giou_loss_coef: float = 2.0,
                 cls_loss_coef: float = 1.0,
                 aux_loss: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_loss = aux_loss
        
        if weight_dict is None:
            weight_dict = {
                'loss_ce': cls_loss_coef,
                'loss_bbox': bbox_loss_coef,
                'loss_giou': giou_loss_coef
            }
        
        self.weight_dict = weight_dict
        
        # Create empty weight dict for aux losses
        if aux_loss:
            aux_weight_dict = {}
            for i in range(6):  # Assuming 6 decoder layers
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)
    
    def loss_labels(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], 
                   indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: int) -> Dict[str, torch.Tensor]:
        """Classification loss (focal loss)"""
        assert 'pred_logits' in outputs
        
        src_logits = outputs['pred_logits']  # [bs, num_queries, num_classes]
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # Focal loss
        src_logits_flatten = src_logits.flatten(0, 1)  # [bs * num_queries, num_classes]
        target_classes_flatten = target_classes.flatten(0, 1)  # [bs * num_queries]
        
        # Create one-hot encoding for targets
        target_classes_onehot = torch.zeros_like(src_logits_flatten)
        target_classes_onehot.scatter_(1, target_classes_flatten.unsqueeze(1), 1)
        
        # Background class
        target_classes_onehot = target_classes_onehot[:, :self.num_classes]
        
        # Focal loss computation
        ce_loss = F.binary_cross_entropy_with_logits(src_logits_flatten, target_classes_onehot, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss_ce = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss
        
        return {'loss_ce': loss_ce.mean()}
    
    def loss_boxes(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], 
                  indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: int) -> Dict[str, torch.Tensor]:
        """Bounding box losses"""
        assert 'pred_boxes' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        losses = {}
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        # GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
    
    def loss_cardinality(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], 
                        indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: int) -> Dict[str, torch.Tensor]:
        """Cardinality loss"""
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        
        tgt_lengths = torch.tensor([len(v['labels']) for v in targets], device=device)
        
        # Count the number of predictions that are not "no-object" class
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        
        return {'cardinality_error': card_err}
    
    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get source permutation indices"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get target permutation indices"""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_loss(self, loss: str, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], 
                indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: int) -> Dict[str, torch.Tensor]:
        """Get specific loss"""
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        
        assert loss in loss_map, f'Loss {loss} not implemented'
        return loss_map[loss](outputs, targets, indices, num_boxes)
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Remove aux outputs from outputs
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boxes across all nodes, for normalization
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses
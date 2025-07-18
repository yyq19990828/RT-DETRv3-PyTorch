"""
RT-DETRv3 Main Model
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from ...core import register
from ...nn.backbone import ResNet, ResNetVD
from ...nn.neck import HybridEncoder
from ...nn.transformer import RTDETRTransformerV3
from ...nn.head import RTDETRHead, RTDETRPostProcessor
from ...nn.criterion import RTDETRCriterion


@register()
class RTDETRv3(nn.Module):
    """RT-DETRv3 Main Model"""
    
    __inject__ = ['backbone', 'neck', 'transformer', 'head', 'post_processor']
    
    def __init__(self,
                 backbone: Optional[nn.Module] = None,
                 neck: Optional[nn.Module] = None,
                 transformer: Optional[nn.Module] = None,
                 head: Optional[nn.Module] = None,
                 post_processor: Optional[nn.Module] = None,
                 num_classes: int = 80,
                 aux_loss: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        
        # Initialize components
        if backbone is None:
            backbone = ResNetVD(depth=50, return_idx=[1, 2, 3])
        
        if neck is None:
            neck = HybridEncoder(
                in_channels=[512, 1024, 2048],
                feat_strides=[8, 16, 32],
                hidden_dim=256,
                use_encoder_idx=[2]
            )
        
        if transformer is None:
            transformer = RTDETRTransformerV3(
                num_classes=num_classes,
                hidden_dim=256,
                num_queries=300,
                num_decoder_layers=6
            )
        
        if head is None:
            head = RTDETRHead(
                num_classes=num_classes,
                hidden_dim=256,
                num_queries=300,
                num_decoder_layers=6,
                aux_loss=aux_loss
            )
        
        if post_processor is None:
            post_processor = RTDETRPostProcessor(num_classes=num_classes)
        
        self.backbone = backbone
        self.neck = neck
        self.transformer = transformer
        self.head = head
        self.post_processor = post_processor
        
        # Initialize query embeddings
        self.query_embed = nn.Embedding(300, 256)
        
        # Initialize positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, 256))
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.query_embed.weight)
        nn.init.xavier_uniform_(self.pos_embed)
    
    def forward(self, 
                images: torch.Tensor,
                targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: Input images [batch_size, 3, H, W]
            targets: Ground truth targets (only during training)
            
        Returns:
            Dictionary of outputs
        """
        # Extract features with backbone
        features = self.backbone(images)
        
        # Apply neck (feature pyramid network)
        features = self.neck(features)
        
        # Prepare inputs for transformer
        batch_size = images.shape[0]
        device = images.device
        
        # Query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply transformer
        if self.training:
            # During training, pass targets for denoising
            transformer_outputs = self.transformer(
                features, 
                pad_mask=None, 
                gt_meta=self._prepare_targets(targets) if targets else None
            )
        else:
            # During inference
            transformer_outputs = self.transformer(features)
        
        # Unpack transformer outputs
        if len(transformer_outputs) == 5:
            out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits, dn_metas = transformer_outputs
        else:
            out_bboxes, out_logits = transformer_outputs[:2]
            enc_topk_bboxes = enc_topk_logits = dn_metas = None
        
        # Prepare outputs
        outputs = []
        for i in range(len(out_bboxes)):
            outputs.append({
                'pred_logits': out_logits[i],
                'pred_boxes': out_bboxes[i]
            })
        
        # Main output
        result = {
            'pred_logits': out_logits[-1],
            'pred_boxes': out_bboxes[-1]
        }
        
        # Add auxiliary outputs if training
        if self.training and self.aux_loss:
            result['aux_outputs'] = outputs[:-1]
        
        # Add encoder outputs
        if enc_topk_bboxes is not None:
            result['enc_outputs'] = {
                'pred_logits': enc_topk_logits,
                'pred_boxes': enc_topk_bboxes
            }
        
        # Add denoising meta information
        if dn_metas is not None:
            result['dn_metas'] = dn_metas
        
        return result
    
    def _prepare_targets(self, targets: List[Dict[str, torch.Tensor]]) -> Dict[str, List]:
        """Prepare targets for transformer"""
        if not targets:
            return {}
        
        gt_bbox = []
        gt_class = []
        
        for target in targets:
            if 'boxes' in target and 'labels' in target:
                gt_bbox.append(target['boxes'].cpu().numpy().tolist())
                gt_class.append(target['labels'].cpu().numpy().tolist())
            else:
                gt_bbox.append([])
                gt_class.append([])
        
        return {
            'gt_bbox': gt_bbox,
            'gt_class': gt_class
        }
    
    def predict(self, 
                images: torch.Tensor,
                image_sizes: Optional[torch.Tensor] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Prediction interface
        
        Args:
            images: Input images [batch_size, 3, H, W]
            image_sizes: Original image sizes [batch_size, 2] (height, width)
            
        Returns:
            List of predictions for each image
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(images)
            
            # Post-process predictions
            if image_sizes is None:
                image_sizes = torch.tensor([[images.shape[2], images.shape[3]]] * images.shape[0])
            
            predictions = self.post_processor([outputs], image_sizes)
        
        return predictions
    
    def deploy(self):
        """Convert model to deployment mode"""
        self.eval()
        
        # Convert RepVGG blocks to deploy mode
        for module in self.modules():
            if hasattr(module, 'convert_to_deploy'):
                module.convert_to_deploy()
        
        return self


# Factory functions for different model variants
def rtdetrv3_r18(**kwargs):
    """RT-DETRv3 with ResNet-18 backbone"""
    backbone = ResNetVD(depth=18, return_idx=[1, 2, 3])
    return RTDETRv3(backbone=backbone, **kwargs)


def rtdetrv3_r34(**kwargs):
    """RT-DETRv3 with ResNet-34 backbone"""
    backbone = ResNetVD(depth=34, return_idx=[1, 2, 3])
    return RTDETRv3(backbone=backbone, **kwargs)


def rtdetrv3_r50(**kwargs):
    """RT-DETRv3 with ResNet-50 backbone"""
    backbone = ResNetVD(depth=50, return_idx=[1, 2, 3])
    return RTDETRv3(backbone=backbone, **kwargs)


def rtdetrv3_r101(**kwargs):
    """RT-DETRv3 with ResNet-101 backbone"""
    backbone = ResNetVD(depth=101, return_idx=[1, 2, 3])
    return RTDETRv3(backbone=backbone, **kwargs)


# Register model variants
@register()
class RTDETRv3_R18(RTDETRv3):
    def __init__(self, **kwargs):
        backbone = ResNetVD(depth=18, return_idx=[1, 2, 3])
        super().__init__(backbone=backbone, **kwargs)


@register()
class RTDETRv3_R34(RTDETRv3):
    def __init__(self, **kwargs):
        backbone = ResNetVD(depth=34, return_idx=[1, 2, 3])
        super().__init__(backbone=backbone, **kwargs)


@register()
class RTDETRv3_R50(RTDETRv3):
    def __init__(self, **kwargs):
        backbone = ResNetVD(depth=50, return_idx=[1, 2, 3])
        super().__init__(backbone=backbone, **kwargs)


@register()
class RTDETRv3_R101(RTDETRv3):
    def __init__(self, **kwargs):
        backbone = ResNetVD(depth=101, return_idx=[1, 2, 3])
        super().__init__(backbone=backbone, **kwargs)
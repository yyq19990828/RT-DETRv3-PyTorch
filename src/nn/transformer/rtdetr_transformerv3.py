"""
RT-DETR Transformer v3 for PyTorch
Migrated from PaddlePaddle RT-DETRv3 implementation
"""

import math
import copy
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .layers import MLP, MultiHeadAttention, MSDeformableAttention
from .utils import get_sine_pos_embed, inverse_sigmoid, get_contrastive_denoising_training_group

__all__ = ['RTDETRTransformerV3', 'TransformerDecoderLayer', 'TransformerDecoder']


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer for RT-DETRv3"""
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 n_levels: int = 4,
                 n_points: int = 4):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (Multi-Scale Deformable Attention)
        self.cross_attn = MSDeformableAttention(
            d_model, nhead, n_levels, n_points, dropout
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = self._get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _get_activation_fn(self, activation: str):
        """Get activation function"""
        if activation == 'relu':
            return F.relu
        elif activation == 'gelu':
            return F.gelu
        elif activation == 'glu':
            return F.glu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        """Add positional embedding to tensor"""
        return tensor if pos is None else tensor + pos
    
    def forward(self,
                tgt: torch.Tensor,
                reference_points: torch.Tensor,
                memory: torch.Tensor,
                memory_spatial_shapes: torch.Tensor,
                memory_level_start_index: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                query_pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Self-attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        
        if attn_mask is not None:
            attn_mask = torch.where(
                attn_mask.bool(),
                torch.zeros_like(attn_mask, dtype=tgt.dtype),
                torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype)
            )
        
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_level_start_index,
            memory_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward network
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerDecoder(nn.Module):
    """Transformer decoder for RT-DETRv3"""
    
    def __init__(self,
                 hidden_dim: int,
                 decoder_layer: TransformerDecoderLayer,
                 num_layers: int,
                 eval_idx: int = -1):
        super().__init__()
        
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
    
    def forward(self,
                tgt: torch.Tensor,
                ref_points_unact: torch.Tensor,
                memory: torch.Tensor,
                memory_spatial_shapes: torch.Tensor,
                memory_level_start_index: torch.Tensor,
                bbox_head: nn.ModuleList,
                score_head: nn.ModuleList,
                query_pos_head: nn.Module,
                attn_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                query_pos_head_inv_sig: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        
        ref_points_detach = torch.sigmoid(ref_points_unact)
        
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(inverse_sigmoid(ref_points_detach))
            
            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed
            )
            
            inter_ref_bbox = torch.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
            )
            
            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points))
                    )
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox
        
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register()
class RTDETRTransformerV3(nn.Module):
    """RT-DETR Transformer v3 with hierarchical dense positive supervision"""
    
    __share__ = ['num_classes', 'hidden_dim', 'eval_size', 'o2m_branch', 'num_queries_o2m']
    
    def __init__(self,
                 num_classes: int = 80,
                 hidden_dim: int = 256,
                 num_queries: int = 300,
                 position_embed_type: str = 'sine',
                 backbone_feat_channels: List[int] = [512, 1024, 2048],
                 feat_strides: List[int] = [8, 16, 32],
                 num_levels: int = 3,
                 num_decoder_points: int = 4,
                 nhead: int = 8,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 num_denoising: int = 100,
                 label_noise_ratio: float = 0.5,
                 box_noise_scale: float = 1.0,
                 learnt_init_query: bool = True,
                 query_pos_head_inv_sig: bool = False,
                 eval_size: Optional[List[int]] = None,
                 eval_idx: int = -1,
                 num_noises: int = 0,
                 num_noise_queries: List[int] = [],
                 num_noise_denoising: int = 100,
                 o2m_branch: bool = False,
                 num_queries_o2m: int = 450,
                 eps: float = 1e-2):
        super().__init__()
        
        assert position_embed_type in ['sine', 'learned']
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        assert len(num_noise_queries) == num_noises
        
        # Extend feat_strides if needed
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = [num_queries]
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        
        # Handle noise queries
        self.num_noises = num_noises
        self.num_noise_denoising = num_noise_denoising
        self.num_groups = 1
        
        if num_noises > 0:
            self.num_queries.extend(num_noise_queries)
            self.num_groups += num_noises
        
        # Handle o2m branch
        self.o2m_branch = o2m_branch
        self.num_queries_o2m = num_queries_o2m
        if o2m_branch:
            self.num_queries.append(num_queries_o2m)
            self.num_groups += 1
        
        # Build input projection layers
        self._build_input_proj_layer(backbone_feat_channels)
        
        # Build decoder
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_levels, num_decoder_points
        )
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)
        
        # Denoising components
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        
        # Query initialization
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_inv_sig = query_pos_head_inv_sig
        
        # Encoder output heads
        self.enc_output = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(self.num_groups)
        ])
        
        self.enc_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(self.num_groups)
        ])
        
        self.enc_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(self.num_groups)
        ])
        
        # Memory mapping
        self.map_memory = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Decoder heads
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)
        ])
        
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)
        ])
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        # Initialize classification heads with proper bias
        bias_cls = -math.log((1 - 0.01) / 0.01)
        
        for enc_score_head in self.enc_score_head:
            nn.init.constant_(enc_score_head.bias, bias_cls)
        
        for enc_bbox_head in self.enc_bbox_head:
            nn.init.constant_(enc_bbox_head.layers[-1].weight, 0)
            nn.init.constant_(enc_bbox_head.layers[-1].bias, 0)
        
        for cls_head, reg_head in zip(self.dec_score_head, self.dec_bbox_head):
            nn.init.constant_(cls_head.bias, bias_cls)
            nn.init.constant_(reg_head.layers[-1].weight, 0)
            nn.init.constant_(reg_head.layers[-1].bias, 0)
        
        # Initialize other components
        for enc_output in self.enc_output:
            nn.init.xavier_uniform_(enc_output[0].weight)
        
        nn.init.xavier_uniform_(self.map_memory[0].weight)
        
        if self.learnt_init_query:
            nn.init.xavier_uniform_(self.tgt_embed.weight)
        
        nn.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)
        
        # Initialize anchors for evaluation
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()
    
    def _build_input_proj_layer(self, backbone_feat_channels: List[int]):
        """Build input projection layers"""
        self.input_proj = nn.ModuleList()
        
        for in_channels in backbone_feat_channels:
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim)
            ))
        
        # Add extra layers if needed
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim)
            ))
            in_channels = self.hidden_dim
    
    def _get_encoder_input(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[List[int]], List[int]]:
        """Get encoder input from features"""
        # Project features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # Add extra levels if needed
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))
        
        # Flatten features
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        
        for feat in proj_feats:
            b, c, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).transpose(1, 2))  # [B, H*W, C]
            spatial_shapes.append([h, w])
            level_start_index.append(h * w + level_start_index[-1])
        
        feat_flatten = torch.cat(feat_flatten, dim=1)  # [B, Total_HW, C]
        level_start_index = level_start_index[:-1]
        
        return feat_flatten, spatial_shapes, level_start_index
    
    def _generate_anchors(self,
                         spatial_shapes: Optional[List[List[int]]] = None,
                         grid_size: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate anchors for evaluation"""
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]
        
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing='ij'
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            
            valid_WH = torch.tensor([w, h], dtype=torch.float32)
            grid_xy = (grid_xy + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            
            anchors.append(torch.cat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))
        
        anchors = torch.cat(anchors, dim=1)
        valid_mask = ((anchors > self.eps) & (anchors < 1 - self.eps)).all(dim=-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.full_like(anchors, float('inf')))
        
        return anchors, valid_mask
    
    def forward(self,
                feats: List[torch.Tensor],
                pad_mask: Optional[torch.Tensor] = None,
                gt_meta: Optional[Dict] = None,
                is_teacher: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
        """Forward pass"""
        # Get encoder input
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)
        
        # Prepare denoising training
        if self.training:
            denoising_classes, denoising_bbox_unacts, attn_masks, dn_metas = [], [], [], []
            
            for g_id in range(self.num_noises + 1):
                num_denoising = self.num_denoising if g_id == 0 else self.num_noise_denoising
                
                denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                    get_contrastive_denoising_training_group(
                        gt_meta,
                        self.num_classes,
                        self.num_queries[g_id],
                        self.denoising_class_embed.weight,
                        num_denoising,
                        self.label_noise_ratio,
                        self.box_noise_scale
                    )
                
                denoising_classes.append(denoising_class)
                denoising_bbox_unacts.append(denoising_bbox_unact)
                attn_masks.append(attn_mask)
                dn_metas.append(dn_meta)
        else:
            denoising_classes = denoising_bbox_unacts = attn_masks = dn_metas = None
        
        # Get decoder input
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_classes, denoising_bbox_unacts, is_teacher)
        
        # Multi-group noise attention
        if self.training:
            new_size = target.shape[1]
            new_attn_mask = torch.ones([new_size, new_size], dtype=torch.bool, device=target.device)
            begin, end = 0, 0
            
            for g_id in range(self.num_groups):
                if self.o2m_branch and g_id == self.num_groups - 1:
                    end = begin + self.num_queries_o2m
                    new_mask = torch.rand([self.num_queries[g_id], self.num_queries[g_id]], device=target.device) >= 0.0
                    new_attn_mask[begin:end, begin:end] = new_mask
                else:
                    end = begin + attn_masks[g_id].shape[1]
                    dn_size, q_size = dn_metas[g_id]['dn_num_split']
                    
                    if g_id > 0:
                        new_mask = torch.rand([self.num_queries[g_id], self.num_queries[g_id]], device=target.device) > 0.1
                    else:
                        new_mask = torch.rand([self.num_queries[g_id], self.num_queries[g_id]], device=target.device) >= 0.0
                    
                    attn_masks[g_id][dn_size:dn_size + q_size, dn_size:dn_size + q_size] = new_mask
                    new_attn_mask[begin:end, begin:end] = attn_masks[g_id]
                
                begin = end
            
            attn_masks = new_attn_mask
        
        # Decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            torch.tensor(spatial_shapes, dtype=torch.long, device=memory.device),
            torch.tensor(level_start_index, dtype=torch.long, device=memory.device),
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_masks,
            memory_mask=None,
            query_pos_head_inv_sig=self.query_pos_head_inv_sig
        )
        
        return out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits, dn_metas
    
    def _get_decoder_input(self,
                          memory: torch.Tensor,
                          spatial_shapes: List[List[int]],
                          denoising_classes: Optional[List[torch.Tensor]] = None,
                          denoising_bbox_unacts: Optional[List[torch.Tensor]] = None,
                          is_teacher: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get decoder input"""
        bs = memory.shape[0]
        
        # Generate anchors
        if self.training or self.eval_size is None or is_teacher:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        
        anchors = anchors.to(memory.device)
        valid_mask = valid_mask.to(memory.device)
        
        memory = torch.where(valid_mask, memory, torch.zeros_like(memory))
        map_memory = self.map_memory(memory.detach())
        
        targets, reference_points_unacts, enc_topk_bboxes, enc_topk_logits = [], [], [], []
        
        for g_id in range(self.num_groups):
            output_memory = self.enc_output[g_id](memory)
            enc_outputs_class = self.enc_score_head[g_id](output_memory)
            enc_outputs_coord_unact = self.enc_bbox_head[g_id](output_memory) + anchors
            
            # Get top-k proposals
            _, topk_ind = torch.topk(enc_outputs_class.max(dim=-1)[0], self.num_queries[g_id], dim=1)
            
            batch_ind = torch.arange(bs, dtype=topk_ind.dtype, device=topk_ind.device)
            batch_ind = batch_ind.unsqueeze(-1).expand(-1, self.num_queries[g_id])
            
            reference_points_unact = enc_outputs_coord_unact[batch_ind, topk_ind]
            enc_topk_bbox = torch.sigmoid(reference_points_unact)
            enc_topk_logit = enc_outputs_class[batch_ind, topk_ind]
            
            # Add denoising if available
            if denoising_bbox_unacts is not None and not (self.o2m_branch and g_id == self.num_groups - 1):
                reference_points_unact = torch.cat([denoising_bbox_unacts[g_id], reference_points_unact], dim=1)
            
            if self.training:
                reference_points_unact = reference_points_unact.detach()
            
            # Extract region features
            if self.learnt_init_query:
                target = self.tgt_embed.weight.unsqueeze(0).expand(bs, -1, -1)
            else:
                if g_id == 0:
                    target = output_memory[batch_ind, topk_ind]
                    if self.training:
                        target = target.detach()
                else:
                    target = map_memory[batch_ind, topk_ind]
            
            if denoising_classes is not None and not (self.o2m_branch and g_id == self.num_groups - 1):
                target = torch.cat([denoising_classes[g_id], target], dim=1)
            
            if not self.training:
                return target, reference_points_unact, enc_topk_bbox, enc_topk_logit
            
            targets.append(target)
            reference_points_unacts.append(reference_points_unact)
            enc_topk_bboxes.append(enc_topk_bbox)
            enc_topk_logits.append(enc_topk_logit)
        
        targets = torch.cat(targets, dim=1)
        reference_points_unacts = torch.cat(reference_points_unacts, dim=1)
        enc_topk_bboxes = torch.cat(enc_topk_bboxes, dim=1)
        enc_topk_logits = torch.cat(enc_topk_logits, dim=1)
        
        return targets, reference_points_unacts, enc_topk_bboxes, enc_topk_logits
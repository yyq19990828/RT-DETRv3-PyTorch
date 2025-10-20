"""
RT-DETRv3 Transformer Decoder

This module implements the transformer decoder for RT-DETRv3, following PaddlePaddle's
implementation for numerical equivalence.

Key components:
- TransformerDecoderLayer: Single decoder layer with self-attention, cross-attention, and FFN
- TransformerDecoder: Stack of decoder layers with iterative refinement
- RTDETRTransformerv3: Main transformer module with encoder query selection and decoder

Reference:
- PaddlePaddle RT-DETR: ppdet/modeling/transformers/rtdetr_transformerv3.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
import math

from .attention import MSDeformableAttention
from .utils import MLP, inverse_sigmoid
from ppdet.core.workspace import register


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for self-attention in decoder

    This is standard multi-head attention (not deformable).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N, C)
            key: (B, M, C)
            value: (B, M, C)
            attn_mask: (B, N, M) or (N, M), additive mask (0 for valid, -inf for invalid)

        Returns:
            output: (B, N, C)
        """
        B, N, C = query.shape
        M = key.shape[1]

        # Project and reshape to (B, num_heads, N/M, head_dim)
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # (1, N, M)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, M) for broadcasting
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        output = output.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        output = self.out_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer

    Structure:
    1. Self-Attention (with residual + norm)
    2. Cross-Attention (deformable, with residual + norm)
    3. Feed-Forward Network (with residual + norm)

    Following PaddlePaddle's implementation exactly.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_points: int = 4
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (deformable)
        self.cross_attn = MSDeformableAttention(
            embed_dim=d_model,
            num_heads=n_head,
            num_levels=n_levels,
            num_points=n_points
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        """Add positional embedding to tensor"""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """Feed-forward network"""
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: torch.Tensor,
        memory_level_start_index: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        query_pos_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target queries (B, N, C)
            reference_points: Reference points (B, N, n_levels, 2)
            memory: Encoder memory (B, M, C)
            memory_spatial_shapes: (n_levels, 2)
            memory_level_start_index: (n_levels,)
            attn_mask: Self-attention mask (optional)
            memory_mask: Cross-attention mask (optional)
            query_pos_embed: Query positional embedding (B, N, C)

        Returns:
            Updated target queries (B, N, C)
        """
        # Self-attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            # Convert boolean mask to additive mask
            attn_mask = torch.where(
                attn_mask.bool(),
                torch.zeros_like(attn_mask, dtype=tgt.dtype),
                torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype)
            )
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention (deformable)
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

        # FFN
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder with iterative refinement

    Features:
    - Multiple decoder layers
    - Iterative bounding box refinement
    - Returns predictions at each layer during training
    - Returns only final layer prediction during eval
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        eval_idx: int = -1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=decoder_layer.self_attn.embed_dim,
                n_head=decoder_layer.self_attn.num_heads,
                dim_feedforward=decoder_layer.linear1.out_features,
                dropout=decoder_layer.dropout1.p,
                activation='relu',  # Default from PaddlePaddle
                n_levels=decoder_layer.cross_attn.num_levels,
                n_points=decoder_layer.cross_attn.num_points
            )
            for _ in range(num_layers)
        ])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
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
        query_pos_head_inv_sig: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tgt: Initial target queries (B, N, C)
            ref_points_unact: Unactivated reference points (B, N, 4), will be sigmoid-ed
            memory: Encoder memory (B, M, C)
            memory_spatial_shapes: (n_levels, 2)
            memory_level_start_index: (n_levels,)
            bbox_head: List of bbox regression heads (one per layer)
            score_head: List of classification heads (one per layer)
            query_pos_head: Query position embedding head
            attn_mask: Attention mask
            memory_mask: Memory mask
            query_pos_head_inv_sig: Whether to use inverse_sigmoid for query_pos_head input

        Returns:
            dec_out_bboxes: Stacked bbox predictions (num_layers, B, N, 4)
            dec_out_logits: Stacked classification logits (num_layers, B, N, num_classes)
        """
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)  # (B, N, 1, 4) -> (B, N, n_levels, 4)

            # Generate query positional embedding
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(inverse_sigmoid(ref_points_detach))

            # Apply decoder layer
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

            # Iterative refinement: predict bbox offset and refine reference points
            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
            )

            # Collect outputs
            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points))
                    )
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            # Update reference points for next layer
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


# Note: build_transformer_decoder() function removed.
# Following PaddlePaddle style: decoder is instantiated directly in RTDETRTransformerv3.__init__
# PaddlePaddle reference: ppdet/modeling/transformers/rtdetr_transformerv3.py:330-334


@register
class RTDETRTransformerv3(nn.Module):
    """
    Complete RT-DETRv3 Transformer with multi-group queries and self-attention perturbation

    This module implements the full transformer architecture including:
    - Multi-group query mechanism (one-to-one, noise, one-to-many)
    - Encoder query selection (top-k from encoder features)
    - Multi-group encoder heads
    - Self-attention perturbation masks
    - Denoising queries (inherited from RT-DETRv2)

    Following PaddlePaddle's implementation for numerical equivalence.

    Reference:
    - PaddlePaddle: ppdet/modeling/transformers/rtdetr_transformerv3.py:263-653
    """

    __category__ = 'transformer'
    __inject__ = []  # No component dependencies (receives features from neck)
    __shared__ = ['num_classes', 'hidden_dim', 'o2m_branch', 'num_queries_o2m']  # Shared config following PaddlePaddle

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        num_levels: int = 3,
        num_decoder_points: int = 4,
        num_noises: int = 1,
        num_noise_queries: List[int] = [100],
        o2m_branch: bool = False,
        num_queries_o2m: int = 450,
        eval_idx: int = -1,
        # Denoising parameters (inherited from RT-DETRv2)
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0
    ):
        """
        Args:
            num_classes: Number of object classes
            hidden_dim: Hidden dimension for transformer
            num_queries: Number of one-to-one queries (default: 300)
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            num_levels: Number of feature pyramid levels
            num_decoder_points: Number of sampling points in deformable attention
            num_noises: Number of noise groups (default: 1)
            num_noise_queries: List of query counts for each noise group (default: [100])
            o2m_branch: Enable one-to-many branch
            num_queries_o2m: Number of one-to-many queries (default: 450)
            eval_idx: Decoder layer index for evaluation (-1 for last layer)
            num_denoising: Number of denoising queries
            label_noise_ratio: Label noise ratio for denoising
            box_noise_scale: Box noise scale for denoising
        """
        super().__init__()

        # Basic configuration
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.num_decoder_layers = num_decoder_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_decoder_layers + eval_idx

        # Multi-group query configuration
        # Following PaddlePaddle: rtdetr_transformerv3.py:308-324
        self.num_queries = [num_queries]  # Start with o2o group
        self.num_noises = num_noises
        self.num_groups = 1

        if num_noises > 0:
            # Add noise groups
            self.num_queries.extend(num_noise_queries)
            self.num_groups += num_noises

        self.o2m_branch = o2m_branch
        self.num_queries_o2m = num_queries_o2m
        if o2m_branch:
            # Add o2m group
            self.num_queries.append(num_queries_o2m)
            self.num_groups += 1

        # Denoising configuration
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Multi-group encoder heads
        # Following PaddlePaddle: rtdetr_transformerv3.py:353-369
        self.enc_output = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(self.num_groups)
        ])

        self.enc_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(self.num_groups)
        ])

        self.enc_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(self.num_groups)
        ])

        # Decoder (following PaddlePaddle: ppdet/modeling/transformers/rtdetr_transformerv3.py:330-334)
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            n_head=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_levels,
            n_points=num_decoder_points
        )
        self.decoder = TransformerDecoder(
            hidden_dim=hidden_dim,
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            eval_idx=eval_idx
        )

        # Decoder bbox and score heads (shared across all groups)
        # Following PaddlePaddle: rtdetr_transformerv3.py:330-334
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])

        # Query position head
        self.query_pos_head = MLP(4, 512, hidden_dim, num_layers=2)

        # Learnable query embeddings for each group
        # Following PaddlePaddle: rtdetr_transformerv3.py:371-379
        self.tgt_embed = nn.ParameterList([
            nn.Parameter(torch.randn(num_q, hidden_dim))
            for num_q in self.num_queries
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        # Initialize query embeddings
        for tgt in self.tgt_embed:
            nn.init.normal_(tgt)

        # Initialize encoder heads
        for module in self.enc_output:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        for head in self.enc_score_head:
            nn.init.constant_(head.bias, -math.log((1 - 0.01) / 0.01))

        # Initialize decoder heads
        for head in self.dec_score_head:
            nn.init.constant_(head.bias, -math.log((1 - 0.01) / 0.01))

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        """
        Flatten and concatenate multi-scale features into encoder memory

        Args:
            feats: List of features [(B, C, H1, W1), (B, C, H2, W2), (B, C, H3, W3)]

        Returns:
            memory: (B, sum(Hi*Wi), C)
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)
        """
        from .utils import get_encoder_memory_and_spatial_shapes
        return get_encoder_memory_and_spatial_shapes(feats)

    def _generate_anchors(
        self,
        spatial_shapes: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate anchor points for all feature levels

        Args:
            spatial_shapes: (num_levels, 2) containing (H, W) for each level
            device: Device to create anchors on

        Returns:
            anchors: (sum(Hi*Wi), 4) in format [cx, cy, w, h], normalized to [0, 1]
        """
        anchors_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            # Create grid
            y, x = torch.meshgrid(
                torch.arange(H, dtype=torch.float32, device=device),
                torch.arange(W, dtype=torch.float32, device=device),
                indexing='ij'
            )

            # Normalize to [0, 1]
            x = (x + 0.5) / W
            y = (y + 0.5) / H

            # Create anchor in format [cx, cy, w, h]
            # Initial width/height are small values
            anchors = torch.stack([
                x.flatten(),
                y.flatten(),
                torch.full_like(x.flatten(), 0.05),
                torch.full_like(y.flatten(), 0.05)
            ], dim=-1)

            anchors_list.append(anchors)

        return torch.cat(anchors_list, dim=0)

    def _select_topk(
        self,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        anchors: torch.Tensor,
        group_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-K proposals from encoder memory for a specific group

        Following PaddlePaddle: rtdetr_transformerv3.py:605-619

        Args:
            memory: Encoder memory (B, sum(Hi*Wi), C)
            spatial_shapes: (num_levels, 2)
            anchors: (sum(Hi*Wi), 4)
            group_id: Query group index

        Returns:
            enc_topk_bboxes: (B, K, 4) top-K bboxes
            enc_topk_logits: (B, K, num_classes) top-K logits
            target: (B, K, C) top-K memory features
            ref_points_unact: (B, K, 4) unactivated reference points
        """
        B = memory.shape[0]
        K = self.num_queries[group_id]

        # Process memory through group-specific encoder head
        output_memory = self.enc_output[group_id](memory)  # (B, sum(Hi*Wi), C)

        # Get classification scores and bbox predictions
        enc_outputs_class = self.enc_score_head[group_id](output_memory)  # (B, sum(Hi*Wi), num_classes)
        enc_outputs_coord_unact = self.enc_bbox_head[group_id](output_memory)  # (B, sum(Hi*Wi), 4)
        enc_outputs_coord_unact = enc_outputs_coord_unact + inverse_sigmoid(anchors.unsqueeze(0))

        # Select top-K based on max class score
        max_scores, _ = enc_outputs_class.max(dim=-1)  # (B, sum(Hi*Wi))
        _, topk_ind = torch.topk(max_scores, K, dim=-1)  # (B, K)

        # Gather top-K features
        topk_ind_expand = topk_ind.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        target = torch.gather(output_memory, 1, topk_ind_expand)  # (B, K, C)

        # Gather top-K predictions
        topk_ind_expand_cls = topk_ind.unsqueeze(-1).expand(-1, -1, self.num_classes)
        enc_topk_logits = torch.gather(enc_outputs_class, 1, topk_ind_expand_cls)  # (B, K, num_classes)

        topk_ind_expand_bbox = topk_ind.unsqueeze(-1).expand(-1, -1, 4)
        ref_points_unact = torch.gather(enc_outputs_coord_unact, 1, topk_ind_expand_bbox)  # (B, K, 4)
        enc_topk_bboxes = torch.sigmoid(ref_points_unact)

        return enc_topk_bboxes, enc_topk_logits, target, ref_points_unact

    def _generate_perturbation_mask(
        self,
        num_queries_list: List[int],
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate self-attention perturbation masks for training

        Following PaddlePaddle: rtdetr_transformerv3.py:518-539

        Args:
            num_queries_list: List of query counts for each group
            device: Device to create mask on

        Returns:
            attn_mask: (total_queries, total_queries) boolean mask
                      False = attend, True = mask (block attention)
        """
        total_queries = sum(num_queries_list)
        attn_mask = torch.zeros(total_queries, total_queries, dtype=torch.bool, device=device)

        begin = 0
        for g_id, num_q in enumerate(num_queries_list):
            end = begin + num_q

            # Generate random mask for this group
            rand_mask = torch.rand(num_q, num_q, device=device)

            if self.o2m_branch and g_id == len(num_queries_list) - 1:
                # O2M branch: no perturbation (p=0)
                group_mask = rand_mask >= 0.0  # All True (attend all)
            elif g_id > 0:
                # Noise group: 10% perturbation (p=0.1)
                group_mask = rand_mask > 0.1  # True with prob 0.9
            else:
                # One-to-one group: no perturbation (p=0)
                group_mask = rand_mask >= 0.0  # All True (attend all)

            # Apply mask to this group's region
            attn_mask[begin:end, begin:end] = ~group_mask  # Invert: True = block

            begin = end

        return attn_mask

    def forward(
        self,
        feats: List[torch.Tensor],
        targets: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through RT-DETRv3 Transformer

        Args:
            feats: List of multi-scale features from neck
                   [(B, C, H1, W1), (B, C, H2, W2), (B, C, H3, W3)]
            targets: Training targets (optional, only used in training mode)

        Returns:
            dec_out_bboxes: (num_layers, B, total_queries, 4) decoder bbox predictions
            dec_out_logits: (num_layers, B, total_queries, num_classes) decoder logits
            enc_topk_bboxes: (B, total_queries, 4) encoder top-k bboxes (concatenated from all groups)
            enc_topk_logits: (B, total_queries, num_classes) encoder top-k logits
            dn_meta: Denoising metadata (None for now)
        """
        # Flatten multi-scale features
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)
        B, _, C = memory.shape
        device = memory.device

        # Generate anchors
        anchors = self._generate_anchors(spatial_shapes, device)

        # Select top-K from encoder for each group
        targets_list = []
        ref_points_list = []
        enc_topk_bboxes_list = []
        enc_topk_logits_list = []

        for g_id in range(self.num_groups):
            enc_topk_bboxes, enc_topk_logits, target, ref_points_unact = self._select_topk(
                memory, spatial_shapes, anchors, g_id
            )

            # Initialize target with learned embeddings + top-K features
            tgt_embed = self.tgt_embed[g_id].unsqueeze(0).expand(B, -1, -1)  # (B, K, C)
            target = target + tgt_embed

            targets_list.append(target)
            ref_points_list.append(ref_points_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # Concatenate all groups
        target = torch.cat(targets_list, dim=1)  # (B, total_queries, C)
        ref_points_unact = torch.cat(ref_points_list, dim=1)  # (B, total_queries, 4)
        enc_topk_bboxes = torch.cat(enc_topk_bboxes_list, dim=1)
        enc_topk_logits = torch.cat(enc_topk_logits_list, dim=1)

        # Generate perturbation mask for training
        attn_mask = None
        if self.training:
            attn_mask = self._generate_perturbation_mask(self.num_queries, device)

        # Run decoder
        dec_out_bboxes, dec_out_logits = self.decoder(
            target,
            ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask
        )

        # Denoising metadata (not implemented yet)
        dn_meta = None

        return dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], global_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Build RTDETRTransformerv3 from config (PaddlePaddle-style).

        Following PaddlePaddle: ppdet/modeling/architectures/rtdetrv3.py:62-64
        The transformer receives input_shape from neck but doesn't need to create sub-components.

        Args:
            cfg: Transformer configuration dict
            global_config: Global configuration for shared values

        Returns:
            Empty dict (no special construction logic needed, all done in __init__)
        """
        # PaddlePaddle pattern: transformer = create(cfg['transformer'], **kwargs)
        # where kwargs = {'input_shape': neck.out_shape}
        # RTDETRTransformerv3 doesn't use input_shape, so we just return empty dict
        return {}


# Note: build_rtdetr_transformer() function removed.
# Use create('RTDETRTransformerv3', **kwargs) for PaddlePaddle-style instantiation instead.

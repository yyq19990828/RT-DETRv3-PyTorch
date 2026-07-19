# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025 PyTorch Migration. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppdet_pytorch.core.workspace import register

from ..batch_norm import ContiguousGradBatchNorm2d
from ..initializer import (
    bias_init_with_prob,
    constant_,
    linear_init_,
    xavier_uniform_,
)
from ..layers import MultiHeadAttention
from .attention import MSDeformableAttention
from .utils import (
    MLP,
    _get_clones,
    get_contrastive_denoising_training_group,
    inverse_sigmoid,
)

__all__ = ["RTDETRTransformerv3"]


class PPMSDeformableAttention(MSDeformableAttention):
    """
    PaddlePaddle-style MSDeformableAttention wrapper for compatibility
    """

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (Tensor): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.unsqueeze(-1).float()
        value = value.view(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = (
                value_spatial_shapes.flip([1])
                .view(1, 1, 1, self.num_levels, 1, 2)
                .float()
            )
            sampling_locations = (
                reference_points.view(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        # Use the core deformable attention function
        output = self.ms_deform_attn_core(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with self-attention, cross-attention, and FFN
    """

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = PPMSDeformableAttention(
            d_model, n_head, n_levels, n_points, 1.0
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(
        self, tensor: torch.Tensor, pos: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
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
        query_pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            # Convert boolean mask to additive mask
            attn_mask = torch.where(
                attn_mask.bool(),
                torch.zeros_like(attn_mask, dtype=tgt.dtype),
                torch.full_like(attn_mask, float("-inf"), dtype=tgt.dtype),
            )
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_level_start_index,
            memory_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder with multiple layers
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: nn.Module,
        num_layers: int,
        eval_idx: int = -1,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
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
        query_pos_head_inv_sig: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        ref_points = ref_points_detach
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
                query_pos_embed,
            )

            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
            )

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

            ref_points = inter_ref_bbox
            ref_points_detach = (
                inter_ref_bbox.detach() if self.training else inter_ref_bbox
            )

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register
class RTDETRTransformerv3(nn.Module):
    """
    RT-DETR Transformer v3

    This is the main transformer module for RT-DETRv3, implementing:
    - Multi-scale deformable attention
    - Contrastive denoising training
    - One-to-many (O2M) branch
    - Multiple noise groups
    """

    __shared__ = [
        "num_classes",
        "hidden_dim",
        "eval_size",
        "o2m_branch",
        "num_queries_o2m",
    ]

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        position_embed_type: str = "sine",
        backbone_feat_channels: List[int] = [512, 1024, 2048],
        feat_strides: List[int] = [8, 16, 32],
        num_levels: int = 3,
        num_decoder_points: int = 4,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
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
        eps: float = 1e-2,
    ):
        super(RTDETRTransformerv3, self).__init__()
        assert position_embed_type in ["sine", "learned"], (
            f"ValueError: position_embed_type not supported {position_embed_type}!"
        )
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        assert len(num_noise_queries) == num_noises

        # Extend feat_strides if needed
        feat_strides = feat_strides.copy()
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

        self.num_noises = num_noises
        self.num_noise_denoising = num_noise_denoising
        self.num_groups = 1
        if num_noises > 0:
            self.num_queries.extend(num_noise_queries)
            self.num_groups += num_noises

        self.o2m_branch = o2m_branch
        self.num_queries_o2m = num_queries_o2m
        if o2m_branch:
            self.num_queries.append(num_queries_o2m)
            self.num_groups += 1

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_decoder_points,
        )
        self.decoder = TransformerDecoder(
            hidden_dim, decoder_layer, num_decoder_layers, eval_idx
        )

        # denoising part
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)
        nn.init.normal_(self.denoising_class_embed.weight)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_inv_sig = query_pos_head_inv_sig

        # encoder head
        self.enc_output = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
                )
                for _ in range(self.num_groups)
            ]
        )
        self.enc_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.num_groups)]
        )
        self.enc_bbox_head = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, 4, num_layers=3)
                for _ in range(self.num_groups)
            ]
        )

        self.map_memory = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )

        # decoder head
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, 4, num_layers=3)
                for _ in range(num_decoder_layers)
            ]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        for enc_score_head in self.enc_score_head:
            linear_init_(enc_score_head)
            constant_(enc_score_head.bias, bias_cls)
        for enc_bbox_head in self.enc_bbox_head:
            constant_(enc_bbox_head.layers[-1].weight)
            constant_(enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        for enc_output in self.enc_output:
            linear_init_(enc_output[0])
            xavier_uniform_(enc_output[0].weight)
        linear_init_(self.map_memory[0])
        xavier_uniform_(self.map_memory[0].weight)

        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"backbone_feat_channels": [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels: List[int]):
        self.input_proj = nn.ModuleList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    in_channels,
                                    self.hidden_dim,
                                    kernel_size=1,
                                    bias=False,
                                ),
                            ),
                            ("norm", ContiguousGradBatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )

        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    in_channels,
                                    self.hidden_dim,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                ),
                            ),
                            ("norm", ContiguousGradBatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(
        self, feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[List[int]], List[int]]:
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.cat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def forward(
        self,
        feats: List[torch.Tensor],
        pad_mask: Optional[torch.Tensor] = None,
        gt_meta: Optional[dict] = None,
        is_teacher: bool = False,
    ) -> Tuple:
        # input projection and embedding
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            denoising_classes, denoising_bbox_unacts, attn_masks, dn_metas = (
                [],
                [],
                [],
                [],
            )
            for g_id in range(self.num_noises + 1):
                if g_id == 0:
                    num_denoising = self.num_denoising
                else:
                    num_denoising = self.num_noise_denoising
                denoising_class, denoising_bbox_unact, attn_mask, dn_meta = (
                    get_contrastive_denoising_training_group(
                        gt_meta,
                        self.num_classes,
                        self.num_queries[g_id],
                        self.denoising_class_embed.weight,
                        num_denoising,
                        self.label_noise_ratio,
                        self.box_noise_scale,
                    )
                )
                denoising_classes.append(denoising_class)
                denoising_bbox_unacts.append(denoising_bbox_unact)
                attn_masks.append(attn_mask)
                dn_metas.append(dn_meta)
        else:
            denoising_classes, denoising_bbox_unacts, attn_masks, dn_metas = (
                None,
                None,
                None,
                None,
            )

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = (
            self._get_decoder_input(
                memory,
                spatial_shapes,
                denoising_classes,
                denoising_bbox_unacts,
                is_teacher,
            )
        )

        # multi group noise attention
        if self.training:
            new_size = target.shape[1]
            # True means allowed by TransformerDecoderLayer. Queries from
            # different groups must remain blocked, matching Paddle's
            # initially-false block-diagonal mask.
            new_attn_mask = torch.zeros(
                new_size, new_size, dtype=torch.bool, device=target.device
            )
            begin, end = 0, 0

            for g_id in range(self.num_groups):
                new_mask = torch.rand(
                    self.num_queries[g_id], self.num_queries[g_id], device=target.device
                )
                if self.o2m_branch and g_id == self.num_groups - 1:
                    end = end + self.num_queries_o2m
                    new_mask = new_mask >= 0.0
                    new_attn_mask[begin:end, begin:end] = new_mask
                else:
                    end = end + attn_masks[g_id].shape[1]
                    dn_size, q_size = dn_metas[g_id]["dn_num_split"]
                    if g_id > 0:
                        new_mask = new_mask > 0.1
                    else:
                        new_mask = new_mask >= 0.0
                    attn_masks[g_id][
                        dn_size : dn_size + q_size, dn_size : dn_size + q_size
                    ] = new_mask
                    new_attn_mask[begin:end, begin:end] = attn_masks[g_id]
                begin = end
            attn_masks = new_attn_mask

        # decoder
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
            query_pos_head_inv_sig=self.query_pos_head_inv_sig,
        )

        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits, dn_metas)

    def _generate_anchors(
        self,
        spatial_shapes: Optional[List[List[int]]] = None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]

        if device is None:
            device = next(self.parameters()).device

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=dtype, device=device),
                torch.arange(w, dtype=dtype, device=device),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))

        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > self.eps) & (anchors < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(
            valid_mask, anchors, torch.tensor(float("inf"), device=device)
        )
        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: List[List[int]],
        denoising_classes: Optional[List[torch.Tensor]] = None,
        denoising_bbox_unacts: Optional[List[torch.Tensor]] = None,
        is_teacher: bool = False,
    ) -> Tuple:
        bs = memory.shape[0]
        device = memory.device

        # prepare input for decoder
        if self.training or self.eval_size is None or is_teacher:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=device)
        else:
            anchors, valid_mask = self.anchors.to(device), self.valid_mask.to(device)

        memory = torch.where(valid_mask, memory, torch.tensor(0.0, device=device))
        map_memory = self.map_memory(memory.detach())
        targets, reference_points_unacts, enc_topk_bboxes, enc_topk_logits = (
            [],
            [],
            [],
            [],
        )

        for g_id in range(self.num_groups):
            output_memory = self.enc_output[g_id](memory)
            enc_outputs_class = self.enc_score_head[g_id](output_memory)
            enc_outputs_coord_unact = self.enc_bbox_head[g_id](output_memory) + anchors

            _, topk_ind = torch.topk(
                enc_outputs_class.max(-1)[0], self.num_queries[g_id], dim=1
            )

            # extract region proposal boxes
            batch_ind = torch.arange(bs, dtype=topk_ind.dtype, device=device)
            batch_ind = batch_ind.unsqueeze(-1).repeat(1, self.num_queries[g_id])

            # Gather using advanced indexing
            reference_points_unact = enc_outputs_coord_unact[
                batch_ind, topk_ind
            ]  # unsigmoided
            enc_topk_bbox = torch.sigmoid(reference_points_unact)
            enc_topk_logit = enc_outputs_class[batch_ind, topk_ind]

            if denoising_bbox_unacts is not None and not (
                self.o2m_branch and g_id == self.num_groups - 1
            ):
                reference_points_unact = torch.cat(
                    [denoising_bbox_unacts[g_id], reference_points_unact], 1
                )
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            # extract region features
            if self.learnt_init_query:
                target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            else:
                if g_id == 0:
                    target = output_memory[batch_ind, topk_ind]
                    if self.training:
                        target = target.detach()
                else:
                    target = map_memory[batch_ind, topk_ind]

            if denoising_classes is not None and not (
                self.o2m_branch and g_id == self.num_groups - 1
            ):
                target = torch.cat([denoising_classes[g_id], target], 1)

            if not self.training:
                return target, reference_points_unact, enc_topk_bbox, enc_topk_logit

            targets.append(target)
            reference_points_unacts.append(reference_points_unact)
            enc_topk_bboxes.append(enc_topk_bbox)
            enc_topk_logits.append(enc_topk_logit)

        targets = torch.cat(targets, 1)
        reference_points_unacts = torch.cat(reference_points_unacts, 1)
        enc_topk_bboxes = torch.cat(enc_topk_bboxes, 1)
        enc_topk_logits = torch.cat(enc_topk_logits, 1)
        return targets, reference_points_unacts, enc_topk_bboxes, enc_topk_logits

"""DEIMv2 transformer decoder (RMSNorm + SwiGLU + shared query position).

Ported from Intellindust-AI-Lab/DEIMv2@add5bcd (engine/deim/deim_decoder.py and
engine/deim/deim_utils.py, Apache-2.0). FDR/LQE/CDN and the deformable attention
core are reused from the verified D-FINE primitives in this repository.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from detrs.core.workspace import register

from .dfine_decoder import LQE, MLP, Integral, MSDeformableAttention
from .dfine_support import get_contrastive_denoising_training_group
from .dfine_utils import distance2bbox, weighting_function
from .utils import inverse_sigmoid

__all__ = ["DEIMTransformer"]


def _bias_init_with_prob(prior_prob: float) -> float:
    return -math.log((1 - prior_prob) / prior_prob)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.scale

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.constant_(self.w12.bias, 0)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.constant_(self.w3.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class DEIMv2Gate(nn.Module):
    def __init__(self, d_model: int, use_rmsnorm: bool = False):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        nn.init.constant_(self.gate.bias, _bias_init_with_prob(0.5))
        nn.init.constant_(self.gate.weight, 0)
        self.norm: nn.Module = (
            RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gate1, gate2 = torch.sigmoid(self.gate(torch.cat([x1, x2], -1))).chunk(
            2, dim=-1
        )
        return self.norm(gate1 * x1 + gate2 * x2)


class DEIMv2TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
        use_gateway=False,
    ):
        super().__init__()
        if layer_scale is not None:
            d_model = round(layer_scale * d_model)
            dim_feedforward = round(layer_scale * dim_feedforward)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = DEIMv2Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        target,
        reference_points,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos_embed=None,
    ):
        query = self.with_pos_embed(target, query_pos_embed)
        attended, _ = self.self_attn(query, query, target, attn_mask=attn_mask)
        target = self.norm1(target + self.dropout1(attended))

        update = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )
        if self.use_gateway:
            target = self.gateway(target, self.dropout2(update))
        else:
            target = self.norm2(target + self.dropout2(update))

        update = self.swish_ffn(target)
        target = self.norm3((target + self.dropout4(update)).clamp(-65504, 65504))
        return target


class DEIMv2TransformerDecoder(nn.Module):
    """FDR decoder with an eval-index layer split and deploy-time pruning."""

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        act="relu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [
                copy.deepcopy(decoder_layer_wide)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)]
        )

    def value_op(
        self,
        memory: torch.Tensor,
        value_proj,
        value_scale,
        memory_mask,
        memory_spatial_shapes,
    ):
        value = value_proj(memory) if value_proj is not None else memory
        value = (
            F.interpolate(memory, size=value_scale)
            if value_scale is not None
            else value
        )
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        # persistent=False keeps the key out of state_dict so official
        # checkpoints still strict-load; a bare tensor attribute would be
        # frozen on the trace device and break CUDA TorchScript inference.
        self.register_buffer(
            "project",
            weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True),
            persistent=False,
        )
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
    ):
        del dn_meta
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if hasattr(self, "project"):
            project = self.project
        else:
            project = weighting_function(self.reg_max, up, reg_scale)

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for index, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if index >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(
                    query_pos_embed, scale_factor=self.layer_scale
                )
                value = self.value_op(
                    memory,
                    None,
                    query_pos_embed.shape[-1],
                    memory_mask,
                    spatial_shapes,
                )
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(
                output,
                ref_points_input,
                value,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
            )

            if index == 0:
                pre_bboxes = F.sigmoid(
                    pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[index](output + output_detach) + (
                pred_corners_undetach
            )
            inter_ref_bbox = distance2bbox(
                ref_points_initial, integral(pred_corners, project), reg_scale
            )

            if self.training or index == self.eval_idx:
                scores = score_head[index](output)
                scores = self.lqe_layers[index](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


@register
class DEIMTransformer(nn.Module):
    """DEIMv2 decoder head with shared heads and deploy-time eval pruning.

    Refines queries like `DFINETransformer`; additionally shares box/score
    heads across decoder layers and can drop gateway branches at eval time
    for deployment.

    Args:
        num_classes (int): Number of foreground classes.
        hidden_dim (int): Embedding dimension of queries and decoder.
        num_queries (int): Number of object queries.
        feat_channels (tuple): Channels of the input feature levels.
        feat_strides (tuple): Strides of the input feature levels.
        num_levels (int): Number of feature levels used.
        num_points (int): Sampling points per level for deformable attention.
        nhead (int): Attention heads.
        num_layers (int): Decoder layers.
        dim_feedforward (int): Width of the decoder FFN.
        dropout (float): Dropout rate.
        activation (str): FFN activation.
        num_denoising (int): Number of contrastive denoising queries.
        label_noise_ratio (float): Label noise ratio for denoising.
        box_noise_scale (float): Box noise scale for denoising.
        learn_query_content (bool): Learn query content embeddings.
        eval_spatial_size (tuple|None): Fixed input size for export.
        eval_idx (int): Decoder layer returned at inference.
        eps (float): Numerical epsilon.
        aux_loss (bool): All decoder layers produce loss outputs.
        cross_attn_method (str): Cross-attention implementation variant.
        query_select_method (str): Top-query selection implementation.
        reg_max (int): Discrete bins of the box distribution.
        reg_scale (float): Scale mapping distributions to box offsets.
        layer_scale (int): Isotropic layer scaling factor.
        mlp_act (str): Activation of the box MLP.
        use_gateway (bool): Use gated attention units; the gate is pruned
            at eval time.
        share_bbox_head (bool): Share the box head across decoder layers.
        share_score_head (bool): Share the score head across decoder layers.
        input_shape (dict|None): Explicit input feature shapes overriding
            `feat_channels`/`feat_strides`.
    """

    __shared__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32),
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        input_shape=None,
    ):
        super().__init__()
        feat_channels = list(feat_channels)
        feat_strides = list(feat_strides)
        del input_shape
        if len(feat_strides) != len(feat_channels):
            raise ValueError("feat_strides must match feat_channels")
        if len(feat_channels) > num_levels:
            raise ValueError("feat_channels cannot exceed num_levels")
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        if query_select_method not in ("default", "one2many", "agnostic"):
            raise ValueError(f"unsupported query_select_method: {query_select_method}")
        if cross_attn_method not in ("default", "discrete"):
            raise ValueError(f"unsupported cross_attn_method: {cross_attn_method}")
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self.use_gateway = use_gateway
        self.share_bbox_head = share_bbox_head
        self.share_score_head = share_score_head

        self._build_input_proj_layer(feat_channels)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = DEIMv2TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            use_gateway=use_gateway,
        )
        decoder_layer_wide = DEIMv2TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            use_gateway=use_gateway,
        )
        self.decoder = DEIMv2TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=activation,
        )

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            nn.init.normal_(self.denoising_class_embed.weight[:-1])

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=mlp_act)
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.integral = Integral(self.reg_max)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        dec_score_head = nn.Linear(hidden_dim, num_classes)
        self.dec_score_head = nn.ModuleList(
            [
                (dec_score_head if share_score_head else copy.deepcopy(dec_score_head))
                for _ in range(self.eval_idx + 1)
            ]
            + [
                copy.deepcopy(dec_score_head)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        dec_bbox_head = MLP(
            hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                (dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head))
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(
                    scaled_dim,
                    scaled_dim,
                    4 * (self.reg_max + 1),
                    3,
                    act=mlp_act,
                )
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[index] if index <= self.eval_idx else nn.Identity()
                for index in range(len(self.dec_bbox_head))
            ]
        )

    def _reset_parameters(self, feat_channels):
        bias = _bias_init_with_prob(0.01)
        nn.init.constant_(self.enc_score_head.bias, bias)
        nn.init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        nn.init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        nn.init.constant_(self.pre_bbox_head.layers[-1].bias, 0)
        for cls_head, reg_head in zip(self.dec_score_head, self.dec_bbox_head):
            nn.init.constant_(cls_head.bias, bias)
            if hasattr(reg_head, "layers"):
                nn.init.constant_(reg_head.layers[-1].weight, 0)
                nn.init.constant_(reg_head.layers[-1].bias, 0)
        if self.learn_query_content:
            nn.init.xavier_uniform_(self.tgt_embed.weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for projection, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim and hasattr(projection, "conv"):
                nn.init.xavier_uniform_(projection[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels, self.hidden_dim, 1, bias=False
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
        last_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if last_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        last_channels,
                                        self.hidden_dim,
                                        3,
                                        2,
                                        padding=1,
                                        bias=False,
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                last_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[index](feat) for index, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            base = len(proj_feats)
            for index in range(base, self.num_levels):
                if index == base:
                    proj_feats.append(self.input_proj[index](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[index](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for feat in proj_feats:
            _, _, height, width = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([height, width])
        return torch.concat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes=None,
        grid_size=0.05,
        dtype=torch.float32,
        device="cpu",
    ):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for stride in self.feat_strides:
                spatial_shapes.append([int(eval_h / stride), int(eval_w / stride)])

        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height), torch.arange(width), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor(
                [width, height], dtype=dtype
            )
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            level_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(
                -1, height * width, 4
            )
            anchors.append(level_anchors)

        anchor_tensor = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchor_tensor > self.eps) * (anchor_tensor < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchor_tensor = torch.log(anchor_tensor / (1 - anchor_tensor))
        anchor_tensor = torch.where(valid_mask, anchor_tensor, torch.inf)
        return anchor_tensor, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        denoising_logits=None,
        denoising_bbox_unact=None,
    ):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory
        enc_outputs_logits = self.enc_score_head(memory)

        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            memory, enc_outputs_logits, anchors, self.num_queries
        )
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes_list.append(F.sigmoid(enc_topk_bbox_unact))
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int,
    ):
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        else:
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]),
        )
        topk_logits = (
            outputs_logits.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]),
            )
            if self.training
            else None
        )
        topk_memory = memory.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]),
        )
        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )

        decoder_input = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )
        init_ref_contents = decoder_input[0]
        init_ref_points_unact = decoder_input[1]
        enc_topk_bboxes_list = decoder_input[2]
        enc_topk_logits_list = decoder_input[3]

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = (
            self.decoder(
                init_ref_contents,
                init_ref_points_unact,
                memory,
                spatial_shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                self.pre_bbox_head,
                self.integral,
                self.up,
                self.reg_scale,
                attn_mask=attn_mask,
                dn_meta=dn_meta,
            )
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(
                pre_logits, dn_meta["dn_num_split"], dim=1
            )
            dn_pre_bboxes, pre_bboxes = torch.split(
                pre_bboxes, dn_meta["dn_num_split"], dim=1
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_corners, out_corners = torch.split(
                out_corners, dn_meta["dn_num_split"], dim=2
            )
            dn_out_refs, out_refs = torch.split(
                out_refs, dn_meta["dn_num_split"], dim=2
            )

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(
                enc_topk_logits_list, enc_topk_bboxes_list
            )
            out["pre_outputs"] = {
                "pred_logits": pre_logits,
                "pred_boxes": pre_bboxes,
            }
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}
            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                }
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {"pred_logits": class_item, "pred_boxes": coord_item}
            for class_item, coord_item in zip(outputs_class, outputs_coord)
        ]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        return [
            {
                "pred_logits": class_item,
                "pred_boxes": coord_item,
                "pred_corners": corner_item,
                "ref_points": ref_item,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for class_item, coord_item, corner_item, ref_item in zip(
                outputs_class, outputs_coord, outputs_corners, outputs_ref
            )
        ]

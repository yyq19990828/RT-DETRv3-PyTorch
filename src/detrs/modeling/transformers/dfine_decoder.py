"""Fine-grained distribution refinement decoder primitives."""

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from detrs.core.workspace import register

from .dfine_support import get_contrastive_denoising_training_group
from .dfine_utils import _validate_reg_max, distance2bbox, weighting_function


def _activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(left, right) for left, right in zip(dims[:-1], dims[1:])
        )
        self.act = _activation(act)

    def forward(self, value):
        for index, layer in enumerate(self.layers):
            value = (
                self.act(layer(value)) if index < len(self.layers) - 1 else layer(value)
            )
        return value


def deformable_attention_core_func_v2(
    value, spatial_shapes, sampling_locations, attention_weights, num_points_list
):
    batch_size, num_heads, channels, _ = value[0].shape
    query_length = sampling_locations.shape[1]
    grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    grids = grids.split(num_points_list, dim=-2)
    sampled = []
    for level, (height, width) in enumerate(spatial_shapes):
        level_value = value[level].reshape(
            batch_size * num_heads, channels, height, width
        )
        sampled.append(
            F.grid_sample(
                level_value,
                grids[level],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, query_length, sum(num_points_list)
    )
    output = (torch.cat(sampled, -1) * weights).sum(-1)
    return output.reshape(batch_size, num_heads * channels, query_length).permute(
        0, 2, 1
    )


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        points = (
            list(num_points)
            if isinstance(num_points, (list, tuple))
            else [num_points] * num_levels
        )
        if len(points) != num_levels:
            raise ValueError("num_points level count must equal num_levels")
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points_list = points
        if method not in ("default", "discrete"):
            raise ValueError(f"unsupported cross-attention method: {method}")
        self.method = method
        self.offset_scale = offset_scale
        scales = [1 / count for count in points for _ in range(count)]
        self.register_buffer("num_points_scale", torch.tensor(scales))
        total = num_heads * sum(points)
        self.sampling_offsets = nn.Linear(embed_dim, total * 2)
        self.attention_weights = nn.Linear(embed_dim, total)
        nn.init.constant_(self.sampling_offsets.weight, 0)
        theta = torch.arange(num_heads) * (2 * math.pi / num_heads)
        grid = torch.stack([theta.cos(), theta.sin()], -1)
        grid = grid / grid.abs().max(-1, keepdim=True).values
        grid = grid.reshape(num_heads, 1, 2).tile(1, sum(points), 1)
        scale = torch.cat([torch.arange(1, count + 1) for count in points]).reshape(
            1, -1, 1
        )
        self.sampling_offsets.bias.data.copy_((grid * scale).flatten())
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        if method == "discrete":
            self.sampling_offsets.requires_grad_(False)

    def forward(self, query, reference_points, value, spatial_shapes):
        batch_size, query_length = query.shape[:2]
        offsets = self.sampling_offsets(query).reshape(
            batch_size, query_length, self.num_heads, sum(self.num_points_list), 2
        )
        weights = F.softmax(
            self.attention_weights(query).reshape(
                batch_size, query_length, self.num_heads, sum(self.num_points_list)
            ),
            -1,
        )
        if reference_points.shape[-1] != 4:
            raise ValueError(
                "D-FINE reference_points must contain four box coordinates"
            )
        scale = self.num_points_scale.to(query.dtype).unsqueeze(-1)
        locations = reference_points[:, :, None, :, :2] + (
            offsets * scale * reference_points[:, :, None, :, 2:] * self.offset_scale
        )
        return deformable_attention_core_func_v2(
            value, spatial_shapes, locations, weights, self.num_points_list
        )


class Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        nn.init.constant_(self.gate.bias, 0)
        nn.init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, first, second):
        first_gate, second_gate = torch.sigmoid(
            self.gate(torch.cat([first, second], -1))
        ).chunk(2, -1)
        return self.norm(first_gate * first + second_gate * second)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
    ):
        super().__init__()
        if layer_scale is not None:
            d_model = round(layer_scale * d_model)
            dim_feedforward = round(layer_scale * dim_feedforward)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        target,
        reference_points,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos=None,
    ):
        query = target if query_pos is None else target + query_pos
        attended = self.self_attn(query, query, target, attn_mask=attn_mask)[0]
        target = self.norm1(target + self.dropout1(attended))
        query = target if query_pos is None else target + query_pos
        target = self.gateway(
            target,
            self.dropout2(
                self.cross_attn(query, reference_points, value, spatial_shapes)
            ),
        )
        update = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        return self.norm3((target + self.dropout4(update)).clamp(-65504, 65504))


class Integral(nn.Module):
    def __init__(self, reg_max=32):
        super().__init__()
        _validate_reg_max(reg_max)
        self.reg_max = reg_max

    def forward(self, value, project):
        shape = value.shape
        probabilities = F.softmax(value.reshape(-1, self.reg_max + 1), 1)
        # project comes from a buffer (or is derived from model parameters),
        # so it already shares value's device; a .to(value.device) here would
        # freeze the trace-time device and break cross-device TorchScript.
        result = F.linear(probabilities, project).reshape(-1, 4)
        return result.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act="relu"):
        super().__init__()
        _validate_reg_max(reg_max)
        if k > reg_max + 1:
            raise ValueError("k cannot exceed the number of regression bins")
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        nn.init.constant_(self.reg_conf.layers[-1].bias, 0)
        nn.init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        batch_size, length = pred_corners.shape[:2]
        probability = F.softmax(
            pred_corners.reshape(batch_size, length, 4, self.reg_max + 1), -1
        )
        topk = probability.topk(self.k, dim=-1).values
        statistics = torch.cat([topk, topk.mean(-1, keepdim=True)], -1)
        return scores + self.reg_conf(statistics.reshape(batch_size, length, -1))


class TransformerDecoder(nn.Module):
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
        _validate_reg_max(reg_max)
        self.num_head = num_head
        self.up = up
        self.reg_scale = reg_scale
        self.reg_max = reg_max
        self.layer_scale = layer_scale
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        if not 0 <= self.eval_idx < num_layers:
            raise ValueError("eval_idx must select a decoder layer")
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [
                copy.deepcopy(decoder_layer_wide)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.lqe_layers = nn.ModuleList(
            [LQE(4, 64, 2, reg_max, act=act) for _ in range(num_layers)]
        )

    def value_op(self, memory, memory_mask, spatial_shapes):
        value = memory
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        return value.permute(0, 2, 3, 1).split([h * w for h, w in spatial_shapes], -1)

    def convert_to_deploy(self):
        if hasattr(self, "project"):
            return self
        self.register_buffer(
            "project",
            weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True),
        )
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]]
        )
        return self

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
        value = self.value_op(memory, memory_mask, spatial_shapes)
        boxes, logits, corners, refs = [], [], [], []
        project = getattr(
            self, "project", weighting_function(self.reg_max, up, reg_scale)
        )
        ref_points_detach = ref_points_unact.sigmoid()
        for index, layer in enumerate(self.layers):
            query_pos = query_pos_head(ref_points_detach).clamp(-10, 10)
            output = layer(
                output,
                ref_points_detach.unsqueeze(2),
                value,
                spatial_shapes,
                attn_mask,
                query_pos,
            )
            if index == 0:
                pre_bboxes = (
                    pre_bbox_head(output) + torch.logit(ref_points_detach, eps=1e-5)
                ).sigmoid()
                pre_scores = score_head[0](output)
                initial_refs = pre_bboxes.detach()
            pred_corners = (
                bbox_head[index](output + output_detach) + pred_corners_undetach
            )
            refined = distance2bbox(
                initial_refs, integral(pred_corners, project), reg_scale
            )
            if self.training or index == self.eval_idx:
                scores = self.lqe_layers[index](score_head[index](output), pred_corners)
                boxes.append(refined)
                logits.append(scores)
                corners.append(pred_corners)
                refs.append(initial_refs)
                if not self.training:
                    break
            pred_corners_undetach = pred_corners
            ref_points_detach = refined.detach()
            output_detach = output.detach()
        return tuple(torch.stack(items) for items in (boxes, logits, corners, refs)) + (
            pre_bboxes,
            pre_scores,
        )


@register
class DFINETransformer(nn.Module):
    """D-FINE decoder head: fine-grained distribution refinement (FDR).

    Selects top encoder features as initial queries, then refines box
    distributions over `num_layers` decoder layers with optional denoising
    groups and a shared/self-scaling localization design.

    Args:
        num_classes (int): Number of foreground classes.
        hidden_dim (int): Embedding dimension of queries and decoder.
        num_queries (int): Number of object queries.
        feat_channels (tuple): Channels of the input feature levels.
        feat_strides (tuple): Strides of the input feature levels.
        num_levels (int): Number of feature levels used.
        num_points (int): Sampling points per level for deformable attention.
        nhead (int): Attention heads.
        num_layers (int): Decoder (refinement) layers.
        dim_feedforward (int): Width of the decoder FFN.
        dropout (float): Dropout rate.
        activation (str): FFN activation.
        num_denoising (int): Number of contrastive denoising queries; `0`
            disables denoising.
        label_noise_ratio (float): Label noise ratio for denoising.
        box_noise_scale (float): Box noise scale for denoising.
        learn_query_content (bool): Learn query content embeddings instead
            of using encoder features.
        eval_spatial_size (tuple|None): Fixed input size used to precompute
            positional encoding for export.
        eval_idx (int): Index of the decoder layer returned at inference;
            `-1` uses the last layer.
        eps (float): Numerical epsilon.
        aux_loss (bool): Whether all decoder layers produce loss outputs.
        cross_attn_method (str): Cross-attention implementation variant.
        query_select_method (str): Top-query selection implementation.
        reg_max (int): Discrete bins of the box distribution.
        reg_scale (float): Scale factor mapping distributions to box
            offsets.
        layer_scale (int): Isotropic layer scaling factor.
        mlp_act (str): Activation of the box MLP.
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
    ):
        super().__init__()
        feat_channels = list(feat_channels)
        feat_strides = list(feat_strides)
        if len(feat_channels) > num_levels:
            raise ValueError("feat_channels cannot exceed num_levels")
        if len(feat_strides) != len(feat_channels):
            raise ValueError("feat_strides and feat_channels must have equal length")
        if num_levels <= 0 or not feat_channels:
            raise ValueError("at least one feature level is required")
        _validate_reg_max(reg_max)
        if query_select_method not in ("default", "one2many", "agnostic"):
            raise ValueError("unsupported D-FINE query_select_method")
        if cross_attn_method not in ("default", "discrete"):
            raise ValueError("unsupported D-FINE cross_attn_method")
        while len(feat_strides) < num_levels:
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.feat_channels = feat_channels
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
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self._build_input_proj_layer(feat_channels)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
        )
        self.decoder = TransformerDecoder(
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
            act=mlp_act,
        )
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed: nn.Module = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            nn.init.normal_(self.denoising_class_embed.weight[:-1])

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)
        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    ("norm", nn.LayerNorm(hidden_dim)),
                ]
            )
        )
        self.enc_score_head = nn.Linear(
            hidden_dim, 1 if query_select_method == "agnostic" else num_classes
        )
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        if not 0 <= self.eval_idx < num_layers:
            raise ValueError("eval_idx must select a decoder layer")
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [
                nn.Linear(scaled_dim, num_classes)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(
                    hidden_dim,
                    hidden_dim,
                    4 * (reg_max + 1),
                    3,
                    act=mlp_act,
                )
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(
                    scaled_dim,
                    scaled_dim,
                    4 * (reg_max + 1),
                    3,
                    act=mlp_act,
                )
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.integral = Integral(reg_max)
        if eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)
        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        if getattr(self, "_deployed", False):
            return self
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                head if index <= self.eval_idx else nn.Identity()
                for index, head in enumerate(self.dec_bbox_head)
            ]
        )
        if hasattr(self, "denoising_class_embed"):
            self.denoising_class_embed = nn.Identity()
        self.decoder.convert_to_deploy()
        self._deployed = True
        return self

    def _reset_parameters(self, feat_channels):
        bias = float(-math.log(99))
        nn.init.constant_(self.enc_score_head.bias, bias)
        nn.init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        nn.init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        nn.init.constant_(self.pre_bbox_head.layers[-1].bias, 0)
        for score, box in zip(self.dec_score_head, self.dec_bbox_head):
            nn.init.constant_(score.bias, bias)
            nn.init.constant_(box.layers[-1].weight, 0)
            nn.init.constant_(box.layers[-1].bias, 0)
        nn.init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            nn.init.xavier_uniform_(self.tgt_embed.weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for projection, channels in zip(self.input_proj, feat_channels):
            if channels != self.hidden_dim:
                nn.init.xavier_uniform_(projection[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for channels in feat_channels:
            if channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(channels, self.hidden_dim, 1, bias=False),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
        channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        channels,
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
                channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        projected = [
            self.input_proj[index](feature) for index, feature in enumerate(feats)
        ]
        for index in range(len(projected), self.num_levels):
            source = feats[-1] if index == len(feats) else projected[-1]
            projected.append(self.input_proj[index](source))
        memory, spatial_shapes = [], []
        for feature in projected:
            height, width = feature.shape[-2:]
            memory.append(feature.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([height, width])
        return torch.cat(memory, 1), spatial_shapes

    def _generate_anchors(
        self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"
    ):
        if spatial_shapes is None:
            if self.eval_spatial_size is None:
                raise ValueError("spatial_shapes or eval_spatial_size is required")
            height, width = self.eval_spatial_size
            spatial_shapes = [
                [int(height / stride), int(width / stride)]
                for stride in self.feat_strides
            ]
        anchor_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height), torch.arange(width), indexing="ij"
            )
            grid = torch.stack([grid_x, grid_y], -1)
            grid = (grid.unsqueeze(0) + 0.5) / torch.tensor(
                [width, height], dtype=dtype
            )
            size = torch.ones_like(grid) * grid_size * (2.0**level)
            anchor_list.append(
                torch.cat([grid, size], -1).reshape(-1, height * width, 4)
            )
        anchors = torch.cat(anchor_list, 1).to(device)
        valid = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.where(valid, torch.log(anchors / (1 - anchors)), torch.inf)
        return anchors, valid

    def _select_topk(self, memory, logits, anchors, topk):
        if topk > memory.shape[1]:
            raise ValueError("num_queries cannot exceed available encoder positions")
        if self.query_select_method == "default":
            indices = torch.topk(logits.max(-1).values, topk, dim=-1).indices
        elif self.query_select_method == "one2many":
            indices = torch.topk(logits.flatten(1), topk, dim=-1).indices
            indices = indices // self.num_classes
        else:
            indices = torch.topk(logits.squeeze(-1), topk, dim=-1).indices
        values = (memory, logits, anchors)
        selected = tuple(
            value.gather(1, indices.unsqueeze(-1).repeat(1, 1, value.shape[-1]))
            for value in values
        )
        return selected[0], selected[1] if self.training else None, selected[2]

    def _get_decoder_input(
        self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None
    ):
        if self.training or self.eval_spatial_size is None:
            anchors, valid = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors, valid = self.anchors, self.valid_mask
        anchors = anchors.expand(memory.shape[0], -1, -1)
        output_memory = self.enc_output(valid.to(memory.dtype) * memory)
        encoder_logits = self.enc_score_head(output_memory)
        topk_memory, topk_logits, topk_anchors = self._select_topk(
            output_memory, encoder_logits, anchors, self.num_queries
        )
        topk_bbox_unact = self.enc_bbox_head(topk_memory) + topk_anchors
        encoder_boxes, encoder_scores = [], []
        if self.training:
            encoder_boxes.append(topk_bbox_unact.sigmoid())
            encoder_scores.append(topk_logits)
        content = (
            self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
            if self.learn_query_content
            else topk_memory.detach()
        )
        topk_bbox_unact = topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            topk_bbox_unact = torch.cat([denoising_bbox_unact, topk_bbox_unact], 1)
            content = torch.cat([denoising_logits, content], 1)
        return content, topk_bbox_unact, encoder_boxes, encoder_scores

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0:
            if targets is None:
                raise ValueError("targets are required for D-FINE denoising training")
            denoising_logits, denoising_boxes, mask, metadata = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    self.num_denoising,
                    self.label_noise_ratio,
                    1.0,
                )
            )
        else:
            denoising_logits = denoising_boxes = mask = metadata = None
        content, references, encoder_boxes, encoder_logits = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_boxes
        )
        boxes, logits, corners, refs, pre_boxes, pre_logits = self.decoder(
            content,
            references,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=mask,
            dn_meta=metadata,
        )
        if self.training and metadata is not None and metadata["dn_num_split"][0] > 0:
            dn_pre_logits, pre_logits = torch.split(
                pre_logits, metadata["dn_num_split"], 1
            )
            dn_pre_boxes, pre_boxes = torch.split(
                pre_boxes, metadata["dn_num_split"], 1
            )
            dn_boxes, boxes = torch.split(boxes, metadata["dn_num_split"], 2)
            dn_logits, logits = torch.split(logits, metadata["dn_num_split"], 2)
            dn_corners, corners = torch.split(corners, metadata["dn_num_split"], 2)
            dn_refs, refs = torch.split(refs, metadata["dn_num_split"], 2)
        else:
            metadata = None

        if self.training:
            output = {
                "pred_logits": logits[-1],
                "pred_boxes": boxes[-1],
                "pred_corners": corners[-1],
                "ref_points": refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            output = {"pred_logits": logits[-1], "pred_boxes": boxes[-1]}
        if self.training and self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss2(
                logits[:-1],
                boxes[:-1],
                corners[:-1],
                refs[:-1],
                corners[-1],
                logits[-1],
            )
            output["enc_aux_outputs"] = self._set_aux_loss(
                encoder_logits, encoder_boxes
            )
            output["pre_outputs"] = {
                "pred_logits": pre_logits,
                "pred_boxes": pre_boxes,
            }
            output["enc_meta"] = {
                "class_agnostic": self.query_select_method == "agnostic"
            }
            if metadata is not None:
                output["dn_outputs"] = self._set_aux_loss2(
                    dn_logits,
                    dn_boxes,
                    dn_corners,
                    dn_refs,
                    dn_corners[-1],
                    dn_logits[-1],
                )
                output["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_boxes,
                }
                output["dn_meta"] = metadata
        return output

    @torch.jit.unused
    def _set_aux_loss(self, classes, coordinates):
        return [
            {"pred_logits": class_value, "pred_boxes": coordinate_value}
            for class_value, coordinate_value in zip(classes, coordinates)
        ]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        classes,
        coordinates,
        corners,
        refs,
        teacher_corners=None,
        teacher_logits=None,
    ):
        return [
            {
                "pred_logits": class_value,
                "pred_boxes": coordinate_value,
                "pred_corners": corner_value,
                "ref_points": ref_value,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for class_value, coordinate_value, corner_value, ref_value in zip(
                classes, coordinates, corners, refs
            )
        ]


__all__ = [
    "DFINETransformer",
    "Integral",
    "LQE",
    "MSDeformableAttention",
    "TransformerDecoder",
    "TransformerDecoderLayer",
]

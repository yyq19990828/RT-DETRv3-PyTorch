"""Minimal RT-DETRv2 decoder slice used by DEIM models."""

from __future__ import annotations

import copy
import functools
import math
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from detrs.core.workspace import register

from .utils import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, inverse_sigmoid

__all__ = ["RTDETRTransformerv2"]


_VARIANT_PROFILES = {
    "r18vd": {
        "depth": 18,
        "backbone_channels": (128, 256, 512),
        "feat_channels": (256, 256, 256),
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "num_layers": 3,
        "eval_idx": -1,
    },
    "r34vd": {
        "depth": 34,
        "backbone_channels": (128, 256, 512),
        "feat_channels": (256, 256, 256),
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "num_layers": 4,
        "eval_idx": -1,
    },
    "r50vd_m": {
        "depth": 50,
        "backbone_channels": (512, 1024, 2048),
        "feat_channels": (256, 256, 256),
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "num_layers": 3,
        "eval_idx": 2,
    },
    "r50vd": {
        "depth": 50,
        "backbone_channels": (512, 1024, 2048),
        "feat_channels": (256, 256, 256),
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "num_layers": 6,
        "eval_idx": -1,
    },
    "r101vd": {
        "depth": 101,
        "backbone_channels": (512, 1024, 2048),
        "feat_channels": (384, 384, 384),
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "num_layers": 6,
        "eval_idx": -1,
    },
}
_V3_CONFIG_KEYS = {
    "backbone_feat_channels",
    "num_decoder_layers",
    "num_decoder_points",
    "learnt_init_query",
    "o2m_branch",
    "num_noises",
}


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    raise ValueError("unsupported RT-DETRv2 activation: {}".format(name))


class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip([input_dim] + hidden, hidden + [output_dim])
        )
        self.num_layers = num_layers
        self.act = _activation(act)

    def forward(self, value):
        for index, layer in enumerate(self.layers):
            value = (
                self.act(layer(value)) if index < self.num_layers - 1 else layer(value)
            )
        return value


def _deformable_attention_core_v2(
    value,
    spatial_shapes,
    sampling_locations,
    attention_weights,
    num_points_list,
    method="default",
    value_shape="default",
):
    if value_shape == "default":
        batch_size, num_heads, channels, _ = value[0].shape
    elif value_shape == "reshape":
        batch_size, _, num_heads, channels = value.shape
        split_shapes = [height * width for height, width in spatial_shapes]
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shapes, dim=-1)
    else:
        raise ValueError("unsupported RT-DETRv2 value_shape: {}".format(value_shape))

    query_length = sampling_locations.shape[1]
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        raise ValueError("unsupported RT-DETRv2 attention method: {}".format(method))
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    location_groups = sampling_grids.split(num_points_list, dim=-2)

    sampled_values = []
    for level, (height, width) in enumerate(spatial_shapes):
        level_value = value[level].reshape(
            batch_size * num_heads, channels, height, width
        )
        level_grid = location_groups[level]
        if method == "default":
            sampled = F.grid_sample(
                level_value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        else:
            size = torch.tensor([[width, height]], device=level_value.device)
            coordinates = (level_grid * size + 0.5).to(torch.int64)
            coordinates = coordinates.clamp(0, height - 1).reshape(
                batch_size * num_heads, query_length * num_points_list[level], 2
            )
            batch_indices = (
                torch.arange(coordinates.shape[0], device=level_value.device)
                .unsqueeze(-1)
                .repeat(1, coordinates.shape[1])
            )
            sampled = (
                level_value[batch_indices, :, coordinates[..., 1], coordinates[..., 0]]
                .permute(0, 2, 1)
                .reshape(
                    batch_size * num_heads,
                    channels,
                    query_length,
                    num_points_list[level],
                )
            )
        sampled_values.append(sampled)

    weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, query_length, sum(num_points_list)
    )
    output = (torch.cat(sampled_values, dim=-1) * weights).sum(-1)
    return output.reshape(batch_size, num_heads * channels, query_length).permute(
        0, 2, 1
    )


class _MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
        value_shape="default",
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if isinstance(num_points, int):
            num_points = [num_points] * num_levels
        if len(num_points) != num_levels or any(point <= 0 for point in num_points):
            raise ValueError("num_points must contain one positive value per level")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points_list = list(num_points)
        self.offset_scale = offset_scale
        self.total_points = num_heads * sum(self.num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads
        point_scale = [1 / count for count in num_points for _ in range(count)]
        self.register_buffer(
            "num_points_scale", torch.tensor(point_scale, dtype=torch.float32)
        )
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.core = functools.partial(
            _deformable_attention_core_v2,
            method=method,
            value_shape=value_shape,
        )
        self._reset_parameters()
        if method == "discrete":
            for parameter in self.sampling_offsets.parameters():
                parameter.requires_grad = False

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (
            2 * math.pi / self.num_heads
        )
        grid = torch.stack([angles.cos(), angles.sin()], -1)
        grid = grid / grid.abs().max(-1, keepdim=True).values
        grid = grid.reshape(self.num_heads, 1, 2).tile(
            [1, sum(self.num_points_list), 1]
        )
        scaling = torch.cat(
            [torch.arange(1, count + 1) for count in self.num_points_list]
        ).reshape(1, -1, 1)
        self.sampling_offsets.bias.data.copy_((grid * scaling).flatten())
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, spatial_shapes, value_mask=None):
        batch_size, query_length = query.shape[:2]
        value_length = value.shape[1]
        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(batch_size, value_length, self.num_heads, self.head_dim)
        offsets = self.sampling_offsets(query).reshape(
            batch_size,
            query_length,
            self.num_heads,
            sum(self.num_points_list),
            2,
        )
        weights = F.softmax(
            self.attention_weights(query).reshape(
                batch_size,
                query_length,
                self.num_heads,
                sum(self.num_points_list),
            ),
            dim=-1,
        )
        if reference_points.shape[-1] == 2:
            normalizer = (
                torch.as_tensor(spatial_shapes, device=query.device, dtype=query.dtype)
                .flip([1])
                .reshape(1, 1, 1, self.num_levels, 1, 2)
            )
            locations = (
                reference_points.reshape(
                    batch_size, query_length, 1, self.num_levels, 1, 2
                )
                + offsets / normalizer
            )
        elif reference_points.shape[-1] == 4:
            point_scale = self.num_points_scale.to(query.dtype).unsqueeze(-1)
            offset = (
                offsets
                * point_scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError("reference_points must end in 2 or 4 coordinates")
        return self.output_proj(
            self.core(value, spatial_shapes, locations, weights, self.num_points_list)
        )


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        dim_feedforward,
        dropout,
        activation,
        n_levels,
        n_points,
        cross_attn_method,
        value_shape,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = _MSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
            value_shape=value_shape,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(
        self,
        target,
        reference_points,
        memory,
        spatial_shapes,
        attn_mask=None,
        memory_mask=None,
        query_pos_embed=None,
    ):
        query = key = target if query_pos_embed is None else target + query_pos_embed
        target2, _ = self.self_attn(query, key, target, attn_mask=attn_mask)
        target = self.norm1(target + self.dropout1(target2))
        query = target if query_pos_embed is None else target + query_pos_embed
        target2 = self.cross_attn(
            query, reference_points, memory, spatial_shapes, memory_mask
        )
        target = self.norm2(target + self.dropout2(target2))
        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        return self.norm3(target + self.dropout4(target2))


class _Decoder(nn.Module):
    def __init__(self, hidden_dim, layer, num_layers, eval_idx):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
    ):
        output = target
        boxes, logits = [], []
        detached_reference = ref_points_unact.sigmoid()
        reference = detached_reference
        for index, layer in enumerate(self.layers):
            query_position = query_pos_head(detached_reference)
            output = layer(
                output,
                detached_reference.unsqueeze(2),
                memory,
                spatial_shapes,
                attn_mask,
                memory_mask,
                query_position,
            )
            intermediate = (
                bbox_head[index](output) + inverse_sigmoid(detached_reference)
            ).sigmoid()
            if self.training:
                logits.append(score_head[index](output))
                boxes.append(
                    intermediate
                    if index == 0
                    else (
                        bbox_head[index](output) + inverse_sigmoid(reference)
                    ).sigmoid()
                )
            elif index == self.eval_idx:
                logits.append(score_head[index](output))
                boxes.append(intermediate)
                break
            reference = intermediate
            detached_reference = intermediate.detach()
        return torch.stack(boxes), torch.stack(logits)


def _denoising_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising,
    label_noise_ratio,
    box_noise_scale,
):
    if num_denoising <= 0:
        return None, None, None, None
    counts = [len(target["labels"]) for target in targets]
    max_count = max(counts)
    if max_count == 0:
        return None, None, None, None
    device = targets[0]["labels"].device
    groups = max(1, num_denoising // max_count)
    classes = torch.full(
        [len(targets), max_count], num_classes, dtype=torch.int32, device=device
    )
    boxes = torch.zeros([len(targets), max_count, 4], device=device)
    valid = torch.zeros([len(targets), max_count], dtype=torch.bool, device=device)
    for index, target in enumerate(targets):
        count = counts[index]
        if count:
            classes[index, :count] = target["labels"]
            boxes[index, :count] = target["boxes"]
            valid[index, :count] = True
    classes = classes.tile([1, 2 * groups])
    boxes = boxes.tile([1, 2 * groups, 1])
    valid = valid.tile([1, 2 * groups])
    negative = torch.zeros([len(targets), max_count * 2, 1], device=device)
    negative[:, max_count:] = 1
    negative = negative.tile([1, groups, 1])
    positive = (1 - negative).squeeze(-1) * valid
    positive_indices = torch.split(
        torch.nonzero(positive)[:, 1], [count * groups for count in counts]
    )
    denoising_count = max_count * 2 * groups
    if label_noise_ratio > 0:
        noise = torch.rand_like(classes, dtype=torch.float) < label_noise_ratio * 0.5
        replacements = torch.randint_like(classes, 0, num_classes)
        classes = torch.where(noise & valid, replacements, classes)
    if box_noise_scale > 0:
        xyxy = bbox_cxcywh_to_xyxy(boxes)
        difference = boxes[..., 2:].tile([1, 1, 2]) * 0.5 * box_noise_scale
        signs = torch.randint_like(boxes, 0, 2) * 2.0 - 1.0
        random = torch.rand_like(boxes)
        random = (random + 1.0) * negative + random * (1 - negative)
        boxes = bbox_xyxy_to_cxcywh((xyxy + signs * random * difference).clip(0, 1))
        boxes[boxes < 0] *= -1
        boxes = inverse_sigmoid(boxes)
    logits = class_embed(classes)
    mask = torch.zeros(
        [denoising_count + num_queries] * 2, dtype=torch.bool, device=device
    )
    mask[denoising_count:, :denoising_count] = True
    for index in range(groups):
        start, end = max_count * 2 * index, max_count * 2 * (index + 1)
        mask[start:end, end:denoising_count] = True
        mask[start:end, :start] = True
    metadata = {
        "dn_positive_idx": positive_indices,
        "dn_num_group": groups,
        "dn_num_split": [denoising_count, num_queries],
    }
    return logits, boxes, mask, metadata


@register
class RTDETRTransformerv2(nn.Module):
    """Pinned DEIM RT-DETRv2 decoder, restricted to the five planned variants.

    Standard RT-DETR decoder (no D-FINE distribution refinement); channel,
    depth and FFN defaults resolve from `variant` when not set explicitly.

    Args:
        variant (str): One of the pinned sizes `s`, `m`, `m-star`, `l`, `x`.
        num_classes (int): Number of foreground classes.
        hidden_dim (int|None): Embedding dimension; derived from `variant`
            when omitted.
        num_queries (int): Number of object queries.
        feat_channels (tuple|None): Input channels; derived from `variant`
            when omitted.
        feat_strides (tuple): Strides of the input feature levels.
        num_levels (int): Number of feature levels used.
        num_points (tuple): Sampling points per level for deformable
            attention.
        nhead (int): Attention heads.
        num_layers (int|None): Decoder layers; derived from `variant` when
            omitted.
        dim_feedforward (int|None): FFN width; derived from `variant` when
            omitted.
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
        value_shape (str): Reshape strategy of the value tensors.
        mlp_act (str): Activation of the box MLP.
        query_pos_method (str): Query positional encoding method.
    """

    __shared__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        variant: str,
        num_classes=80,
        hidden_dim=None,
        num_queries=300,
        feat_channels=None,
        feat_strides=(8, 16, 32),
        num_levels=3,
        num_points=(4, 4, 4),
        nhead=8,
        num_layers=None,
        dim_feedforward=None,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=None,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        value_shape="reshape",
        mlp_act="relu",
        query_pos_method="default",
    ):
        super().__init__()
        if variant not in _VARIANT_PROFILES:
            raise ValueError(
                "unsupported RT-DETRv2 depth/variant {!r}; expected one of {}".format(
                    variant, ", ".join(_VARIANT_PROFILES)
                )
            )
        profile = _VARIANT_PROFILES[variant]
        hidden_dim = profile["hidden_dim"] if hidden_dim is None else hidden_dim
        dim_feedforward = (
            profile["dim_feedforward"] if dim_feedforward is None else dim_feedforward
        )
        feat_channels = tuple(
            feat_channels or cast(tuple[int, ...], profile["feat_channels"])
        )
        num_layers = profile["num_layers"] if num_layers is None else num_layers
        eval_idx = profile["eval_idx"] if eval_idx is None else eval_idx
        expected = (
            profile["feat_channels"],
            profile["hidden_dim"],
            profile["dim_feedforward"],
            profile["num_layers"],
            profile["eval_idx"],
        )
        actual = (feat_channels, hidden_dim, dim_feedforward, num_layers, eval_idx)
        if actual != expected:
            raise ValueError(
                "RT-DETRv2 {} config mismatch: expected feat_channels={}, hidden_dim={}, dim_feedforward={}, num_layers={}, eval_idx={}".format(
                    variant, *expected
                )
            )
        if len(feat_channels) > num_levels or len(feat_strides) != len(feat_channels):
            raise ValueError(
                "RT-DETRv2 feature levels do not match the selected profile"
            )
        if query_select_method not in ("default", "one2many", "agnostic"):
            raise ValueError("unsupported RT-DETRv2 query_select_method")
        if cross_attn_method not in ("default", "discrete"):
            raise ValueError("unsupported RT-DETRv2 cross_attn_method")

        self.variant = variant
        self.backbone_depth = profile["depth"]
        self.backbone_channels = profile["backbone_channels"]
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = list(feat_strides)
        while len(self.feat_strides) < num_levels:
            self.feat_strides.append(self.feat_strides[-1] * 2)
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self._build_input_proj(feat_channels)
        layer = _DecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method,
            value_shape,
        )
        self.decoder = _Decoder(hidden_dim, layer, num_layers, eval_idx)
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
        if query_pos_method == "as_reg":
            self.query_pos_head = _MLP(4, hidden_dim, hidden_dim, 3, act=mlp_act)
        elif query_pos_method == "default":
            self.query_pos_head = _MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)
        else:
            raise ValueError("unsupported RT-DETRv2 query_pos_method")
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
        self.enc_bbox_head = _MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [_MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act) for _ in range(num_layers)]
        )
        if eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)
        self._reset_parameters()

    @classmethod
    def from_config(cls, cfg, input_shape=None):
        bad_keys = sorted(_V3_CONFIG_KEYS.intersection(cfg))
        if bad_keys or str(cfg.get("family", "")).lower() in ("rtdetrv3", "v3"):
            raise ValueError(
                "RTDETRTransformerv2 rejects RT-DETRv3 config{}".format(
                    ": " + ", ".join(bad_keys) if bad_keys else ""
                )
            )
        variant = cfg.get("variant")
        if variant not in _VARIANT_PROFILES:
            raise ValueError(
                "RTDETRTransformerv2 requires one of the five planned variants"
            )
        if input_shape is not None:
            channels = tuple(shape.channels for shape in input_shape)
            expected = _VARIANT_PROFILES[variant]["feat_channels"]
            if channels != expected:
                raise ValueError(
                    "RT-DETRv2 {} encoder channels must be {}, got {}".format(
                        variant, expected, channels
                    )
                )
        return {}

    def _build_input_proj(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for channels in feat_channels:
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

    def _reset_parameters(self):
        bias = float(-math.log(99))
        nn.init.constant_(self.enc_score_head.bias, bias)
        nn.init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for score, box in zip(self.dec_score_head, self.dec_bbox_head):
            nn.init.constant_(score.bias, bias)
            nn.init.constant_(box.layers[-1].weight, 0)
            nn.init.constant_(box.layers[-1].bias, 0)
        nn.init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            nn.init.xavier_uniform_(self.tgt_embed.weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for projection in self.input_proj:
            nn.init.xavier_uniform_(projection[0].weight)

    def _encoder_input(self, features):
        projected = [
            self.input_proj[index](feature) for index, feature in enumerate(features)
        ]
        for index in range(len(projected), self.num_levels):
            source = features[-1] if index == len(features) else projected[-1]
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
            grid = torch.stack([grid_x, grid_y], dim=-1)
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

    def _select_topk(self, memory, logits, coordinates, topk):
        if self.query_select_method == "default":
            indices = torch.topk(logits.max(-1).values, topk, dim=-1).indices
        elif self.query_select_method == "one2many":
            indices = torch.topk(logits.flatten(1), topk, dim=-1).indices
            indices = indices // self.num_classes
        else:
            indices = torch.topk(logits.squeeze(-1), topk, dim=-1).indices
        return tuple(
            value.gather(1, indices.unsqueeze(-1).repeat(1, 1, value.shape[-1]))
            for value in (memory, logits, coordinates)
        )

    def _decoder_input(
        self, memory, spatial_shapes, denoising_logits=None, denoising_boxes=None
    ):
        if self.training or self.eval_spatial_size is None:
            anchors, valid = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors, valid = self.anchors, self.valid_mask
        memory = valid.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        encoder_logits = self.enc_score_head(output_memory)
        encoder_coordinates = self.enc_bbox_head(output_memory) + anchors
        topk_memory, topk_logits, topk_coordinates = self._select_topk(
            output_memory, encoder_logits, encoder_coordinates, self.num_queries
        )
        encoder_boxes, encoder_scores = [], []
        if self.training:
            encoder_boxes.append(topk_coordinates.sigmoid())
            encoder_scores.append(topk_logits)
        content = (
            self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
            if self.learn_query_content
            else topk_memory.detach()
        )
        topk_coordinates = topk_coordinates.detach()
        if denoising_boxes is not None:
            topk_coordinates = torch.cat([denoising_boxes, topk_coordinates], 1)
            content = torch.cat([denoising_logits, content], 1)
        return content, topk_coordinates, encoder_boxes, encoder_scores

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._encoder_input(feats)
        if self.training and self.num_denoising > 0:
            if targets is None:
                raise ValueError("targets are required for RT-DETRv2 training")
            denoising_logits, denoising_boxes, mask, metadata = _denoising_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                self.num_denoising,
                self.label_noise_ratio,
                self.box_noise_scale,
            )
        else:
            denoising_logits = denoising_boxes = mask = metadata = None
        content, references, encoder_boxes, encoder_logits = self._decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_boxes
        )
        boxes, logits = self.decoder(
            content,
            references,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=mask,
        )
        if self.training and metadata is not None:
            denoising_boxes, boxes = torch.split(boxes, metadata["dn_num_split"], dim=2)
            denoising_logits, logits = torch.split(
                logits, metadata["dn_num_split"], dim=2
            )
        output = {"pred_logits": logits[-1], "pred_boxes": boxes[-1]}
        if self.training and self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(logits[:-1], boxes[:-1])
            output["enc_aux_outputs"] = self._set_aux_loss(
                encoder_logits, encoder_boxes
            )
            output["enc_meta"] = {
                "class_agnostic": self.query_select_method == "agnostic"
            }
            if metadata is not None:
                output["dn_outputs"] = self._set_aux_loss(
                    denoising_logits, denoising_boxes
                )
                output["dn_meta"] = metadata
        return output

    @torch.jit.unused
    def _set_aux_loss(self, classes, coordinates):
        return [
            {"pred_logits": class_value, "pred_boxes": coordinate_value}
            for class_value, coordinate_value in zip(classes, coordinates)
        ]

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        keys = set(state_dict)
        if any(
            key.startswith(("map_memory.", "enc_output.0.", "enc_score_head.0."))
            for key in keys
        ):
            raise ValueError("RTDETRTransformerv2 rejects RT-DETRv3 checkpoint")
        expected = self.state_dict()
        missing = sorted(set(expected).difference(keys))
        unexpected = sorted(keys.difference(expected))
        mismatched = sorted(
            key
            for key in set(expected).intersection(keys)
            if not isinstance(state_dict[key], torch.Tensor)
            or state_dict[key].shape != expected[key].shape
        )
        if strict and (missing or unexpected or mismatched):
            raise RuntimeError(
                "RT-DETRv2 checkpoint validation failed before mutation: missing={}, unexpected={}, mismatched={}".format(
                    missing, unexpected, mismatched
                )
            )
        if mismatched:
            raise RuntimeError(
                "RT-DETRv2 checkpoint tensor layout mismatch before mutation: {}".format(
                    mismatched
                )
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

"""DEIMv2 LiteEncoder for the Pico/Femto/Atto two-scale variants.

Ported from Intellindust-AI-Lab/DEIMv2@add5bcd (engine/deim/lite_encoder.py,
Apache-2.0; GAP bi-fusion from Meituan YOLOv6). Reuses the verified
ConvNormLayer_fuse and RepNCSPELAN4 primitives from the D-FINE encoder.
"""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from detrs.core.workspace import register
from detrs.modeling.shape_spec import ShapeSpec

from .dfine_hybrid_encoder import ConvNormLayer_fuse, RepNCSPELAN4, _activation

__all__ = ["LiteEncoder"]


class GAPFusion(nn.Module):
    """Bi-Fusion block adding global average pooled context."""

    def __init__(self, in_channels, out_channels, act=None):
        super().__init__()
        self.cv = ConvNormLayer_fuse(out_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1)
        return self.cv(x + gap)


@register
class LiteEncoder(nn.Module):
    """Single-scale two-level FPN/PAN encoder for DEIMv2 tiny variants."""

    __shared__ = ["eval_spatial_size"]

    def __init__(
        self,
        in_channels=(512,),
        feat_strides=(16,),
        hidden_dim=256,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        csp_type="csp2",
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels) + 1)]
        self.out_strides = [*feat_strides, feat_strides[-1] * 2]

        self.input_proj = nn.ModuleList()
        for channels in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(channels, hidden_dim, 1, bias=False),
                            ),
                            ("norm", nn.BatchNorm2d(hidden_dim)),
                        ]
                    )
                )
            )

        down_sample = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            _activation(act),
        )
        self.down_sample1 = copy.deepcopy(down_sample)
        self.down_sample2 = copy.deepcopy(down_sample)

        self.bi_fusion = GAPFusion(hidden_dim, hidden_dim, act=act)

        c1, c2, c3, c4, num_blocks = (
            hidden_dim,
            hidden_dim,
            hidden_dim * 2,
            round(expansion * hidden_dim // 2),
            round(3 * depth_mult),
        )
        fuse_block = RepNCSPELAN4(c1, c2, c3, c4, num_blocks, csp_type=csp_type)
        self.fpn_block = copy.deepcopy(fuse_block)
        self.pan_block = copy.deepcopy(fuse_block)

    def forward(self, feats):
        if len(feats) != len(self.in_channels):
            raise ValueError(
                f"expected {len(self.in_channels)} feature levels, got {len(feats)}"
            )
        projected = [module(feat) for module, feat in zip(self.input_proj, feats)]
        projected.append(self.down_sample1(projected[-1]))
        projected[-1] = self.bi_fusion(projected[-1])

        outputs = []
        fused = projected[0] + F.interpolate(
            projected[1], scale_factor=2.0, mode="nearest"
        )
        outputs.append(self.fpn_block(fused))

        fused = projected[1] + self.down_sample2(outputs[-1])
        outputs.append(self.pan_block(fused))
        return outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_channels": [shape.channels for shape in input_shape],
            "feat_strides": [shape.stride for shape in input_shape],
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=self.hidden_dim, stride=stride)
            for stride in self.out_strides
        ]

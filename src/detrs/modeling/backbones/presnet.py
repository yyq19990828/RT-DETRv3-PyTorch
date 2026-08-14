"""Checkpoint-compatible PResNet backbone used by DEIM-RT-DETRv2."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from torch import nn
from torch.nn import functional as F

from ...core.workspace import register
from ..shape_spec import ShapeSpec
from .hgnetv2 import FrozenBatchNorm2d

__all__ = ["PResNet"]

_DEPTHS = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
}


def _activation(name):
    if name is None:
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"unsupported PResNet activation: {name}")


class ConvNormLayer(nn.Module):
    def __init__(
        self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = _activation(act)

    def forward(self, inputs):
        return self.act(self.norm(self.conv(inputs)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="b"):
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            self.short: nn.Module
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1)
        self.act = _activation(act)

    def forward(self, inputs):
        output = self.branch2b(self.branch2a(inputs))
        return self.act(output + (inputs if self.shortcut else self.short(inputs)))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="b"):
        super().__init__()
        stride1, stride2 = (stride, 1) if variant == "a" else (1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(ch_out, ch_out * self.expansion, 1, 1)
        self.shortcut = shortcut
        if not shortcut:
            self.short: nn.Module
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            (
                                "conv",
                                ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1),
                            ),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)
        self.act = _activation(act)

    def forward(self, inputs):
        output = self.branch2c(self.branch2b(self.branch2a(inputs)))
        return self.act(output + (inputs if self.shortcut else self.short(inputs)))


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act="relu", variant="b"):
        super().__init__()
        self.blocks = nn.ModuleList()
        for index in range(count):
            self.blocks.append(
                block(
                    ch_in,
                    ch_out,
                    stride=2 if index == 0 and stage_num != 2 else 1,
                    shortcut=index != 0,
                    variant=variant,
                    act=act,
                )
            )
            if index == 0:
                ch_in = ch_out * block.expansion

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


@register
class PResNet(nn.Module):
    def __init__(
        self,
        depth,
        variant="d",
        num_stages=4,
        return_idx=(0, 1, 2, 3),
        act="relu",
        freeze_at=-1,
        freeze_norm=True,
        pretrained=None,
        local_model_dir=None,
    ):
        super().__init__()
        if depth not in _DEPTHS:
            raise ValueError(f"unsupported PResNet depth: {depth}")
        if variant not in ("a", "b", "c", "d"):
            raise ValueError(f"unsupported PResNet variant: {variant}")
        if num_stages not in range(1, 5):
            raise ValueError("PResNet num_stages must be between 1 and 4")
        if not return_idx or any(
            index not in range(num_stages) for index in return_idx
        ):
            raise ValueError("PResNet return_idx selects an unavailable stage")
        if len(set(return_idx)) != len(return_idx):
            raise ValueError("PResNet return_idx must contain no duplicates")
        if pretrained not in (None, False):
            raise ValueError(
                "PResNet pretrained initialization must be loaded explicitly"
            )
        del local_model_dir

        channels = 64
        definitions = (
            [
                (3, channels // 2, 3, 2, "conv1_1"),
                (channels // 2, channels // 2, 3, 1, "conv1_2"),
                (channels // 2, channels, 3, 1, "conv1_3"),
            ]
            if variant in ("c", "d")
            else [(3, channels, 7, 2, "conv1_1")]
        )
        self.conv1 = nn.Sequential(
            OrderedDict(
                (name, ConvNormLayer(source, target, kernel, stride, act=act))
                for source, target, kernel, stride, name in definitions
            )
        )
        block = BottleNeck if depth >= 50 else BasicBlock
        stage_channels = (64, 128, 256, 512)
        output_channels = [block.expansion * value for value in stage_channels]
        self.res_layers = nn.ModuleList()
        for index in range(num_stages):
            self.res_layers.append(
                Blocks(
                    block,
                    channels,
                    stage_channels[index],
                    _DEPTHS[depth][index],
                    index + 2,
                    act,
                    variant,
                )
            )
            channels = output_channels[index]
        self.return_idx = tuple(return_idx)
        self.out_channels = [output_channels[index] for index in self.return_idx]
        self.out_strides = [(4, 8, 16, 32)[index] for index in self.return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for index in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[index])
        if freeze_norm:
            self._freeze_norm(self)

    @staticmethod
    def _freeze_parameters(module):
        for parameter in module.parameters():
            parameter.requires_grad = False

    @classmethod
    def _freeze_norm(cls, module):
        if isinstance(module, nn.BatchNorm2d):
            return FrozenBatchNorm2d(module.num_features)
        for name, child in module.named_children():
            replacement = cls._freeze_norm(child)
            if replacement is not child:
                setattr(module, name, replacement)
        return module

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=channels, stride=stride)
            for channels, stride in zip(self.out_channels, self.out_strides)
        ]

    def forward(self, inputs):
        if isinstance(inputs, Mapping):
            inputs = inputs["image"]
        inputs = F.max_pool2d(self.conv1(inputs), 3, 2, 1)
        outputs = []
        for index, stage in enumerate(self.res_layers):
            inputs = stage(inputs)
            if index in self.return_idx:
                outputs.append(inputs)
        return outputs

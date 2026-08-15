"""HGNetv2 backbones used by the D-FINE-derived model families."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union, cast

import torch
from torch import nn
from torch.nn import functional as F

from ...core.workspace import register
from ..shape_spec import ShapeSpec

__all__ = ["HGNetv2"]


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + self.eps).rsqrt()
        bias = bias - running_mean * scale
        return inputs * scale + bias


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]))
        self.bias = nn.Parameter(torch.tensor([bias_value]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.scale * inputs + self.bias


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        padding: str = "",
        use_act: bool = True,
        use_lab: bool = False,
    ):
        super().__init__()
        self.conv: nn.Module
        if padding == "same":
            self.conv = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False,
                ),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if use_act else nn.Identity()
        self.lab = LearnableAffineBlock() if use_act and use_lab else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.lab(self.act(self.bn(self.conv(inputs))))


class LightConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_lab: bool = False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=1, use_act=False)
        self.conv2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_lab=use_lab,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(inputs))


class StemBlock(nn.Module):
    def __init__(
        self, in_channels: int, mid_channels: int, out_channels: int, use_lab: bool
    ):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels, mid_channels, kernel_size=3, stride=2, use_lab=use_lab
        )
        self.stem2a = ConvBNAct(
            mid_channels, mid_channels // 2, kernel_size=2, use_lab=use_lab
        )
        self.stem2b = ConvBNAct(
            mid_channels // 2, mid_channels, kernel_size=2, use_lab=use_lab
        )
        self.stem3 = ConvBNAct(
            mid_channels * 2,
            mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_channels, out_channels, kernel_size=1, use_lab=use_lab
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.stem1(inputs)
        padded = F.pad(inputs, (0, 1, 0, 1))
        branch = self.stem2b(F.pad(self.stem2a(padded), (0, 1, 0, 1)))
        inputs = torch.cat([self.pool(padded), branch], dim=1)
        return self.stem4(self.stem3(inputs))


class EseModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.sigmoid(self.conv(inputs.mean((2, 3), keepdim=True)))
        return inputs * scale


class HGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        layer_num: int,
        kernel_size: int,
        residual: bool,
        light_block: bool,
        use_lab: bool,
        aggregation: str,
    ):
        super().__init__()
        self.residual = residual
        block = LightConvBNAct if light_block else ConvBNAct
        self.layers = nn.ModuleList(
            block(
                in_channels if index == 0 else mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                use_lab=use_lab,
            )
            for index in range(layer_num)
        )
        total_channels = in_channels + layer_num * mid_channels
        if aggregation == "se":
            self.aggregation = nn.Sequential(
                ConvBNAct(
                    total_channels, out_channels // 2, kernel_size=1, use_lab=use_lab
                ),
                ConvBNAct(
                    out_channels // 2, out_channels, kernel_size=1, use_lab=use_lab
                ),
            )
        else:
            self.aggregation = nn.Sequential(
                ConvBNAct(total_channels, out_channels, kernel_size=1, use_lab=use_lab),
                EseModule(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        outputs = [inputs]
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)
        inputs = self.aggregation(torch.cat(outputs, dim=1))
        return inputs + identity if self.residual else inputs


class HGStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        block_num: int,
        layer_num: int,
        downsample: bool,
        light_block: bool,
        kernel_size: int,
        use_lab: bool,
        aggregation: str = "se",
    ):
        super().__init__()
        self.downsample = (
            ConvBNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
            )
            if downsample
            else nn.Identity()
        )
        self.blocks = nn.Sequential(
            *[
                HGBlock(
                    in_channels if index == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    layer_num,
                    kernel_size,
                    residual=index != 0,
                    light_block=light_block,
                    use_lab=use_lab,
                    aggregation=aggregation,
                )
                for index in range(block_num)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(inputs))


@register
class HGNetv2(nn.Module):
    """HGNetv2 B0/B2/B4/B5 with the pinned D-FINE state layout.

    Args:
        name (str): Variant tag such as `B0`, `B2`, `B4`, `B5` plus the
            pruned DEIMv2 tags `Atto`/`Femto`/`Pico`/`N`.
        use_lab (bool): Use the label-assignment (Lab) layout variant.
        return_idx (tuple): Stage indices whose outputs are returned.
        freeze_stem_only (bool): Freeze only the stem convolution.
        freeze_at (int): Freeze stages up to this index; `0` freezes
            nothing.
        freeze_norm (bool): Whether to freeze normalization parameters.
        pretrained: Unused; state loading stays explicit through
            `load_checkpoint`.
    """

    # Pruned DEIMv2 variants reuse the B0 stage-1 weights with partial loading.
    PRUNED_VARIANTS = ("Atto", "Femto", "Pico")

    ARCH_CONFIGS = {
        "Atto": (
            [3, 16, 16],
            [
                [16, 16, 64, 1, False, False, 3, 3],
                [64, 32, 256, 1, True, False, 3, 3],
                [256, 64, 256, 1, True, True, 3, 3],
            ],
        ),
        "Femto": (
            [3, 16, 16],
            [
                [16, 16, 64, 1, False, False, 3, 3],
                [64, 32, 256, 1, True, False, 3, 3],
                [256, 64, 512, 1, True, True, 5, 3],
            ],
        ),
        "Pico": (
            [3, 16, 16],
            [
                [16, 16, 64, 1, False, False, 3, 3],
                [64, 32, 256, 1, True, False, 3, 3],
                [256, 64, 512, 2, True, True, 5, 3],
            ],
        ),
        "B0": (
            [3, 16, 16],
            [
                [16, 16, 64, 1, False, False, 3, 3],
                [64, 32, 256, 1, True, False, 3, 3],
                [256, 64, 512, 2, True, True, 5, 3],
                [512, 128, 1024, 1, True, True, 5, 3],
            ],
        ),
        "B2": (
            [3, 24, 32],
            [
                [32, 32, 96, 1, False, False, 3, 4],
                [96, 64, 384, 1, True, False, 3, 4],
                [384, 128, 768, 3, True, True, 5, 4],
                [768, 256, 1536, 1, True, True, 5, 4],
            ],
        ),
        "B4": (
            [3, 32, 48],
            [
                [48, 48, 128, 1, False, False, 3, 6],
                [128, 96, 512, 1, True, False, 3, 6],
                [512, 192, 1024, 3, True, True, 5, 6],
                [1024, 384, 2048, 1, True, True, 5, 6],
            ],
        ),
        "B5": (
            [3, 32, 64],
            [
                [64, 64, 128, 1, False, False, 3, 6],
                [128, 128, 512, 2, True, False, 3, 6],
                [512, 256, 1024, 5, True, True, 5, 6],
                [1024, 512, 2048, 2, True, True, 5, 6],
            ],
        ),
    }

    def __init__(
        self,
        name: str,
        use_lab: bool = False,
        return_idx: tuple[int, ...] = (1, 2, 3),
        freeze_stem_only: bool = True,
        freeze_at: int = 0,
        freeze_norm: bool = True,
        pretrained: Union[str, Path, None] = None,
    ):
        super().__init__()
        if name not in self.ARCH_CONFIGS:
            raise ValueError(
                "unsupported HGNetv2 variant {!r}; expected one of {}".format(
                    name, ", ".join(self.ARCH_CONFIGS)
                )
            )
        if not return_idx:
            raise ValueError("return_idx must not be empty")
        if len(set(return_idx)) != len(return_idx) or tuple(
            sorted(return_idx)
        ) != tuple(return_idx):
            raise ValueError("return_idx must be sorted and contain no duplicates")

        self.name = name
        self.return_idx = tuple(return_idx)
        stem_channels, raw_stage_configs = self.ARCH_CONFIGS[name]
        stage_configs = cast(
            list[tuple[int, int, int, int, bool, bool, int, int]],
            raw_stage_configs,
        )
        stage_count = len(stage_configs)
        if any(index not in range(stage_count) for index in self.return_idx):
            raise ValueError(
                "return_idx must contain HGNetv2 stage indices in [0, {}]".format(
                    stage_count - 1
                )
            )
        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [config[2] for config in stage_configs]
        self.stem = StemBlock(
            *cast(tuple[int, int, int], stem_channels), use_lab=use_lab
        )
        self.stages = nn.ModuleList()
        for config in stage_configs:
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                downsample,
                light_block,
                kernel_size,
                layer_num,
            ) = config
            self.stages.append(
                HGStage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                )
            )

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for stage in self.stages[: min(freeze_at + 1, len(self.stages))]:
                    self._freeze_parameters(stage)
        if freeze_norm:
            self._freeze_norm(self)
        if pretrained is not None:
            self.load_pretrained(pretrained)

    @property
    def out_shape(self) -> list[ShapeSpec]:
        return [
            ShapeSpec(
                channels=self._out_channels[index], stride=self._out_strides[index]
            )
            for index in self.return_idx
        ]

    @staticmethod
    def _freeze_parameters(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False

    @classmethod
    def _freeze_norm(cls, module: nn.Module) -> nn.Module:
        if isinstance(module, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(module.num_features, module.eps)
            frozen.load_state_dict(module.state_dict())
            return frozen
        for child_name, child in module.named_children():
            frozen_child = cls._freeze_norm(child)
            if frozen_child is not child:
                setattr(module, child_name, frozen_child)
        return module

    def load_pretrained(self, path: Union[str, Path]) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError("HGNetv2 checkpoint must be a tensor state dict")

        target = self.state_dict()
        if self.name in self.PRUNED_VARIANTS:
            # Pruned DEIMv2 variants initialize from the B0 stage-1 weights by
            # keeping only tensors whose key and shape both match this graph.
            matched = {
                key: value
                for key, value in state.items()
                if key in target and value.shape == target[key].shape
            }
            if not matched:
                raise ValueError(
                    "HGNetv2 pruned variant {} found no matching tensors in {}".format(
                        self.name, path
                    )
                )
            self.load_state_dict(matched, strict=False)
            return
        missing = sorted(
            key
            for key in set(target) - set(state)
            if not key.endswith("num_batches_tracked")
        )
        unexpected = sorted(set(state) - set(target))
        if missing or unexpected:
            raise ValueError(
                "HGNetv2 checkpoint keys do not match {}: missing={}, unexpected={}".format(
                    self.name, missing[:3], unexpected[:3]
                )
            )
        for key, value in state.items():
            expected = target[key]
            if value.shape != expected.shape:
                raise ValueError(
                    "HGNetv2 checkpoint tensor {} has shape {}, expected {}".format(
                        key, tuple(value.shape), tuple(expected.shape)
                    )
                )
            if value.dtype != expected.dtype:
                raise ValueError(
                    "HGNetv2 checkpoint tensor {} has dtype {}, expected {}".format(
                        key, value.dtype, expected.dtype
                    )
                )
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(
                    "HGNetv2 checkpoint tensor {} is non-finite".format(key)
                )
        self.load_state_dict(state, strict=True)

    def forward(self, inputs: Union[torch.Tensor, Mapping[str, torch.Tensor]]):
        if isinstance(inputs, Mapping):
            inputs = inputs["image"]
        inputs = self.stem(inputs)
        outputs = []
        for index, stage in enumerate(self.stages):
            inputs = stage(inputs)
            if index in self.return_idx:
                outputs.append(inputs)
        return outputs

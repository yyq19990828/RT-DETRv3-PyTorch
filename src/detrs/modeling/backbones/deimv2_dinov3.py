"""DEIMv2 DINOv3/ViT-Tiny backbones with the STA spatial tuning adapter.

Ported from Intellindust-AI-Lab/DEIMv2@add5bcd (engine/backbone/dinov3_adapter.py,
Apache-2.0; the wrapped ViT forward is vendored under the Meta DINOv3 License).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union

import torch
import torch.nn.functional as F
from torch import nn

from detrs.core.workspace import register
from detrs.modeling.shape_spec import ShapeSpec

from .dinov3 import DinoVisionTransformer
from .vit_tiny import VisionTransformer

__all__ = ["DINOv3STAs"]


class SpatialPriorModulev2(nn.Module):
    """Lite convolutional spatial prior pyramid (stride 8/16/32).

    Upstream uses SyncBatchNorm; this port uses BatchNorm2d for identical
    single-process math and the same state layout.
    """

    def __init__(self, inplanes: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                inplanes,
                2 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(2 * inplanes),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                2 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(4 * inplanes),
        )
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                4 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(4 * inplanes),
        )

    def forward(self, x: torch.Tensor):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c2, c3, c4


@register
class DINOv3STAs(nn.Module):
    """DINOv3 (or distilled ViT-Tiny) with a bi-fused spatial prior pyramid."""

    def __init__(
        self,
        name: str,
        weights_path: Union[str, Path, None] = None,
        interaction_indexes: tuple[int, ...] = (),
        finetune: bool = True,
        embed_dim: int = 192,
        num_heads: int = 3,
        patch_size: int = 16,
        use_sta: bool = True,
        conv_inplane: int = 16,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if "dinov3" in name:
            self.dinov3: nn.Module = DinoVisionTransformer(name)
            state = self._read_state(weights_path)
            if state is not None:
                self.dinov3.load_state_dict(state)
        else:
            self.dinov3 = VisionTransformer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                return_layers=list(interaction_indexes),
            )
            state = self._read_state(weights_path)
            if state is not None:
                self.dinov3._model.load_state_dict(state)
        reported_dim: object = getattr(self.dinov3, "embed_dim", None)
        if not isinstance(reported_dim, int) or reported_dim <= 0:
            raise ValueError("ViT backbone reported no embedding dimension")
        backbone_dim = reported_dim
        self.interaction_indexes = list(interaction_indexes)
        self.patch_size = int(patch_size)
        self.out_dim = hidden_dim if hidden_dim is not None else backbone_dim

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        self.use_sta = bool(use_sta)
        if self.use_sta:
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        prior = conv_inplane * 2 if self.use_sta else 0
        prior_deep = conv_inplane * 4 if self.use_sta else 0
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    backbone_dim + prior,
                    self.out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Conv2d(
                    backbone_dim + prior_deep,
                    self.out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Conv2d(
                    backbone_dim + prior_deep,
                    self.out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.BatchNorm2d(self.out_dim),
                nn.BatchNorm2d(self.out_dim),
                nn.BatchNorm2d(self.out_dim),
            ]
        )

    @staticmethod
    def _read_state(weights_path: Union[str, Path, None]):
        if weights_path is None:
            return None
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(
                "DINOv3STAs weights_path does not exist: {}".format(path)
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("DINOv3STAs backbone weights must be a state dict")
        return state

    @property
    def out_shape(self) -> list[ShapeSpec]:
        return [
            ShapeSpec(channels=self.out_dim, stride=8),
            ShapeSpec(channels=self.out_dim, stride=16),
            ShapeSpec(channels=self.out_dim, stride=32),
        ]

    def forward(self, x: Union[torch.Tensor, Mapping[str, torch.Tensor]]):
        if isinstance(x, Mapping):
            x = x["image"]
        h_tokens, w_tokens = x.shape[2] // 16, x.shape[3] // 16
        batch, _, _, _ = x.shape

        if self.interaction_indexes and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)
        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        semantic_features = []
        num_scales = len(all_layers) - 2
        for index, layer in enumerate(all_layers):
            patch_tokens, _ = layer
            feature = patch_tokens.transpose(1, 2).reshape(
                batch, -1, h_tokens, w_tokens
            )
            target_h = int(h_tokens * 2 ** (num_scales - index))
            target_w = int(w_tokens * 2 ** (num_scales - index))
            semantic_features.append(
                F.interpolate(
                    feature,
                    size=[target_h, target_w],
                    mode="bilinear",
                    align_corners=False,
                )
            )

        if self.use_sta:
            detail_features = self.sta(x)
            fused = [
                torch.cat([semantic, detail], dim=1)
                for semantic, detail in zip(semantic_features, detail_features)
            ]
        else:
            fused = semantic_features

        c2 = self.norms[0](self.convs[0](fused[0]))
        c3 = self.norms[1](self.convs[1](fused[1]))
        c4 = self.norms[2](self.convs[2](fused[2]))
        return [c2, c3, c4]

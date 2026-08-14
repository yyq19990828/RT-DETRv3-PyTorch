# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Vendored (trimmed) from facebookresearch/dinov3 via
# Intellindust-AI-Lab/DEIMv2@add5bcdb499bf7b8a366bfeac1a47d3dc278de27.
# Only the forward path required by the DEIMv2 ViT-S/16 backbones is kept;
# fp8, sparse-linear, causal-attention and clustering utilities are removed.

from __future__ import annotations

from .vision_transformer import DinoVisionTransformer, configs

__all__ = ["DinoVisionTransformer", "configs"]

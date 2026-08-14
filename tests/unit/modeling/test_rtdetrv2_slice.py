from __future__ import annotations

import pytest
import torch

from detrs.core.workspace import get_registered_modules
from detrs.modeling.transformers.rtdetr_transformerv2 import (
    RTDETRTransformerv2,
)

CASES = {
    "r18vd": (18, (128, 256, 512), (256, 256, 256), 256, 1024, 3, -1),
    "r34vd": (34, (128, 256, 512), (256, 256, 256), 256, 1024, 4, -1),
    "r50vd_m": (50, (512, 1024, 2048), (256, 256, 256), 256, 1024, 3, 2),
    "r50vd": (50, (512, 1024, 2048), (256, 256, 256), 256, 1024, 6, -1),
    "r101vd": (101, (512, 1024, 2048), (384, 384, 384), 256, 1024, 6, -1),
}


@pytest.mark.parametrize(
    "variant", CASES, ids=("r18vd", "r34vd", "r50vd_m", "r50vd", "r101vd")
)
def test_planned_variant_build_contract(variant):
    (
        depth,
        backbone_channels,
        feat_channels,
        hidden_dim,
        dim_feedforward,
        layers,
        eval_idx,
    ) = CASES[variant]
    model = RTDETRTransformerv2(variant=variant, num_classes=3, num_queries=5)

    assert model.backbone_depth == depth
    assert model.backbone_channels == backbone_channels
    assert tuple(layer.conv.in_channels for layer in model.input_proj) == feat_channels
    assert model.hidden_dim == hidden_dim
    assert model.decoder.layers[0].linear1.out_features == dim_feedforward
    assert model.num_layers == layers
    assert model.decoder.eval_idx == (layers - 1 if eval_idx == -1 else eval_idx)


def test_registry_has_unique_decoder_without_standalone_architecture():
    modules = get_registered_modules()

    assert modules["RTDETRTransformerv2"].cls is RTDETRTransformerv2
    assert "RTDETRV2" not in modules
    assert "DEIMRTDETRV2" not in modules


def test_rejects_v3_config_before_build():
    with pytest.raises(ValueError, match="rejects RT-DETRv3 config"):
        RTDETRTransformerv2.from_config(
            {"variant": "r18vd", "num_decoder_layers": 3, "family": "rtdetrv3"}
        )


def test_rejects_v3_checkpoint_without_partial_mutation():
    model = RTDETRTransformerv2(
        variant="r18vd", num_classes=3, num_queries=5, num_denoising=0
    )
    before = {key: value.clone() for key, value in model.state_dict().items()}
    v3_state = dict(before)
    v3_state["map_memory.0.weight"] = torch.randn(4, 4)

    with pytest.raises(ValueError, match="rejects RT-DETRv3 checkpoint"):
        model.load_state_dict(v3_state)

    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in before.items()
    )


@pytest.mark.parametrize("variant", ["r152vd", "r18", "r50vd_x"])
def test_rejects_unsupported_depth(variant):
    with pytest.raises(ValueError, match="unsupported RT-DETRv2 depth/variant"):
        RTDETRTransformerv2(variant=variant)


def test_rejects_profile_swap():
    with pytest.raises(ValueError, match="r50vd_m config mismatch"):
        RTDETRTransformerv2(variant="r50vd_m", num_layers=6, eval_idx=-1)

    with pytest.raises(ValueError, match="r101vd config mismatch"):
        RTDETRTransformerv2(variant="r101vd", hidden_dim=384)

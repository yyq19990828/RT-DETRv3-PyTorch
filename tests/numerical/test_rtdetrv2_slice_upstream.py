from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from detrs.modeling.transformers.rtdetr_transformerv2 import (
    RTDETRTransformerv2,
)

UPSTREAM_SHA = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
CASES = {
    "r18vd": {
        "hidden_dim": 256,
        "feat_channels": [256] * 3,
        "num_layers": 3,
        "eval_idx": -1,
    },
    "r34vd": {
        "hidden_dim": 256,
        "feat_channels": [256] * 3,
        "num_layers": 4,
        "eval_idx": -1,
    },
    "r50vd_m": {
        "hidden_dim": 256,
        "feat_channels": [256] * 3,
        "num_layers": 3,
        "eval_idx": 2,
    },
    "r50vd": {
        "hidden_dim": 256,
        "feat_channels": [256] * 3,
        "num_layers": 6,
        "eval_idx": -1,
    },
    "r101vd": {
        "hidden_dim": 256,
        "feat_channels": [384] * 3,
        "num_layers": 6,
        "eval_idx": -1,
    },
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load upstream module {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_decoder():
    root_value = os.environ.get("DEIM_UPSTREAM_ROOT")
    if not root_value:
        pytest.skip("set DEIM_UPSTREAM_ROOT to the pinned DEIM checkout")
    root = Path(root_value).expanduser().resolve()
    source = root / "engine/deim/rtdetrv2_decoder.py"
    if not source.is_file():
        pytest.skip("pinned DEIM RT-DETRv2 source is absent")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == UPSTREAM_SHA

    for package_name in (
        "_deim_reference",
        "_deim_reference.engine",
        "_deim_reference.engine.deim",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
    core = types.ModuleType("_deim_reference.engine.core")
    core.register = lambda: lambda cls: cls
    sys.modules[core.__name__] = core
    denoising = types.ModuleType("_deim_reference.engine.deim.denoising")
    denoising.get_contrastive_denoising_training_group = lambda *args, **kwargs: (
        None,
        None,
        None,
        None,
    )
    sys.modules[denoising.__name__] = denoising
    _load_module("_deim_reference.engine.deim.utils", root / "engine/deim/utils.py")
    return _load_module(
        "_deim_reference.engine.deim.rtdetrv2_decoder", source
    ).RTDETRTransformerv2


@pytest.mark.parametrize(
    "variant", CASES, ids=("r18vd", "r34vd", "r50vd_m", "r50vd", "r101vd")
)
def test_state_activation_and_output_parity(upstream_decoder, variant):
    profile = CASES[variant]
    kwargs = {
        **profile,
        "num_classes": 3,
        "num_queries": 5,
        "num_levels": 3,
        "num_points": [4, 4, 4],
        "num_denoising": 0,
        "eval_spatial_size": None,
        "value_shape": "reshape",
    }
    torch.manual_seed(0)
    reference = upstream_decoder(**kwargs).eval()
    torch.manual_seed(0)
    local = RTDETRTransformerv2(variant=variant, **kwargs).eval()

    reference_state = reference.state_dict()
    local_state = local.state_dict()
    assert list(reference_state) == list(local_state)
    for key, value in reference_state.items():
        torch.testing.assert_close(value, local_state[key], rtol=0, atol=0)

    activations = {"reference": {}, "local": {}}

    def capture(group, name):
        def hook(_module, _inputs, output):
            activations[group][name] = output.detach()

        return hook

    handles = [
        reference.input_proj[0].register_forward_hook(
            capture("reference", "projection")
        ),
        local.input_proj[0].register_forward_hook(capture("local", "projection")),
        reference.decoder.layers[0].register_forward_hook(
            capture("reference", "decoder")
        ),
        local.decoder.layers[0].register_forward_hook(capture("local", "decoder")),
    ]
    generator = torch.Generator().manual_seed(1)
    features = [
        torch.randn(1, profile["feat_channels"][0], 2, 2, generator=generator),
        torch.randn(1, profile["feat_channels"][1], 1, 1, generator=generator),
        torch.randn(1, profile["feat_channels"][2], 1, 1, generator=generator),
    ]
    with torch.inference_mode():
        reference_output = reference([value.clone() for value in features])
        local_output = local([value.clone() for value in features])
    for handle in handles:
        handle.remove()

    for name in ("projection", "decoder"):
        torch.testing.assert_close(
            activations["reference"][name],
            activations["local"][name],
            rtol=1e-5,
            atol=1e-6,
        )
    for name in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(
            reference_output[name], local_output[name], rtol=1e-5, atol=1e-6
        )

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from detrs.modeling.transformers.dfine_decoder import (
    LQE,
    Integral,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from detrs.modeling.transformers.dfine_hybrid_encoder import DFINEHybridEncoder
from detrs.modeling.transformers.dfine_utils import (
    bbox2distance,
    distance2bbox,
    weighting_function,
)

PINNED_SHA = "267a6da6d04c8ad52e54120692896515b9e55981"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_modules():
    root_value = os.environ.get("DFINE_UPSTREAM_ROOT")
    if not root_value:
        pytest.skip("set DFINE_UPSTREAM_ROOT to the pinned D-FINE checkout")
    upstream = Path(root_value).expanduser().resolve()
    if not (upstream / "src/zoo/dfine/dfine_decoder.py").is_file():
        pytest.skip("pinned D-FINE primitive sources are unavailable")

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert sha == PINNED_SHA

    for package in ("upstream_dfine", "upstream_dfine.zoo", "upstream_dfine.zoo.dfine"):
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    core = types.ModuleType("upstream_dfine.core")

    def register():
        return lambda cls: cls

    core.register = register
    sys.modules[core.__name__] = core
    denoising = types.ModuleType("upstream_dfine.zoo.dfine.denoising")
    denoising.get_contrastive_denoising_training_group = None
    sys.modules[denoising.__name__] = denoising
    root = upstream / "src/zoo/dfine"
    box_ops = _load("upstream_dfine.zoo.dfine.box_ops", root / "box_ops.py")
    utils = _load("upstream_dfine.zoo.dfine.utils", root / "utils.py")
    dfine_utils = _load("upstream_dfine.zoo.dfine.dfine_utils", root / "dfine_utils.py")

    decoder = _load("upstream_dfine.zoo.dfine.dfine_decoder", root / "dfine_decoder.py")
    encoder = _load(
        "upstream_dfine.zoo.dfine.hybrid_encoder", root / "hybrid_encoder.py"
    )
    return box_ops, utils, dfine_utils, decoder, encoder


@pytest.mark.numerical
def test_utility_matches_upstream(upstream_modules):
    upstream = upstream_modules[2]
    up = torch.tensor([0.5])
    scale = torch.tensor([4.0])
    assert torch.allclose(
        weighting_function(32, up, scale), upstream.weighting_function(32, up, scale)
    )
    points = torch.rand(2, 5, 4) + 0.2
    distance = torch.randn(2, 5, 4)
    assert torch.allclose(
        distance2bbox(points, distance, scale),
        upstream.distance2bbox(points, distance, scale),
    )
    bbox = torch.tensor([[0.1, 0.2, 0.8, 0.9], [0.2, 0.1, 0.7, 0.6]])
    refs = torch.tensor([[0.4, 0.5, 0.2, 0.3], [0.5, 0.4, 0.3, 0.2]])
    actual = bbox2distance(refs, bbox, 32, scale, up)
    expected = upstream.bbox2distance(refs, bbox, 32, scale, up)
    for left, right in zip(actual, expected):
        assert torch.allclose(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.numerical
def test_integral_and_lqe_match_upstream(upstream_modules):
    decoder = upstream_modules[3]
    logits = torch.randn(2, 5, 132)
    project = weighting_function(32, torch.tensor([0.5]), torch.tensor([4.0]))
    assert torch.allclose(
        Integral(32)(logits, project), decoder.Integral(32)(logits, project)
    )

    torch.manual_seed(0)
    expected = decoder.LQE(4, 16, 2, 32)
    actual = LQE(4, 16, 2, 32)
    actual.load_state_dict(expected.state_dict())
    scores = torch.randn(2, 5, 7)
    assert torch.allclose(
        actual(scores, logits), expected(scores, logits), rtol=1e-5, atol=1e-6
    )


@pytest.mark.numerical
def test_real_decoder_layer_state_and_activation_match_upstream(upstream_modules):
    upstream = upstream_modules[3]
    kwargs = dict(
        d_model=8,
        n_head=2,
        dim_feedforward=16,
        dropout=0.0,
        activation="relu",
        n_levels=2,
        n_points=[2, 3],
    )
    torch.manual_seed(7)
    expected = upstream.TransformerDecoderLayer(**kwargs).eval()
    torch.manual_seed(7)
    actual = TransformerDecoderLayer(**kwargs).eval()
    assert list(actual.state_dict()) == list(expected.state_dict())
    actual.load_state_dict(expected.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(11)
    target = torch.randn(1, 4, 8, generator=generator)
    references = torch.rand(1, 4, 1, 4, generator=generator)
    memory = torch.randn(1, 2, 4, 5, generator=generator).split([2, 3], dim=-1)
    position = torch.randn(1, 4, 8, generator=generator)
    args = (target, references, memory, [[1, 2], [1, 3]], None, position)
    with torch.inference_mode():
        expected_output = expected(*args)
        actual_output = actual(*args)
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)


class _Layer(torch.nn.Module):
    def forward(
        self, target, reference_points, value, spatial_shapes, attn_mask, query_pos
    ):
        return target + query_pos * 0.01


class _Head(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, value):
        return self.linear(value)


@pytest.mark.numerical
@pytest.mark.parametrize("training", [True, False])
def test_decoder_activations_match_upstream(upstream_modules, training):
    upstream_decoder = upstream_modules[3].TransformerDecoder
    kwargs = dict(
        hidden_dim=8,
        decoder_layer=_Layer(),
        decoder_layer_wide=_Layer(),
        num_layers=3,
        num_head=2,
        reg_max=8,
        reg_scale=torch.tensor([4.0]),
        up=torch.tensor([0.5]),
        eval_idx=1,
        layer_scale=1,
    )
    torch.manual_seed(0)
    expected = upstream_decoder(**kwargs)
    actual = TransformerDecoder(**kwargs)
    actual.load_state_dict(expected.state_dict())
    heads = (
        torch.nn.ModuleList([_Head(8, 36) for _ in range(3)]),
        torch.nn.ModuleList([_Head(8, 6) for _ in range(3)]),
        _Head(4, 8),
        _Head(8, 4),
    )
    args = (
        torch.randn(2, 5, 8),
        torch.randn(2, 5, 4),
        torch.randn(2, 5, 8),
        [[1, 2], [1, 3]],
        *heads,
        Integral(8),
        torch.tensor([0.5]),
        torch.tensor([4.0]),
    )
    expected.train(training)
    actual.train(training)
    expected_outputs = expected(*args)
    actual_outputs = actual(*args)
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        assert torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-6)


@pytest.mark.numerical
def test_encoder_activations_match_upstream(upstream_modules):
    upstream_cls = upstream_modules[4].HybridEncoder
    kwargs = dict(
        in_channels=[4, 8],
        feat_strides=[8, 16],
        hidden_dim=8,
        nhead=2,
        dim_feedforward=16,
        use_encoder_idx=[1],
        num_encoder_layers=1,
        expansion=0.5,
        depth_mult=0.34,
    )
    torch.manual_seed(0)
    expected = upstream_cls(**kwargs).eval()
    actual = DFINEHybridEncoder(**kwargs).eval()
    actual.load_state_dict(expected.state_dict())
    feats = [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4)]
    with torch.no_grad():
        expected_outputs = expected(feats)
        actual_outputs = actual(feats)
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        assert torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-6)

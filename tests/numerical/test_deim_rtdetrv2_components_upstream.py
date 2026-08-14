from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from detrs.modeling.backbones.presnet import PResNet
from detrs.modeling.post_process import DETRPostProcess
from detrs.modeling.transformers.dfine_hybrid_encoder import (
    RTDETRV2HybridEncoder,
)

PINNED_SHA = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_components():
    root_value = os.environ.get("DEIM_UPSTREAM_ROOT")
    if not root_value:
        pytest.skip("set DEIM_UPSTREAM_ROOT to the pinned DEIM checkout")
    root = Path(root_value).expanduser().resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == PINNED_SHA
    assert not dirty

    engine = root / "engine"
    _package("upstream_deim", engine)
    core = _package("upstream_deim.core", engine / "core")
    core.register = lambda: lambda cls: cls
    _package("upstream_deim.backbone", engine / "backbone")
    _load("upstream_deim.backbone.common", engine / "backbone/common.py")
    presnet = _load("upstream_deim.backbone.presnet", engine / "backbone/presnet.py")
    _package("upstream_deim.deim", engine / "deim")
    _load("upstream_deim.deim.utils", engine / "deim/utils.py")
    encoder = _load(
        "upstream_deim.deim.hybrid_encoder", engine / "deim/hybrid_encoder.py"
    )
    postprocessor = _load(
        "upstream_deim.deim.postprocessor", engine / "deim/postprocessor.py"
    )
    return presnet.PResNet, encoder.HybridEncoder, postprocessor.PostProcessor


@pytest.mark.numerical
@pytest.mark.parametrize("depth", [18, 34, 50, 101])
def test_presnet_state_and_activation_match_upstream(upstream_components, depth):
    upstream_cls, _, _ = upstream_components
    kwargs = dict(
        depth=depth,
        variant="d",
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        pretrained=False,
    )
    torch.manual_seed(0)
    expected = upstream_cls(**kwargs).eval()
    torch.manual_seed(0)
    actual = PResNet(**kwargs).eval()
    assert list(actual.state_dict()) == list(expected.state_dict())
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)

    image = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        expected_outputs = expected(image)
        actual_outputs = actual({"image": image})
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


@pytest.mark.numerical
@pytest.mark.parametrize(
    "in_channels,hidden_dim,feedforward,expansion",
    [
        ([128, 256, 512], 256, 1024, 0.5),
        ([512, 1024, 2048], 256, 1024, 1.0),
        ([512, 1024, 2048], 384, 2048, 1.0),
    ],
)
def test_encoder_state_and_activation_match_upstream(
    upstream_components, in_channels, hidden_dim, feedforward, expansion
):
    _, upstream_cls, _ = upstream_components
    kwargs = dict(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        dim_feedforward=feedforward,
        expansion=expansion,
        version="rt_detrv2",
        eval_spatial_size=None,
    )
    torch.manual_seed(0)
    expected = upstream_cls(**kwargs).eval()
    torch.manual_seed(0)
    actual = RTDETRV2HybridEncoder(**kwargs).eval()
    assert list(actual.state_dict()) == list(expected.state_dict())
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)

    generator = torch.Generator().manual_seed(2)
    features = [
        torch.randn(1, channels, size, size, generator=generator)
        for channels, size in zip(in_channels, [8, 4, 2])
    ]
    with torch.inference_mode():
        expected_outputs = expected(features)
        actual_outputs = actual(features)
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


@pytest.mark.numerical
def test_focal_postprocessor_matches_upstream(upstream_components):
    _, _, upstream_cls = upstream_components
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 20, 80, generator=generator)
    boxes = torch.rand(2, 20, 4, generator=generator)
    image_shapes = torch.tensor([[480.0, 640.0], [720.0, 1280.0]])

    expected = upstream_cls(num_classes=80, use_focal_loss=True, num_top_queries=300)(
        {"pred_logits": logits, "pred_boxes": boxes}, image_shapes.flip(-1)
    )
    actual, bbox_num, _ = DETRPostProcess(
        num_classes=80, use_focal_loss=True, num_top_queries=300
    )((boxes, logits, None), image_shapes, torch.ones_like(image_shapes))

    assert bbox_num.tolist() == [300, 300]
    actual = actual.reshape(2, 300, 6)
    for batch, expected_batch in zip(actual, expected):
        torch.testing.assert_close(
            batch[:, 0].to(torch.long), expected_batch["labels"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            batch[:, 1], expected_batch["scores"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            batch[:, 2:], expected_batch["boxes"], rtol=0, atol=0
        )

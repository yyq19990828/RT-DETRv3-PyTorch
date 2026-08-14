import os
import subprocess
import sys
import types
from importlib import util
from pathlib import Path

import pytest
import torch

from ppdet_pytorch.modeling.post_process import DETRPostProcess
from ppdet_pytorch.modeling.transformers.dfine_support import (
    DFINEHungarianMatcher,
    get_contrastive_denoising_training_group,
)

PINNED_SHA = "267a6da6d04c8ad52e54120692896515b9e55981"


def _load_module(name, path):
    spec = util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module {path}")
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_modules():
    root_value = os.environ.get("DFINE_UPSTREAM_ROOT")
    if not root_value:
        pytest.skip("set DFINE_UPSTREAM_ROOT to the pinned D-FINE checkout")
    root = Path(root_value).expanduser().resolve()
    if not (root / "src/zoo/dfine/denoising.py").is_file():
        pytest.skip("pinned D-FINE support sources are unavailable")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == PINNED_SHA

    package = "_pinned_dfine"
    for package_name in (package, f"{package}.zoo", f"{package}.zoo.dfine"):
        package_module = types.ModuleType(package_name)
        package_module.__path__ = []
        sys.modules[package_name] = package_module

    core_module = types.ModuleType(f"{package}.core")
    core_module.register = lambda: lambda cls: cls
    sys.modules[core_module.__name__] = core_module

    box_module = _load_module(
        f"{package}.zoo.dfine.box_ops", root / "src/zoo/dfine/box_ops.py"
    )

    utils_module = types.ModuleType(f"{package}.zoo.dfine.utils")
    utils_module.inverse_sigmoid = lambda value, eps=1e-5: torch.log(
        value.clamp(min=eps) / (1 - value).clamp(min=eps)
    )
    sys.modules[utils_module.__name__] = utils_module

    denoising = _load_module(
        f"{package}.zoo.dfine.denoising", root / "src/zoo/dfine/denoising.py"
    )
    matcher = _load_module(
        f"{package}.zoo.dfine.matcher", root / "src/zoo/dfine/matcher.py"
    )
    postprocessor = _load_module(
        f"{package}.zoo.dfine.postprocessor",
        root / "src/zoo/dfine/postprocessor.py",
    )
    return denoising, matcher, postprocessor, box_module


@pytest.mark.numerical
def test_denoising_matches_pinned_upstream(upstream_modules):
    upstream = upstream_modules[0]
    targets = [
        {
            "labels": torch.tensor([1, 0]),
            "boxes": torch.tensor([[0.75, 0.75, 0.20, 0.20], [0.20, 0.20, 0.10, 0.10]]),
        },
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)},
    ]
    embedding = torch.nn.Embedding(3, 5)

    torch.manual_seed(0)
    expected = upstream.get_contrastive_denoising_training_group(
        targets, 2, 3, embedding, 4, 0.5, 1.0
    )
    torch.manual_seed(0)
    actual = get_contrastive_denoising_training_group(
        targets, 2, 3, embedding, 4, 0.5, 1.0
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert torch.equal(actual[2], expected[2])
    assert actual[3]["dn_num_group"] == expected[3]["dn_num_group"]
    assert actual[3]["dn_num_split"] == expected[3]["dn_num_split"]
    for actual_indices, expected_indices in zip(
        actual[3]["dn_positive_idx"], expected[3]["dn_positive_idx"]
    ):
        assert torch.equal(actual_indices, expected_indices)


@pytest.mark.numerical
def test_matcher_assignments_match_pinned_upstream(upstream_modules):
    upstream = upstream_modules[1]
    targets = [
        {
            "labels": torch.tensor([1, 0]),
            "boxes": torch.tensor([[0.75, 0.75, 0.20, 0.20], [0.20, 0.20, 0.10, 0.10]]),
        },
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)},
    ]
    outputs = {
        "pred_logits": torch.tensor(
            [
                [[-3.0, 5.0], [4.0, -2.0], [-1.0, -1.0]],
                [[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]],
            ]
        ),
        "pred_boxes": torch.tensor(
            [
                [[0.75, 0.75, 0.20, 0.20], [0.20, 0.20, 0.10, 0.10], [0.5] * 4],
                [[0.1] * 4, [0.2] * 4, [0.3] * 4],
            ]
        ),
    }
    weights = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}

    expected = upstream.HungarianMatcher(weights, use_focal_loss=True)(outputs, targets)
    actual = DFINEHungarianMatcher(weights, use_focal_loss=True)(outputs, targets)

    for actual_pair, expected_pair in zip(actual["indices"], expected["indices"]):
        assert torch.equal(actual_pair[0].cpu(), expected_pair[0])
        assert torch.equal(actual_pair[1].cpu(), expected_pair[1])


@pytest.mark.numerical
def test_postprocess_top300_matches_pinned_upstream(upstream_modules):
    upstream = upstream_modules[2]
    logits = torch.arange(600, dtype=torch.float32).reshape(1, 2, 300) / 100
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.25], [0.25, 0.25, 0.1, 0.2]]])
    original_sizes = torch.tensor([[640.0, 480.0]])

    expected = upstream.DFINEPostProcessor(
        num_classes=300, num_top_queries=300, use_focal_loss=True
    )({"pred_logits": logits, "pred_boxes": boxes}, original_sizes)[0]
    bbox, bbox_num, mask = DETRPostProcess(
        num_classes=300, num_top_queries=300, use_focal_loss=True
    )(
        (boxes, logits, None),
        im_shape=torch.tensor([[240.0, 320.0]]),
        scale_factor=torch.tensor([[0.5, 0.5]]),
    )

    assert torch.equal(bbox[:, 0].to(torch.int64), expected["labels"])
    torch.testing.assert_close(bbox[:, 1], expected["scores"], rtol=0, atol=0)
    torch.testing.assert_close(bbox[:, 2:], expected["boxes"], rtol=0, atol=0)
    assert torch.equal(bbox_num, torch.tensor([300], dtype=torch.int32))
    assert mask is None

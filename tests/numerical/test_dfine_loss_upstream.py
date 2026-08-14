import os
import subprocess
import sys
import types
from importlib import util
from pathlib import Path

import pytest
import torch

from ppdet_pytorch.modeling.losses.dfine_loss import DFINECriterion
from ppdet_pytorch.modeling.transformers.dfine_support import DFINEHungarianMatcher

PINNED_SHA = "267a6da6d04c8ad52e54120692896515b9e55981"


def _load(name, path):
    spec = util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module {path}")
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_criterion():
    root_value = os.environ.get("DFINE_UPSTREAM_ROOT")
    if not root_value:
        pytest.skip("set DFINE_UPSTREAM_ROOT to the pinned D-FINE checkout")
    root = Path(root_value).expanduser().resolve()
    source = root / "src/zoo/dfine/dfine_criterion.py"
    if not source.is_file():
        pytest.skip("pinned D-FINE criterion source is unavailable")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == PINNED_SHA

    package = "_pinned_dfine_criterion"
    for name in (package, f"{package}.zoo", f"{package}.zoo.dfine"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    core = types.ModuleType(f"{package}.core")
    core.register = lambda: lambda cls: cls
    sys.modules[core.__name__] = core
    misc = types.ModuleType(f"{package}.misc")
    misc.__path__ = []
    sys.modules[misc.__name__] = misc
    dist_utils = types.ModuleType(f"{package}.misc.dist_utils")
    dist_utils.get_world_size = lambda: 1
    dist_utils.is_dist_available_and_initialized = lambda: False
    sys.modules[dist_utils.__name__] = dist_utils
    source_root = root / "src/zoo/dfine"
    _load(f"{package}.zoo.dfine.box_ops", source_root / "box_ops.py")
    _load(f"{package}.zoo.dfine.dfine_utils", source_root / "dfine_utils.py")
    return _load(f"{package}.zoo.dfine.dfine_criterion", source)


def _prediction(generator, batch=2, queries=4, classes=3, local=False):
    boxes = torch.rand(batch, queries, 4, generator=generator) * 0.35 + 0.2
    prediction = {
        "pred_logits": torch.randn(batch, queries, classes, generator=generator),
        "pred_boxes": boxes,
    }
    if local:
        prediction.update(
            pred_corners=torch.randn(batch, queries, 36, generator=generator),
            ref_points=torch.rand(batch, queries, 4, generator=generator) * 0.3 + 0.2,
            teacher_corners=torch.randn(batch, queries, 36, generator=generator),
            teacher_logits=torch.randn(batch, queries, classes, generator=generator),
        )
    return prediction


def _fixture():
    generator = torch.Generator().manual_seed(19)
    outputs = _prediction(generator, local=True)
    outputs.update(
        aux_outputs=[_prediction(generator, local=True)],
        pre_outputs=_prediction(generator),
        enc_aux_outputs=[_prediction(generator)],
        enc_meta={"class_agnostic": False},
        dn_outputs=[_prediction(generator, queries=2, local=True)],
        dn_pre_outputs=_prediction(generator, queries=2),
        dn_meta={
            "dn_positive_idx": [torch.tensor([0, 1]), torch.tensor([0])],
            "dn_num_group": 1,
        },
        up=torch.tensor([0.5]),
        reg_scale=torch.tensor([4.0]),
    )
    targets = [
        {
            "labels": torch.tensor([0, 2]),
            "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]]),
        },
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.3, 0.6, 0.2, 0.1]])},
    ]
    return outputs, targets


def _clone_predictions(value, leaves):
    if isinstance(value, torch.Tensor):
        clone = value.detach().clone()
        if value.is_floating_point() and value.ndim >= 2:
            clone.requires_grad_()
            leaves.append(clone)
        return clone
    if isinstance(value, dict):
        return {key: _clone_predictions(item, leaves) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_predictions(item, leaves) for item in value]
    return value


def _build(criterion_class):
    weights = {
        "loss_vfl": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
    }
    matcher = DFINEHungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}, use_focal_loss=True
    )
    return criterion_class(
        matcher,
        weights,
        ["vfl", "boxes", "local"],
        num_classes=3,
        reg_max=8,
    )


@pytest.mark.numerical
def test_all_losses_and_prediction_gradients_match_upstream(upstream_criterion):
    outputs, targets = _fixture()
    expected_leaves = []
    actual_leaves = []
    expected_outputs = _clone_predictions(outputs, expected_leaves)
    actual_outputs = _clone_predictions(outputs, actual_leaves)

    expected = _build(upstream_criterion.DFINECriterion)(expected_outputs, targets)
    actual = _build(DFINECriterion)(actual_outputs, targets)

    assert set(actual) == set(expected)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-6)
    sum(expected.values()).backward()
    sum(actual.values()).backward()
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        if actual_leaf.grad is None or expected_leaf.grad is None:
            assert actual_leaf.grad is expected_leaf.grad is None
        else:
            torch.testing.assert_close(
                actual_leaf.grad, expected_leaf.grad, rtol=1e-4, atol=1e-6
            )


@pytest.mark.numerical
@pytest.mark.parametrize("loss_name", ["vfl", "focal"])
def test_empty_targets_match_upstream(upstream_criterion, loss_name, monkeypatch):
    outputs, _ = _fixture()
    outputs.pop("dn_outputs")
    outputs.pop("dn_pre_outputs")
    outputs.pop("dn_meta")
    targets = [
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)}
        for _ in range(2)
    ]
    expected_leaves = []
    actual_leaves = []
    expected_outputs = _clone_predictions(outputs, expected_leaves)
    actual_outputs = _clone_predictions(outputs, actual_leaves)
    weights = {"loss_vfl": 1.0, "loss_focal": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    matcher = DFINEHungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}, use_focal_loss=True
    )
    expected_criterion = upstream_criterion.DFINECriterion(
        matcher, weights, [loss_name, "boxes"], num_classes=3, reg_max=8
    )
    actual_criterion = DFINECriterion(
        matcher, weights, [loss_name, "boxes"], num_classes=3, reg_max=8
    )

    if loss_name == "focal":
        original_focal = upstream_criterion.torchvision.ops.sigmoid_focal_loss

        def focal_with_current_dtype(inputs, target, *args, **kwargs):
            return original_focal(inputs, target.to(inputs.dtype), *args, **kwargs)

        monkeypatch.setattr(
            upstream_criterion.torchvision.ops,
            "sigmoid_focal_loss",
            focal_with_current_dtype,
        )
    expected = expected_criterion(expected_outputs, targets)
    monkeypatch.undo()
    actual = actual_criterion(actual_outputs, targets)

    assert set(actual) == set(expected)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-6)

import os
import subprocess
import sys
import types
from importlib import util
from pathlib import Path

import pytest
import torch
from torch import nn

from detrs.modeling.losses.deim_loss import DEIMCriterion

PINNED_SHA = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"


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
    root_value = os.environ.get("DEIM_UPSTREAM_ROOT")
    if not root_value:
        pytest.skip("set DEIM_UPSTREAM_ROOT to the pinned DEIM checkout")
    root = Path(root_value).expanduser().resolve()
    source = root / "engine/deim/deim_criterion.py"
    if not source.is_file():
        pytest.skip("pinned DEIM criterion source is unavailable")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == PINNED_SHA

    package = "_pinned_deim_criterion"
    for name in (package, f"{package}.deim"):
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
    source_root = root / "engine/deim"
    _load(f"{package}.deim.box_ops", source_root / "box_ops.py")
    _load(f"{package}.deim.dfine_utils", source_root / "dfine_utils.py")
    return _load(f"{package}.deim.deim_criterion", source)


def _fixture():
    generator = torch.Generator().manual_seed(47)
    logits = torch.randn(2, 5, 3, generator=generator)
    boxes = torch.rand(2, 5, 4, generator=generator) * 0.35 + 0.2
    targets = [
        {
            "labels": torch.tensor([0, 2]),
            "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]]),
        },
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.3, 0.6, 0.2, 0.1]]),
        },
    ]
    indices = [
        (torch.tensor([1, 4]), torch.tensor([0, 1])),
        (torch.tensor([2]), torch.tensor([0])),
    ]
    return logits, boxes, targets, indices


@pytest.mark.numerical
@pytest.mark.parametrize("mal_alpha", [None, 0.5])
def test_mal_value_and_gradients_match_upstream(upstream_criterion, mal_alpha):
    logits, boxes, targets, indices = _fixture()
    expected_logits = logits.clone().requires_grad_()
    expected_boxes = boxes.clone().requires_grad_()
    actual_logits = logits.clone().requires_grad_()
    actual_boxes = boxes.clone().requires_grad_()
    kwargs = dict(
        matcher=None,
        weight_dict={"loss_mal": 1},
        losses=["mal"],
        gamma=1.5,
        num_classes=3,
        mal_alpha=mal_alpha,
        use_uni_set=False,
    )
    expected_criterion = upstream_criterion.DEIMCriterion(**kwargs)
    actual_criterion = DEIMCriterion(**kwargs)

    expected = expected_criterion.loss_labels_mal(
        {"pred_logits": expected_logits, "pred_boxes": expected_boxes},
        targets,
        indices,
        3,
    )["loss_mal"]
    actual = actual_criterion.loss_labels_mal(
        {"pred_logits": actual_logits, "pred_boxes": actual_boxes},
        targets,
        indices,
        3,
    )["loss_mal"]

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(
        actual_logits.grad, expected_logits.grad, rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(
        actual_boxes.grad, expected_boxes.grad, rtol=1e-4, atol=1e-6
    )


class _SequenceMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_index = 0

    def forward(self, outputs, targets):
        del outputs, targets
        query = self.call_index
        self.call_index += 1
        return {
            "indices": [
                (torch.tensor([query]), torch.tensor([0])),
                (torch.tensor([(query + 1) % 4]), torch.tensor([0])),
            ]
        }


def _forward_fixture():
    generator = torch.Generator().manual_seed(83)

    def prediction():
        return {
            "pred_logits": torch.randn(2, 4, 3, generator=generator).requires_grad_(),
            "pred_boxes": (
                torch.rand(2, 4, 4, generator=generator) * 0.3 + 0.3
            ).requires_grad_(),
        }

    outputs = {
        **prediction(),
        "aux_outputs": [prediction()],
        "enc_aux_outputs": [prediction()],
        "enc_meta": {"class_agnostic": False},
        "dn_outputs": [prediction()],
        "dn_meta": {
            "dn_positive_idx": [torch.tensor([0]), torch.tensor([0])],
            "dn_num_group": 1,
        },
    }
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]])},
        {"labels": torch.tensor([2]), "boxes": torch.tensor([[0.6, 0.6, 0.1, 0.2]])},
    ]
    return outputs, targets


def _clone_outputs(outputs):
    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone().requires_grad_(value.requires_grad)
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return value

    return clone(outputs)


def _prediction_tensors(outputs):
    predictions = [
        outputs,
        *outputs["aux_outputs"],
        *outputs["enc_aux_outputs"],
        *outputs["dn_outputs"],
    ]
    return [
        value
        for prediction in predictions
        for value in (prediction["pred_logits"], prediction["pred_boxes"])
    ]


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("use_uni_set", "boxes_weight_format"), [(False, None), (True, "iou")]
)
def test_rtdetrv2_graph_full_forward_matches_upstream(
    upstream_criterion, use_uni_set, boxes_weight_format
):
    outputs, targets = _forward_fixture()
    expected_outputs = _clone_outputs(outputs)
    actual_outputs = _clone_outputs(outputs)
    kwargs = dict(
        weight_dict={"loss_mal": 1, "loss_bbox": 5, "loss_giou": 2},
        losses=["mal", "boxes"],
        gamma=1.5,
        num_classes=3,
        use_uni_set=use_uni_set,
        boxes_weight_format=boxes_weight_format,
    )
    expected_criterion = upstream_criterion.DEIMCriterion(
        matcher=_SequenceMatcher(), **kwargs
    )
    actual_criterion = DEIMCriterion(matcher=_SequenceMatcher(), **kwargs)

    expected = expected_criterion(expected_outputs, targets)
    actual = actual_criterion(actual_outputs, targets)

    assert actual.keys() == expected.keys()
    assert len(actual) == 12
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=1e-6)
    sum(expected.values()).backward()
    sum(actual.values()).backward()
    for actual_tensor, expected_tensor in zip(
        _prediction_tensors(actual_outputs), _prediction_tensors(expected_outputs)
    ):
        torch.testing.assert_close(
            actual_tensor.grad, expected_tensor.grad, rtol=1e-4, atol=1e-6
        )


def _dfine_fixture():
    generator = torch.Generator().manual_seed(97)

    def prediction(queries=4, local=False):
        value = {
            "pred_logits": torch.randn(2, queries, 3, generator=generator),
            "pred_boxes": torch.rand(2, queries, 4, generator=generator) * 0.3 + 0.3,
        }
        if local:
            value.update(
                pred_corners=torch.randn(2, queries, 36, generator=generator),
                ref_points=torch.rand(2, queries, 4, generator=generator) * 0.2 + 0.4,
                teacher_corners=torch.randn(2, queries, 36, generator=generator),
                teacher_logits=torch.randn(2, queries, 3, generator=generator),
            )
        return value

    outputs = prediction(local=True)
    outputs.update(
        aux_outputs=[prediction(local=True)],
        pre_outputs=prediction(),
        enc_aux_outputs=[prediction()],
        enc_meta={"class_agnostic": False},
        dn_outputs=[prediction(queries=2, local=True)],
        dn_pre_outputs=prediction(queries=2),
        dn_meta={
            "dn_positive_idx": [torch.tensor([0]), torch.tensor([0])],
            "dn_num_group": 1,
        },
        up=torch.tensor([0.5]),
        reg_scale=torch.tensor([4.0]),
    )
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]])},
        {"labels": torch.tensor([2]), "boxes": torch.tensor([[0.6, 0.6, 0.1, 0.2]])},
    ]
    return outputs, targets


def _clone_float_leaves(value, leaves):
    if isinstance(value, torch.Tensor):
        clone = value.detach().clone()
        if value.is_floating_point() and value.ndim >= 2:
            clone.requires_grad_()
            leaves.append(clone)
        return clone
    if isinstance(value, dict):
        return {key: _clone_float_leaves(item, leaves) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_float_leaves(item, leaves) for item in value]
    return value


@pytest.mark.numerical
def test_dfine_graph_full_forward_local_and_cdn_match_upstream(upstream_criterion):
    from detrs.modeling.transformers.dfine_support import (
        DFINEHungarianMatcher,
    )

    outputs, targets = _dfine_fixture()
    expected_leaves, actual_leaves = [], []
    expected_outputs = _clone_float_leaves(outputs, expected_leaves)
    actual_outputs = _clone_float_leaves(outputs, actual_leaves)
    weights = {
        "loss_mal": 1,
        "loss_bbox": 5,
        "loss_giou": 2,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
    }

    def criterion(criterion_class):
        matcher = DFINEHungarianMatcher(
            {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            use_focal_loss=True,
        )
        return criterion_class(
            matcher=matcher,
            weight_dict=weights,
            losses=["mal", "boxes", "local"],
            gamma=1.5,
            num_classes=3,
            reg_max=8,
            use_uni_set=True,
        )

    expected = criterion(upstream_criterion.DEIMCriterion)(expected_outputs, targets)
    actual = criterion(DEIMCriterion)(actual_outputs, targets)

    assert actual.keys() == expected.keys()
    assert any(key.startswith("loss_fgl") for key in actual)
    assert any(key.startswith("loss_ddf") for key in actual)
    assert any(key.endswith("_pre") for key in actual)
    assert any("_dn_" in key for key in actual)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=1e-6)
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

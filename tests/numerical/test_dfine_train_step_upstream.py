import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ppdet_pytorch.modeling.architectures.dfine import DFINE
from ppdet_pytorch.modeling.losses.dfine_loss import DFINECriterion
from ppdet_pytorch.modeling.post_process import DETRPostProcess
from ppdet_pytorch.modeling.transformers.dfine_decoder import DFINETransformer
from ppdet_pytorch.modeling.transformers.dfine_support import DFINEHungarianMatcher

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
    root = Path(root_value).expanduser().resolve()
    source = root / "src/zoo/dfine"
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert sha == PINNED_SHA

    package = "upstream_dfine_train_step"
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
    denoising = types.ModuleType(f"{package}.zoo.dfine.denoising")
    from ppdet_pytorch.modeling.transformers.dfine_support import (
        get_contrastive_denoising_training_group,
    )

    denoising.get_contrastive_denoising_training_group = (
        get_contrastive_denoising_training_group
    )
    sys.modules[denoising.__name__] = denoising
    _load(f"{package}.zoo.dfine.box_ops", source / "box_ops.py")
    _load(f"{package}.zoo.dfine.utils", source / "utils.py")
    _load(f"{package}.zoo.dfine.dfine_utils", source / "dfine_utils.py")
    decoder = _load(f"{package}.zoo.dfine.dfine_decoder", source / "dfine_decoder.py")
    criterion = _load(
        f"{package}.zoo.dfine.dfine_criterion", source / "dfine_criterion.py"
    )
    return decoder, criterion


class _FeatureBackbone(nn.Module):
    def forward(self, inputs):
        image = inputs["image"]
        return [image, F.adaptive_avg_pool2d(image, 1)]


class _IdentityEncoder(nn.Module):
    def forward(self, features):
        return features


def _decoder_config(num_denoising=4):
    return {
        "num_classes": 3,
        "hidden_dim": 8,
        "num_queries": 4,
        "feat_channels": [8, 8],
        "feat_strides": [8, 16],
        "num_levels": 2,
        "num_points": [2, 2],
        "nhead": 2,
        "num_layers": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "num_denoising": num_denoising,
        "reg_max": 8,
    }


def _criterion_config(matcher):
    return {
        "matcher": matcher,
        "weight_dict": {
            "loss_vfl": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        "losses": ["vfl", "boxes", "local"],
        "alpha": 0.75,
        "gamma": 2,
        "num_classes": 3,
        "reg_max": 8,
    }


def _matcher():
    return DFINEHungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2,
    )


def _batch(empty=False):
    generator = torch.Generator().manual_seed(19)
    if empty:
        labels = [torch.empty(0, 1, dtype=torch.int64) for _ in range(2)]
        boxes = [torch.empty(0, 4) for _ in range(2)]
    else:
        labels = [torch.tensor([[1]]), torch.tensor([[0], [2]])]
        boxes = [
            torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
            torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.7, 0.6, 0.2, 0.1]]),
        ]
    return {
        "image": torch.randn(2, 8, 2, 2, generator=generator),
        "gt_class": labels,
        "gt_bbox": boxes,
        "im_shape": torch.tensor([[16.0, 16.0], [16.0, 16.0]]),
        "scale_factor": torch.ones(2, 2),
    }


def _targets(batch):
    return [
        {"labels": labels.reshape(-1), "boxes": boxes}
        for labels, boxes in zip(batch["gt_class"], batch["gt_bbox"])
    ]


def _model(decoder, criterion, exclude_post_process=False):
    return DFINE(
        backbone=_FeatureBackbone(),
        encoder=_IdentityEncoder(),
        decoder=decoder,
        criterion=criterion,
        post_process=DETRPostProcess(
            num_classes=3, num_top_queries=6, use_focal_loss=True
        ),
        exclude_post_process=exclude_post_process,
    )


@pytest.mark.numerical
def test_train_losses_and_gradients_match_pinned_upstream(upstream_modules):
    upstream_decoder, upstream_criterion = upstream_modules
    torch.manual_seed(5)
    expected_decoder = upstream_decoder.DFINETransformer(**_decoder_config()).train()
    actual_decoder = DFINETransformer(**_decoder_config()).train()
    actual_decoder.load_state_dict(expected_decoder.state_dict(), strict=True)
    expected_criterion = upstream_criterion.DFINECriterion(
        **_criterion_config(_matcher())
    )
    actual_criterion = DFINECriterion(**_criterion_config(_matcher()))
    actual_model = _model(actual_decoder, actual_criterion).train()
    batch = _batch()

    torch.manual_seed(23)
    expected_outputs = expected_decoder(_FeatureBackbone()(batch), _targets(batch))
    expected_losses = expected_criterion(expected_outputs, _targets(batch))
    expected_total = sum(expected_losses.values())
    expected_total.backward()

    torch.manual_seed(23)
    actual_losses = actual_model(batch)
    actual_losses["loss"].backward()

    assert actual_losses.keys() == expected_losses.keys() | {"loss"}, set(
        actual_losses
    ) ^ (set(expected_losses) | {"loss"})
    for key, expected in expected_losses.items():
        torch.testing.assert_close(actual_losses[key], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual_losses["loss"], expected_total, rtol=1e-5, atol=1e-6
    )
    for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
        actual_decoder.named_parameters(), expected_decoder.named_parameters()
    ):
        assert actual_name == expected_name
        if expected_parameter.grad is not None:
            torch.testing.assert_close(
                actual_parameter.grad,
                expected_parameter.grad,
                rtol=2e-5,
                atol=2e-6,
                msg=actual_name,
            )


@pytest.mark.numerical
def test_train_empty_targets_remain_finite(upstream_modules):
    upstream_decoder, _ = upstream_modules
    torch.manual_seed(5)
    expected_decoder = upstream_decoder.DFINETransformer(**_decoder_config()).train()
    actual_decoder = DFINETransformer(**_decoder_config()).train()
    actual_decoder.load_state_dict(expected_decoder.state_dict(), strict=True)
    model = _model(
        actual_decoder, DFINECriterion(**_criterion_config(_matcher()))
    ).train()

    torch.manual_seed(23)
    losses = model(_batch(empty=True))
    assert losses
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in actual_decoder.parameters()
    )


@pytest.mark.numerical
def test_eval_raw_and_final_detections_match_pinned_upstream(upstream_modules):
    upstream_decoder, _ = upstream_modules
    torch.manual_seed(7)
    expected_decoder = upstream_decoder.DFINETransformer(**_decoder_config(0)).eval()
    actual_decoder = DFINETransformer(**_decoder_config(0)).eval()
    actual_decoder.load_state_dict(expected_decoder.state_dict(), strict=True)
    raw_model = _model(
        actual_decoder, DFINECriterion(**_criterion_config(_matcher())), True
    ).eval()
    final_model = _model(actual_decoder, raw_model.criterion).eval()
    batch = _batch()

    with torch.inference_mode():
        expected_raw = expected_decoder(_FeatureBackbone()(batch))
        actual_raw = raw_model(batch)
        actual_final = final_model(batch)
        expected_bbox, expected_bbox_num, _ = final_model.post_process(
            (expected_raw["pred_boxes"], expected_raw["pred_logits"], None),
            batch["im_shape"],
            batch["scale_factor"],
        )
    for key in expected_raw:
        torch.testing.assert_close(
            actual_raw[key], expected_raw[key], rtol=1e-5, atol=1e-6
        )
    torch.testing.assert_close(
        actual_final["bbox"], expected_bbox, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(actual_final["bbox_num"], expected_bbox_num)

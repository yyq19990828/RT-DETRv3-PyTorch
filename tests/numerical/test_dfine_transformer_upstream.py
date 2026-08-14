import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from detrs.modeling.transformers.dfine_decoder import DFINETransformer
from detrs.modeling.transformers.dfine_support import (
    get_contrastive_denoising_training_group,
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
def upstream_decoder():
    root_value = os.environ.get("DFINE_UPSTREAM_ROOT")
    if not root_value:
        pytest.skip("set DFINE_UPSTREAM_ROOT to the pinned D-FINE checkout")
    upstream = Path(root_value).expanduser().resolve()
    source_root = upstream / "src/zoo/dfine"
    if not (source_root / "dfine_decoder.py").is_file():
        pytest.skip("pinned D-FINE decoder sources are unavailable")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert sha == PINNED_SHA

    package = "upstream_dfine_transformer"
    for name in (package, f"{package}.zoo", f"{package}.zoo.dfine"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    core = types.ModuleType(f"{package}.core")
    core.register = lambda: lambda cls: cls
    sys.modules[core.__name__] = core
    denoising = types.ModuleType(f"{package}.zoo.dfine.denoising")
    denoising.get_contrastive_denoising_training_group = (
        get_contrastive_denoising_training_group
    )
    sys.modules[denoising.__name__] = denoising
    _load(f"{package}.zoo.dfine.box_ops", source_root / "box_ops.py")
    _load(f"{package}.zoo.dfine.utils", source_root / "utils.py")
    _load(f"{package}.zoo.dfine.dfine_utils", source_root / "dfine_utils.py")
    return _load(f"{package}.zoo.dfine.dfine_decoder", source_root / "dfine_decoder.py")


def _config(num_denoising=4, box_noise_scale=1.0):
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
        "box_noise_scale": box_noise_scale,
        "reg_max": 8,
    }


def _inputs():
    generator = torch.Generator().manual_seed(19)
    features = [
        torch.randn(2, 8, 2, 2, generator=generator),
        torch.randn(2, 8, 1, 1, generator=generator),
    ]
    targets = [
        {
            "labels": torch.tensor([1], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
        },
        {
            "labels": torch.tensor([0, 2], dtype=torch.long),
            "boxes": torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.7, 0.6, 0.2, 0.1]]),
        },
    ]
    return features, targets


def _assert_nested_close(actual, expected, path="output"):
    assert type(actual) is type(expected), path
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_nested_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_close(actual_item, expected_item, f"{path}[{index}]")
    else:
        assert actual == expected, path


@pytest.mark.numerical
def test_real_top_level_train_every_output_matches_pinned_upstream(upstream_decoder):
    torch.manual_seed(5)
    expected_model = upstream_decoder.DFINETransformer(**_config()).train()
    actual_model = DFINETransformer(**_config()).train()
    assert list(actual_model.state_dict()) == list(expected_model.state_dict())
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    features, targets = _inputs()

    torch.manual_seed(23)
    expected = expected_model(features, targets)
    torch.manual_seed(23)
    actual = actual_model(features, targets)
    _assert_nested_close(actual, expected)


@pytest.mark.numerical
def test_nondefault_denoising_box_noise_matches_pinned_upstream(upstream_decoder):
    config = _config(box_noise_scale=0.25)
    torch.manual_seed(13)
    expected_model = upstream_decoder.DFINETransformer(**config).train()
    actual_model = DFINETransformer(**config).train()
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    features, targets = _inputs()

    torch.manual_seed(29)
    expected = expected_model(features, targets)
    torch.manual_seed(29)
    actual = actual_model(features, targets)
    _assert_nested_close(actual, expected)

    default_model = DFINETransformer(**_config(box_noise_scale=1.0)).train()
    default_model.load_state_dict(expected_model.state_dict(), strict=True)
    torch.manual_seed(29)
    default_output = default_model(features, targets)
    _assert_nested_close(actual, default_output)


@pytest.mark.numerical
def test_real_top_level_eval_matches_pinned_upstream(upstream_decoder):
    torch.manual_seed(7)
    expected_model = upstream_decoder.DFINETransformer(**_config(0)).eval()
    actual_model = DFINETransformer(**_config(0)).eval()
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    features, _ = _inputs()
    with torch.inference_mode():
        expected = expected_model(features)
        actual = actual_model(features)
    _assert_nested_close(actual, expected)


@pytest.mark.numerical
def test_top_level_deploy_conversion_preserves_pinned_eval_result(upstream_decoder):
    torch.manual_seed(11)
    expected_model = upstream_decoder.DFINETransformer(**_config(0)).eval()
    actual_model = DFINETransformer(**_config(0)).eval()
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    features, _ = _inputs()
    with torch.inference_mode():
        expected = expected_model(features)
        actual_model.convert_to_deploy()
        actual = actual_model(features)
    _assert_nested_close(actual, expected)
    assert "decoder.project" in actual_model.state_dict()

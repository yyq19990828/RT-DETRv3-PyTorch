from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from ppdet_pytorch.modeling.backbones.hgnetv2 import HGNetv2

UPSTREAM_SHA = "267a6da6d04c8ad52e54120692896515b9e55981"
CASES = [
    (
        "B0",
        True,
        (2, 3),
        7555621,
        "70a372e8cbc59b34c5da2943261ecb633faf304a58e7e05461a27bd8d8b7f3d1",
    ),
    (
        "B2",
        True,
        (1, 2, 3),
        24362501,
        "41272985db6136ac11732b246c7ea794dcc203d0a8cbc463152c840b9d9f22d1",
    ),
    (
        "B4",
        False,
        (1, 2, 3),
        54559385,
        "a72ad8d32902c90f5fa07f642034955d0ed9149c46d4d97f0e2ec36344d24bea",
    ),
    (
        "B5",
        False,
        (1, 2, 3),
        133945533,
        "812d5cde50e415abfb1ea1dd27121fa4f861522a327e48033a3cea8d604b3545",
    ),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load upstream module {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_hgnetv2():
    root_value = os.environ.get("DFINE_UPSTREAM_ROOT")
    if not root_value:
        pytest.skip("set DFINE_UPSTREAM_ROOT to the pinned D-FINE checkout")
    root = Path(root_value).expanduser().resolve()
    source = root / "src/nn/backbone/hgnetv2.py"
    if not source.is_file():
        pytest.skip("pinned D-FINE HGNetv2 source is absent")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == UPSTREAM_SHA

    package_names = (
        "_dfine_reference",
        "_dfine_reference.nn",
        "_dfine_reference.nn.backbone",
    )
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
    core = types.ModuleType("_dfine_reference.core")
    core.register = lambda: lambda cls: cls
    sys.modules[core.__name__] = core
    _load_module(
        "_dfine_reference.nn.backbone.common",
        root / "src/nn/backbone/common.py",
    )
    return _load_module("_dfine_reference.nn.backbone.hgnetv2", source).HGNetv2


@pytest.mark.parametrize(
    ("name", "use_lab", "return_idx", "expected_size", "expected_sha256"), CASES
)
def test_hgnetv2_official_state_and_stage_parity(
    upstream_hgnetv2,
    name,
    use_lab,
    return_idx,
    expected_size,
    expected_sha256,
):
    checkpoint_root_value = os.environ.get("HGNETV2_CHECKPOINT_ROOT")
    if not checkpoint_root_value:
        pytest.skip("set HGNETV2_CHECKPOINT_ROOT to the official stage-1 weights")
    checkpoint = Path(
        checkpoint_root_value
    ).expanduser().resolve() / "PPHGNetV2_{}_stage1.pth".format(name)
    if not checkpoint.is_file():
        pytest.skip("official {} stage-1 checkpoint is absent".format(name))
    assert checkpoint.stat().st_size == expected_size
    assert _sha256(checkpoint) == expected_sha256

    reference = upstream_hgnetv2(
        name=name,
        use_lab=use_lab,
        return_idx=list(return_idx),
        freeze_at=-1,
        freeze_norm=False,
        pretrained=False,
    ).eval()
    local = HGNetv2(
        name=name,
        use_lab=use_lab,
        return_idx=return_idx,
        freeze_at=-1,
        freeze_norm=False,
    ).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference.load_state_dict(state, strict=True)
    local.load_pretrained(checkpoint)

    assert list(reference.state_dict()) == list(local.state_dict())
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, local.state_dict()[key], rtol=0, atol=0)

    generator = torch.Generator().manual_seed(0)
    reference_value = torch.randn(1, 3, 64, 64, generator=generator)
    local_value = reference_value.clone()
    with torch.no_grad():
        reference_value = reference.stem(reference_value)
        local_value = local.stem(local_value)
        torch.testing.assert_close(reference_value, local_value, rtol=1e-5, atol=1e-6)
        for reference_stage, local_stage in zip(reference.stages, local.stages):
            reference_value = reference_stage(reference_value)
            local_value = local_stage(local_value)
            torch.testing.assert_close(
                reference_value, local_value, rtol=1e-5, atol=1e-6
            )

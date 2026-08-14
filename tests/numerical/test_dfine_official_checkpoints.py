"""Opt-in parity checks for official D-FINE N/S/M/L/X checkpoints."""

import importlib.util
import os
import shutil
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dfine_checkpoint_parity", ROOT / "tools/dev/dfine_checkpoint_parity.py"
)
assert SPEC is not None and SPEC.loader is not None
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)

pytestmark = [pytest.mark.numerical, pytest.mark.slow]


def _assets():
    checkpoint_value = os.environ.get("UPSTREAM_CHECKPOINT_ROOT")
    upstream_value = os.environ.get("DFINE_UPSTREAM_ROOT")
    if not checkpoint_value or not upstream_value:
        pytest.skip("set UPSTREAM_CHECKPOINT_ROOT and DFINE_UPSTREAM_ROOT")
    return Path(checkpoint_value), Path(upstream_value)


@pytest.mark.parametrize("variant", "nsmlx")
def test_official_dfine_checkpoint_matches_pinned_upstream(variant):
    checkpoint_root, upstream_root = _assets()
    result = PARITY.validate_variant(variant, checkpoint_root, upstream_root)
    assert result["status"] == "APPROVE"
    assert result["key_mapping"] == "identity"
    assert result["checks"]


def test_rejects_swapped_variant_before_prediction():
    checkpoint_root, _ = _assets()
    manifest = PARITY.load_manifest()
    _, state = PARITY.preflight_artifact(checkpoint_root, "n", manifest)
    with pytest.raises(
        ValueError, match="identity mapping mismatch.*missing=.*unexpected="
    ):
        PARITY.build_local_model("s", state)


def test_rejects_modified_tensor_before_prediction(tmp_path):
    checkpoint_root, _ = _assets()
    manifest = PARITY.load_manifest()
    source = checkpoint_root / manifest["models"]["n"]["filename"]
    target = tmp_path / source.name
    shutil.copyfile(source, target)
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    key = next(iter(checkpoint["model"]))
    checkpoint["model"][key].view(-1)[0] += 1
    torch.save(checkpoint, target)
    with pytest.raises(ValueError, match="(?:size|SHA-256) mismatch"):
        PARITY.preflight_artifact(tmp_path, "n", manifest)

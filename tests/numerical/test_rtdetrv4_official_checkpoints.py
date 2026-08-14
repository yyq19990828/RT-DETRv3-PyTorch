"""Opt-in parity checks for official RT-DETRv4 S/M/L/X checkpoints."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PARITY_PATH = ROOT / "tools/dev/rtdetrv4_checkpoint_parity.py"

pytestmark = [pytest.mark.numerical, pytest.mark.slow]


def _parity_module():
    sys.path.insert(0, str(PARITY_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "rtdetrv4_checkpoint_parity", PARITY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", tuple("smlx"))
def test_official_rtdetrv4_checkpoint_matches_pinned_upstream(variant):
    checkpoint_root = os.environ.get("RTDETRV4_CHECKPOINT_ROOT")
    upstream_root = os.environ.get("RTDETRV4_UPSTREAM_ROOT")
    if not checkpoint_root or not upstream_root:
        pytest.skip("set RTDETRV4_CHECKPOINT_ROOT and RTDETRV4_UPSTREAM_ROOT")

    result = _parity_module().validate_variant(
        variant, Path(checkpoint_root), Path(upstream_root)
    )

    assert result["status"] == "APPROVE"
    assert all(check["status"] == "APPROVE" for check in result["checks"])


def test_rejects_swapped_rtdetrv4_variant_before_prediction():
    checkpoint_root = os.environ.get("RTDETRV4_CHECKPOINT_ROOT")
    if not checkpoint_root:
        pytest.skip("set RTDETRV4_CHECKPOINT_ROOT")
    parity = _parity_module()
    manifest = parity.load_manifest()
    _, state = parity.preflight_artifact(Path(checkpoint_root), "s", manifest)

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key|size mismatch"):
        parity.build_local_model("m", state)

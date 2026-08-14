"""Opt-in parity checks for official DEIM-RT-DETRv2 S/M/M*/L/X checkpoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.numerical, pytest.mark.slow]


def _assets():
    checkpoint_value = os.environ.get("DEIM_RTDETRV2_CHECKPOINT_ROOT")
    upstream_value = os.environ.get("DEIM_UPSTREAM_ROOT")
    if not checkpoint_value or not upstream_value:
        pytest.skip("set DEIM_RTDETRV2_CHECKPOINT_ROOT and DEIM_UPSTREAM_ROOT")
    return Path(checkpoint_value), Path(upstream_value)


@pytest.mark.parametrize("variant", ["s", "m", "m-star", "l", "x"])
def test_official_checkpoint_matches_pinned_upstream(variant):
    sys.path.insert(0, str(ROOT / "tools/dev"))
    from deim_rtdetrv2_checkpoint_parity import validate_variant

    checkpoint_root, upstream_root = _assets()
    result = validate_variant(variant, checkpoint_root, upstream_root)

    assert result["status"] == "APPROVE"
    assert result["key_mapping"] == "identity"
    assert result["checks"]
    assert all(check["max_abs_error"] == 0 for check in result["checks"])


def test_rejects_r50_variant_swap_before_prediction():
    import torch

    from detrs.core.workspace import create, load_config

    checkpoint_root, _ = _assets()
    state = torch.load(
        checkpoint_root / "deim_r50vd_m_60e_coco.pth",
        map_location="cpu",
        weights_only=False,
    )["model"]
    model = create(
        load_config(ROOT / "configs/deim/rtdetrv2/deim_r50vd_60e_coco.yml").architecture
    )

    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state, strict=True)

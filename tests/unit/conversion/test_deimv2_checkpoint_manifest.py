"""DEIMv2 checkpoint manifest completeness and model-graph alignment."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from detrs import data, engine, modeling  # noqa: F401
from detrs.core.workspace import create, load_config

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "configs/checkpoints/deimv2_coco.yml"

EXPECTED = {
    "x": 0.578,
    "l": 0.560,
    "m": 0.530,
    "s": 0.509,
    "n": 0.430,
    "pico": 0.385,
    "femto": 0.310,
    "atto": 0.238,
}
CONFIGS = {
    "x": "configs/deimv2/deimv2_dinov3_x_coco.yml",
    "l": "configs/deimv2/deimv2_dinov3_l_coco.yml",
    "m": "configs/deimv2/deimv2_dinov3_m_coco.yml",
    "s": "configs/deimv2/deimv2_dinov3_s_coco.yml",
    "n": "configs/deimv2/deimv2_hgnetv2_n_coco.yml",
    "pico": "configs/deimv2/deimv2_hgnetv2_pico_coco.yml",
    "femto": "configs/deimv2/deimv2_hgnetv2_femto_coco.yml",
    "atto": "configs/deimv2/deimv2_hgnetv2_atto_coco.yml",
}
TENSOR_COUNTS = {
    "x": 989,
    "l": 809,
    "m": 769,
    "s": 673,
    "n": 639,
    "pico": 551,
    "femto": 493,
    "atto": 493,
}


def _manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_covers_all_eight_official_variants():
    manifest = _manifest()
    assert manifest["schema_version"] == 2
    assert manifest["family"] == "deimv2"
    assert manifest["hosting"] == "upstream"
    assert manifest["source_repository"]["revision"] == (
        "add5bcdb499bf7b8a366bfeac1a47d3dc278de27"
    )
    assert set(manifest["models"]) == set(EXPECTED)
    for variant, entry in manifest["models"].items():
        assert entry["alias"] == f"deimv2-{variant}"
        assert entry["config"] == CONFIGS[variant]
        assert entry["official_bbox_ap"] == EXPECTED[variant]
        assert entry["container_keys"] == ["model"]
        assert entry["evaluation_state"] == "model"
        assert entry["state_tensor_count"] == TENSOR_COUNTS[variant]
        assert entry["key_mapping"] == "identity"
        assert entry["source_size_bytes"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", entry["source_sha256"])
        # Google Drive artifacts are list/verify only; no direct download URL.
        assert "download_url" not in entry
        assert entry["source_url"].startswith("https://drive.google.com/")


def test_manifest_tensor_counts_match_built_models(isolated_workspace):
    for variant, expected in TENSOR_COUNTS.items():
        config = load_config(ROOT / CONFIGS[variant])
        model = create(config.architecture)
        assert len(model.state_dict()) == expected, variant

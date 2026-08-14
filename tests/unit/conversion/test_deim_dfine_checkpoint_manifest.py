import hashlib
import os
from pathlib import Path

import pytest
import torch
import yaml

from detrs.core.workspace import create, load_config

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "configs/checkpoints/deim_dfine_coco.yml"


def _manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_deim_dfine_checkpoint_manifest_is_complete_and_target_aware():
    manifest = _manifest()
    official_bbox_ap = {"n": 0.430, "s": 0.490, "m": 0.527, "l": 0.547, "x": 0.565}

    assert manifest["schema_version"] == 2
    assert manifest["family"] == "deim-dfine"
    assert manifest["hosting"] == "upstream"
    assert manifest["source_repository"]["revision"] == (
        "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
    )
    assert set(manifest["models"]) == set("nsmlx")
    for variant, model in manifest["models"].items():
        assert model["alias"] == f"deim-dfine-{variant}"
        assert model["artifact_format"] == "pytorch-checkpoint"
        assert (ROOT / model["config"]).is_file()
        assert model["filename"] == f"deim_hgnetv2_{variant}_coco.pth"
        assert model["official_bbox_ap"] == official_bbox_ap[variant]
        assert model["source_url"].startswith("https://drive.google.com/file/d/")
        assert model["source_size_bytes"] > 0
        assert len(model["source_sha256"]) == 64
        int(model["source_sha256"], 16)
        assert model["container_keys"] == ["model"]
        assert model["state_tensor_count"] > 0
        assert model["key_mapping"] == "identity"
        assert model["tensor_layout"] == "pytorch-native"


@pytest.mark.parametrize("variant", "nsmlx")
def test_official_deim_dfine_state_strictly_matches_configured_graph(
    variant, isolated_workspace
):
    root_value = os.environ.get("DEIM_DFINE_CHECKPOINT_ROOT")
    if not root_value:
        pytest.skip("set DEIM_DFINE_CHECKPOINT_ROOT")
    manifest = _manifest()
    entry = manifest["models"][variant]
    path = Path(root_value) / entry["filename"]
    if not path.is_file():
        pytest.skip(f"missing official checkpoint: {path}")
    assert path.stat().st_size == entry["source_size_bytes"]
    assert _sha256(path) == entry["source_sha256"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert list(checkpoint) == entry["container_keys"]
    assert len(checkpoint["model"]) == entry["state_tensor_count"]

    config = load_config(ROOT / entry["config"])
    model = create(config.architecture)
    model.load_state_dict(checkpoint["model"], strict=True)

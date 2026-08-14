import hashlib
import os
from pathlib import Path

import pytest
import torch
import yaml

from ppdet_pytorch.core.workspace import create, load_config

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "configs/checkpoints/rtdetrv4_coco.yml"


def _manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_rtdetrv4_checkpoint_manifest_is_complete_and_target_aware():
    manifest = _manifest()
    official_bbox_ap = {"s": 0.498, "m": 0.537, "l": 0.554, "x": 0.570}
    tensor_counts = {"s": 796, "m": 1055, "l": 1255, "x": 1573}

    assert manifest["schema_version"] == 2
    assert manifest["family"] == "rtdetrv4"
    assert manifest["hosting"] == "upstream"
    assert manifest["source_repository"]["revision"] == (
        "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"
    )
    assert set(manifest["models"]) == set("smlx")
    for variant, model in manifest["models"].items():
        assert model["alias"] == f"rtdetrv4-{variant}"
        assert model["artifact_format"] == "pytorch-checkpoint"
        assert (ROOT / model["config"]).is_file()
        assert model["filename"] == f"RTv4-{variant.upper()}-hgnet.pth"
        assert model["official_bbox_ap"] == official_bbox_ap[variant]
        assert model["source_url"].startswith("https://drive.google.com/file/d/")
        assert model["source_size_bytes"] > 0
        assert len(model["source_sha256"]) == 64
        int(model["source_sha256"], 16)
        assert model["container_keys"] == [
            "date",
            "last_epoch",
            "model",
            "criterion",
            "postprocessor",
            "ema",
            "scaler",
            "optimizer",
            "lr_warmup_scheduler",
        ]
        assert model["evaluation_state"] == "ema.module"
        assert model["state_tensor_count"] == tensor_counts[variant]
        assert model["key_mapping"] == "identity"
        assert model["tensor_layout"] == "pytorch-native"


@pytest.mark.parametrize("variant", tuple("smlx"))
def test_official_rtdetrv4_state_strictly_matches_configured_graph(
    variant, isolated_workspace
):
    root_value = os.environ.get("RTDETRV4_CHECKPOINT_ROOT")
    if not root_value:
        pytest.skip("set RTDETRV4_CHECKPOINT_ROOT")
    manifest = _manifest()
    entry = manifest["models"][variant]
    path = Path(root_value) / entry["filename"]
    if not path.is_file():
        pytest.skip(f"missing official checkpoint: {path}")
    assert path.stat().st_size == entry["source_size_bytes"]
    assert _sha256(path) == entry["source_sha256"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert list(checkpoint) == entry["container_keys"]
    state = checkpoint["ema"]["module"]
    assert len(state) == entry["state_tensor_count"]

    config = load_config(ROOT / entry["config"])
    model = create(config.architecture)
    model.load_state_dict(state, strict=True)

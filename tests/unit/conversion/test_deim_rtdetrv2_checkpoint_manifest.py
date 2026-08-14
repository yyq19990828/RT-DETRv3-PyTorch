import hashlib
import os
from pathlib import Path

import pytest
import torch
import yaml

from detrs.core.workspace import create, load_config
from detrs.utils.checkpoint import load_pretrain_weight

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "configs/checkpoints/deim_rtdetrv2_coco.yml"


def _manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_is_complete_and_target_aware():
    manifest = _manifest()
    official_ap = {"s": 0.490, "m": 0.509, "m-star": 0.532, "l": 0.543, "x": 0.555}

    assert manifest["schema_version"] == 2
    assert manifest["family"] == "deim-rtdetrv2"
    assert manifest["hosting"] == "upstream"
    assert manifest["source_repository"]["revision"] == (
        "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
    )
    assert set(manifest["models"]) == set(official_ap)
    for variant, model in manifest["models"].items():
        assert model["alias"] == f"deim-rtv2-{variant}"
        assert model["artifact_format"] == "pytorch-checkpoint"
        assert (ROOT / model["config"]).is_file()
        assert model["official_bbox_ap"] == official_ap[variant]
        assert model["source_url"].startswith("https://drive.google.com/file/d/")
        assert model["source_size_bytes"] > 0
        assert len(model["source_sha256"]) == 64
        int(model["source_sha256"], 16)
        assert model["container_keys"] == ["model"]
        assert model["state_tensor_count"] > 0
        assert model["key_mapping"] == "identity"
        assert model["tensor_layout"] == "pytorch-native"

    pretrained = manifest["pretrained_backbones"]
    assert set(pretrained) == {"r18vd", "r34vd", "r50vd", "r101vd"}
    for model in pretrained.values():
        assert model["source_url"].startswith(
            "https://github.com/lyuwenyu/storage/releases/download/v0.1/"
        )
        assert model["source_size_bytes"] > 0
        assert len(model["source_sha256"]) == 64
        int(model["source_sha256"], 16)
        assert model["container_keys"] == []
        assert model["state_tensor_count"] > 0
        assert model["key_mapping"] == "backbone-prefix"


@pytest.mark.parametrize("variant", ["s", "m", "m-star", "l", "x"])
def test_official_state_strictly_matches_configured_graph(variant, isolated_workspace):
    root_value = os.environ.get("DEIM_RTDETRV2_CHECKPOINT_ROOT")
    if not root_value:
        pytest.skip("set DEIM_RTDETRV2_CHECKPOINT_ROOT")
    entry = _manifest()["models"][variant]
    path = Path(root_value) / entry["filename"]
    if not path.is_file():
        pytest.skip(f"missing official checkpoint: {path}")
    assert path.stat().st_size == entry["source_size_bytes"]
    assert _sha256(path) == entry["source_sha256"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert list(checkpoint) == entry["container_keys"]
    assert len(checkpoint["model"]) == entry["state_tensor_count"]

    model = create(load_config(ROOT / entry["config"]).architecture)
    model.load_state_dict(checkpoint["model"], strict=True)


@pytest.mark.parametrize("variant", ["r18vd", "r34vd", "r50vd", "r101vd"])
def test_official_backbone_pretrain_state_is_complete_and_loads(
    variant, isolated_workspace
):
    root_value = os.environ.get("DEIM_RTDETRV2_PRETRAINED_ROOT")
    if not root_value:
        pytest.skip("set DEIM_RTDETRV2_PRETRAINED_ROOT")
    manifest = _manifest()
    entry = manifest["pretrained_backbones"][variant]
    path = Path(root_value) / entry["filename"]
    if not path.is_file():
        pytest.skip(f"missing official backbone checkpoint: {path}")
    assert path.stat().st_size == entry["source_size_bytes"]
    assert _sha256(path) == entry["source_sha256"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert len(state) == entry["state_tensor_count"]

    config_entry = manifest["models"][
        {"r18vd": "s", "r34vd": "m", "r50vd": "l", "r101vd": "x"}[variant]
    ]
    model = create(load_config(ROOT / config_entry["config"]).architecture)
    head_before = model.decoder.enc_score_head.weight.detach().clone()
    load_pretrain_weight(model, str(path))

    missing = set(model.backbone.state_dict()) - set(state)
    assert missing
    assert all(key.endswith(".num_batches_tracked") for key in missing)
    assert not set(state) - set(model.backbone.state_dict())
    for key, value in state.items():
        torch.testing.assert_close(
            model.backbone.state_dict()[key], value, rtol=0, atol=0
        )
    torch.testing.assert_close(
        model.decoder.enc_score_head.weight, head_before, rtol=0, atol=0
    )

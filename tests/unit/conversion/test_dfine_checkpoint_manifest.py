from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "configs/checkpoints/dfine_coco.yml"


def test_dfine_checkpoint_manifest_is_complete_and_target_aware():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    official_bbox_ap = {"n": 0.428, "s": 0.485, "m": 0.523, "l": 0.540, "x": 0.558}

    assert manifest["schema_version"] == 2
    assert manifest["family"] == "dfine"
    assert manifest["hosting"] == "upstream"
    assert manifest["source_repository"]["revision"] == (
        "267a6da6d04c8ad52e54120692896515b9e55981"
    )
    assert set(manifest["models"]) == set("nsmlx")
    for variant, model in manifest["models"].items():
        assert model["alias"] == f"dfine-{variant}"
        assert model["artifact_format"] == "pytorch-checkpoint"
        assert (ROOT / model["config"]).is_file()
        assert model["filename"] == f"dfine_{variant}_coco.pth"
        assert model["official_bbox_ap"] == official_bbox_ap[variant]
        assert model["source_url"].startswith(
            "https://api.github.com/repos/Peterande/storage/releases/assets/"
        )
        assert model["source_size_bytes"] > 0
        assert len(model["source_sha256"]) == 64
        int(model["source_sha256"], 16)
        assert model["container_keys"] == ["model"]
        assert model["key_mapping"] == "identity"
        assert model["tensor_layout"] == "pytorch-native"

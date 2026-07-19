import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "configs/checkpoints/rtdetrv3_coco.yml"
PYPROJECT = ROOT / "pyproject.toml"


def test_checkpoint_manifest_references_repository_configs():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert len(manifest["source_repository"]["revision"]) == 40
    assert manifest["distribution"] == {
        "repository": "https://github.com/yyq19990828/RT-DETRv3-PyTorch",
        "release_tag": "v0.1.0",
    }
    package_version = re.search(
        r'^version = "([^"]+)"$',
        PYPROJECT.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert package_version is not None
    assert manifest["distribution"]["release_tag"] == f"v{package_version.group(1)}"
    assert set(manifest["models"]) == {
        "rtdetrv3_r18vd_6x_coco",
        "rtdetrv3_r34vd_6x_coco",
        "rtdetrv3_r50vd_6x_coco",
    }

    file_ids = set()
    for model in manifest["models"].values():
        assert (ROOT / model["config"]).is_file()
        assert model["source_url"].startswith("https://drive.google.com/")
        assert model["source_file_id"] not in file_ids
        file_ids.add(model["source_file_id"])


def test_verified_checkpoint_has_sha256_and_size():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    verified = [
        model for model in manifest["models"].values() if model["status"] == "verified"
    ]

    assert verified
    for model in verified:
        assert model["source_size_bytes"] > 0
        assert len(model["source_sha256"]) == 64
        int(model["source_sha256"], 16)
        assert model["source_path"].startswith("pretrained_models/paddle/")
        converted = model["converted_artifact"]
        assert converted["path"].startswith("pretrained_models/pytorch/")
        assert converted["size_bytes"] > 0
        assert len(converted["sha256"]) == 64
        int(converted["sha256"], 16)
        assert converted["mapping_report"].startswith("pretrained_models/reports/")
        assert converted["mapping_size_bytes"] > 0
        assert len(converted["mapping_sha256"]) == 64
        int(converted["mapping_sha256"], 16)
        assert converted["mapping_count"] > 0
        assert converted["distribution_status"] == "published"
        assert converted["download_url"] == (
            f"{manifest['distribution']['repository']}/releases/download/"
            f"{manifest['distribution']['release_tag']}/"
            f"{Path(converted['path']).name}"
        )


def test_r18_pretraining_manifest_is_target_aware():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    pretraining = manifest["pretraining"]["resnet18_vd"]

    assert pretraining["used_by"] == "rtdetrv3_r18vd_6x_coco"
    assert (ROOT / pretraining["target_config"]).is_file()
    assert pretraining["source_url"].endswith("/ResNet18_vd_pretrained.pdparams")
    assert pretraining["source_size_bytes"] == 44850756
    assert len(pretraining["source_sha256"]) == 64
    int(pretraining["source_sha256"], 16)
    assert pretraining["converted_tensor_count"] == 115
    assert pretraining["converted_artifact"]["alias"] == "r18-backbone"
    assert pretraining["converted_artifact"]["mapping_size_bytes"] == 55344
    assert len(pretraining["converted_artifact"]["mapping_sha256"]) == 64
    int(pretraining["converted_artifact"]["mapping_sha256"], 16)
    assert pretraining["converted_artifact"]["mapping_count"] == 115
    converted = pretraining["converted_artifact"]
    assert converted["distribution_status"] == "published"
    assert converted["download_url"] == (
        f"{manifest['distribution']['repository']}/releases/download/"
        f"{manifest['distribution']['release_tag']}/"
        f"{Path(converted['path']).name}"
    )

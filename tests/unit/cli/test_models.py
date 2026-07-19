import hashlib
import io
import json
from pathlib import Path

import pytest
import yaml

from ppdet_pytorch.cli import models as models_cli


def _write_manifest(
    path: Path,
    payload: bytes,
    *,
    published: bool = False,
) -> None:
    models = {}
    for alias, name in models_cli.MODEL_ALIASES.items():
        artifact = {
            "path": f"pretrained_models/pytorch/{alias}.pth",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "distribution_status": "published" if published else "unpublished",
        }
        if published:
            artifact["download_url"] = f"https://example.com/{alias}.pth"
        models[name] = {
            "config": f"configs/rtdetrv3/{alias}.yml",
            "converted_artifact": artifact,
        }
    path.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")


def test_default_manifest_lists_all_models_as_unpublished(capsys):
    assert models_cli.main(["list", "--json"]) == 0

    records = json.loads(capsys.readouterr().out)
    assert [record["alias"] for record in records] == ["r18", "r34", "r50"]
    assert {record["distribution_status"] for record in records} == {"unpublished"}
    assert all(record["download_url"] is None for record in records)


def test_verify_checks_size_and_sha256(tmp_path):
    payload = b"converted checkpoint"
    manifest = tmp_path / "manifest.yml"
    checkpoint = tmp_path / "r18.pth"
    _write_manifest(manifest, payload)
    checkpoint.write_bytes(payload)
    artifacts = models_cli.load_artifacts(manifest)

    result = models_cli.verify_artifact(checkpoint, artifacts["r18"])

    assert result["verified"] is True
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    checkpoint.write_bytes(b"wrong-size")
    with pytest.raises(ValueError, match="size mismatch"):
        models_cli.verify_artifact(checkpoint, artifacts["r18"])


def test_download_refuses_unpublished_model(tmp_path, capsys):
    payload = b"checkpoint"
    manifest = tmp_path / "manifest.yml"
    _write_manifest(manifest, payload)

    return_code = models_cli.main(
        [
            "--manifest",
            str(manifest),
            "download",
            "r18",
            "--output",
            str(tmp_path / "model.pth"),
        ]
    )

    assert return_code == 1
    assert "not published" in capsys.readouterr().err
    assert not (tmp_path / "model.pth").exists()


def test_download_is_atomic_and_verifies_content(tmp_path, monkeypatch):
    payload = b"published checkpoint"
    manifest = tmp_path / "manifest.yml"
    destination = tmp_path / "models" / "r18.pth"
    _write_manifest(manifest, payload, published=True)
    artifact = models_cli.load_artifacts(manifest)["r18"]

    class Response(io.BytesIO):
        def geturl(self):
            return artifact.download_url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(models_cli, "_open_url", lambda request: Response(payload))

    result = models_cli.download_artifact(artifact, destination, force=False)

    assert result["verified"] is True
    assert destination.read_bytes() == payload
    assert list(destination.parent.glob("*.part")) == []


def test_download_preserves_mismatched_existing_file_without_force(
    tmp_path, monkeypatch
):
    payload = b"published checkpoint"
    manifest = tmp_path / "manifest.yml"
    destination = tmp_path / "r18.pth"
    destination.write_bytes(b"user data")
    _write_manifest(manifest, payload, published=True)
    artifact = models_cli.load_artifacts(manifest)["r18"]
    monkeypatch.setattr(
        models_cli,
        "_open_url",
        lambda request: pytest.fail("download should not start"),
    )

    with pytest.raises(FileExistsError, match="--force"):
        models_cli.download_artifact(artifact, destination, force=False)

    assert destination.read_bytes() == b"user data"

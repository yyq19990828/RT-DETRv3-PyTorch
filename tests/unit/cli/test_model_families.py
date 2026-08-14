import hashlib
import io
import json
from pathlib import Path
from typing import Optional

import pytest
import yaml

from detrs.cli import models as models_cli

ROOT = Path(__file__).resolve().parents[3]


def _write_v2_manifest(
    path: Path,
    payload: bytes,
    *,
    alias: str = "dfine-n",
    artifact_path: str = "pretrained_models/upstream/dfine/dfine_n_coco.pth",
    source_url: str = "https://example.com/official/dfine_n_coco.pth",
    download_url: Optional[str] = None,
):
    document = {
        "schema_version": 2,
        "family": "dfine",
        "hosting": "upstream",
        "models": {
            "n": {
                "alias": alias,
                "config": "configs/dfine/dfine_hgnetv2_n_coco.yml",
                "path": artifact_path,
                "artifact_format": "pytorch-checkpoint",
                "source_url": source_url,
                "source_size_bytes": len(payload),
                "source_sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    if download_url is not None:
        document["models"]["n"]["download_url"] = download_url
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


@pytest.mark.parametrize(
    ("family", "aliases"),
    [
        ("dfine", [f"dfine-{variant}" for variant in "nsmlx"]),
        ("deim-dfine", [f"deim-dfine-{variant}" for variant in "nsmlx"]),
        (
            "deimv2",
            [
                "deimv2-x",
                "deimv2-l",
                "deimv2-m",
                "deimv2-s",
                "deimv2-n",
                "deimv2-pico",
                "deimv2-femto",
                "deimv2-atto",
            ],
        ),
        (
            "deim-rtdetrv2",
            [
                "deim-rtv2-s",
                "deim-rtv2-m",
                "deim-rtv2-m-star",
                "deim-rtv2-l",
                "deim-rtv2-x",
            ],
        ),
        ("rtdetrv4", [f"rtdetrv4-{variant}" for variant in "smlx"]),
    ],
)
def test_family_list_uses_collision_free_schema_v2_aliases(family, aliases, capsys):
    assert models_cli.main(["--family", family, "list", "--json"]) == 0

    records = json.loads(capsys.readouterr().out)
    assert [record["alias"] for record in records] == aliases
    assert {record["hosting"] for record in records} == {"upstream"}
    assert {record["artifact_format"] for record in records} == {"pytorch-checkpoint"}


def test_manifest_has_priority_over_family(tmp_path, capsys):
    payload = b"custom"
    manifest = tmp_path / "custom.yml"
    _write_v2_manifest(manifest, payload, alias="custom-model")

    assert (
        models_cli.main(
            ["--family", "rtdetrv4", "--manifest", str(manifest), "list", "--json"]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)[0]["alias"] == "custom-model"


def test_schema_v2_verify_checks_upstream_artifact(tmp_path):
    payload = b"official upstream checkpoint"
    manifest = tmp_path / "manifest.yml"
    checkpoint = tmp_path / "checkpoint.pth"
    _write_v2_manifest(manifest, payload)
    checkpoint.write_bytes(payload)

    artifact = models_cli.load_artifacts(manifest)["dfine-n"]
    assert models_cli.verify_artifact(checkpoint, artifact)["verified"] is True

    checkpoint.write_bytes(b"official upstream checkpoinu")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        models_cli.verify_artifact(checkpoint, artifact)


def test_schema_v2_atomic_download_is_direct_and_verified(tmp_path, monkeypatch):
    payload = b"direct upstream checkpoint"
    manifest = tmp_path / "manifest.yml"
    destination = tmp_path / "models" / "checkpoint.pth"
    download_url = "https://example.com/download/checkpoint.pth"
    _write_v2_manifest(manifest, payload, download_url=download_url)
    artifact = models_cli.load_artifacts(manifest)["dfine-n"]

    class Response(io.BytesIO):
        def geturl(self):
            return download_url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(models_cli, "_open_url", lambda request: Response(payload))

    result = models_cli.download_artifact(artifact, destination, force=False)

    assert result["verified"] is True
    assert destination.read_bytes() == payload
    assert list(destination.parent.glob("*.part")) == []


def test_rejects_gated_download_with_official_url_and_no_partial(tmp_path, capsys):
    payload = b"gated checkpoint"
    manifest = tmp_path / "manifest.yml"
    destination = tmp_path / "model.pth"
    source_url = "https://drive.google.com/file/d/official"
    _write_v2_manifest(manifest, payload, source_url=source_url)

    result = models_cli.main(
        [
            "--manifest",
            str(manifest),
            "download",
            "dfine-n",
            "--output",
            str(destination),
        ]
    )

    assert result == 1
    assert source_url in capsys.readouterr().err
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("alias", "DFINE-N", "invalid artifact alias"),
        ("path", "../checkpoint.pth", "repository-relative"),
        ("source_url", "http://example.com/checkpoint.pth", "must use HTTPS"),
        ("source_sha256", "0" * 63, "invalid n SHA-256"),
    ],
)
def test_rejects_path_traversal_checksum_non_https_and_invalid_alias(
    tmp_path, field, value, message
):
    manifest = tmp_path / "manifest.yml"
    _write_v2_manifest(manifest, b"checkpoint")
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["models"]["n"][field] = value
    manifest.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        models_cli.load_artifacts(manifest)


def test_rejects_duplicate_alias_across_schema_v2_entries(tmp_path):
    manifest = tmp_path / "manifest.yml"
    _write_v2_manifest(manifest, b"checkpoint")
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["models"]["s"] = dict(document["models"]["n"])
    manifest.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate artifact alias"):
        models_cli.load_artifacts(manifest)


def test_aliases_are_globally_unique_across_families():
    aliases = []
    for family in models_cli.FAMILY_MANIFESTS:
        aliases.extend(
            models_cli.load_artifacts(models_cli.default_manifest_path(family))
        )

    assert len(aliases) == len(set(aliases))


def test_rejects_unknown_family_before_manifest_lookup(tmp_path):
    with pytest.raises(SystemExit) as error:
        models_cli.main(
            ["--family", "unknown", "list", "--json", "--manifest", str(tmp_path)]
        )

    assert error.value.code == 2

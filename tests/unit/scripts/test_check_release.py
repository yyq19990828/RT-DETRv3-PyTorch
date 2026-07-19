import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts/check_release.py"


def load_script(monkeypatch):
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_repository_release_metadata_and_manifest_are_valid(monkeypatch):
    script = load_script(monkeypatch)

    summary = script.validate_repository(require_models=False)

    assert summary["manifest_entries"] == 4
    assert summary["distribution_artifacts"] == 4
    assert summary["checked_model_files"] >= 0


def test_release_model_assets_include_weights_and_mapping_reports(monkeypatch):
    script = load_script(monkeypatch)

    assets = [path.name for path in script.release_model_assets()]

    assert assets == [
        "rtdetrv3_r18vd_6x_coco.pth",
        "rtdetrv3_r34vd_6x_coco.pth",
        "rtdetrv3_r50vd_6x_coco.pth",
        "ResNet18_vd_pretrained.pth",
        "rtdetrv3_r18vd_6x_coco.mapping.json",
        "rtdetrv3_r34vd_6x_coco.mapping.json",
        "rtdetrv3_r50vd_6x_coco.mapping.json",
        "ResNet18_vd_pretrained.mapping.json",
    ]


def test_write_sha256sums_is_atomic_and_uses_flat_asset_names(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    output = tmp_path / "SHA256SUMS"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    count = script.write_sha256sums([first, second], output)

    assert count == 2
    assert output.read_text(encoding="utf-8") == (
        f"{hashlib.sha256(b'first').hexdigest()}  first.bin\n"
        f"{hashlib.sha256(b'second').hexdigest()}  second.bin\n"
    )
    assert not list(tmp_path.glob(".SHA256SUMS.*.tmp"))


def test_write_sha256sums_rejects_duplicate_asset_names(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    first = tmp_path / "first" / "model.pth"
    second = tmp_path / "second" / "model.pth"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    with pytest.raises(ValueError, match="basenames must be unique"):
        script.write_sha256sums([first, second], tmp_path / "SHA256SUMS")


def test_checksum_generation_requires_models_and_archives(monkeypatch, tmp_path):
    script = load_script(monkeypatch)

    with pytest.raises(ValueError, match="requires --require-models"):
        script.main(["--write-sha256sums", str(tmp_path / "SHA256SUMS")])


@pytest.mark.parametrize("name", ["../payload", "/absolute/payload"])
def test_archive_validation_rejects_unsafe_paths(monkeypatch, name):
    script = load_script(monkeypatch)

    with pytest.raises(ValueError):
        script._validate_archive_names([name], "fixture")

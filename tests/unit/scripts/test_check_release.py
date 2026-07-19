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


def create_release_directory(script, monkeypatch, tmp_path):
    release_directory = tmp_path / "release"
    release_directory.mkdir()
    contents = {
        "model.pth": b"model",
        "model.mapping.json": b"mapping",
        "rtdetrv3_pytorch-0.1.0-py3-none-any.whl": b"wheel",
        "rtdetrv3_pytorch-0.1.0.tar.gz": b"sdist",
    }
    for filename, content in contents.items():
        (release_directory / filename).write_bytes(content)

    monkeypatch.setattr(
        script,
        "release_model_asset_specs",
        lambda: [
            (
                tmp_path / "manifest" / "model.pth",
                len(contents["model.pth"]),
                hashlib.sha256(contents["model.pth"]).hexdigest(),
            ),
            (
                tmp_path / "manifest" / "model.mapping.json",
                len(contents["model.mapping.json"]),
                hashlib.sha256(contents["model.mapping.json"]).hexdigest(),
            ),
        ],
    )
    monkeypatch.setattr(script, "validate_wheel", lambda path: None)
    monkeypatch.setattr(script, "validate_sdist", lambda path: None)
    script.write_sha256sums(
        [release_directory / filename for filename in contents],
        release_directory / "SHA256SUMS",
    )
    return release_directory


def create_staging_sources(script, monkeypatch, tmp_path):
    source_directory = tmp_path / "sources"
    source_directory.mkdir()
    contents = {
        "model.pth": b"model",
        "model.mapping.json": b"mapping",
        "rtdetrv3_pytorch-0.1.0-py3-none-any.whl": b"wheel",
        "rtdetrv3_pytorch-0.1.0.tar.gz": b"sdist",
    }
    for filename, content in contents.items():
        (source_directory / filename).write_bytes(content)

    monkeypatch.setattr(
        script,
        "release_model_asset_specs",
        lambda: [
            (
                source_directory / "model.pth",
                len(contents["model.pth"]),
                hashlib.sha256(contents["model.pth"]).hexdigest(),
            ),
            (
                source_directory / "model.mapping.json",
                len(contents["model.mapping.json"]),
                hashlib.sha256(contents["model.mapping.json"]).hexdigest(),
            ),
        ],
    )
    monkeypatch.setattr(script, "validate_wheel", lambda path: None)
    monkeypatch.setattr(script, "validate_sdist", lambda path: None)
    return (
        source_directory / "rtdetrv3_pytorch-0.1.0-py3-none-any.whl",
        source_directory / "rtdetrv3_pytorch-0.1.0.tar.gz",
    )


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


def test_release_directory_verifies_exact_inventory_and_checksums(
    monkeypatch, tmp_path
):
    script = load_script(monkeypatch)
    release_directory = create_release_directory(script, monkeypatch, tmp_path)

    summary = script.validate_release_directory(release_directory)

    assert summary == {"release_assets": 5, "checksummed_assets": 4}


def test_release_directory_rejects_tampered_asset(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    release_directory = create_release_directory(script, monkeypatch, tmp_path)
    (release_directory / "model.pth").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="release checksum mismatch: model.pth"):
        script.validate_release_directory(release_directory)


def test_release_directory_rejects_unexpected_asset(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    release_directory = create_release_directory(script, monkeypatch, tmp_path)
    (release_directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="release directory inventory mismatch"):
        script.validate_release_directory(release_directory)


def test_stage_release_directory_creates_verified_inventory(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    wheel, sdist = create_staging_sources(script, monkeypatch, tmp_path)
    destination = tmp_path / "staged"

    summary = script.stage_release_directory(wheel, sdist, destination)

    assert summary == {"release_assets": 5, "checksummed_assets": 4}
    assert {path.name for path in destination.iterdir()} == {
        "model.pth",
        "model.mapping.json",
        wheel.name,
        sdist.name,
        "SHA256SUMS",
    }
    assert not list(tmp_path.glob(".staged.*"))


def test_stage_release_directory_refuses_existing_destination(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    wheel, sdist = create_staging_sources(script, monkeypatch, tmp_path)
    destination = tmp_path / "staged"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="staging destination already exists"):
        script.stage_release_directory(wheel, sdist, destination)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".staged.*"))


def test_stage_release_directory_cleans_partial_output(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    wheel, sdist = create_staging_sources(script, monkeypatch, tmp_path)
    destination = tmp_path / "staged"

    def fail_validation(path):
        raise ValueError("validation failed")

    monkeypatch.setattr(script, "validate_release_directory", fail_validation)

    with pytest.raises(ValueError, match="validation failed"):
        script.stage_release_directory(wheel, sdist, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".staged.*"))


def test_verify_release_directory_cli_must_be_used_alone(monkeypatch, tmp_path):
    script = load_script(monkeypatch)

    with pytest.raises(ValueError, match="must be used on its own"):
        script.main(
            [
                "--verify-release-dir",
                str(tmp_path),
                "--wheel",
                str(tmp_path / "package.whl"),
            ]
        )


def test_release_staging_requires_models_and_archives(monkeypatch, tmp_path):
    script = load_script(monkeypatch)

    with pytest.raises(ValueError, match="requires --require-models"):
        script.main(["--stage-release-dir", str(tmp_path / "staged")])


def test_read_sha256sums_rejects_unsafe_asset_name(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    checksums = tmp_path / "SHA256SUMS"
    checksums.write_text(f"{'0' * 64}  ../payload\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe asset name"):
        script._read_sha256sums(checksums)


@pytest.mark.parametrize("name", ["../payload", "/absolute/payload"])
def test_archive_validation_rejects_unsafe_paths(monkeypatch, name):
    script = load_script(monkeypatch)

    with pytest.raises(ValueError):
        script._validate_archive_names([name], "fixture")

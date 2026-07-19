#!/usr/bin/env python3
"""Validate release metadata, model provenance, and distribution contents."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import tarfile
import tempfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence
from zipfile import ZipFile

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs/checkpoints/rtdetrv3_coco.yml"
EXPECTED_CONSOLE_SCRIPTS = {
    "rtdetrv3-convert",
    "rtdetrv3-eval",
    "rtdetrv3-export",
    "rtdetrv3-infer",
    "rtdetrv3-models",
    "rtdetrv3-train",
}


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate release metadata and optional wheel/sdist artifacts. "
            "Model files are checked when present."
        )
    )
    parser.add_argument("--wheel", type=Path, help="Wheel artifact to inspect")
    parser.add_argument("--sdist", type=Path, help="Source archive to inspect")
    parser.add_argument(
        "--require-models",
        action="store_true",
        help="Fail unless every source, converted model, and mapping report exists",
    )
    parser.add_argument(
        "--write-sha256sums",
        type=Path,
        metavar="PATH",
        help=(
            "Atomically write checksums for converted models, mapping reports, "
            "wheel, and sdist"
        ),
    )
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{label} is not SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value


def _validate_distribution_alias(value: Any, label: str) -> str:
    _require(isinstance(value, str), f"{label} has an invalid distribution alias")
    parts = value.split("-")
    _require(
        all(
            part and part.isascii() and part.isalnum() and part == part.lower()
            for part in parts
        ),
        f"{label} has an invalid distribution alias",
    )
    return value


def _validate_revision(value: Any) -> str:
    _require(isinstance(value, str) and len(value) == 40, "invalid Git revision")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("Git revision is not hexadecimal") from error
    return value


def _positive_int(value: Any, label: str) -> int:
    _require(isinstance(value, int) and value > 0, f"{label} must be positive")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def _repository_path(value: Any, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path")
    path = Path(value)
    _require(not path.is_absolute(), f"{label} must be repository-relative")
    _require(".." not in path.parts, f"{label} cannot escape the repository")
    return REPO_ROOT / path


def _validate_local_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    required: bool,
) -> bool:
    if not path.is_file():
        _require(not required, f"required release file is missing: {path}")
        return False
    _require(path.stat().st_size == expected_size, f"unexpected size: {path}")
    _require(_sha256(path) == expected_sha256, f"checksum mismatch: {path}")
    return True


def _validate_manifest_entry(
    name: str,
    entry: dict[str, Any],
    *,
    config_key: str,
    require_models: bool,
) -> tuple[int, str]:
    _require(entry.get("status") == "verified", f"{name} is not verified")
    config_path = _repository_path(entry.get(config_key), f"{name}.{config_key}")
    _require(config_path.is_file(), f"config is missing: {config_path}")
    source_url = entry.get("source_url")
    _require(
        isinstance(source_url, str) and source_url.startswith("https://"),
        f"{name}.source_url must use HTTPS",
    )

    source_path = _repository_path(entry.get("source_path"), f"{name}.source_path")
    source_size = _positive_int(entry.get("source_size_bytes"), f"{name} source size")
    source_digest = _validate_digest(entry.get("source_sha256"), f"{name} source")
    checked = int(
        _validate_local_file(
            source_path,
            expected_size=source_size,
            expected_sha256=source_digest,
            required=require_models,
        )
    )

    converted = _mapping(entry.get("converted_artifact"), f"{name}.converted_artifact")
    alias = _validate_distribution_alias(converted.get("alias"), name)
    artifact_path = _repository_path(converted.get("path"), f"{name} artifact")
    artifact_size = _positive_int(converted.get("size_bytes"), f"{name} artifact size")
    artifact_digest = _validate_digest(converted.get("sha256"), f"{name} artifact")
    distribution_status = converted.get("distribution_status", "unpublished")
    _require(
        distribution_status in {"unpublished", "published"},
        f"{name} has an invalid distribution status",
    )
    download_url = converted.get("download_url")
    if distribution_status == "published":
        _require(
            isinstance(download_url, str) and download_url.startswith("https://"),
            f"{name} published artifact must have an HTTPS URL",
        )
    else:
        _require(download_url is None, f"{name} unpublished artifact has a URL")
    checked += int(
        _validate_local_file(
            artifact_path,
            expected_size=artifact_size,
            expected_sha256=artifact_digest,
            required=require_models,
        )
    )

    mapping_path = _repository_path(
        converted.get("mapping_report"), f"{name} mapping report"
    )
    mapping_count = _positive_int(
        converted.get("mapping_count"), f"{name} mapping count"
    )
    if mapping_path.is_file():
        mapping = _mapping(
            json.loads(mapping_path.read_text(encoding="utf-8")),
            f"mapping report {mapping_path}",
        )
        _require(
            isinstance(mapping, dict) and isinstance(mapping.get("mappings"), list),
            f"invalid mapping report: {mapping_path}",
        )
        _require(
            len(mapping["mappings"]) == mapping_count,
            f"mapping count mismatch: {mapping_path}",
        )
        checked += 1
    else:
        _require(not require_models, f"mapping report is missing: {mapping_path}")
    return checked, alias


def validate_repository(*, require_models: bool) -> dict[str, int]:
    for filename in ("LICENSE", "NOTICE"):
        _require((REPO_ROOT / filename).is_file(), f"{filename} is missing")

    manifest = _mapping(
        yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")),
        "checkpoint manifest",
    )
    _require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    source_repository = _mapping(manifest.get("source_repository"), "source repository")
    _require(
        source_repository.get("license") == "Apache-2.0",
        "source repository license must be Apache-2.0",
    )
    _validate_revision(source_repository.get("revision"))

    entries: list[tuple[str, dict[str, Any], str]] = []
    pretraining = _mapping(manifest.get("pretraining"), "pretraining manifest")
    models = _mapping(manifest.get("models"), "model manifest")
    _require(bool(models), "model manifest is empty")
    for name, entry in pretraining.items():
        entries.append(
            (name, _mapping(entry, f"{name} manifest entry"), "target_config")
        )
    for name, entry in models.items():
        entries.append((name, _mapping(entry, f"{name} manifest entry"), "config"))

    checked_files = 0
    distribution_aliases: set[str] = set()
    for name, entry, config_key in entries:
        checked, alias = _validate_manifest_entry(
            name,
            entry,
            config_key=config_key,
            require_models=require_models,
        )
        _require(
            alias not in distribution_aliases,
            f"duplicate distribution alias: {alias}",
        )
        distribution_aliases.add(alias)
        checked_files += checked
    return {
        "manifest_entries": len(entries),
        "distribution_artifacts": len(distribution_aliases),
        "checked_model_files": checked_files,
    }


def release_model_assets() -> list[Path]:
    """Return converted weights and audit reports in deterministic release order."""
    manifest = _mapping(
        yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")),
        "checkpoint manifest",
    )
    weights: list[Path] = []
    reports: list[Path] = []
    for section_name in ("models", "pretraining"):
        section = _mapping(manifest.get(section_name), f"{section_name} manifest")
        for name, value in section.items():
            entry = _mapping(value, f"{name} manifest entry")
            converted = _mapping(
                entry.get("converted_artifact"), f"{name}.converted_artifact"
            )
            weights.append(_repository_path(converted.get("path"), f"{name} artifact"))
            reports.append(
                _repository_path(
                    converted.get("mapping_report"), f"{name} mapping report"
                )
            )
    return weights + reports


def write_sha256sums(paths: Sequence[Path], output_path: Path) -> int:
    """Write a flat, GitHub-Release-compatible checksum file atomically."""
    resolved_paths = [path.resolve() for path in paths]
    for path in resolved_paths:
        _require(path.is_file(), f"release asset is missing: {path}")

    names = [path.name for path in resolved_paths]
    _require(len(names) == len(set(names)), "release asset basenames must be unique")

    output_path = output_path.resolve()
    _require(
        output_path not in resolved_paths, "checksum output cannot replace an asset"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for path in resolved_paths:
                output_file.write(f"{_sha256(path)}  {path.name}\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(resolved_paths)


def _validate_archive_names(names: Sequence[str], label: str) -> None:
    for name in names:
        path = PurePosixPath(name)
        _require(not path.is_absolute(), f"{label} contains an absolute path: {name}")
        _require(".." not in path.parts, f"{label} contains path traversal: {name}")


def _expected_packaged_configs() -> set[str]:
    config_root = REPO_ROOT / "configs"
    return {
        f"ppdet_pytorch/configs/{path.relative_to(config_root).as_posix()}"
        for path in config_root.rglob("*")
        if path.is_file()
    }


def validate_wheel(path: Path) -> None:
    _require(path.is_file(), f"wheel is missing: {path}")
    with ZipFile(path) as archive:
        names = archive.namelist()
        _validate_archive_names(names, "wheel")
        name_set = set(names)
        missing_configs = _expected_packaged_configs() - name_set
        _require(not missing_configs, f"wheel configs are missing: {missing_configs}")
        _require(
            not any(
                part in {"third-party", "pretrained_models", "tests"}
                for name in names
                for part in PurePosixPath(name).parts
            ),
            "wheel contains development-only files",
        )
        _require(
            not any(name.endswith((".pdparams", ".pth")) for name in names),
            "wheel must not bundle model checkpoints",
        )

        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        _require(len(metadata_names) == 1, "wheel must contain one METADATA file")
        metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(metadata_names[0])
        )
        _require(metadata.get("Name") == "rtdetrv3-pytorch", "unexpected package name")
        _require(metadata.get("License-Expression") == "Apache-2.0", "license metadata")
        license_files = set(metadata.get_all("License-File", []))
        _require({"LICENSE", "NOTICE"} <= license_files, "license files metadata")
        project_urls = {
            value.split(",", maxsplit=1)[0]
            for value in metadata.get_all("Project-URL", [])
        }
        _require(
            {"Documentation", "Issues", "Repository"} <= project_urls,
            "project URLs are incomplete",
        )

        entry_names = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        _require(len(entry_names) == 1, "wheel entry points are missing")
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(entry_names[0]).decode("utf-8"))
        scripts = set(entry_points["console_scripts"])
        _require(scripts == EXPECTED_CONSOLE_SCRIPTS, "unexpected console scripts")

        legal_names = {PurePosixPath(name).name for name in names}
        _require({"LICENSE", "NOTICE"} <= legal_names, "wheel legal files are missing")


def validate_sdist(path: Path) -> None:
    _require(path.is_file(), f"sdist is missing: {path}")
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        _validate_archive_names(names, "sdist")
        _require(
            not any(member.issym() or member.islnk() for member in members),
            "sdist contains links",
        )
        roots = {PurePosixPath(name).parts[0] for name in names if name}
        _require(len(roots) == 1, "sdist must have one root directory")
        root = next(iter(roots))
        name_set = set(names)
        required = {
            f"{root}/LICENSE",
            f"{root}/NOTICE",
            f"{root}/README.md",
            f"{root}/pyproject.toml",
            f"{root}/configs/checkpoints/rtdetrv3_coco.yml",
        }
        _require(
            required <= name_set, f"sdist files are missing: {required - name_set}"
        )
        _require(
            any(name.startswith(f"{root}/src/ppdet_pytorch/") for name in names),
            "sdist package source is missing",
        )
        forbidden = (f"{root}/third-party/", f"{root}/pretrained_models/")
        _require(
            not any(name.startswith(forbidden) for name in names),
            "sdist contains excluded repository assets",
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.write_sha256sums is not None:
        _require(
            args.require_models,
            "--write-sha256sums requires --require-models",
        )
        _require(args.wheel is not None, "--write-sha256sums requires --wheel")
        _require(args.sdist is not None, "--write-sha256sums requires --sdist")

    summary = validate_repository(require_models=args.require_models)
    if args.wheel is not None:
        validate_wheel(args.wheel.resolve())
    if args.sdist is not None:
        validate_sdist(args.sdist.resolve())
    checksum_count = 0
    if args.write_sha256sums is not None:
        checksum_count = write_sha256sums(
            release_model_assets() + [args.wheel.resolve(), args.sdist.resolve()],
            args.write_sha256sums,
        )
    print(
        "release checks passed: "
        f"{summary['manifest_entries']} manifest entries, "
        f"{summary['distribution_artifacts']} distribution artifacts, "
        f"{summary['checked_model_files']} local model files/reports"
    )
    if args.write_sha256sums is not None:
        print(
            f"wrote SHA256SUMS for {checksum_count} release assets: "
            f"{args.write_sha256sums.resolve()}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(f"release check failed: {error}")
        raise SystemExit(1) from error

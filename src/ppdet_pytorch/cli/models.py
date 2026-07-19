"""List, verify, and download published RT-DETRv3 model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

MANIFEST_RELATIVE_PATH = Path("configs/checkpoints/rtdetrv3_coco.yml")
MODEL_ALIASES = {
    "r18": "rtdetrv3_r18vd_6x_coco",
    "r34": "rtdetrv3_r34vd_6x_coco",
    "r50": "rtdetrv3_r50vd_6x_coco",
}


@dataclass(frozen=True)
class ModelArtifact:
    alias: str
    name: str
    config: str
    path: str
    size_bytes: int
    sha256: str
    distribution_status: str
    download_url: Optional[str]


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List, verify, or download RT-DETRv3 PyTorch model weights."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Checkpoint manifest; defaults to the repository or packaged manifest.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List known model artifacts")
    list_parser.add_argument("--json", action="store_true", help="Emit JSON")

    verify_parser = subparsers.add_parser(
        "verify", help="Verify a local model against the manifest"
    )
    verify_parser.add_argument("model", help="Model alias: r18, r34, or r50")
    verify_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Local file; defaults to the path recorded in the manifest",
    )

    download_parser = subparsers.add_parser(
        "download", help="Download and verify a published model"
    )
    download_parser.add_argument("model", help="Model alias: r18, r34, or r50")
    download_parser.add_argument(
        "--output",
        type=Path,
        help="Destination; defaults to the path recorded in the manifest",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file that does not match the manifest",
    )
    return parser


def default_manifest_path() -> Path:
    package_manifest = Path(__file__).resolve().parents[1] / MANIFEST_RELATIVE_PATH
    if package_manifest.is_file():
        return package_manifest
    repository_manifest = Path(__file__).resolve().parents[3] / MANIFEST_RELATIVE_PATH
    if repository_manifest.is_file():
        return repository_manifest
    raise FileNotFoundError(
        "checkpoint manifest is unavailable; pass --manifest explicitly"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"invalid {label} SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"invalid {label} SHA-256") from error
    return value


def load_artifacts(manifest_path: Path) -> dict[str, ModelArtifact]:
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(document, dict), "checkpoint manifest must be a mapping")
    models = document.get("models")
    _require(isinstance(models, dict), "checkpoint manifest is missing models")

    aliases_by_name = {name: alias for alias, name in MODEL_ALIASES.items()}
    artifacts: dict[str, ModelArtifact] = {}
    for name, entry in models.items():
        _require(isinstance(name, str), "model name must be a string")
        _require(isinstance(entry, dict), f"invalid model entry: {name}")
        alias = aliases_by_name.get(name)
        if alias is None:
            continue
        converted = entry.get("converted_artifact")
        _require(isinstance(converted, dict), f"missing converted artifact: {name}")
        size_bytes = converted.get("size_bytes")
        _require(
            isinstance(size_bytes, int) and size_bytes > 0,
            f"invalid artifact size: {name}",
        )
        status = converted.get("distribution_status", "unpublished")
        _require(
            status in {"unpublished", "published"},
            f"invalid distribution status: {name}",
        )
        download_url = converted.get("download_url")
        _require(
            download_url is None
            or (isinstance(download_url, str) and download_url.startswith("https://")),
            f"download URL must use HTTPS: {name}",
        )
        if status == "published":
            _require(download_url is not None, f"published model has no URL: {name}")
        else:
            _require(download_url is None, f"unpublished model has a URL: {name}")
        artifacts[alias] = ModelArtifact(
            alias=alias,
            name=name,
            config=str(entry["config"]),
            path=str(converted["path"]),
            size_bytes=size_bytes,
            sha256=_validate_sha256(converted.get("sha256"), name),
            distribution_status=status,
            download_url=download_url,
        )

    missing_aliases = set(MODEL_ALIASES) - set(artifacts)
    _require(
        not missing_aliases, f"manifest models are missing: {sorted(missing_aliases)}"
    )
    return artifacts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, artifact: ModelArtifact) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != artifact.size_bytes:
        raise ValueError(
            f"size mismatch for {path}: expected {artifact.size_bytes}, got {actual_size}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != artifact.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {artifact.sha256}, "
            f"got {actual_sha256}"
        )
    return {
        "model": artifact.alias,
        "path": str(path),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "verified": True,
    }


def _open_url(request: urllib.request.Request) -> Any:
    return urllib.request.urlopen(request, timeout=60)  # nosec B310


def download_artifact(
    artifact: ModelArtifact,
    destination: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    if artifact.download_url is None:
        raise ValueError(
            f"{artifact.alias} is not published; no download_url is recorded"
        )
    if destination.exists():
        try:
            return verify_artifact(destination, artifact)
        except ValueError:
            if not force:
                raise FileExistsError(
                    f"existing file does not match the manifest: {destination}; "
                    "use --force to replace it"
                )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            request = urllib.request.Request(
                artifact.download_url,
                headers={"User-Agent": "rtdetrv3-pytorch-model-downloader/0.1"},
            )
            with _open_url(request) as response:
                final_url = response.geturl()
                if not final_url.startswith("https://"):
                    raise ValueError("download redirected to a non-HTTPS URL")
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)
                    if temporary_file.tell() > artifact.size_bytes:
                        raise ValueError("download exceeds the manifest size")
        result = verify_artifact(temporary_path, artifact)
        os.replace(temporary_path, destination)
        temporary_path = None
        result["path"] = str(destination)
        return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _list_artifacts(artifacts: dict[str, ModelArtifact], as_json: bool) -> None:
    ordered = [artifacts[alias] for alias in MODEL_ALIASES]
    if as_json:
        print(json.dumps([asdict(item) for item in ordered], indent=2))
        return
    print("MODEL  STATUS       SIZE (BYTES)  SHA-256")
    for artifact in ordered:
        print(
            f"{artifact.alias:<6} {artifact.distribution_status:<12} "
            f"{artifact.size_bytes:<13} {artifact.sha256}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    try:
        manifest_path = (
            args.manifest.resolve() if args.manifest else default_manifest_path()
        )
        artifacts = load_artifacts(manifest_path)
        if args.command == "list":
            _list_artifacts(artifacts, args.json)
            return 0

        alias = args.model.lower()
        if alias not in artifacts:
            raise ValueError(
                f"unknown model {args.model!r}; choose from {', '.join(artifacts)}"
            )
        artifact = artifacts[alias]
        if args.command == "verify":
            path = args.path or Path(artifact.path)
            result = verify_artifact(path, artifact)
        elif args.command == "download":
            destination = args.output or Path(artifact.path)
            result = download_artifact(artifact, destination, force=args.force)
        else:
            raise ValueError(f"unsupported command: {args.command}")
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"model artifact error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""List, verify, and download model artifacts by family."""

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
from typing import Any, Optional, Sequence, cast

import yaml
from rich import box
from rich.panel import Panel
from rich.table import Table

from detrs.core.workspace import load_config
from detrs.utils.cli import DetrsHelpFormatter
from detrs.utils.console import get_console

DEFAULT_FAMILY = "rtdetrv3"
FAMILY_MANIFESTS = {
    "rtdetrv3": Path("configs/checkpoints/rtdetrv3_coco.yml"),
    "dfine": Path("configs/checkpoints/dfine_coco.yml"),
    "deim-dfine": Path("configs/checkpoints/deim_dfine_coco.yml"),
    "deim-rtdetrv2": Path("configs/checkpoints/deim_rtdetrv2_coco.yml"),
    "rtdetrv4": Path("configs/checkpoints/rtdetrv4_coco.yml"),
    "deimv2": Path("configs/checkpoints/deimv2_coco.yml"),
}
DISTRIBUTION_SECTIONS = (("models", "config"), ("pretraining", "target_config"))


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
    hosting: str = "project"
    artifact_format: str = "pytorch-checkpoint"
    source_url: Optional[str] = None


def _add_manifest_options(
    parser: argparse.ArgumentParser, suppress_defaults: bool = False
) -> None:
    # On nested subcommands the defaults are suppressed so a value parsed on
    # the parent parser (``models --family x list``) survives unless the user
    # overrides it after the subcommand (``models list --family x``).
    parser.add_argument(
        "--family",
        choices=tuple(FAMILY_MANIFESTS),
        default=argparse.SUPPRESS if suppress_defaults else DEFAULT_FAMILY,
        help="Model family; defaults to rtdetrv3.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="Checkpoint manifest; defaults to the repository or packaged manifest.",
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List, verify, or download PyTorch model weights.",
        formatter_class=DetrsHelpFormatter,
    )
    _add_manifest_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="List known model artifacts",
        formatter_class=DetrsHelpFormatter,
    )
    _add_manifest_options(list_parser, suppress_defaults=True)
    list_parser.add_argument("--json", action="store_true", help="Emit JSON")

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify a local model against the manifest",
        formatter_class=DetrsHelpFormatter,
    )
    _add_manifest_options(verify_parser, suppress_defaults=True)
    verify_parser.add_argument(
        "model", help="Artifact alias: r18, r34, r50, or r18-backbone"
    )
    verify_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Local file; defaults to the path recorded in the manifest",
    )
    verify_parser.add_argument("--json", action="store_true", help="Emit JSON")

    download_parser = subparsers.add_parser(
        "download",
        help="Download and verify a published model",
        formatter_class=DetrsHelpFormatter,
    )
    _add_manifest_options(download_parser, suppress_defaults=True)
    download_parser.add_argument(
        "model", help="Artifact alias: r18, r34, r50, or r18-backbone"
    )
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
    download_parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser


def _resolve_repository_file(relative: Path) -> Optional[Path]:
    """Locate a repository-relative file beside the package or the checkout."""
    module_file = Path(__file__).resolve()
    for base in (module_file.parents[1], module_file.parents[3]):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return None


def default_manifest_path(family: str = DEFAULT_FAMILY) -> Path:
    resolved = _resolve_repository_file(FAMILY_MANIFESTS[family])
    if resolved is None:
        raise FileNotFoundError(
            "checkpoint manifest is unavailable; pass --manifest explicitly"
        )
    return resolved


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


def _validate_alias(value: Any, label: str) -> str:
    _require(isinstance(value, str), f"invalid artifact alias: {label}")
    parts = value.split("-")
    _require(
        all(
            part and part.isascii() and part.isalnum() and part == part.lower()
            for part in parts
        ),
        f"invalid artifact alias: {label}",
    )
    return value


def _validate_artifact_path(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"invalid artifact path: {label}")
    path = Path(value)
    _require(
        not path.is_absolute() and ".." not in path.parts,
        f"artifact path must be repository-relative: {label}",
    )
    return value


def _load_schema_v1(document: dict[str, Any]) -> dict[str, ModelArtifact]:
    artifacts: dict[str, ModelArtifact] = {}
    for section_name, config_key in DISTRIBUTION_SECTIONS:
        section = document.get(section_name)
        _require(
            isinstance(section, dict),
            f"checkpoint manifest is missing {section_name}",
        )
        section = cast(dict[str, Any], section)
        for name, entry in section.items():
            _require(isinstance(name, str), "artifact name must be a string")
            _require(isinstance(entry, dict), f"invalid model entry: {name}")
            converted = entry.get("converted_artifact")
            _require(isinstance(converted, dict), f"missing converted artifact: {name}")
            alias = _validate_alias(converted.get("alias"), name)
            _require(alias not in artifacts, f"duplicate artifact alias: {alias}")
            config = entry.get(config_key)
            _require(isinstance(config, str), f"missing config for artifact: {name}")
            artifact_path = _validate_artifact_path(converted.get("path"), name)
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
                or (
                    isinstance(download_url, str)
                    and download_url.startswith("https://")
                ),
                f"download URL must use HTTPS: {name}",
            )
            if status == "published":
                _require(
                    download_url is not None, f"published model has no URL: {name}"
                )
            else:
                _require(download_url is None, f"unpublished model has a URL: {name}")
            status = cast(str, status)
            artifacts[alias] = ModelArtifact(
                alias=alias,
                name=name,
                config=config,
                path=artifact_path,
                size_bytes=size_bytes,
                sha256=_validate_sha256(converted.get("sha256"), name),
                distribution_status=status,
                download_url=download_url,
            )

    return artifacts


def _load_schema_v2(document: dict[str, Any]) -> dict[str, ModelArtifact]:
    family = document.get("family")
    _require(family in FAMILY_MANIFESTS, "invalid schema-v2 model family")
    hosting = document.get("hosting")
    _require(hosting in {"project", "upstream"}, "invalid artifact hosting")
    hosting = cast(str, hosting)
    section = document.get("models")
    _require(
        isinstance(section, dict) and bool(section),
        "checkpoint manifest is missing models",
    )
    section = cast(dict[str, Any], section)
    artifacts: dict[str, ModelArtifact] = {}
    for name, entry in section.items():
        _require(isinstance(name, str), "artifact name must be a string")
        _require(isinstance(entry, dict), f"invalid model entry: {name}")
        alias = _validate_alias(entry.get("alias"), name)
        _require(alias not in artifacts, f"duplicate artifact alias: {alias}")
        config = entry.get("config")
        _require(isinstance(config, str), f"missing config for artifact: {name}")
        artifact_path = _validate_artifact_path(entry.get("path"), name)
        size_bytes = entry.get("source_size_bytes")
        _require(
            isinstance(size_bytes, int) and size_bytes > 0,
            f"invalid artifact size: {name}",
        )
        artifact_format = entry.get("artifact_format")
        _require(
            artifact_format == "pytorch-checkpoint",
            f"invalid artifact format: {name}",
        )
        source_url = entry.get("source_url")
        _require(
            isinstance(source_url, str) and source_url.startswith("https://"),
            f"source URL must use HTTPS: {name}",
        )
        download_url = entry.get("download_url")
        _require(
            download_url is None
            or (isinstance(download_url, str) and download_url.startswith("https://")),
            f"download URL must use HTTPS: {name}",
        )
        artifacts[alias] = ModelArtifact(
            alias=alias,
            name=name,
            config=config,
            path=artifact_path,
            size_bytes=size_bytes,
            sha256=_validate_sha256(entry.get("source_sha256"), name),
            distribution_status=hosting,
            download_url=download_url,
            hosting=hosting,
            artifact_format=artifact_format,
            source_url=source_url,
        )
    return artifacts


def load_artifacts(manifest_path: Path) -> dict[str, ModelArtifact]:
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(document, dict), "checkpoint manifest must be a mapping")
    schema_version = document.get("schema_version", 1)
    if schema_version == 1:
        return _load_schema_v1(document)
    if schema_version == 2:
        return _load_schema_v2(document)
    raise ValueError(f"unsupported checkpoint manifest schema: {schema_version}")


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
        if artifact.hosting == "upstream" and artifact.source_url:
            raise ValueError(
                f"{artifact.alias} is hosted upstream; automatic download is "
                f"unavailable; official source: {artifact.source_url}"
            )
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
                headers={
                    "Accept": "application/octet-stream",
                    "User-Agent": "detrs-model-downloader/0.1",
                },
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
    ordered = list(artifacts.values())
    if as_json:
        print(json.dumps([asdict(item) for item in ordered], indent=2))
        return
    table = Table(box=box.SIMPLE)
    table.add_column("MODEL", style="bold")
    table.add_column("STATUS")
    table.add_column("SIZE (BYTES)", justify="right")
    table.add_column("SHA-256", style="dim")
    for artifact in ordered:
        status = artifact.distribution_status
        style = "green" if status == "published" else "yellow"
        table.add_row(
            artifact.alias,
            f"[{style}]{status}[/]",
            str(artifact.size_bytes),
            artifact.sha256,
        )
    get_console().print(table)


def _find_first_config_value(node: Any, key: str) -> Any:
    """Depth-first search for ``key`` anywhere in a merged config tree."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find_first_config_value(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_first_config_value(item, key)
            if found is not None:
                return found
    return None


def _count_parameters(path: Path) -> Optional[int]:
    """Best-effort parameter count; the file was hash-verified beforehand."""
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict: Any = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if isinstance(checkpoint.get(key), dict):
                    state_dict = checkpoint[key]
                    break
        return sum(
            tensor.numel()
            for tensor in state_dict.values()
            if isinstance(tensor, torch.Tensor)
        )
    except Exception:
        # Not a loadable torch checkpoint (or an unexpected layout); the
        # verification result itself stays authoritative.
        return None


def _artifact_details(artifact: ModelArtifact, path: Path) -> dict[str, Any]:
    """Best-effort display metadata; never affects verification results."""
    details: dict[str, Any] = {
        "name": artifact.name,
        "config": artifact.config,
        "hosting": artifact.hosting,
        "input_shape": None,
        "num_queries": None,
        "num_classes": None,
        "params": None,
    }
    config_path = _resolve_repository_file(Path(artifact.config))
    if config_path is not None:
        try:
            config = load_config(str(config_path))
            spatial = _find_first_config_value(config, "eval_spatial_size")
            if isinstance(spatial, (list, tuple)) and len(spatial) == 2:
                details["input_shape"] = [3, int(spatial[0]), int(spatial[1])]
            queries = _find_first_config_value(config, "num_queries")
            if isinstance(queries, int) and not isinstance(queries, bool):
                details["num_queries"] = queries
            classes = _find_first_config_value(config, "num_classes")
            if isinstance(classes, int) and not isinstance(classes, bool):
                details["num_classes"] = classes
        except Exception:
            # A broken or exotic config must not fail an already verified file.
            pass
    details["params"] = _count_parameters(path)
    return details


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.2f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    if num_bytes >= 1_000:
        return f"{num_bytes / 1_000:.1f} KB"
    return f"{num_bytes} B"


def _format_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f} B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f} M"
    if count >= 1_000:
        return f"{count / 1_000:.1f} K"
    return str(count)


def _render_verification(
    result: dict[str, Any], details: dict[str, Any], action: str
) -> None:
    rows = Table(box=None, show_header=False, pad_edge=False)
    rows.add_column(style="bold", no_wrap=True)
    rows.add_column(overflow="fold")
    rows.add_row("name", str(details["name"]))
    rows.add_row("config", str(details["config"]))
    rows.add_row("hosting", str(details["hosting"]))
    rows.add_row("file", str(result["path"]))
    size_bytes = int(result["size_bytes"])
    rows.add_row("size", f"{size_bytes:,} bytes ({_format_bytes(size_bytes)})")
    rows.add_row("sha256", str(result["sha256"]))
    input_shape = details["input_shape"]
    if input_shape is not None:
        rows.add_row("input", " × ".join(str(d) for d in input_shape) + " (C × H × W)")
    queries = details["num_queries"]
    classes = details["num_classes"]
    if queries is not None and classes is not None:
        rows.add_row(
            "output",
            f"{queries} × {classes + 4} ({queries} queries, {classes} classes + 4 box)",
        )
    params = details["params"]
    if params is not None:
        rows.add_row("params", _format_count(params))
    panel = Panel(
        rows,
        title=f"[green]✔[/] {result['model']} · {action}",
        title_align="left",
        box=box.ROUNDED,
        expand=False,
    )
    get_console().print(panel)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    try:
        manifest_path = (
            args.manifest.resolve()
            if args.manifest
            else default_manifest_path(args.family)
        )
        artifacts = load_artifacts(manifest_path)
        if args.command == "list":
            _list_artifacts(artifacts, args.json)
            return 0

        alias = args.model.lower()
        if alias not in artifacts:
            raise ValueError(
                f"unknown artifact {args.model!r}; choose from {', '.join(artifacts)}"
            )
        artifact = artifacts[alias]
        if args.command == "verify":
            path = args.path or Path(artifact.path)
            result = verify_artifact(path, artifact)
            action = "verified"
        elif args.command == "download":
            destination = args.output or Path(artifact.path)
            result = download_artifact(artifact, destination, force=args.force)
            action = "downloaded"
        else:
            raise ValueError(f"unsupported command: {args.command}")
        details = _artifact_details(artifact, Path(result["path"]))
        if args.json:
            print(json.dumps({**result, **details}, indent=2))
        else:
            _render_verification(result, details, action)
        return 0
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"model artifact error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

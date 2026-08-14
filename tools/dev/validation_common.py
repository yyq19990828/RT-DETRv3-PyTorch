"""Shared deterministic contracts for model-family validation drivers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

APPROVE = "APPROVE"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
EXIT_CODES = {APPROVE: 0, FAIL: 1, BLOCKED: 2}
SCHEMA_VERSION = 1
DEFAULT_PLAN = Path(".omo/plans/rtdetrv4-merge.md")

_TASK_MARKER = re.compile(rb"(?m)^- \[[ xX]\] ((?:[1-9][0-9]*|F[1-9][0-9]*)\.)")


class DriverArgumentParser(argparse.ArgumentParser):
    """Keep exit code 2 reserved for externally blocked validation."""

    def error(self, message: str) -> None:
        self.print_usage()
        raise ValueError(message)


def normalized_plan_identity(path: Path) -> str:
    """Hash a plan while ignoring only column-zero task checkbox state."""
    raw = path.read_bytes()
    normalized = _TASK_MARKER.sub(rb"- [ ] \1", raw)
    return hashlib.sha256(normalized).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid {} SHA-256: {}".format(label, value))
    return value


def parse_csv(value: str, label: str) -> list[str]:
    values = value.split(",")
    if not values or any(not item or item.strip() != item for item in values):
        raise ValueError(
            "{} must be a comma-separated list without empty items".format(label)
        )
    if len(set(values)) != len(values):
        raise ValueError("{} contains duplicate items".format(label))
    return values


def atomic_write_text(path: Path, content: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".{}-".format(path.name), suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        temporary_path.unlink(missing_ok=True)


def deterministic_json(document: dict[str, Any]) -> str:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def publish_json(path: Path, document: dict[str, Any]) -> None:
    atomic_write_text(path, deterministic_json(document))


def result_document(
    plan_identity: str,
    *,
    family_results: Optional[list[dict[str, Any]]] = None,
    negatives: Optional[list[dict[str, Any]]] = None,
    status: str,
) -> dict[str, Any]:
    return {
        "family_results": family_results or [],
        "negatives": negatives or [],
        "plan_identity": plan_identity,
        "schema_version": SCHEMA_VERSION,
        "status": status,
    }


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def run_main(main: Any, argv: Optional[Sequence[str]] = None) -> int:
    """Convert argument/preflight errors into the shared FAIL exit code."""
    try:
        return int(main(argv))
    except (FileNotFoundError, OSError, ValueError) as error:
        print("FAIL: {}".format(error))
        return EXIT_CODES[FAIL]

#!/usr/bin/env python3
"""Run non-Paddle tests and enforce documented coverage regression floors."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_PACKAGE_MINIMUM = 50.0
DIRECT_MAINTAINED_MINIMUM = 85.0
DIRECT_MAINTAINED_PREFIXES = (
    Path("src/ppdet_pytorch/cli"),
    Path("src/ppdet_pytorch/conversion"),
    Path("src/ppdet_pytorch/core"),
    Path("src/ppdet_pytorch/deploy"),
)


@dataclass(frozen=True)
class CoverageSummary:
    statements: int
    covered: int

    @property
    def percent(self) -> float:
        if self.statements == 0:
            return 100.0
        return self.covered / self.statements * 100.0


def resolve_pytest() -> str:
    """Resolve pytest from the active virtual environment or PATH."""
    executable_directory = Path(sys.executable).parent
    for executable_name in ("pytest", "pytest.exe"):
        candidate = executable_directory / executable_name
        if candidate.is_file():
            return str(candidate)
    resolved = shutil.which("pytest")
    if resolved is None:
        raise FileNotFoundError(
            "pytest is unavailable; run `uv sync --extra test` or "
            "`uv sync --extra dev`."
        )
    return resolved


def build_command(report_path: Path) -> list[str]:
    return [
        resolve_pytest(),
        "-p",
        "no:cacheprovider",
        "-q",
        "-m",
        "not paddle",
        "--cov=ppdet_pytorch",
        "--cov-report=term",
        f"--cov-report=json:{report_path}",
    ]


def _is_directly_maintained(file_name: str) -> bool:
    path = Path(file_name)
    if path.is_absolute():
        try:
            path = path.relative_to(REPO_ROOT)
        except ValueError:
            return False
    return any(
        prefix == path or prefix in path.parents
        for prefix in DIRECT_MAINTAINED_PREFIXES
    )


def _file_counts(record: object) -> tuple[int, int]:
    if not isinstance(record, dict):
        raise ValueError("coverage file entry must be an object")
    summary = record.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("coverage file entry is missing its summary")
    statements = summary.get("num_statements")
    covered = summary.get("covered_lines")
    if not isinstance(statements, int) or not isinstance(covered, int):
        raise ValueError("coverage summary has invalid statement counts")
    return statements, covered


def summarize_files(
    files: Mapping[str, object],
) -> tuple[CoverageSummary, CoverageSummary]:
    full_statements = 0
    full_covered = 0
    direct_statements = 0
    direct_covered = 0

    for file_name, record in files.items():
        statements, covered = _file_counts(record)
        full_statements += statements
        full_covered += covered
        if _is_directly_maintained(file_name):
            direct_statements += statements
            direct_covered += covered

    if full_statements == 0 or direct_statements == 0:
        raise ValueError("coverage report does not contain the expected package scopes")

    return (
        CoverageSummary(full_statements, full_covered),
        CoverageSummary(direct_statements, direct_covered),
    )


def load_summaries(report_path: Path) -> tuple[CoverageSummary, CoverageSummary]:
    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict):
        raise ValueError("coverage report must be a JSON object")
    files = report.get("files")
    if not isinstance(files, dict):
        raise ValueError("coverage report is missing its files table")
    return summarize_files(files)


def threshold_failures(full: CoverageSummary, direct: CoverageSummary) -> list[str]:
    failures = []
    if full.percent < FULL_PACKAGE_MINIMUM:
        failures.append(
            f"full package {full.percent:.2f}% is below {FULL_PACKAGE_MINIMUM:.2f}%"
        )
    if direct.percent < DIRECT_MAINTAINED_MINIMUM:
        failures.append(
            "direct-maintained scope "
            f"{direct.percent:.2f}% is below {DIRECT_MAINTAINED_MINIMUM:.2f}%"
        )
    return failures


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rtdetrv3-coverage-") as temp_directory:
        temp_path = Path(temp_directory)
        report_path = temp_path / "coverage.json"
        command = build_command(report_path)
        environment = os.environ.copy()
        environment["COVERAGE_FILE"] = str(temp_path / ".coverage")

        print(f"+ {shlex.join(command)}", flush=True)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode

        try:
            full, direct = load_summaries(report_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"coverage report error: {error}", file=sys.stderr)
            return 2

    print(
        f"full package: {full.covered}/{full.statements} statements "
        f"({full.percent:.2f}%, minimum {FULL_PACKAGE_MINIMUM:.2f}%)"
    )
    print(
        f"direct-maintained scope: {direct.covered}/{direct.statements} statements "
        f"({direct.percent:.2f}%, minimum {DIRECT_MAINTAINED_MINIMUM:.2f}%)"
    )

    failures = threshold_failures(full, direct)
    for failure in failures:
        print(f"coverage threshold failed: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

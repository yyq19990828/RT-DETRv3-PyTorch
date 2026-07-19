#!/usr/bin/env python3
"""Run repository-wide Ruff and incremental Mypy quality checks."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUFF_TARGETS = (".",)
MYPY_TARGETS = (
    "scripts/check_coverage.py",
    "scripts/check_quality.py",
    "scripts/check_release.py",
    "scripts/render_prediction_comparison.py",
    "scripts/run_framework_benchmark.py",
    "scripts/run_stability_experiment.py",
    "src/ppdet_pytorch",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repository-wide Ruff and incremental Mypy quality checks."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Format files and apply Ruff's safe lint fixes before type checking.",
    )
    return parser.parse_args(argv)


def resolve_tool(name: str) -> str:
    """Resolve a tool from the active virtual environment or PATH."""
    executable_directory = Path(sys.executable).parent
    for executable_name in (name, f"{name}.exe"):
        candidate = executable_directory / executable_name
        if candidate.is_file():
            return str(candidate)
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(
            f"{name} is unavailable; run `uv sync --extra quality` or "
            "`uv sync --extra dev`."
        )
    return resolved


def build_commands(fix: bool) -> list[list[str]]:
    ruff = resolve_tool("ruff")
    mypy = resolve_tool("mypy")
    if fix:
        return [
            [ruff, "format", *RUFF_TARGETS],
            [ruff, "check", "--fix", *RUFF_TARGETS],
            [mypy, *MYPY_TARGETS],
        ]
    return [
        [ruff, "format", "--check", *RUFF_TARGETS],
        [ruff, "check", *RUFF_TARGETS],
        [mypy, *MYPY_TARGETS],
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    for command in build_commands(args.fix):
        print(f"+ {shlex.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

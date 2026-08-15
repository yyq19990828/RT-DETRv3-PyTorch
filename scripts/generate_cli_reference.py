#!/usr/bin/env python3
"""Generate the CLI reference page from live ``detrs`` help output.

Writes ``docs/guides/cli-reference.md`` by rendering the top-level and every
subcommand ``--help`` through the real dispatch parsers, so the page can
never drift from the actual argument definitions. Width is pinned via
``COLUMNS`` to keep output deterministic across environments.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "guides" / "cli-reference.md"
COLUMNS = "100"

HEADER = """\
# CLI 参考

本页由 [`scripts/generate_cli_reference.py`](../../scripts/generate_cli_reference.py)
从 `detrs` 各命令的 `--help` 输出自动生成,**请勿手改**。参数变更后重新运行
`uv run python scripts/generate_cli_reference.py` 并提交结果;CI 会对生成结果做
一致性检查。
"""


def _help_text(parser, argv: list[str]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            parser.parse_args(argv)
        except SystemExit:
            pass
    return buffer.getvalue().rstrip("\n")


def generate_reference() -> str:
    from detrs.cli.main import COMMANDS, create_argument_parser

    os.environ["COLUMNS"] = COLUMNS
    sections = [HEADER]
    sections.append("## detrs\n")
    sections.append("```text")
    sections.append(_help_text(create_argument_parser(), ["--help"]))
    sections.append("```\n")
    for command in COMMANDS:
        parser = create_argument_parser(command)
        sections.append(f"## detrs {command}\n")
        sections.append("```text")
        sections.append(_help_text(parser, [command, "--help"]))
        sections.append("```\n")
    return "\n".join(sections)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the generated page is up to date instead of writing it",
    )
    args = parser.parse_args(argv)

    content = generate_reference() + "\n"
    if args.check:
        current = (
            OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        )
        if current != content:
            print("cli-reference.md is out of date; rerun generate_cli_reference.py")
            return 1
        print("cli-reference.md is up to date")
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

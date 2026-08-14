#!/usr/bin/env python3
"""Audit task/final evidence against the byte-sensitive execution plan."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from validation_common import (
    APPROVE,
    DEFAULT_PLAN,
    EXIT_CODES,
    DriverArgumentParser,
    atomic_write_text,
    normalized_plan_identity,
    parse_csv,
    result_document,
    run_main,
)


def _parse_tasks(value: str) -> list[str]:
    tasks = set()
    for part in parse_csv(value, "require-tasks"):
        match = re.fullmatch(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?", part)
        if not match:
            raise ValueError("invalid task range: {}".format(part))
        start = int(match.group(1))
        stop = int(match.group(2) or start)
        if stop < start:
            raise ValueError("task range is descending: {}".format(part))
        tasks.update(str(number) for number in range(start, stop + 1))
    return sorted(tasks, key=int)


def _load_receipt(path: Path, identity: str) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = ("schema_version", "plan_identity", "command", "git_revision", "status")
    missing = [key for key in required if key not in receipt]
    if missing:
        raise ValueError(
            "{} missing receipt fields: {}".format(path.name, ",".join(missing))
        )
    if receipt["plan_identity"] != identity:
        raise ValueError("{} has stale plan identity".format(path.name))
    if receipt["status"] != APPROVE:
        raise ValueError("{} is not APPROVE".format(path.name))
    if not isinstance(receipt["command"], list) or not receipt["command"]:
        raise ValueError("{} has no executable command".format(path.name))
    if not re.fullmatch(r"[0-9a-f]{40}", receipt["git_revision"]):
        raise ValueError("{} has invalid git revision".format(path.name))
    return receipt


def create_parser() -> argparse.ArgumentParser:
    parser = DriverArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--require-tasks", required=True)
    parser.add_argument("--require-finals")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    identity = normalized_plan_identity(args.plan)
    tasks = _parse_tasks(args.require_tasks)
    finals = (
        parse_csv(args.require_finals, "require-finals") if args.require_finals else []
    )
    if any(not re.fullmatch(r"F[1-9][0-9]*", item) for item in finals):
        raise ValueError("final IDs must use the F<number> form")

    requested = [("task", item) for item in tasks] + [
        ("final", item) for item in finals
    ]
    audited = []
    for kind, evidence_id in requested:
        if kind == "task":
            matches = list(
                args.attempt_dir.glob("task-{}-rtdetrv4-merge.json".format(evidence_id))
            )
        else:
            matches = list(args.attempt_dir.glob("final-{}-*.json".format(evidence_id)))
        if len(matches) != 1:
            raise ValueError(
                "{} {} requires exactly one JSON receipt, found {}".format(
                    kind, evidence_id, len(matches)
                )
            )
        receipt = _load_receipt(matches[0], identity)
        audited.append(
            {
                "evidence_id": evidence_id,
                "evidence_kind": kind,
                "git_revision": receipt["git_revision"],
                "path": matches[0].name,
                "plan_identity_match": True,
                "status": APPROVE,
            }
        )

    document = result_document(identity, family_results=audited, status=APPROVE)
    lines = [
        "# Plan Evidence Audit",
        "",
        "Plan identity: `{}`".format(identity),
        "",
        "| Evidence | Kind | Revision | Status |",
        "|---|---|---|---|",
    ]
    lines.extend(
        "| {} | {} | `{}` | {} |".format(
            item["evidence_id"],
            item["evidence_kind"],
            item["git_revision"],
            item["status"],
        )
        for item in audited
    )
    lines.extend(
        ["", "```json", json.dumps(document, sort_keys=True), "```", "", APPROVE, ""]
    )
    atomic_write_text(args.output, "\n".join(lines))
    print(APPROVE)
    return EXIT_CODES[APPROVE]


if __name__ == "__main__":
    raise SystemExit(run_main(main))

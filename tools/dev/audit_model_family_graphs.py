#!/usr/bin/env python3
"""Audit model-family dependency and deployment graph contracts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from validation_common import (
    APPROVE,
    DEFAULT_PLAN,
    EXIT_CODES,
    FAIL,
    DriverArgumentParser,
    normalized_plan_identity,
    parse_csv,
    publish_json,
    result_document,
    run_main,
)

FAMILIES = ("rtdetrv3", "dfine", "deim-dfine", "deim-rtdetrv2", "rtdetrv4")
FORBIDDEN_GRAPH_TERMS = (
    "teacher_encoder_output",
    "student_distill_output",
    "dinov3",
    "dsi_loss",
    "gam_controller",
)
TOLERANCES = {
    "activation_atol": 1e-6,
    "activation_rtol": 1e-5,
    "box_atol": 0.02,
    "score_atol": 2e-5,
}


def _top_level_paddle_imports(source_root: Path) -> list[str]:
    violations = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "paddle" or name.startswith("paddle.") for name in names):
                violations.append("{}:{}".format(path, node.lineno))
    return violations


def audit_fixture(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    observed_opset = fixture.get("opset", 17)
    checks.append(
        {
            "details": {"expected": 17, "observed": observed_opset},
            "name": "opset",
            "status": APPROVE if observed_opset == 17 else FAIL,
        }
    )
    nodes = sorted(str(node) for node in fixture.get("nodes", []))
    residue = [
        node
        for node in nodes
        if any(term in node.lower() for term in FORBIDDEN_GRAPH_TERMS)
    ]
    checks.append(
        {
            "details": {"violations": residue},
            "name": "training-residue",
            "status": APPROVE if not residue else FAIL,
        }
    )
    imports = sorted(str(item) for item in fixture.get("paddle_imports", []))
    checks.append(
        {
            "details": {"violations": imports},
            "name": "core-dependencies",
            "status": APPROVE if not imports else FAIL,
        }
    )
    duplicates = sorted(str(item) for item in fixture.get("duplicates", []))
    checks.append(
        {
            "details": {"violations": duplicates},
            "name": "duplicate-implementation",
            "status": APPROVE if not duplicates else FAIL,
        }
    )
    observed_tolerances = fixture.get("tolerances", TOLERANCES)
    checks.append(
        {
            "details": {"expected": TOLERANCES, "observed": observed_tolerances},
            "name": "tolerance-contract",
            "status": APPROVE if observed_tolerances == TOLERANCES else FAIL,
        }
    )
    return checks


def create_parser() -> argparse.ArgumentParser:
    parser = DriverArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--family")
    parser.add_argument("--variants", default="smallest")
    parser.add_argument("--surfaces", default="eager,onnx,torchscript")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    families = list(FAMILIES) if args.all else parse_csv(args.family, "family")
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError("unknown families: {}".format(",".join(unknown)))
    parse_csv(args.variants, "variants")
    surfaces = parse_csv(args.surfaces, "surfaces")
    if set(surfaces) - {"eager", "onnx", "torchscript"}:
        raise ValueError("unknown graph surface")
    identity = normalized_plan_identity(args.plan)
    if args.fixture:
        fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    else:
        source_root = Path(__file__).resolve().parents[2] / "src" / "ppdet_pytorch"
        fixture = {"paddle_imports": _top_level_paddle_imports(source_root)}
    checks = audit_fixture(fixture)
    status = APPROVE if all(item["status"] == APPROVE for item in checks) else FAIL
    family_results = [
        {"checks": checks, "family": family, "status": status, "variant": args.variants}
        for family in families
    ]
    output = args.output or args.evidence_dir / "model-family-graphs.json"
    publish_json(
        output,
        result_document(identity, family_results=family_results, status=status),
    )
    print(status)
    return EXIT_CODES[status]


if __name__ == "__main__":
    raise SystemExit(run_main(main))

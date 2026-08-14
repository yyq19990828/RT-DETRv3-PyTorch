#!/usr/bin/env python3
"""Validate current model documentation against manifests and evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "rtdetrv3": ("r18", "r34", "r50", "r18-backbone"),
    "dfine": tuple(f"dfine-{variant}" for variant in "nsmlx"),
    "deim-dfine": tuple(f"deim-dfine-{variant}" for variant in "nsmlx"),
    "deim-rtdetrv2": (
        "deim-rtv2-s",
        "deim-rtv2-m",
        "deim-rtv2-m-star",
        "deim-rtv2-l",
        "deim-rtv2-x",
    ),
    "rtdetrv4": tuple(f"rtdetrv4-{variant}" for variant in "smlx"),
}
MANIFESTS = {
    "rtdetrv3": "rtdetrv3_coco.yml",
    "dfine": "dfine_coco.yml",
    "deim-dfine": "deim_dfine_coco.yml",
    "deim-rtdetrv2": "deim_rtdetrv2_coco.yml",
    "rtdetrv4": "rtdetrv4_coco.yml",
}
MODEL_REPORT_FILES = ("validation-report.md", "metrics.md", "evidence-index.md")
ATTRIBUTIONS = {
    "https://github.com/Peterande/D-FINE": (
        "267a6da6d04c8ad52e54120692896515b9e55981",
        "Apache-2.0",
    ),
    "https://github.com/Intellindust-AI-Lab/DEIM": (
        "09d35d53d39ee3145a1e61e3a989b28b9468d1dd",
        "Apache-2.0",
    ),
    "https://github.com/RT-DETRs/RT-DETRv4": (
        "55fefaaed7efe2a5f72d0a18fd4e05965e35c292",
        "Apache-2.0",
    ),
    "https://github.com/facebookresearch/dinov3": (
        "346f38fee679c56a6888f91c51670fae61d364e0",
        "DINOv3 License",
    ),
}
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:^|[\s`(])(?:/home/[^\s`)]+|/Users/[^\s`)]+|[A-Za-z]:\\[^\s`)]+)",
    re.MULTILINE,
)
INTERNAL_WORKFLOW_PATTERN = re.compile(r"\b(?:task|todo|F[1-4])\b", re.IGNORECASE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_expected_variants(actual: Sequence[str], family: str) -> None:
    expected = FAMILIES[family]
    _require(tuple(actual) == expected, f"stale variant matrix for {family}")


def require_model_report_layout(models_root: Path) -> None:
    for family in FAMILIES:
        family_root = models_root / family
        for filename in MODEL_REPORT_FILES:
            _require(
                (family_root / filename).is_file(),
                f"missing model report: {family}/{filename}",
            )


def reject_internal_workflow_terms(documents: Mapping[str, str]) -> None:
    for path, content in documents.items():
        _require(
            INTERNAL_WORKFLOW_PATTERN.search(content) is None,
            f"internal workflow term in model documentation: {path}",
        )


def require_attributions(text: str) -> None:
    for url, fields in ATTRIBUTIONS.items():
        _require(url in text, f"missing attribution URL: {url}")
        for field in fields:
            _require(field in text, f"missing attribution field for {url}: {field}")
    for phrase in ("gated", "not included", "not redistributed"):
        _require(phrase in text.lower(), f"missing DINOv3 boundary: {phrase}")


def reject_absolute_paths(documents: Mapping[str, str]) -> None:
    for name, text in documents.items():
        match = ABSOLUTE_PATH_PATTERN.search(text)
        _require(
            match is None,
            f"workstation absolute path in {name}: {match.group(0) if match else ''}",
        )


def validate_teacher_graph_claim(
    documentation: str, evidence: Mapping[str, Any]
) -> None:
    _require("student-only" in documentation, "RT-DETRv4 student-only claim is missing")
    exports = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "training_residue" in value:
                exports.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(evidence)
    _require(len(exports) == 4, "RT-DETRv4 graph evidence must cover four variants")
    _require(
        all(item.get("training_residue") is False for item in exports),
        "teacher graph contradiction: training residue is present",
    )
    _require(
        all(item.get("opset") == 17 for item in exports),
        "teacher graph contradiction: export is not opset 17",
    )


def _manifest_aliases(document: Mapping[str, Any]) -> tuple[str, ...]:
    if document.get("schema_version") == 1:
        aliases = []
        for section_name in ("models", "pretraining"):
            for entry in document[section_name].values():
                aliases.append(entry["converted_artifact"]["alias"])
        return tuple(aliases)
    return tuple(entry["alias"] for entry in document["models"].values())


def _current_markdown_files(root: Path, plan: Path) -> list[Path]:
    files = [root / "README.md", root / "ROADMAP.md", plan]
    for directory in (root / "docs/models", root / "docs/migrations"):
        files.extend(directory.rglob("*.md"))
    files.extend((root / "docs/README.md", root / "docs/plans/README.md"))
    return sorted(set(path.resolve() for path in files))


def _validate_links(paths: Sequence[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(text):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            _require(resolved.exists(), f"broken relative link in {path}: {target}")


def validate_repository(plan: Path, evidence_dir: Path) -> dict[str, int]:
    plan = plan.resolve()
    evidence_dir = evidence_dir.resolve()
    aliases = []
    config_count = 0
    require_model_report_layout(REPO_ROOT / "docs/models")
    for family, filename in MANIFESTS.items():
        path = REPO_ROOT / "configs/checkpoints" / filename
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        family_aliases = _manifest_aliases(document)
        require_expected_variants(family_aliases, family)
        aliases.extend(family_aliases)
        for entry in document["models"].values():
            _require(
                (REPO_ROOT / entry["config"]).is_file(),
                f"missing config: {entry['config']}",
            )
            if family != "rtdetrv3":
                config_count += 1
    _require(len(aliases) == len(set(aliases)), "model aliases are not globally unique")
    _require(config_count == 19, "current support matrix must contain 19 new variants")

    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    require_attributions(notice)
    markdown = _current_markdown_files(REPO_ROOT, plan)
    _validate_links(markdown)
    documents = {
        str(path.relative_to(REPO_ROOT)): path.read_text(encoding="utf-8")
        for path in markdown
    }
    documents["NOTICE"] = notice
    reject_absolute_paths(documents)
    reject_internal_workflow_terms(
        {
            path: content
            for path, content in documents.items()
            if path.startswith("docs/models/")
        }
    )

    graph_evidence = json.loads(
        (evidence_dir / "task-20-rtdetrv4-merge.json").read_text(encoding="utf-8")
    )
    validate_teacher_graph_claim(
        (REPO_ROOT / "docs/models/rtdetrv4/README.md").read_text(encoding="utf-8"),
        graph_evidence,
    )
    _require("- [x] 23." in plan.read_text(encoding="utf-8"), "Task 23 is not checked")
    return {
        "families": len(FAMILIES),
        "artifacts": len(aliases),
        "new_variants": config_count,
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    summary = validate_repository(args.plan, args.evidence_dir)
    print(
        "documentation checks passed: "
        f"{summary['families']} families, {summary['artifacts']} artifacts, "
        f"{summary['new_variants']} new variants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

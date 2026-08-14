#!/usr/bin/env python3
"""Validate repository-owned documentation, indexes, and model contracts."""

from __future__ import annotations

import argparse
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
    "deimv2": (
        "deimv2-x",
        "deimv2-l",
        "deimv2-m",
        "deimv2-s",
        "deimv2-n",
        "deimv2-pico",
        "deimv2-femto",
        "deimv2-atto",
    ),
}
MANIFESTS = {
    "rtdetrv3": "rtdetrv3_coco.yml",
    "dfine": "dfine_coco.yml",
    "deim-dfine": "deim_dfine_coco.yml",
    "deim-rtdetrv2": "deim_rtdetrv2_coco.yml",
    "rtdetrv4": "rtdetrv4_coco.yml",
    "deimv2": "deimv2_coco.yml",
}
MODEL_REPORT_FILES = ("validation-report.md", "metrics.md", "evidence-index.md")
MODEL_DOCUMENTATION_FAMILIES = ("rtdetrv3", "dfine", "deim", "rtdetrv4", "deimv2")
LEGACY_MODEL_DOCUMENTATION_DIRECTORIES = ("deim-dfine", "deim-rtdetrv2")
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
    "https://github.com/Intellindust-AI-Lab/DEIMv2": (
        "add5bcdb499bf7b8a366bfeac1a47d3dc278de27",
        "Apache-2.0",
    ),
    "https://github.com/facebookresearch/dinov3": (
        "346f38fee679c56a6888f91c51670fae61d364e0",
        "DINOv3 License",
    ),
}
LINK_PATTERN = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
FENCE_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
PLAN_STATUS_PATTERN = re.compile(r"^- 状态：`([^`]+)`$", re.MULTILINE)
PLAN_STATES = frozenset(
    {"draft", "in-progress", "deferred", "blocked", "completed", "cancelled"}
)
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
    for family in MODEL_DOCUMENTATION_FAMILIES:
        family_root = models_root / family
        for filename in MODEL_REPORT_FILES:
            _require(
                (family_root / filename).is_file(),
                f"missing model report: {family}/{filename}",
            )
    for directory in LEGACY_MODEL_DOCUMENTATION_DIRECTORIES:
        _require(
            not (models_root / directory).exists(),
            f"legacy model documentation directory remains: {directory}",
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


def _strip_inline_code(line: str) -> str:
    """Replace inline code spans while preserving the rest of a Markdown line."""

    output = []
    index = 0
    while index < len(line):
        if line[index] != "`":
            output.append(line[index])
            index += 1
            continue
        end_of_run = index
        while end_of_run < len(line) and line[end_of_run] == "`":
            end_of_run += 1
        marker = line[index:end_of_run]
        closing = line.find(marker, end_of_run)
        if closing == -1:
            output.append(marker)
            index = end_of_run
            continue
        output.append(" " * (closing + len(marker) - index))
        index = closing + len(marker)
    return "".join(output)


def _markdown_without_code(text: str) -> str:
    """Remove fenced, indented, and inline code before parsing Markdown links."""

    output = []
    fence_character: Optional[str] = None
    fence_length = 0
    for line in text.splitlines(keepends=True):
        fence = FENCE_PATTERN.match(line)
        if fence_character is not None:
            if (
                fence
                and fence.group(1)[0] == fence_character
                and len(fence.group(1)) >= fence_length
            ):
                fence_character = None
                fence_length = 0
            output.append("\n" if line.endswith("\n") else "")
            continue
        if fence:
            fence_character = fence.group(1)[0]
            fence_length = len(fence.group(1))
            output.append("\n" if line.endswith("\n") else "")
            continue
        if line.startswith("    ") or line.startswith("\t"):
            output.append("\n" if line.endswith("\n") else "")
            continue
        output.append(_strip_inline_code(line))
    return "".join(output)


def _repository_markdown_files(root: Path) -> list[Path]:
    files = [root / "README.md", root / "ROADMAP.md"]
    contributing = root / "CONTRIBUTING.md"
    if contributing.is_file():
        files.append(contributing)
    for directory in (root / "docs", root / "tests"):
        files.extend(directory.rglob("*.md"))
    return sorted(set(path.resolve() for path in files))


def _validate_links(paths: Sequence[Path]) -> None:
    for path in paths:
        text = _markdown_without_code(path.read_text(encoding="utf-8"))
        for target in LINK_PATTERN.findall(text):
            target = target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            target = target.split(maxsplit=1)[0]
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "data:")):
                continue
            resolved = (path.parent / target).resolve()
            _require(resolved.exists(), f"broken relative link in {path}: {target}")


def _require_indexed(index: Path, paths: Sequence[Path]) -> None:
    text = _markdown_without_code(index.read_text(encoding="utf-8"))
    for path in paths:
        _require(
            f"]({path.name})" in text,
            f"missing index entry in {index}: {path.name}",
        )


def _validate_indexes(root: Path) -> None:
    plans_root = root / "docs/plans"
    _require_indexed(
        plans_root / "README.md",
        sorted(
            path
            for path in plans_root.glob("*.md")
            if path.name not in {"README.md", "TEMPLATE.md"}
        ),
    )
    migrations_root = root / "docs/migrations"
    _require_indexed(
        migrations_root / "README.md",
        sorted(
            path for path in migrations_root.glob("*.md") if path.name != "README.md"
        ),
    )
    docs_index = (root / "docs/README.md").read_text(encoding="utf-8")
    for target in (
        "guides/README.md",
        "development/README.md",
        "models/README.md",
        "migrations/README.md",
        "plans/README.md",
        "archive/README.md",
    ):
        _require(f"]({target})" in docs_index, f"missing docs index entry: {target}")


def _validate_plan_states(root: Path) -> None:
    for path in sorted((root / "docs/plans").glob("*.md")):
        if path.name == "README.md":
            continue
        states = PLAN_STATUS_PATTERN.findall(path.read_text(encoding="utf-8"))
        _require(len(states) == 1, f"plan must declare exactly one status: {path}")
        _require(
            states[0] in PLAN_STATES, f"invalid plan status in {path}: {states[0]}"
        )


def _validate_archive_notices(root: Path) -> None:
    for path in sorted((root / "docs/archive").rglob("*.md")):
        preamble = "\n".join(path.read_text(encoding="utf-8").splitlines()[:12])
        _require(
            any(term in preamble for term in ("历史", "归档", "快照")),
            f"archive document is missing a dated snapshot notice: {path}",
        )


def validate_repository(plan: Optional[Path] = None) -> dict[str, int]:
    if plan is not None:
        plan = plan.resolve()
        _require(plan.is_file(), f"plan does not exist: {plan}")
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
    _require(config_count == 27, "current support matrix must contain 27 new variants")

    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    require_attributions(notice)
    markdown = _repository_markdown_files(REPO_ROOT)
    _validate_links(markdown)
    _validate_indexes(REPO_ROOT)
    _validate_plan_states(REPO_ROOT)
    _validate_archive_notices(REPO_ROOT)
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

    _require(
        "student-only"
        in (REPO_ROOT / "docs/models/rtdetrv4/README.md").read_text(encoding="utf-8"),
        "RT-DETRv4 student-only claim is missing",
    )
    return {
        "families": len(FAMILIES),
        "artifacts": len(aliases),
        "new_variants": config_count,
        "markdown_files": len(markdown),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        help="optional compatibility argument; plan-specific evidence is checked separately",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    summary = validate_repository(args.plan)
    print(
        "documentation checks passed: "
        f"{summary['families']} families, {summary['artifacts']} artifacts, "
        f"{summary['new_variants']} new variants, "
        f"{summary['markdown_files']} markdown files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

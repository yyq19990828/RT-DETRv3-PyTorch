import copy

import pytest

from scripts import check_docs


def test_rejects_stale_variant():
    with pytest.raises(ValueError, match="stale variant"):
        check_docs.require_expected_variants(("dfine-n", "dfine-s"), "dfine")


def test_rejects_missing_model_report(tmp_path):
    for family in check_docs.MODEL_DOCUMENTATION_FAMILIES:
        family_root = tmp_path / family
        family_root.mkdir()
        for filename in check_docs.MODEL_REPORT_FILES:
            (family_root / filename).touch()
    (tmp_path / "dfine" / "metrics.md").unlink()

    with pytest.raises(ValueError, match="missing model report: dfine/metrics.md"):
        check_docs.require_model_report_layout(tmp_path)


def test_rejects_legacy_deim_documentation_directory(tmp_path):
    for family in check_docs.MODEL_DOCUMENTATION_FAMILIES:
        family_root = tmp_path / family
        family_root.mkdir()
        for filename in check_docs.MODEL_REPORT_FILES:
            (family_root / filename).touch()
    (tmp_path / "deim-dfine").mkdir()

    with pytest.raises(ValueError, match="legacy model documentation directory"):
        check_docs.require_model_report_layout(tmp_path)


def test_rejects_missing_attribution():
    with pytest.raises(ValueError, match="missing attribution"):
        check_docs.require_attributions("Apache-2.0")


def test_rejects_absolute_path():
    with pytest.raises(ValueError, match="absolute path"):
        check_docs.reject_absolute_paths({"README.md": "asset: /home/user/model.pth"})


def test_link_check_ignores_code_spans_and_fences(tmp_path):
    document = tmp_path / "README.md"
    document.write_text(
        "inline `[index](missing)`\n\n"
        "```python\n"
        "output = layer[index](memory)\n"
        "[label](also-missing)\n"
        "```\n",
        encoding="utf-8",
    )

    check_docs._validate_links((document,))


def test_link_check_rejects_broken_relative_link(tmp_path):
    document = tmp_path / "README.md"
    document.write_text("[missing](missing.md)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="broken relative link"):
        check_docs._validate_links((document,))


def test_rejects_unknown_plan_status(tmp_path):
    plans = tmp_path / "docs/plans"
    plans.mkdir(parents=True)
    (plans / "invalid.md").write_text("- 状态：`done`\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid plan status"):
        check_docs._validate_plan_states(tmp_path)


@pytest.mark.parametrize("term", ("Task 12", "F3", "TODO"))
def test_rejects_internal_workflow_term(term):
    with pytest.raises(ValueError, match="internal workflow term"):
        check_docs.reject_internal_workflow_terms({"docs/models/dfine/README.md": term})


def test_rejects_teacher_graph_contradiction():
    evidence = {
        "variants": [
            {"opset": 17, "training_residue": False},
            {"opset": 17, "training_residue": False},
            {"opset": 17, "training_residue": False},
            {"opset": 17, "training_residue": False},
        ]
    }
    contradictory = copy.deepcopy(evidence)
    contradictory["variants"][2]["training_residue"] = True
    with pytest.raises(ValueError, match="teacher graph contradiction"):
        check_docs.validate_teacher_graph_claim("student-only export", contradictory)

import copy

import pytest

from scripts import check_docs


def test_rejects_stale_variant():
    with pytest.raises(ValueError, match="stale variant"):
        check_docs.require_expected_variants(("dfine-n", "dfine-s"), "dfine")


def test_rejects_missing_model_report(tmp_path):
    for family in check_docs.FAMILIES:
        family_root = tmp_path / family
        family_root.mkdir()
        for filename in check_docs.MODEL_REPORT_FILES:
            (family_root / filename).touch()
    (tmp_path / "dfine" / "metrics.md").unlink()

    with pytest.raises(ValueError, match="missing model report: dfine/metrics.md"):
        check_docs.require_model_report_layout(tmp_path)


def test_rejects_missing_attribution():
    with pytest.raises(ValueError, match="missing attribution"):
        check_docs.require_attributions("Apache-2.0")


def test_rejects_absolute_path():
    with pytest.raises(ValueError, match="absolute path"):
        check_docs.reject_absolute_paths({"README.md": "asset: /home/user/model.pth"})


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

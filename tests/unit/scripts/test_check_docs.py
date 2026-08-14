import copy

import pytest

from scripts import check_docs


def test_rejects_stale_variant():
    with pytest.raises(ValueError, match="stale variant"):
        check_docs.require_expected_variants(("dfine-n", "dfine-s"), "dfine")


def test_rejects_missing_attribution():
    with pytest.raises(ValueError, match="missing attribution"):
        check_docs.require_attributions("Apache-2.0")


def test_rejects_absolute_path():
    with pytest.raises(ValueError, match="absolute path"):
        check_docs.reject_absolute_paths({"README.md": "asset: /home/user/model.pth"})


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

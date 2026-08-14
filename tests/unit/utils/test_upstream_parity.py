import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/dev/compare_upstream_pytorch.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("compare_upstream_pytorch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_identical_named_tensors_are_approved(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    driver = _load_driver()

    result = driver.compare_named_tensors(
        {"encoder.stage3": torch.tensor([1.0, 2.0])},
        {"encoder.stage3": torch.tensor([1.0, 2.0])},
    )

    assert result == [
        {
            "atol": 1e-6,
            "candidate_dtype": "torch.float32",
            "max_abs_error": 0.0,
            "max_error_flat_index": 0,
            "max_rel_error": 0.0,
            "name": "encoder.stage3",
            "reference_dtype": "torch.float32",
            "rtol": 1e-5,
            "shape": [2],
            "status": "APPROVE",
        }
    ]


def test_rejects_tensor_above_tolerance_with_exact_name(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    driver = _load_driver()

    result = driver.compare_named_tensors(
        {"decoder.logits": torch.tensor([1.0, 2.0])},
        {"decoder.logits": torch.tensor([1.0, 2.01])},
    )

    assert result[0]["status"] == "FAIL"
    assert result[0]["name"] == "decoder.logits"
    assert result[0]["max_abs_error"] > 0.009
    assert result[0]["max_error_flat_index"] == 1


def test_reports_missing_and_shape_mismatched_tensors(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    driver = _load_driver()

    result = driver.compare_named_tensors(
        {"missing": torch.ones(1), "shape": torch.ones(2)},
        {"shape": torch.ones(3)},
    )

    assert [item["name"] for item in result] == ["missing", "shape"]
    assert all(item["status"] == "FAIL" for item in result)

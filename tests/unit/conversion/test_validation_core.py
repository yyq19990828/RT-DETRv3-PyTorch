"""Core model-output validation tests that do not require Paddle."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from detrs.conversion.validation import (
    ForwardPassResult,
    ModelOutputValidator,
)


class PaddleValue:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.shape = self.value.shape

    def numpy(self):
        return self.value


class TorchValue:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


@pytest.fixture
def validator():
    return ModelOutputValidator(rtol=1e-4, atol=1e-5)


def test_validator_and_result_contract():
    validator = ModelOutputValidator(rtol=1e-6, atol=1e-7)
    result = ForwardPassResult(True, 1e-6, 1e-7, 1e-5, (1, 4), "details")

    assert validator.rtol == 1e-6
    assert validator.atol == 1e-7
    assert result.output_shape == (1, 4)


def test_compare_tensors_reports_matching_and_mismatching_values(validator):
    expected = np.ones((2, 3), dtype=np.float32)

    matching = validator._compare_tensors(expected, expected.copy(), "boxes")
    mismatching = validator._compare_tensors(expected, expected * 2, "boxes")

    assert matching.passed is True
    assert matching.max_abs_diff == 0.0
    assert "PASSED" in matching.details
    assert mismatching.passed is False
    assert mismatching.max_abs_diff == pytest.approx(1.0)
    assert "FAILED" in mismatching.details


def test_compare_tensors_rejects_shape_mismatch(validator):
    result = validator._compare_tensors(
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
        "boxes",
    )

    assert result.passed is False
    assert result.max_abs_diff == float("inf")
    assert "Shape mismatch" in result.details


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_compare_tensors_rejects_non_finite_values_on_both_sides(validator, non_finite):
    values = np.asarray([non_finite], dtype=np.float32)

    result = validator._compare_tensors(values, values.copy(), "scores")

    assert result.passed is False
    assert result.max_abs_diff == float("inf")
    assert "Non-finite" in result.details


def test_compare_tensors_accepts_matching_empty_outputs(validator):
    empty = np.empty((0, 6), dtype=np.float32)

    result = validator._compare_tensors(empty, empty.copy(), "detections")

    assert result.passed is True
    assert result.output_shape == (0, 6)
    assert result.max_abs_diff == 0.0


def test_compare_dict_outputs_aggregates_metrics(validator):
    paddle_output = {
        "pred_boxes": PaddleValue([[1.0, 2.0]]),
        "pred_logits": PaddleValue([[0.4, 0.6]]),
    }
    torch_output = {
        "pred_boxes": TorchValue([[1.0, 2.0]]),
        "pred_logits": TorchValue([[0.4, 0.7]]),
    }

    result = validator._compare_dict_outputs(paddle_output, torch_output)

    assert result.passed is False
    assert result.output_shape == (1, 2)
    assert result.max_abs_diff == pytest.approx(0.1)
    assert "MATCH pred_boxes" in result.details
    assert "MISMATCH pred_logits" in result.details


def test_compare_dict_outputs_rejects_missing_or_extra_keys(validator):
    value = PaddleValue([[1.0]])
    torch_value = TorchValue([[1.0]])

    missing = validator._compare_dict_outputs(
        {"pred_boxes": value, "pred_logits": value},
        {"pred_boxes": torch_value},
    )
    extra = validator._compare_dict_outputs(
        {"pred_boxes": value},
        {"pred_boxes": torch_value, "aux": torch_value},
    )

    assert missing.passed is False
    assert "pred_logits' missing in PyTorch" in missing.details
    assert extra.passed is False
    assert "aux' missing in Paddle" in extra.details


def test_compare_dict_outputs_accepts_two_empty_dictionaries(validator):
    result = validator._compare_dict_outputs({}, {})

    assert result.passed is True
    assert result.output_shape == ()
    assert result.details == "Both output dictionaries are empty"


def _install_fake_paddle(monkeypatch):
    fake_paddle = SimpleNamespace(
        to_tensor=lambda value, dtype=None: PaddleValue(value),
        no_grad=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)


def test_validate_forward_pass_normalizes_generic_sequence_outputs(
    validator, monkeypatch
):
    _install_fake_paddle(monkeypatch)

    class PaddleModel:
        def eval(self):
            return self

        def __call__(self, value):
            return [PaddleValue(value.numpy() * 2)]

    class TorchModel:
        def eval(self):
            return self

        def __call__(self, value):
            return (value * 2,)

    sample = np.ones((1, 3, 2, 2), dtype=np.float32)

    result = validator.validate_forward_pass(PaddleModel(), TorchModel(), sample)

    assert result.passed is True
    assert result.output_shape == sample.shape


def test_validate_forward_pass_extracts_rtdetr_transformer_outputs(
    validator, monkeypatch
):
    _install_fake_paddle(monkeypatch)
    boxes = np.asarray([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)
    logits = np.asarray([[[0.25, 0.75]]], dtype=np.float32)

    class PaddleModel:
        def eval(self):
            return self

        def backbone(self, inputs):
            assert set(inputs) == {"image", "im_shape", "scale_factor"}
            return "body"

        def neck(self, body):
            assert body == "body"
            return "neck"

        def transformer(self, neck, pad_mask, inputs):
            assert (neck, pad_mask, inputs) == ("neck", None, None)
            return None, None, PaddleValue(boxes), PaddleValue(logits)

    class TorchModel:
        def eval(self):
            return self

        def __call__(self, inputs):
            assert set(inputs) == {"image", "im_shape", "scale_factor"}
            return {
                "pred_boxes": torch.from_numpy(boxes.copy()),
                "pred_logits": torch.from_numpy(logits.copy()),
            }

    sample = np.ones((1, 3, 2, 2), dtype=np.float32)

    result = validator.validate_forward_pass(PaddleModel(), TorchModel(), sample)

    assert result.passed is True
    assert result.output_shape == boxes.shape


def test_validate_forward_pass_rejects_incompatible_output_structures(
    validator, monkeypatch
):
    _install_fake_paddle(monkeypatch)

    class PaddleModel:
        def eval(self):
            return self

        def __call__(self, value):
            return {"output": PaddleValue(value.numpy())}

    class TorchModel:
        def eval(self):
            return self

        def __call__(self, value):
            return value

    with pytest.raises(TypeError, match="incompatible output structures"):
        validator.validate_forward_pass(
            PaddleModel(),
            TorchModel(),
            np.ones((1, 3, 2, 2), dtype=np.float32),
        )


@pytest.mark.parametrize("passed", [True, False])
def test_print_validation_report_includes_status_and_details(validator, capsys, passed):
    result = ForwardPassResult(
        passed=passed,
        max_abs_diff=1e-6,
        mean_abs_diff=1e-7,
        max_rel_diff=1e-5,
        output_shape=(1, 4),
        details="layer details",
    )

    validator.print_validation_report(result)

    output = capsys.readouterr().out
    assert ("✅ PASSED" if passed else "❌ FAILED") in output
    assert "layer details" in output

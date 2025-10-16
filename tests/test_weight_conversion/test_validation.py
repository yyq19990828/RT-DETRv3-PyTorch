"""Unit tests for model output validation

Tests for validation module in tools/weight_conversion/validation.py
"""

import numpy as np
import pytest
import torch
import paddle

from tools.weight_conversion.validation import (
    ModelOutputValidator,
    ForwardPassResult
)


class DummyPaddleModel(paddle.nn.Layer):
    """Dummy Paddle model for testing"""
    def forward(self, x):
        # Simple computation: y = x + 1
        return {'pred_boxes': x + 1.0, 'pred_logits': x * 2.0}


class DummyTorchModel(torch.nn.Module):
    """Dummy PyTorch model for testing"""
    def forward(self, x):
        # Same computation as Paddle model
        return {'pred_boxes': x + 1.0, 'pred_logits': x * 2.0}


class TestModelOutputValidator:
    """Test suite for ModelOutputValidator"""

    @pytest.fixture
    def validator(self):
        """Create ModelOutputValidator instance"""
        return ModelOutputValidator(rtol=1e-4, atol=1e-5)

    @pytest.fixture
    def matching_models(self):
        """Create matching Paddle and PyTorch models"""
        paddle_model = DummyPaddleModel()
        torch_model = DummyTorchModel()
        return paddle_model, torch_model

    def test_validator_initialization(self):
        """Test validator initialization with custom tolerances"""
        validator = ModelOutputValidator(rtol=1e-6, atol=1e-7)
        assert validator.rtol == 1e-6
        assert validator.atol == 1e-7

    def test_forward_pass_result_dataclass(self):
        """Test ForwardPassResult dataclass"""
        result = ForwardPassResult(
            passed=True,
            max_abs_diff=1e-6,
            mean_abs_diff=1e-7,
            max_rel_diff=1e-5,
            output_shape=(1, 300, 4),
            details="Test details"
        )

        assert result.passed is True
        assert result.max_abs_diff == 1e-6
        assert result.output_shape == (1, 300, 4)

    def test_validate_forward_pass_matching(self, validator, matching_models):
        """Test forward pass validation with matching models"""
        paddle_model, torch_model = matching_models

        # Create sample input
        sample_input = np.random.randn(1, 3, 64, 64).astype(np.float32)

        # Run validation
        result = validator.validate_forward_pass(
            paddle_model,
            torch_model,
            sample_input
        )

        assert result.passed is True
        assert result.max_abs_diff < 1e-6

    def test_validate_forward_pass_dict_output(self, validator):
        """Test validation with dictionary outputs (RT-DETRv3 style)"""
        paddle_model = DummyPaddleModel()
        torch_model = DummyTorchModel()

        sample_input = np.random.randn(2, 3, 32, 32).astype(np.float32)

        result = validator.validate_forward_pass(
            paddle_model,
            torch_model,
            sample_input
        )

        assert result.passed is True
        assert 'pred_boxes' in result.details
        assert 'pred_logits' in result.details

    def test_print_validation_report(self, validator, capsys):
        """Test validation report printing"""
        result = ForwardPassResult(
            passed=True,
            max_abs_diff=1e-6,
            mean_abs_diff=1e-7,
            max_rel_diff=1e-5,
            output_shape=(1, 300, 4),
            details="Sample validation details"
        )

        validator.print_validation_report(result)

        # Capture output
        captured = capsys.readouterr()

        assert "MODEL OUTPUT VALIDATION REPORT" in captured.out
        assert "PASSED" in captured.out
        assert "1.00e-06" in captured.out  # max_abs_diff

    def test_compare_tensors_matching(self, validator):
        """Test tensor comparison with matching arrays"""
        arr1 = np.random.randn(10, 20).astype(np.float32)
        arr2 = arr1.copy()

        result = validator._compare_tensors(arr1, arr2, "test_tensor")

        assert result.passed is True
        assert result.max_abs_diff < 1e-10

    def test_compare_tensors_shape_mismatch(self, validator):
        """Test tensor comparison with shape mismatch"""
        arr1 = np.random.randn(10, 20).astype(np.float32)
        arr2 = np.random.randn(10, 30).astype(np.float32)

        result = validator._compare_tensors(arr1, arr2, "test_tensor")

        assert result.passed is False
        assert "Shape mismatch" in result.details

    def test_compare_tensors_value_mismatch(self, validator):
        """Test tensor comparison with value differences"""
        arr1 = np.ones((10, 20), dtype=np.float32)
        arr2 = np.ones((10, 20), dtype=np.float32) * 2.0  # Completely different

        result = validator._compare_tensors(arr1, arr2, "test_tensor")

        assert result.passed is False
        assert result.max_abs_diff > 0.9  # Should be around 1.0

    def test_compare_tensors_small_diff(self, validator):
        """Test tensor comparison with small differences"""
        arr1 = np.ones((10, 20), dtype=np.float32)
        arr2 = arr1 + 1e-8  # Very small difference

        result = validator._compare_tensors(arr1, arr2, "test_tensor")

        assert result.passed is True  # Should pass with default tolerances
        assert result.max_abs_diff < 1e-7

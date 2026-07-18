"""Unit tests for tensor conversion utilities

Tests for ppdet_pytorch.conversion.tensor_utils.
"""

import pytest
import numpy as np
import torch

from ppdet_pytorch.conversion.tensor_utils import (
    paddle_to_numpy,
    numpy_to_torch,
    validate_tensor_shape,
    detect_dtype,
    convert_paddle_to_torch_tensor,
    check_shape_compatibility,
)


@pytest.fixture
def paddle_module():
    """Return PaddlePaddle or skip only the tests that require it."""
    return pytest.importorskip(
        "paddle", reason="requires the PaddlePaddle development extra"
    )


class TestTensorConversion:
    """Test suite for tensor conversion utilities"""

    @pytest.mark.paddle
    def test_convert_tensor(self, paddle_module):
        """Test complete tensor conversion pipeline

        T013: Unit test for tensor conversion (paddle→numpy→torch)
        Verifies the full conversion pipeline works correctly
        """
        # Create paddle tensor
        paddle_tensor = paddle_module.randn([2, 3, 4], dtype='float32')
        original_data = paddle_tensor.numpy()

        # Convert through full pipeline
        torch_tensor = convert_paddle_to_torch_tensor(paddle_tensor, "test_param")

        # Verify conversion
        assert isinstance(torch_tensor, torch.Tensor)
        assert torch_tensor.shape == (2, 3, 4)
        assert torch_tensor.dtype == torch.float32

        # Verify data is preserved
        np.testing.assert_allclose(
            torch_tensor.numpy(),
            original_data,
            rtol=1e-6,
            atol=1e-6
        )

    @pytest.mark.paddle
    def test_paddle_to_numpy(self, paddle_module):
        """Test PaddlePaddle tensor to NumPy conversion"""
        paddle_tensor = paddle_module.randn([5, 10])
        numpy_array = paddle_to_numpy(paddle_tensor)

        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape == (5, 10)
        assert numpy_array.flags['C_CONTIGUOUS']

    def test_numpy_to_torch(self):
        """Test NumPy array to PyTorch tensor conversion"""
        numpy_array = np.random.randn(3, 7).astype(np.float32)
        torch_tensor = numpy_to_torch(numpy_array)

        assert isinstance(torch_tensor, torch.Tensor)
        assert torch_tensor.shape == (3, 7)
        assert torch_tensor.dtype == torch.float32

        # Verify data is preserved
        np.testing.assert_array_equal(
            torch_tensor.numpy(),
            numpy_array
        )

    def test_validate_tensor_shape_match(self):
        """Test shape validation with matching shapes"""
        tensor_shape = (3, 4, 5)
        expected_shape = (3, 4, 5)

        # Should return True and not raise
        result = validate_tensor_shape(tensor_shape, expected_shape, "test_param", strict=False)
        assert result is True

    def test_validate_tensor_shape_mismatch_non_strict(self):
        """Test shape validation with mismatch in non-strict mode"""
        tensor_shape = (3, 4)
        expected_shape = (5, 6)

        # Should return False but not raise
        result = validate_tensor_shape(tensor_shape, expected_shape, "test_param", strict=False)
        assert result is False

    def test_validate_tensor_shape_mismatch_strict(self):
        """Test shape validation with mismatch in strict mode raises error"""
        tensor_shape = (3, 4)
        expected_shape = (5, 6)

        # Should raise ValueError in strict mode
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_tensor_shape(tensor_shape, expected_shape, "test_param", strict=True)

    @pytest.mark.paddle
    def test_detect_dtype_paddle(self, paddle_module):
        """Test dtype detection for PaddlePaddle tensors"""
        tensor_fp32 = paddle_module.randn([2, 3], dtype='float32')
        tensor_fp64 = paddle_module.randn([2, 3], dtype='float64')

        assert detect_dtype(tensor_fp32) == 'float32'
        assert detect_dtype(tensor_fp64) == 'float64'

    def test_detect_dtype_numpy(self):
        """Test dtype detection for NumPy arrays"""
        array_fp32 = np.random.randn(2, 3).astype(np.float32)
        array_int32 = np.random.randint(0, 10, (2, 3)).astype(np.int32)

        assert detect_dtype(array_fp32) == 'float32'
        assert detect_dtype(array_int32) == 'int32'

    def test_check_shape_compatibility_exact_match(self):
        """Test shape compatibility check with exact match"""
        source_shape = (3, 4, 5)
        target_shape = (3, 4, 5)

        is_compatible, message = check_shape_compatibility(source_shape, target_shape)
        assert is_compatible is True
        assert "match exactly" in message

    def test_check_shape_compatibility_reshapeable(self):
        """Test shape compatibility check with reshapeable tensors"""
        source_shape = (12,)
        target_shape = (3, 4)

        is_compatible, message = check_shape_compatibility(source_shape, target_shape)
        assert is_compatible is True
        assert "reshaped" in message.lower()

    def test_check_shape_compatibility_incompatible(self):
        """Test shape compatibility check with incompatible shapes"""
        source_shape = (3, 5)
        target_shape = (4, 4)

        is_compatible, message = check_shape_compatibility(source_shape, target_shape)
        assert is_compatible is False
        assert "incompatible" in message.lower()

    @pytest.mark.paddle
    def test_convert_different_dtypes(self, paddle_module):
        """Test conversion preserves different data types"""
        # Test float32
        paddle_fp32 = paddle_module.randn([2, 3], dtype='float32')
        torch_fp32 = convert_paddle_to_torch_tensor(paddle_fp32, "fp32_param")
        assert torch_fp32.dtype == torch.float32

        # Test float64
        paddle_fp64 = paddle_module.randn([2, 3], dtype='float64')
        torch_fp64 = convert_paddle_to_torch_tensor(paddle_fp64, "fp64_param")
        assert torch_fp64.dtype == torch.float64

    @pytest.mark.paddle
    def test_convert_various_shapes(self, paddle_module):
        """Test conversion works with various tensor shapes"""
        # Scalar (0-D)
        paddle_0d = paddle_module.to_tensor(3.14)
        torch_0d = convert_paddle_to_torch_tensor(paddle_0d, "scalar")
        assert torch_0d.shape == ()

        # 1-D
        paddle_1d = paddle_module.randn([10])
        torch_1d = convert_paddle_to_torch_tensor(paddle_1d, "1d")
        assert torch_1d.shape == (10,)

        # 4-D (typical conv weights)
        paddle_4d = paddle_module.randn([64, 3, 7, 7])
        torch_4d = convert_paddle_to_torch_tensor(paddle_4d, "4d")
        assert torch_4d.shape == (64, 3, 7, 7)

    @pytest.mark.paddle
    def test_numerical_precision(self, paddle_module):
        """Test that conversion maintains numerical precision"""
        # Create tensor with specific values
        values = np.array([[1.23456789, 2.3456789], [3.456789, 4.56789]], dtype=np.float32)
        paddle_tensor = paddle_module.to_tensor(values)

        # Convert
        torch_tensor = convert_paddle_to_torch_tensor(paddle_tensor, "precision_test")

        # Verify precision is maintained (within floating point tolerance)
        np.testing.assert_allclose(
            torch_tensor.numpy(),
            values,
            rtol=1e-7,
            atol=1e-7
        )

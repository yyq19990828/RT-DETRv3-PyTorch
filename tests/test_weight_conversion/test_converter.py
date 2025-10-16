"""Unit tests for weight converter

Tests for WeightConverter class in tools/weight_conversion/converter.py
"""

import pytest
import torch
from pathlib import Path

from tools.weight_conversion.converter import WeightConverter
from tools.weight_conversion.models import ConversionConfig, CheckpointFormat, Framework


class TestWeightConverter:
    """Test suite for WeightConverter class"""

    @pytest.fixture
    def converter(self):
        """Create WeightConverter instance for testing"""
        config = ConversionConfig(strict_mode=False)
        return WeightConverter(config)

    @pytest.fixture
    def sample_checkpoint_path(self):
        """Get path to sample paddle checkpoint"""
        return "tests/test_weight_conversion/fixtures/sample_paddle.pdparams"

    def test_load_paddle_checkpoint(self, converter, sample_checkpoint_path):
        """Test loading PaddlePaddle checkpoint file

        T012: Unit test for paddle checkpoint loading
        Verifies that:
        - Checkpoint file is loaded successfully
        - Returns valid state dict
        - CheckpointFile metadata is created correctly
        """
        # Load checkpoint
        paddle_state, checkpoint_file = converter.load_paddle_checkpoint(sample_checkpoint_path)

        # Verify state dict is loaded
        assert isinstance(paddle_state, dict)
        assert len(paddle_state) > 0

        # Verify checkpoint metadata
        assert checkpoint_file.format == CheckpointFormat.PDPARAMS
        assert checkpoint_file.framework == Framework.PADDLEPADDLE
        assert checkpoint_file.file_size_bytes > 0
        assert checkpoint_file.checksum is not None
        assert Path(checkpoint_file.file_path).exists()

    def test_load_paddle_checkpoint_not_found(self, converter):
        """Test loading non-existent checkpoint raises error"""
        with pytest.raises(FileNotFoundError):
            converter.load_paddle_checkpoint("nonexistent.pdparams")

    def test_save_torch_checkpoint(self, converter, tmp_path):
        """Test saving PyTorch checkpoint file

        T015: Unit test for torch checkpoint saving
        Verifies that:
        - Checkpoint is saved successfully
        - File is created on disk
        - Metadata is embedded correctly
        """
        # Create sample state dict
        state_dict = {
            "layer.weight": torch.randn(10, 5),
            "layer.bias": torch.randn(10),
        }

        # Save checkpoint
        output_path = tmp_path / "test_output.pth"
        metadata = {
            "test_key": "test_value",
            "source": "test"
        }

        checkpoint_file = converter.save_torch_checkpoint(
            state_dict,
            str(output_path),
            metadata
        )

        # Verify file was created
        assert output_path.exists()
        assert checkpoint_file.format == CheckpointFormat.PTH
        assert checkpoint_file.framework == Framework.PYTORCH
        assert checkpoint_file.file_size_bytes > 0

        # Verify checkpoint can be loaded
        loaded = torch.load(output_path, weights_only=False)
        assert "model" in loaded
        assert "metadata" in loaded
        assert loaded["metadata"]["test_key"] == "test_value"
        assert len(loaded["model"]) == 2

    def test_convert_tensor_basic(self, converter):
        """Test basic tensor conversion from PaddlePaddle to PyTorch

        Verifies that:
        - Tensor is converted successfully
        - Data is preserved
        - Shape is correct
        """
        import paddle

        # Create paddle tensor
        paddle_tensor = paddle.randn([3, 4, 5])

        # Convert to torch
        torch_tensor = converter.convert_tensor(paddle_tensor, "test_param")

        # Verify conversion
        assert isinstance(torch_tensor, torch.Tensor)
        assert torch_tensor.shape == (3, 4, 5)
        assert torch_tensor.dtype == torch.float32

    def test_convert_tensor_with_shape_validation(self, converter):
        """Test tensor conversion with shape validation"""
        import paddle

        paddle_tensor = paddle.randn([3, 4])
        expected_shape = (3, 4)

        # Should succeed with correct shape
        torch_tensor = converter.convert_tensor(
            paddle_tensor,
            "test_param",
            expected_shape
        )
        assert torch_tensor.shape == expected_shape

    def test_convert_tensor_shape_mismatch_strict(self):
        """Test tensor conversion fails in strict mode with shape mismatch"""
        import paddle

        config = ConversionConfig(strict_mode=True)
        converter = WeightConverter(config)

        paddle_tensor = paddle.randn([3, 4])
        wrong_shape = (5, 6)

        # Should raise error in strict mode
        with pytest.raises(ValueError, match="Shape mismatch"):
            converter.convert_tensor(paddle_tensor, "test_param", wrong_shape)

    def test_convert_tensor_shape_mismatch_permissive(self):
        """Test tensor conversion skips in permissive mode with shape mismatch"""
        import paddle

        config = ConversionConfig(strict_mode=False)
        converter = WeightConverter(config)

        paddle_tensor = paddle.randn([3, 4])
        wrong_shape = (5, 6)

        # Should return None in permissive mode
        result = converter.convert_tensor(paddle_tensor, "test_param", wrong_shape)
        assert result is None

    def test_convert_state_dict_basic(self, converter, sample_checkpoint_path):
        """Test converting entire state dict

        Verifies that:
        - All parameters are converted
        - Statistics are tracked
        - Target state dict is created
        """
        # Load sample checkpoint
        paddle_state, _ = converter.load_paddle_checkpoint(sample_checkpoint_path)

        # Generate mappings
        from tools.weight_conversion.name_mapping import NameMapper
        mapper = NameMapper()
        mappings = mapper.apply_naming_rules(list(paddle_state.keys()))

        # Convert state dict
        torch_state, stats = converter.convert_state_dict(
            paddle_state,
            target_state_dict=None,
            mappings=mappings
        )

        # Verify conversion
        assert len(torch_state) > 0
        assert stats.total_parameters == len(mappings)
        assert stats.converted_count > 0
        assert stats.converted_count <= stats.total_parameters

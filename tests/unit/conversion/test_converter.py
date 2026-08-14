"""Unit tests for weight converter

Tests for WeightConverter class in detrs.conversion.converter.
"""

from pathlib import Path

import pytest
import torch

from detrs.conversion.converter import WeightConverter
from detrs.conversion.models import (
    CheckpointFormat,
    ConversionConfig,
    Framework,
)


@pytest.fixture
def paddle_module():
    """Return PaddlePaddle or skip only the tests that require it."""
    return pytest.importorskip(
        "paddle", reason="requires the PaddlePaddle development extra"
    )


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
        return Path(__file__).resolve().parent / "fixtures" / "sample_paddle.pdparams"

    @pytest.mark.paddle
    def test_load_paddle_checkpoint(
        self, converter, sample_checkpoint_path, paddle_module
    ):
        """Test loading PaddlePaddle checkpoint file

        T012: Unit test for paddle checkpoint loading
        Verifies that:
        - Checkpoint file is loaded successfully
        - Returns valid state dict
        - CheckpointFile metadata is created correctly
        """
        # Load checkpoint
        paddle_state, checkpoint_file = converter.load_paddle_checkpoint(
            sample_checkpoint_path
        )

        # Verify state dict is loaded
        assert isinstance(paddle_state, dict)
        assert len(paddle_state) > 0

        # Verify checkpoint metadata
        assert checkpoint_file.format == CheckpointFormat.PDPARAMS
        assert checkpoint_file.framework == Framework.PADDLEPADDLE
        assert checkpoint_file.file_size_bytes > 0
        assert checkpoint_file.checksum is not None
        assert checkpoint_file.checksum_algorithm == "sha256"
        assert len(checkpoint_file.checksum) == 64
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
        metadata = {"test_key": "test_value", "source": "test"}

        checkpoint_file = converter.save_torch_checkpoint(
            state_dict, str(output_path), metadata
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

    def test_save_torch_checkpoint_preserves_existing_file_on_failure(
        self, converter, monkeypatch, tmp_path
    ):
        output_path = tmp_path / "existing.pth"
        output_path.write_bytes(b"existing")

        def fail_after_partial_write(_checkpoint, temporary_path):
            Path(temporary_path).write_bytes(b"partial")
            raise RuntimeError("save failed")

        monkeypatch.setattr(torch, "save", fail_after_partial_write)

        with pytest.raises(ValueError, match="save failed"):
            converter.save_torch_checkpoint({"weight": torch.ones(1)}, str(output_path))

        assert output_path.read_bytes() == b"existing"
        assert not list(tmp_path.glob(".existing.pth.*.tmp"))

    @pytest.mark.paddle
    def test_convert_tensor_basic(self, converter, paddle_module):
        """Test basic tensor conversion from PaddlePaddle to PyTorch

        Verifies that:
        - Tensor is converted successfully
        - Data is preserved
        - Shape is correct
        """
        # Create paddle tensor
        paddle_tensor = paddle_module.randn([3, 4, 5])

        # Convert to torch
        torch_tensor = converter.convert_tensor(paddle_tensor, "test_param")

        # Verify conversion
        assert isinstance(torch_tensor, torch.Tensor)
        assert torch_tensor.shape == (3, 4, 5)
        assert torch_tensor.dtype == torch.float32

    @pytest.mark.paddle
    def test_convert_tensor_with_shape_validation(self, converter, paddle_module):
        """Test tensor conversion with shape validation"""
        paddle_tensor = paddle_module.randn([3, 4])
        expected_shape = (3, 4)

        # Should succeed with correct shape
        torch_tensor = converter.convert_tensor(
            paddle_tensor, "test_param", expected_shape
        )
        assert torch_tensor.shape == expected_shape

    @pytest.mark.paddle
    def test_convert_tensor_honors_target_aware_square_linear_transpose(
        self, converter, paddle_module
    ):
        paddle_tensor = paddle_module.to_tensor([[1.0, 2.0], [3.0, 4.0]])

        torch_tensor = converter.convert_tensor(
            paddle_tensor,
            "square.weight",
            expected_shape=(2, 2),
            transpose=True,
        )

        assert torch.equal(
            torch_tensor,
            torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        )

    @pytest.mark.paddle
    def test_convert_tensor_shape_mismatch_strict(self, paddle_module):
        """Test tensor conversion fails in strict mode with shape mismatch"""
        config = ConversionConfig(strict_mode=True)
        converter = WeightConverter(config)

        paddle_tensor = paddle_module.randn([3, 4])
        wrong_shape = (5, 6)

        # Should raise error in strict mode
        with pytest.raises(ValueError, match="Shape mismatch"):
            converter.convert_tensor(paddle_tensor, "test_param", wrong_shape)

    @pytest.mark.paddle
    def test_convert_tensor_shape_mismatch_permissive(self, paddle_module):
        """Test tensor conversion skips in permissive mode with shape mismatch"""
        config = ConversionConfig(strict_mode=False)
        converter = WeightConverter(config)

        paddle_tensor = paddle_module.randn([3, 4])
        wrong_shape = (5, 6)

        # Should return None in permissive mode
        result = converter.convert_tensor(paddle_tensor, "test_param", wrong_shape)
        assert result is None

    @pytest.mark.paddle
    def test_memory_efficient_conversion_releases_source_tensors(self, paddle_module):
        source_state = {
            "first": paddle_module.to_tensor([1.0, 2.0]),
            "second": paddle_module.to_tensor([3.0, 4.0]),
        }
        converter = WeightConverter(
            ConversionConfig(memory_efficient_mode=True, batch_size=1)
        )

        converted, statistics = converter.convert_state_dict(source_state)

        assert source_state == {}
        assert statistics.converted_count == 2
        assert torch.equal(converted["first"], torch.tensor([1.0, 2.0]))
        assert torch.equal(converted["second"], torch.tensor([3.0, 4.0]))

    @pytest.mark.paddle
    def test_convert_state_dict_basic(
        self, converter, sample_checkpoint_path, paddle_module
    ):
        """Test converting entire state dict

        Verifies that:
        - All parameters are converted
        - Statistics are tracked
        - Target state dict is created
        """
        # Load sample checkpoint
        paddle_state, _ = converter.load_paddle_checkpoint(sample_checkpoint_path)

        # Generate mappings
        from detrs.conversion.name_mapping import NameMapper

        mapper = NameMapper()
        mappings = mapper.apply_naming_rules(list(paddle_state.keys()))

        # Convert state dict
        torch_state, stats = converter.convert_state_dict(
            paddle_state, target_state_dict=None, mappings=mappings
        )

        # Verify conversion
        assert len(torch_state) > 0
        assert stats.total_parameters == len(mappings)
        assert stats.converted_count > 0
        assert stats.converted_count <= stats.total_parameters

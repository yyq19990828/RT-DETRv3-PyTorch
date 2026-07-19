"""Integration tests for full weight conversion workflow

Tests the complete end-to-end conversion process including:
- Loading PaddlePaddle checkpoints
- Name mapping generation
- Tensor conversion
- Saving PyTorch checkpoints
- Mapping export
"""

import json
from pathlib import Path

import pytest
import torch

from ppdet_pytorch.conversion.converter import WeightConverter
from ppdet_pytorch.conversion.models import ConversionConfig, ConversionStatus

pytestmark = pytest.mark.paddle
paddle = pytest.importorskip(
    "paddle", reason="requires the PaddlePaddle development extra"
)


@pytest.mark.integration
class TestFullConversion:
    """Integration test suite for complete conversion workflow"""

    @pytest.fixture
    def sample_checkpoint_path(self):
        """Get path to sample paddle checkpoint"""
        return (
            Path(__file__).resolve().parents[1]
            / "unit"
            / "conversion"
            / "fixtures"
            / "sample_paddle.pdparams"
        )

    def test_convert_r50vd_model(self, sample_checkpoint_path, tmp_path):
        """Test end-to-end conversion workflow

        T016: Integration test for end-to-end conversion
        Simulates converting a complete model checkpoint through the entire pipeline.

        Verifies that:
        - Full conversion pipeline works without errors
        - Output checkpoint is created and loadable
        - Conversion statistics are accurate
        - Session tracking works correctly
        """
        # Setup paths
        output_path = tmp_path / "converted.pth"

        # Create converter with configuration
        config = ConversionConfig(
            strict_mode=False, export_mapping=False, log_level="INFO"
        )
        converter = WeightConverter(config)

        # Run conversion
        session = converter.convert(
            input_path=sample_checkpoint_path,
            output_path=str(output_path),
            target_model_state_dict=None,
        )

        # Verify session completed successfully
        assert session.status == ConversionStatus.COMPLETED
        assert session.end_time is not None
        assert session.duration_seconds > 0

        # Verify output file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify checkpoint can be loaded by PyTorch
        checkpoint = torch.load(output_path, weights_only=False)
        assert "model" in checkpoint
        assert "metadata" in checkpoint

        # Verify converted parameters
        state_dict = checkpoint["model"]
        assert len(state_dict) > 0

        # Verify metadata
        metadata = checkpoint["metadata"]
        assert metadata["source"] == "PaddlePaddle"
        assert "conversion_timestamp" in metadata
        assert "conversion_tool_version" in metadata
        assert "session_id" in metadata

        # Verify conversion statistics
        stats = session.statistics
        assert stats.total_parameters > 0
        assert stats.converted_count > 0
        assert stats.converted_count == metadata["conversion_stats"]["converted"]

        # Verify parameter naming conversion worked
        param_names = list(state_dict.keys())
        # Should have PyTorch-style names (not PaddlePaddle .w_0, ._mean style)
        assert any(".weight" in name for name in param_names)
        assert any(".running_mean" in name or ".bias" in name for name in param_names)
        assert not any(".w_0" in name for name in param_names)
        assert not any("._mean" in name for name in param_names)

    def test_conversion_with_mapping_export(self, sample_checkpoint_path, tmp_path):
        """Test conversion with mapping export enabled

        T029: Integration test for mapping export workflow
        Verifies that:
        - Mapping export works during conversion
        - Exported JSON contains correct information
        - Mapping can be used for debugging
        """
        # Setup paths
        output_path = tmp_path / "converted.pth"
        mapping_path = tmp_path / "mapping.json"

        # Create converter with mapping export
        config = ConversionConfig(
            strict_mode=False,
            export_mapping=True,
            export_mapping_path=str(mapping_path),
            log_level="INFO",
        )
        converter = WeightConverter(config)

        # Run conversion
        session = converter.convert(
            input_path=sample_checkpoint_path,
            output_path=str(output_path),
            target_model_state_dict=None,
        )

        # Verify conversion succeeded
        assert session.status == ConversionStatus.COMPLETED

        # Verify mapping file was created
        assert mapping_path.exists()

        # Load and verify mapping content
        with open(mapping_path, "r") as f:
            mapping_data = json.load(f)

        # Verify mapping structure
        assert "session_id" in mapping_data
        assert "source_checkpoint" in mapping_data
        assert "target_checkpoint" in mapping_data
        assert "timestamp" in mapping_data
        assert "mappings" in mapping_data
        assert "statistics" in mapping_data

        # Verify mappings content
        mappings = mapping_data["mappings"]
        assert len(mappings) > 0

        # Verify each mapping has required fields
        for mapping in mappings:
            assert "source_name" in mapping
            assert "target_name" in mapping
            assert "mapping_type" in mapping
            assert "confidence_score" in mapping
            assert "shape_compatible" in mapping

        # Verify naming conventions were applied
        paddle_names = [m["source_name"] for m in mappings]
        torch_names = [m["target_name"] for m in mappings]

        # Check for PaddlePaddle-style names in source
        assert any(".w_0" in name for name in paddle_names)
        assert any("._mean" in name for name in paddle_names)

        # Check for PyTorch-style names in target
        assert any(".weight" in name for name in torch_names)
        assert any(".running_mean" in name for name in torch_names)

    def test_conversion_with_manual_mapping(self, sample_checkpoint_path, tmp_path):
        """Test conversion with manual mapping override"""
        # Create manual mapping file
        manual_mapping_file = tmp_path / "manual_mapping.json"
        manual_mappings = {
            "version": "1.0",
            "mappings": {"backbone.conv1.w_0": "backbone.conv1.custom_weight"},
        }

        with open(manual_mapping_file, "w") as f:
            json.dump(manual_mappings, f)

        # Setup conversion
        output_path = tmp_path / "converted.pth"
        config = ConversionConfig(
            manual_mapping_file=str(manual_mapping_file), strict_mode=False
        )
        converter = WeightConverter(config)

        # Run conversion
        session = converter.convert(
            input_path=sample_checkpoint_path, output_path=str(output_path)
        )

        # Verify conversion succeeded
        assert session.status == ConversionStatus.COMPLETED

        # Load output and verify manual mapping was applied
        checkpoint = torch.load(output_path, weights_only=False)
        state_dict = checkpoint["model"]

        # Check if manual mapping was applied
        assert "backbone.conv1.custom_weight" in state_dict

    def test_conversion_preserves_values(self, sample_checkpoint_path, tmp_path):
        """Test that conversion preserves parameter values"""
        import numpy as np

        # Load original paddle checkpoint
        paddle_state = paddle.load(str(sample_checkpoint_path))

        # Convert
        output_path = tmp_path / "converted.pth"
        config = ConversionConfig(strict_mode=False)
        converter = WeightConverter(config)

        converter.convert(
            input_path=sample_checkpoint_path, output_path=str(output_path)
        )

        # Load converted checkpoint
        checkpoint = torch.load(output_path, weights_only=False)
        torch_state = checkpoint["model"]

        # Verify at least some parameters have matching values
        # (accounting for name changes)
        verified_count = 0
        for paddle_name, paddle_param in paddle_state.items():
            # Try to find corresponding torch parameter
            # Convert name using same rules
            torch_name = paddle_name
            if "._mean" in torch_name:
                torch_name = torch_name.replace("._mean", ".running_mean")
            if "._variance" in torch_name:
                torch_name = torch_name.replace("._variance", ".running_var")
            if ".w_0" in torch_name:
                torch_name = torch_name.replace(".w_0", ".weight")
            if ".b_0" in torch_name:
                torch_name = torch_name.replace(".b_0", ".bias")
            if "._scale" in torch_name:
                torch_name = torch_name.replace("._scale", ".weight")
            if "._offset" in torch_name:
                torch_name = torch_name.replace("._offset", ".bias")

            if torch_name in torch_state:
                paddle_values = paddle_param.numpy()
                torch_values = torch_state[torch_name].numpy()

                # Verify values match
                np.testing.assert_allclose(
                    torch_values,
                    paddle_values,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"Values mismatch for {paddle_name} -> {torch_name}",
                )
                verified_count += 1

        # Ensure we verified at least some parameters
        assert verified_count > 0, "No parameters were verified"

    def test_conversion_error_handling(self, tmp_path):
        """Test conversion handles errors gracefully"""
        # Try converting non-existent file
        config = ConversionConfig(strict_mode=False)
        converter = WeightConverter(config)

        output_path = tmp_path / "output.pth"

        with pytest.raises(FileNotFoundError):
            converter.convert(
                input_path="nonexistent.pdparams", output_path=str(output_path)
            )

    def test_conversion_with_shape_mismatches(self, sample_checkpoint_path, tmp_path):
        """Test conversion handles shape mismatches appropriately

        T040: Integration test for shape mismatch reporting
        """
        import torch

        # Create a target model with intentional shape mismatch
        target_state_dict = {
            "backbone.conv1.weight": torch.randn(
                32, 3, 7, 7
            ),  # Wrong number of filters
            "backbone.bn1.running_mean": torch.randn(64),
        }

        output_path = tmp_path / "converted.pth"

        # Test permissive mode (should continue despite mismatches)
        config = ConversionConfig(strict_mode=False)
        converter = WeightConverter(config)

        session = converter.convert(
            input_path=sample_checkpoint_path,
            output_path=str(output_path),
            target_model_state_dict=target_state_dict,
        )

        # Should complete but with some skipped parameters
        assert session.status == ConversionStatus.COMPLETED
        assert session.statistics.skipped_count > 0 or session.warnings

    def test_batch_conversion_continues_on_failure(
        self, sample_checkpoint_path, tmp_path
    ):
        """A broken checkpoint does not prevent later batch items from converting."""
        broken_checkpoint = tmp_path / "broken.pdparams"
        broken_checkpoint.write_bytes(b"not a paddle checkpoint")
        output_directory = tmp_path / "converted"
        mapping_directory = tmp_path / "mappings"
        converter = WeightConverter(
            ConversionConfig(
                memory_efficient_mode=True,
                batch_size=2,
            )
        )

        summary = converter.convert_batch(
            input_paths=[str(broken_checkpoint), str(sample_checkpoint_path)],
            output_directory=str(output_directory),
            mapping_directory=str(mapping_directory),
        )

        assert summary.total_count == 2
        assert summary.succeeded_count == 1
        assert summary.failed_count == 1
        failed, succeeded = summary.results
        assert failed.status == ConversionStatus.FAILED
        assert failed.error
        assert not Path(failed.output_path).exists()
        assert succeeded.status == ConversionStatus.COMPLETED
        assert succeeded.converted_count > 0
        assert Path(succeeded.output_path).is_file()
        assert Path(succeeded.mapping_path).is_file()
        assert summary.to_dict()["failed_count"] == 1

        checkpoint = torch.load(succeeded.output_path, weights_only=False)
        assert checkpoint["metadata"]["memory_efficient_mode"] is True
        assert checkpoint["metadata"]["parameter_batch_size"] == 2

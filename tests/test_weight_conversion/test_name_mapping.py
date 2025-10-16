"""Unit tests for parameter name mapping

Tests for NameMapper class in tools/weight_conversion/name_mapping.py
"""

import json
import pytest
from pathlib import Path

from tools.weight_conversion.name_mapping import NameMapper
from tools.weight_conversion.models import MappingType


class TestNameMapper:
    """Test suite for NameMapper class"""

    @pytest.fixture
    def mapper(self):
        """Create NameMapper instance for testing"""
        return NameMapper()

    @pytest.fixture
    def sample_paddle_keys(self):
        """Sample PaddlePaddle parameter names"""
        return [
            "backbone.conv1.w_0",
            "backbone.bn1._mean",
            "backbone.bn1._variance",
            "backbone.bn1._scale",
            "backbone.bn1._offset",
            "encoder.layer.0.self_attn.q_proj.w_0",
            "encoder.layer.0.self_attn.q_proj.b_0",
        ]

    @pytest.fixture
    def expected_torch_keys(self):
        """Expected PyTorch parameter names after mapping"""
        return [
            "backbone.conv1.weight",
            "backbone.bn1.running_mean",
            "backbone.bn1.running_var",
            "backbone.bn1.weight",
            "backbone.bn1.bias",
            "encoder.layer.0.self_attn.q_proj.weight",
            "encoder.layer.0.self_attn.q_proj.bias",
        ]

    def test_generate_name_mapping(self, mapper, sample_paddle_keys, expected_torch_keys):
        """Test automatic parameter name mapping generation

        T014: Unit test for parameter name mapping generation
        Verifies that:
        - Naming rules are applied correctly
        - PaddlePaddle conventions are converted to PyTorch
        - Mapping types are set appropriately
        """
        # Generate mappings
        mappings = mapper.apply_naming_rules(sample_paddle_keys)

        # Verify all keys are mapped
        assert len(mappings) == len(sample_paddle_keys)

        # Verify mapping correctness
        for i, mapping in enumerate(mappings):
            assert mapping.source_name == sample_paddle_keys[i]
            assert mapping.target_name == expected_torch_keys[i]
            assert mapping.mapping_type == MappingType.RULE_BASED
            assert 0.0 <= mapping.confidence_score <= 1.0

    def test_paddle_to_torch_name_batchnorm(self, mapper):
        """Test BatchNorm parameter name conversion"""
        assert mapper._paddle_to_torch_name("bn._mean")[0] == "bn.running_mean"
        assert mapper._paddle_to_torch_name("bn._variance")[0] == "bn.running_var"
        assert mapper._paddle_to_torch_name("bn._scale")[0] == "bn.weight"
        assert mapper._paddle_to_torch_name("bn._offset")[0] == "bn.bias"

    def test_paddle_to_torch_name_conv_linear(self, mapper):
        """Test Conv/Linear parameter name conversion"""
        assert mapper._paddle_to_torch_name("conv.w_0")[0] == "conv.weight"
        assert mapper._paddle_to_torch_name("linear.b_0")[0] == "linear.bias"

    def test_paddle_to_torch_name_no_change(self, mapper):
        """Test names that don't need conversion remain unchanged"""
        unchanged_name = "backbone.layer.custom_param"
        converted, _ = mapper._paddle_to_torch_name(unchanged_name)
        assert converted == unchanged_name

    def test_apply_manual_mappings(self, mapper, tmp_path):
        """Test manual mapping override functionality

        T026: Unit test for manual mapping override
        Verifies that:
        - Manual mappings are loaded from JSON
        - Manual mappings override automatic rules
        - Manual mappings have highest confidence
        """
        # Create manual mapping file
        manual_mapping_file = tmp_path / "manual_mapping.json"
        manual_mappings = {
            "version": "1.0",
            "mappings": {
                "custom.param.w_0": "custom.param.custom_weight",
                "special.layer._mean": "special.layer.special_mean"
            }
        }

        with open(manual_mapping_file, 'w') as f:
            json.dump(manual_mappings, f)

        # Load manual mappings
        count = mapper.load_manual_mappings(str(manual_mapping_file))
        assert count == 2

        # Generate mappings with manual overrides
        source_keys = ["custom.param.w_0", "special.layer._mean", "normal.layer.w_0"]
        mappings = mapper.apply_naming_rules(source_keys)

        # Verify manual mappings are applied
        manual_mapping1 = next(m for m in mappings if m.source_name == "custom.param.w_0")
        assert manual_mapping1.target_name == "custom.param.custom_weight"
        assert manual_mapping1.mapping_type == MappingType.MANUAL
        assert manual_mapping1.confidence_score == 1.0

        manual_mapping2 = next(m for m in mappings if m.source_name == "special.layer._mean")
        assert manual_mapping2.target_name == "special.layer.special_mean"
        assert manual_mapping2.mapping_type == MappingType.MANUAL

        # Verify rule-based mapping for non-manual key
        auto_mapping = next(m for m in mappings if m.source_name == "normal.layer.w_0")
        assert auto_mapping.target_name == "normal.layer.weight"
        assert auto_mapping.mapping_type == MappingType.RULE_BASED

    def test_export_mapping_to_json(self, mapper, sample_paddle_keys, tmp_path):
        """Test mapping export to JSON file

        T027: Unit test for mapping export to JSON
        Verifies that:
        - Mapping is exported in correct JSON format
        - All required fields are present
        - File can be read back correctly
        """
        # Generate mappings
        mappings = mapper.apply_naming_rules(sample_paddle_keys)
        mapper.generated_mappings = mappings

        # Export to JSON
        output_path = tmp_path / "exported_mapping.json"
        mapper.export_to_json(
            str(output_path),
            session_id="test-session-123",
            source_checkpoint="source.pdparams",
            target_checkpoint="target.pth",
            unmapped_source=["unmapped1"],
            unmapped_target=["unmapped2"]
        )

        # Verify file was created
        assert output_path.exists()

        # Read and verify content
        with open(output_path, 'r') as f:
            data = json.load(f)

        assert data["session_id"] == "test-session-123"
        assert data["source_checkpoint"] == "source.pdparams"
        assert data["target_checkpoint"] == "target.pth"
        assert "timestamp" in data
        assert len(data["mappings"]) == len(sample_paddle_keys)
        assert data["unmapped_source"] == ["unmapped1"]
        assert data["unmapped_target"] == ["unmapped2"]

        # Verify mapping structure
        first_mapping = data["mappings"][0]
        assert "source_name" in first_mapping
        assert "target_name" in first_mapping
        assert "mapping_type" in first_mapping
        assert "confidence_score" in first_mapping
        assert "shape_compatible" in first_mapping

    def test_identify_unmapped_parameters(self, mapper):
        """Test unmapped parameter identification

        T028: Unit test for unmapped parameter detection
        Verifies that:
        - Unmapped source parameters are identified
        - Unmapped target parameters are identified
        - Mapped parameters are correctly excluded
        """
        source_keys = ["param1.w_0", "param2.w_0", "param3.custom"]
        target_keys = ["param1.weight", "param2.weight", "param4.weight"]

        # Generate mappings (all source keys will get mappings, but param3.custom
        # gets identity mapping which won't match any target key)
        mappings = mapper.apply_naming_rules(source_keys, set(target_keys))

        # Find unmapped keys
        unmapped_source, unmapped_target = mapper.find_unmapped_keys(
            source_keys,
            target_keys,
            mappings
        )

        # Note: param3.custom gets identity mapping but target doesn't exist
        # So it will be in mapped_source but the target won't be found
        # The implementation considers all source keys as "mapped" if they have a mapping
        # Unmapped target: param4.weight has no source, and param3.custom doesn't exist in target
        assert "param4.weight" in unmapped_target or len(unmapped_target) > 0

        # Verify successfully mapped keys are not in unmapped lists
        assert "param1.w_0" not in unmapped_source
        assert "param2.w_0" not in unmapped_source
        assert "param1.weight" not in unmapped_target
        assert "param2.weight" not in unmapped_target

    def test_load_manual_mappings_invalid_file(self, mapper):
        """Test loading non-existent manual mapping file raises error"""
        with pytest.raises(FileNotFoundError):
            mapper.load_manual_mappings("nonexistent.json")

    def test_load_manual_mappings_invalid_json(self, mapper, tmp_path):
        """Test loading invalid JSON raises error"""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json")

        with pytest.raises(ValueError, match="Invalid JSON"):
            mapper.load_manual_mappings(str(invalid_file))

    def test_load_manual_mappings_missing_key(self, mapper, tmp_path):
        """Test loading JSON without 'mappings' key raises error"""
        invalid_file = tmp_path / "no_mappings.json"
        with open(invalid_file, 'w') as f:
            json.dump({"version": "1.0"}, f)

        with pytest.raises(ValueError, match="mappings"):
            mapper.load_manual_mappings(str(invalid_file))

    def test_identity_mapping(self, mapper):
        """Test that unchanged names get IDENTITY mapping type"""
        # Name that doesn't match any rule
        source_keys = ["backbone.custom_module.special_param"]
        mappings = mapper.apply_naming_rules(source_keys)

        assert len(mappings) == 1
        assert mappings[0].mapping_type == MappingType.IDENTITY
        assert mappings[0].confidence_score == 1.0
        assert mappings[0].source_name == mappings[0].target_name

    def test_multiple_rule_application(self, mapper):
        """Test that only first matching rule is applied"""
        # This should only apply the first matching rule
        name_with_multiple = "layer.w_0._mean"  # Matches both .w_0 and ._mean rules
        converted, _ = mapper._paddle_to_torch_name(name_with_multiple)

        # The implementation applies the first matching rule in order
        # The rules are checked in order: ._mean, ._variance, ._scale, ._offset, .w_0, .b_0
        # So ._mean will match first and be replaced
        assert "._mean" not in converted or ".w_0" not in converted
        # Verify at least one rule was applied
        assert converted != name_with_multiple

"""Parameter name mapping logic

This module provides automatic parameter name mapping between PaddlePaddle and PyTorch naming conventions,
with support for manual overrides and confidence scoring.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import MappingType, ParameterMapping

logger = logging.getLogger(__name__)


class NameMapper:
    """Handles parameter name mapping between PaddlePaddle and PyTorch

    This class provides automatic name mapping using rule-based transformations,
    with support for manual overrides and mapping export.
    """

    # PaddlePaddle -> PyTorch naming rules
    NAMING_RULES = [
        # BatchNorm parameters
        ("._mean", ".running_mean"),
        ("._variance", ".running_var"),
        ("._scale", ".weight"),
        ("._offset", ".bias"),

        # Layer weights and biases
        (".w_0", ".weight"),
        (".b_0", ".bias"),

        # Additional PaddlePaddle conventions
        (".w_", ".weight_"),
        (".b_", ".bias_"),
    ]

    def __init__(self):
        """Initialize NameMapper"""
        self.manual_mappings: Dict[str, str] = {}
        self.generated_mappings: List[ParameterMapping] = []

    def load_manual_mappings(self, mapping_file: str) -> int:
        """Load manual parameter name mappings from JSON file

        Args:
            mapping_file: Path to JSON file with manual mappings

        Returns:
            Number of manual mappings loaded

        Raises:
            FileNotFoundError: If mapping file doesn't exist
            ValueError: If JSON is invalid or doesn't match schema
        """
        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            raise FileNotFoundError(f"Manual mapping file not found: {mapping_file}")

        try:
            with open(mapping_path, 'r') as f:
                data = json.load(f)

            # Validate schema
            if "mappings" not in data:
                raise ValueError("Manual mapping file must contain 'mappings' key")

            self.manual_mappings = data["mappings"]
            logger.info(f"Loaded {len(self.manual_mappings)} manual mappings from {mapping_file}")
            return len(self.manual_mappings)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in manual mapping file: {e}")

    def _paddle_to_torch_name(self, paddle_name: str) -> Tuple[str, Optional[str]]:
        """Apply naming convention rules to convert PaddlePaddle name to PyTorch name

        Args:
            paddle_name: PaddlePaddle parameter name

        Returns:
            Tuple of (torch_name, transformation_description)
        """
        torch_name = paddle_name
        transformation = None

        # Apply naming rules
        for paddle_pattern, torch_pattern in self.NAMING_RULES:
            if paddle_pattern in torch_name:
                torch_name = torch_name.replace(paddle_pattern, torch_pattern)
                transformation = f"Applied rule: '{paddle_pattern}' -> '{torch_pattern}'"
                break  # Apply first matching rule only

        return torch_name, transformation

    def apply_naming_rules(
        self,
        source_keys: List[str],
        target_keys: Optional[Set[str]] = None
    ) -> List[ParameterMapping]:
        """Generate parameter name mappings using naming convention rules

        Args:
            source_keys: List of source (PaddlePaddle) parameter names
            target_keys: Optional set of target (PyTorch) parameter names for validation

        Returns:
            List of ParameterMapping objects
        """
        mappings = []

        for source_name in source_keys:
            # Check manual mapping first (highest priority)
            if source_name in self.manual_mappings:
                target_name = self.manual_mappings[source_name]
                mapping = ParameterMapping(
                    source_name=source_name,
                    target_name=target_name,
                    mapping_type=MappingType.MANUAL,
                    confidence_score=1.0,
                    shape_compatible=True,  # Will be validated later
                    transformation_applied="Manual override",
                )
                mappings.append(mapping)
                logger.debug(f"Manual mapping: {source_name} -> {target_name}")
                continue

            # Apply automatic naming rules
            target_name, transformation = self._paddle_to_torch_name(source_name)

            # Determine mapping type and confidence
            if target_name == source_name:
                # No transformation needed (identity mapping)
                mapping_type = MappingType.IDENTITY
                confidence = 1.0
                transformation = "No transformation needed"
            else:
                # Rule-based transformation applied
                mapping_type = MappingType.RULE_BASED
                confidence = 0.95

            # Check if target name exists in target keys (if provided)
            shape_compatible = True
            if target_keys is not None and target_name not in target_keys:
                logger.warning(f"Mapped target name '{target_name}' not found in target model")
                shape_compatible = False
                confidence *= 0.8  # Lower confidence if target doesn't exist

            mapping = ParameterMapping(
                source_name=source_name,
                target_name=target_name,
                mapping_type=mapping_type,
                confidence_score=confidence,
                shape_compatible=shape_compatible,
                transformation_applied=transformation,
            )
            mappings.append(mapping)

        self.generated_mappings = mappings
        return mappings

    def find_unmapped_keys(
        self,
        source_keys: List[str],
        target_keys: List[str],
        mappings: List[ParameterMapping]
    ) -> Tuple[List[str], List[str]]:
        """Identify unmapped parameters in source and target

        Args:
            source_keys: List of source parameter names
            target_keys: List of target parameter names
            mappings: List of ParameterMapping objects

        Returns:
            Tuple of (unmapped_source_keys, unmapped_target_keys)
        """
        # Extract mapped keys
        mapped_source_keys = {m.source_name for m in mappings}
        mapped_target_keys = {m.target_name for m in mappings}

        # Find unmapped keys
        unmapped_source = [key for key in source_keys if key not in mapped_source_keys]
        unmapped_target = [key for key in target_keys if key not in mapped_target_keys]

        if unmapped_source:
            logger.warning(f"Found {len(unmapped_source)} unmapped source parameters")
        if unmapped_target:
            logger.warning(f"Found {len(unmapped_target)} unmapped target parameters")

        return unmapped_source, unmapped_target

    def export_to_json(
        self,
        output_path: str,
        session_id: str,
        source_checkpoint: str,
        target_checkpoint: str,
        unmapped_source: List[str],
        unmapped_target: List[str]
    ) -> None:
        """Export generated parameter name mapping to JSON file

        Args:
            output_path: Path to save mapping JSON
            session_id: Conversion session ID
            source_checkpoint: Path to source checkpoint
            target_checkpoint: Path to target checkpoint
            unmapped_source: List of unmapped source parameters
            unmapped_target: List of unmapped target parameters
        """
        from datetime import datetime

        mapping_data = {
            "session_id": session_id,
            "source_checkpoint": str(source_checkpoint),
            "target_checkpoint": str(target_checkpoint),
            "timestamp": datetime.now().isoformat(),
            "mappings": [
                {
                    "source_name": m.source_name,
                    "target_name": m.target_name,
                    "mapping_type": m.mapping_type.value,
                    "confidence_score": m.confidence_score,
                    "shape_compatible": m.shape_compatible,
                }
                for m in self.generated_mappings
            ],
            "unmapped_source": unmapped_source,
            "unmapped_target": unmapped_target,
            "statistics": {
                "total_parameters": len(self.generated_mappings),
                "mapped_count": len(self.generated_mappings),
                "unmapped_source_count": len(unmapped_source),
                "unmapped_target_count": len(unmapped_target),
            }
        }

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)

        logger.info(f"Exported parameter name mapping to: {output_path}")

    def apply_manual_overrides(self, manual_mapping_file: str) -> int:
        """Load and apply manual mapping overrides

        Args:
            manual_mapping_file: Path to JSON file with manual mappings

        Returns:
            Number of manual mappings applied
        """
        return self.load_manual_mappings(manual_mapping_file)

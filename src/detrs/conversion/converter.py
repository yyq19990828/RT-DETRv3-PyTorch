"""Weight converter core implementation

This module provides the main WeightConverter class for converting model weights
between PaddlePaddle and PyTorch formats.
"""

import gc
import hashlib
import logging
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .models import (
    BatchConversionResult,
    BatchConversionSummary,
    CheckpointFile,
    CheckpointFormat,
    ConversionConfig,
    ConversionSession,
    ConversionStatistics,
    ConversionStatus,
    Framework,
    ParameterMapping,
)
from .name_mapping import NameMapper
from .tensor_utils import convert_paddle_to_torch_tensor, validate_tensor_shape

logger = logging.getLogger(__name__)


class WeightConverter:
    """Main class for converting model weights between frameworks

    This class orchestrates the entire conversion process:
    1. Load source checkpoint (PaddlePaddle)
    2. Generate parameter name mappings
    3. Convert tensors (Paddle -> NumPy -> PyTorch)
    4. Save target checkpoint (PyTorch)
    """

    def __init__(self, config: Optional[ConversionConfig] = None):
        """Initialize WeightConverter

        Args:
            config: Conversion configuration (uses defaults if None)
        """
        self.config = config or ConversionConfig()
        self.name_mapper = NameMapper()
        self.session: Optional[ConversionSession] = None

    def load_paddle_checkpoint(
        self, checkpoint_path: str
    ) -> Tuple[Dict[str, Any], CheckpointFile]:
        """Load PaddlePaddle checkpoint file

        Args:
            checkpoint_path: Path to .pdparams file

        Returns:
            Tuple of (state_dict, CheckpointFile metadata)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is invalid or cannot be loaded
        """
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        if checkpoint_path_obj.suffix != ".pdparams":
            logger.warning(f"Expected .pdparams file, got {checkpoint_path_obj.suffix}")

        try:
            import paddle

            logger.info(f"Loading PaddlePaddle checkpoint from {checkpoint_path}")
            state_dict = paddle.load(str(checkpoint_path_obj))

            if not isinstance(state_dict, dict):
                raise ValueError(f"Expected state dict, got {type(state_dict)}")

            # Create CheckpointFile metadata
            file_size = checkpoint_path_obj.stat().st_size
            checksum = self._compute_checksum(checkpoint_path)

            checkpoint_file = CheckpointFile(
                file_path=str(checkpoint_path_obj.absolute()),
                format=CheckpointFormat.PDPARAMS,
                file_size_bytes=file_size,
                framework=Framework.PADDLEPADDLE,
                checksum=checksum,
            )

            logger.info(
                f"Loaded {len(state_dict)} parameters from PaddlePaddle checkpoint"
            )
            logger.info(f"Checkpoint size: {file_size / (1024 * 1024):.2f} MB")

            return state_dict, checkpoint_file

        except ImportError:
            raise ValueError("PaddlePaddle is not installed. Run: uv sync --extra dev")
        except Exception as e:
            raise ValueError(f"Failed to load PaddlePaddle checkpoint: {e}")

    def save_torch_checkpoint(
        self,
        state_dict: Dict[str, Any],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointFile:
        """Save PyTorch checkpoint file

        Args:
            state_dict: PyTorch state dict
            output_path: Path to save .pth file
            metadata: Optional metadata to embed in checkpoint

        Returns:
            CheckpointFile metadata

        Raises:
            ValueError: If checkpoint cannot be saved
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None

        try:
            import torch

            # Prepare checkpoint data
            checkpoint_data = {
                "model": state_dict,
            }

            # Add metadata if provided
            if metadata:
                checkpoint_data["metadata"] = metadata

            # Save to the destination directory, then publish atomically.
            logger.info(f"Saving converted checkpoint to {output_path}")
            with tempfile.NamedTemporaryFile(
                dir=output_path_obj.parent,
                prefix=f".{output_path_obj.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            torch.save(checkpoint_data, temporary_path)
            temporary_path.replace(output_path_obj)
            temporary_path = None

            # Create CheckpointFile metadata
            file_size = output_path_obj.stat().st_size
            checksum = self._compute_checksum(output_path)

            checkpoint_file = CheckpointFile(
                file_path=str(output_path_obj.absolute()),
                format=CheckpointFormat.PTH,
                file_size_bytes=file_size,
                framework=Framework.PYTORCH,
                checksum=checksum,
                metadata=metadata,
            )

            logger.info(f"Saved checkpoint: {file_size / (1024 * 1024):.2f} MB")
            return checkpoint_file

        except Exception as e:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise ValueError(f"Failed to save PyTorch checkpoint: {e}")

    def convert_tensor(
        self,
        paddle_tensor: Any,
        param_name: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        transpose: Optional[bool] = None,
    ) -> Any:
        """Convert single parameter tensor from PaddlePaddle to PyTorch

        Args:
            paddle_tensor: PaddlePaddle tensor
            param_name: Parameter name (for logging)
            expected_shape: Expected target shape (for validation)
            transpose: Explicit target-aware Linear transpose decision

        Returns:
            PyTorch tensor or None if conversion fails in permissive mode

        Raises:
            ValueError: If conversion fails or shape validation fails in strict mode
        """
        try:
            # Convert tensor
            torch_tensor = convert_paddle_to_torch_tensor(
                paddle_tensor,
                param_name,
                transpose=transpose,
            )

            # Validate shape if expected shape provided
            if expected_shape is not None:
                shape_matches = validate_tensor_shape(
                    torch_tensor.shape,
                    expected_shape,
                    param_name,
                    strict=self.config.strict_mode,
                )

                # In permissive mode, return None if shape doesn't match
                if not shape_matches and not self.config.strict_mode:
                    return None

            return torch_tensor

        except Exception as e:
            if self.config.strict_mode:
                raise ValueError(f"Failed to convert parameter '{param_name}': {e}")
            else:
                logger.warning(f"Skipping parameter '{param_name}': {e}")
                return None

    def convert_state_dict(
        self,
        paddle_state_dict: Dict[str, Any],
        target_state_dict: Optional[Dict[str, Any]] = None,
        mappings: Optional[List[ParameterMapping]] = None,
        transpose_target_keys: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, Any], ConversionStatistics]:
        """Convert entire state dict from PaddlePaddle to PyTorch

        Args:
            paddle_state_dict: Source PaddlePaddle state dict
            target_state_dict: Optional target PyTorch state dict for shape validation
            mappings: Optional list of parameter mappings (generated if None)
            transpose_target_keys: Target keys backed by PyTorch Linear modules

        Returns:
            Tuple of (torch_state_dict, ConversionStatistics)
        """
        logger.info("Converting state dict...")

        # Generate mappings if not provided
        if mappings is None:
            source_keys = list(paddle_state_dict.keys())
            target_keys = set(target_state_dict.keys()) if target_state_dict else None
            mappings = self.name_mapper.apply_naming_rules(source_keys, target_keys)

        # Initialize statistics
        stats = ConversionStatistics()
        stats.total_parameters = len(mappings)

        # Convert parameters
        torch_state_dict = {}
        memory_batch_size = self.config.batch_size or 64

        for i, mapping in enumerate(mappings):
            source_name = mapping.source_name
            target_name = mapping.target_name

            # Check if source parameter exists
            if source_name not in paddle_state_dict:
                logger.warning(f"Source parameter not found: {source_name}")
                stats.failed_count += 1
                continue

            paddle_tensor = paddle_state_dict[source_name]

            # Get expected shape if target state dict provided
            expected_shape = None
            if target_state_dict and target_name in target_state_dict:
                target_value = target_state_dict[target_name]
                expected_shape = tuple(
                    target_value.shape
                    if hasattr(target_value, "shape")
                    else target_value
                )

            # Convert tensor
            transpose = (
                None
                if transpose_target_keys is None
                else target_name in transpose_target_keys
            )
            torch_tensor = self.convert_tensor(
                paddle_tensor,
                source_name,
                expected_shape,
                transpose=transpose,
            )

            if torch_tensor is not None:
                torch_state_dict[target_name] = torch_tensor
                stats.converted_count += 1
            else:
                stats.skipped_count += 1

            if self.config.memory_efficient_mode:
                paddle_state_dict.pop(source_name, None)
                del paddle_tensor
                if (i + 1) % memory_batch_size == 0 or i + 1 == len(mappings):
                    gc.collect()

            # Progress logging (every 100 parameters)
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(mappings)} ({100 * (i + 1) / len(mappings):.1f}%)"
                )

        logger.info(
            f"Conversion complete: {stats.converted_count}/{stats.total_parameters} parameters"
        )
        return torch_state_dict, stats

    def convert(
        self,
        input_path: str,
        output_path: str,
        target_model_state_dict: Optional[Dict[str, Any]] = None,
        transpose_target_keys: Optional[Set[str]] = None,
    ) -> ConversionSession:
        """Convert weights from PaddlePaddle to PyTorch (full workflow)

        This is the main entry point for weight conversion. It orchestrates:
        1. Load source checkpoint
        2. Generate parameter name mappings
        3. Convert tensors
        4. Save target checkpoint
        5. Export mapping (if configured)

        Args:
            input_path: Path to source .pdparams file
            output_path: Path to target .pth file
            target_model_state_dict: Optional target model state dict for validation
            transpose_target_keys: Target keys backed by PyTorch Linear modules

        Returns:
            ConversionSession with conversion results

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If conversion fails
        """
        from datetime import datetime

        # Initialize session
        self.session = ConversionSession(config=self.config)
        self.session.status = ConversionStatus.LOADING_SOURCE

        try:
            # Step 1: Load source checkpoint
            logger.info("=== Step 1: Load source checkpoint ===")
            paddle_state_dict, source_checkpoint = self.load_paddle_checkpoint(
                input_path
            )
            self.session.source_checkpoint = source_checkpoint

            # Step 2: Generate parameter name mappings
            logger.info("=== Step 2: Generate parameter name mappings ===")
            self.session.status = ConversionStatus.GENERATING_MAPPING

            # Load manual mappings if provided
            if self.config.manual_mapping_file:
                self.name_mapper.apply_manual_overrides(self.config.manual_mapping_file)

            # Generate mappings
            source_keys = list(paddle_state_dict.keys())
            target_keys = (
                set(target_model_state_dict.keys()) if target_model_state_dict else None
            )
            mappings = self.name_mapper.apply_naming_rules(source_keys, target_keys)

            # Find unmapped keys
            target_keys_list = list(target_keys) if target_keys else []
            unmapped_source, unmapped_target = self.name_mapper.find_unmapped_keys(
                source_keys, target_keys_list, mappings
            )

            logger.info(f"Generated {len(mappings)} parameter mappings")
            if unmapped_source:
                logger.warning(f"Unmapped source parameters: {len(unmapped_source)}")
            if unmapped_target:
                logger.warning(f"Unmapped target parameters: {len(unmapped_target)}")

            # Step 3: Convert state dict
            logger.info("=== Step 3: Convert parameters ===")
            self.session.status = ConversionStatus.CONVERTING
            torch_state_dict, stats = self.convert_state_dict(
                paddle_state_dict,
                target_model_state_dict,
                mappings,
                transpose_target_keys,
            )

            # Update session statistics
            self.session.statistics = stats
            self.session.statistics.unmapped_source_keys = unmapped_source
            self.session.statistics.unmapped_target_keys = unmapped_target

            # Step 4: Save target checkpoint
            logger.info("=== Step 4: Save target checkpoint ===")
            self.session.status = ConversionStatus.SAVING

            # Prepare metadata
            metadata = {
                "source": "PaddlePaddle",
                "source_checkpoint": str(input_path),
                "source_checkpoint_size_bytes": source_checkpoint.file_size_bytes,
                "source_checkpoint_sha256": source_checkpoint.checksum,
                "conversion_timestamp": datetime.now().isoformat(),
                "conversion_tool_version": "0.1.0",
                "session_id": self.session.session_id,
                "memory_efficient_mode": self.config.memory_efficient_mode,
                "parameter_batch_size": self.config.batch_size,
                "conversion_stats": {
                    "total": stats.total_parameters,
                    "converted": stats.converted_count,
                    "skipped": stats.skipped_count,
                    "unmapped_source": len(unmapped_source),
                    "unmapped_target": len(unmapped_target),
                },
            }

            if self.config.output_metadata:
                metadata.update(self.config.output_metadata)

            target_checkpoint = self.save_torch_checkpoint(
                torch_state_dict, output_path, metadata
            )
            self.session.target_checkpoint = target_checkpoint

            # Step 5: Export mapping if configured
            if self.config.export_mapping and self.config.export_mapping_path:
                logger.info("=== Step 5: Export parameter mapping ===")
                self.name_mapper.export_to_json(
                    self.config.export_mapping_path,
                    self.session.session_id,
                    str(input_path),
                    str(output_path),
                    unmapped_source,
                    unmapped_target,
                )

            # Mark as completed
            self.session.status = ConversionStatus.COMPLETED
            self.session.end_time = datetime.now()

            logger.info("=== Conversion completed successfully ===")
            logger.info(f"Duration: {self.session.duration_seconds:.2f} seconds")
            logger.info(f"Converted: {stats.converted_count}/{stats.total_parameters}")
            if stats.skipped_count > 0:
                logger.warning(f"Skipped: {stats.skipped_count} parameters")

            return self.session

        except Exception as e:
            self.session.add_error(str(e))
            self.session.end_time = datetime.now()
            logger.error(f"Conversion failed: {e}")
            raise

    def convert_batch(
        self,
        input_paths: Iterable[str],
        output_directory: str,
        target_model_state_dict: Optional[Dict[str, Any]] = None,
        transpose_target_keys: Optional[Set[str]] = None,
        mapping_directory: Optional[str] = None,
        overwrite: bool = False,
    ) -> BatchConversionSummary:
        """Convert checkpoints independently and continue after individual failures."""
        output_directory_path = Path(output_directory)
        output_directory_path.mkdir(parents=True, exist_ok=True)
        mapping_directory_path = (
            Path(mapping_directory) if mapping_directory is not None else None
        )
        if mapping_directory_path is not None:
            mapping_directory_path.mkdir(parents=True, exist_ok=True)

        summary = BatchConversionSummary(output_directory=str(output_directory_path))
        for input_value in input_paths:
            input_path = Path(input_value)
            output_path = output_directory_path / f"{input_path.stem}.pth"
            mapping_path = (
                mapping_directory_path / f"{input_path.stem}.mapping.json"
                if mapping_directory_path is not None
                else None
            )
            output_existed = output_path.exists()
            mapping_existed = mapping_path is not None and mapping_path.exists()

            if (output_existed or mapping_existed) and not overwrite:
                existing_kind = "Output" if output_existed else "Mapping"
                summary.results.append(
                    BatchConversionResult(
                        source_path=str(input_path),
                        output_path=str(output_path),
                        mapping_path=str(mapping_path) if mapping_path else None,
                        status=ConversionStatus.FAILED,
                        error=f"{existing_kind} file already exists",
                    )
                )
                continue

            file_config = replace(
                self.config,
                export_mapping=mapping_path is not None,
                export_mapping_path=str(mapping_path) if mapping_path else None,
            )
            file_converter = WeightConverter(file_config)
            try:
                session = file_converter.convert(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    target_model_state_dict=target_model_state_dict,
                    transpose_target_keys=transpose_target_keys,
                )
                summary.results.append(
                    BatchConversionResult(
                        source_path=str(input_path),
                        output_path=str(output_path),
                        mapping_path=str(mapping_path) if mapping_path else None,
                        status=session.status,
                        session_id=session.session_id,
                        duration_seconds=session.duration_seconds,
                        converted_count=session.statistics.converted_count,
                        skipped_count=session.statistics.skipped_count,
                    )
                )
            except Exception as error:
                if not output_existed:
                    output_path.unlink(missing_ok=True)
                if mapping_path is not None and not mapping_existed:
                    mapping_path.unlink(missing_ok=True)
                failed_session = file_converter.session
                summary.results.append(
                    BatchConversionResult(
                        source_path=str(input_path),
                        output_path=str(output_path),
                        mapping_path=str(mapping_path) if mapping_path else None,
                        status=ConversionStatus.FAILED,
                        session_id=(
                            failed_session.session_id if failed_session else None
                        ),
                        duration_seconds=(
                            failed_session.duration_seconds if failed_session else 0.0
                        ),
                        error=str(error),
                    )
                )
                logger.error("Batch item failed for %s: %s", input_path, error)

        summary.finish()
        return summary

    @staticmethod
    def _compute_checksum(file_path: str) -> str:
        """Compute SHA-256 checksum of file

        Args:
            file_path: Path to file

        Returns:
            SHA-256 checksum as hex string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

"""Data models for weight conversion

This module defines all data structures used in the weight conversion process,
including checkpoint files, parameters, mappings, and conversion statistics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


class CheckpointFormat(Enum):
    """Checkpoint file format enumeration"""

    PDPARAMS = "pdparams"
    PTH = "pth"


class Framework(Enum):
    """Deep learning framework enumeration"""

    PADDLEPADDLE = "PaddlePaddle"
    PYTORCH = "PyTorch"


class MappingType(Enum):
    """Parameter name mapping type"""

    MANUAL = "MANUAL"
    RULE_BASED = "RULE_BASED"
    FUZZY_MATCH = "FUZZY_MATCH"
    IDENTITY = "IDENTITY"


class ConversionStatus(Enum):
    """Conversion session status"""

    INITIALIZING = "INITIALIZING"
    LOADING_SOURCE = "LOADING_SOURCE"
    GENERATING_MAPPING = "GENERATING_MAPPING"
    CONVERTING = "CONVERTING"
    VALIDATING = "VALIDATING"
    SAVING = "SAVING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ShapeMismatchSeverity(Enum):
    """Shape mismatch severity level"""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class CheckpointFile:
    """Represents a serialized model checkpoint file on disk

    Attributes:
        file_path: Absolute path to checkpoint file
        format: File format (pdparams or pth)
        file_size_bytes: Size of file in bytes
        framework: Source framework (PaddlePaddle or PyTorch)
        checksum: File hash for integrity verification (optional)
        checksum_algorithm: Algorithm used for ``checksum``
        metadata: Embedded metadata from checkpoint (optional)
    """

    file_path: str
    format: CheckpointFormat
    file_size_bytes: int
    framework: Framework
    checksum: Optional[str] = None
    checksum_algorithm: str = "sha256"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Parameter:
    """Represents a single named parameter (weight or bias tensor)

    Attributes:
        name: Fully qualified parameter name
        tensor: The actual parameter tensor data (framework-specific)
        shape: Dimensions of the tensor
        dtype: Data type of the tensor
        device: Device location (cpu, cuda:0, etc.)
        requires_grad: Whether parameter requires gradient computation
        is_buffer: Whether parameter is a buffer vs trainable weight
    """

    name: str
    tensor: Any  # Framework-specific tensor type
    shape: Tuple[int, ...]
    dtype: str
    device: str = "cpu"
    requires_grad: bool = False
    is_buffer: bool = False


@dataclass
class ParameterMapping:
    """Represents mapping between source and target parameters

    Attributes:
        source_name: PaddlePaddle parameter name
        target_name: PyTorch parameter name
        mapping_type: How mapping was determined
        confidence_score: Mapping confidence 0.0-1.0
        transformation_applied: Description of name transformation (optional)
        shape_compatible: Whether source and target shapes match
        notes: Additional information or warnings (optional)
    """

    source_name: str
    target_name: str
    mapping_type: MappingType
    confidence_score: float
    shape_compatible: bool
    transformation_applied: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate confidence score range"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}"
            )


@dataclass
class ShapeMismatch:
    """Records a shape incompatibility between source and target

    Attributes:
        parameter_name: Name of the parameter
        source_shape: Shape in source checkpoint
        target_shape: Shape in target model
        severity: Impact level (ERROR or WARNING)
        suggested_fix: Recommended action to resolve mismatch
    """

    parameter_name: str
    source_shape: Tuple[int, ...]
    target_shape: Tuple[int, ...]
    severity: ShapeMismatchSeverity
    suggested_fix: str


@dataclass
class DtypeConversion:
    """Records a data type conversion applied during transfer

    Attributes:
        parameter_name: Name of the parameter
        source_dtype: Original data type
        target_dtype: Converted data type
        precision_loss: Whether conversion loses precision
        justification: Reason for dtype conversion
    """

    parameter_name: str
    source_dtype: str
    target_dtype: str
    precision_loss: bool
    justification: str


@dataclass
class ConversionStatistics:
    """Tracks metrics and statistics for a conversion session

    Attributes:
        total_parameters: Total number of parameters in source checkpoint
        converted_count: Number of parameters successfully converted
        skipped_count: Number of parameters skipped
        failed_count: Number of parameters that failed conversion
        unmapped_source_keys: List of unmapped source parameter names
        unmapped_target_keys: List of unpopulated target parameter names
        shape_mismatches: List of parameters with shape incompatibilities
        dtype_conversions: List of parameters with dtype changes
        total_source_size_bytes: Total size of source parameters in bytes
        total_target_size_bytes: Total size of target parameters in bytes
        peak_memory_usage_bytes: Maximum memory used during conversion
    """

    total_parameters: int = 0
    converted_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    unmapped_source_keys: List[str] = field(default_factory=list)
    unmapped_target_keys: List[str] = field(default_factory=list)
    shape_mismatches: List[ShapeMismatch] = field(default_factory=list)
    dtype_conversions: List[DtypeConversion] = field(default_factory=list)
    total_source_size_bytes: int = 0
    total_target_size_bytes: int = 0
    peak_memory_usage_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (target size / source size)"""
        if self.total_source_size_bytes == 0:
            return 0.0
        return self.total_target_size_bytes / self.total_source_size_bytes

    @property
    def conversion_success_rate(self) -> float:
        """Calculate conversion success rate"""
        if self.total_parameters == 0:
            return 0.0
        return self.converted_count / self.total_parameters

    @property
    def mapping_coverage(self) -> float:
        """Calculate mapping coverage rate"""
        if self.total_parameters == 0:
            return 0.0
        unmapped_count = len(self.unmapped_source_keys)
        return 1.0 - (unmapped_count / self.total_parameters)


@dataclass
class ConversionConfig:
    """Configuration settings for a conversion session

    Attributes:
        strict_mode: Fail on shape/dtype mismatches vs skip them
        validate_values: Perform numerical validation after conversion
        validation_tolerance: Numerical tolerance for value comparison
        manual_mapping_file: Path to JSON file with manual mappings
        export_mapping: Export generated mapping to file
        export_mapping_path: Path to save mapping JSON
        memory_efficient_mode: Use chunked processing to reduce memory
        batch_size: Number of parameters to process per chunk
        log_level: Logging verbosity
        output_metadata: Additional metadata to embed in output
    """

    strict_mode: bool = False
    validate_values: bool = False
    validation_tolerance: float = 1e-5
    manual_mapping_file: Optional[str] = None
    export_mapping: bool = False
    export_mapping_path: Optional[str] = None
    memory_efficient_mode: bool = False
    batch_size: Optional[int] = None
    log_level: str = "INFO"
    output_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration"""
        if self.validation_tolerance <= 0:
            raise ValueError(
                f"Validation tolerance must be positive, got {self.validation_tolerance}"
            )
        if self.export_mapping and not self.export_mapping_path:
            raise ValueError(
                "export_mapping_path must be specified when export_mapping is True"
            )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class ConversionSession:
    """Represents a single execution of the weight conversion process

    Attributes:
        session_id: Unique identifier for this conversion
        source_checkpoint: Source CheckpointFile instance
        target_checkpoint: Target CheckpointFile instance (created during conversion)
        start_time: Conversion start timestamp
        end_time: Conversion completion timestamp
        status: Current conversion status
        config: Conversion configuration settings
        statistics: Conversion statistics
        errors: List of errors encountered
        warnings: List of warnings generated
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    source_checkpoint: Optional[CheckpointFile] = None
    target_checkpoint: Optional[CheckpointFile] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: ConversionStatus = ConversionStatus.INITIALIZING
    config: ConversionConfig = field(default_factory=ConversionConfig)
    statistics: ConversionStatistics = field(default_factory=ConversionStatistics)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Calculate total conversion time in seconds"""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

    def add_error(self, error: str):
        """Add error message to session

        Args:
            error: Error message to record
        """
        self.errors.append(error)
        self.status = ConversionStatus.FAILED

    def add_warning(self, warning: str):
        """Add warning message to session

        Args:
            warning: Warning message to record
        """
        self.warnings.append(warning)


@dataclass
class BatchConversionResult:
    """Outcome for one checkpoint in a batch conversion."""

    source_path: str
    output_path: str
    status: ConversionStatus
    mapping_path: Optional[str] = None
    session_id: Optional[str] = None
    duration_seconds: float = 0.0
    converted_count: int = 0
    skipped_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable result."""
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "mapping_path": self.mapping_path,
            "status": self.status.value,
            "session_id": self.session_id,
            "duration_seconds": self.duration_seconds,
            "converted_count": self.converted_count,
            "skipped_count": self.skipped_count,
            "error": self.error,
        }


@dataclass
class BatchConversionSummary:
    """Aggregate outcome for an isolated multi-checkpoint conversion."""

    output_directory: str
    results: List[BatchConversionResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def succeeded_count(self) -> int:
        return sum(
            result.status == ConversionStatus.COMPLETED for result in self.results
        )

    @property
    def failed_count(self) -> int:
        return self.total_count - self.succeeded_count

    @property
    def duration_seconds(self) -> float:
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()

    def finish(self) -> None:
        self.end_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "output_directory": self.output_directory,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "total_count": self.total_count,
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "results": [result.to_dict() for result in self.results],
        }

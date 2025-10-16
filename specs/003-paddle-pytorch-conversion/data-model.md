# Data Model: Paddle to PyTorch Weight Conversion

**Feature**: 003-paddle-pytorch-conversion
**Date**: 2025-10-16
**Phase**: Phase 1 - Design

## Overview

This document defines the data structures, entities, and their relationships for the weight conversion tool. The tool operates on checkpoint files (source and target), parameter name mappings, conversion statistics, and validation results.

## Core Entities

### 1. CheckpointFile

Represents a serialized model checkpoint file on disk.

**Attributes**:
- `file_path`: Absolute path to the checkpoint file (string)
- `format`: File format ('pdparams' or 'pth') (enum)
- `file_size_bytes`: Size of the file in bytes (integer)
- `framework`: Source framework ('PaddlePaddle' or 'PyTorch') (enum)
- `checksum`: MD5 hash of the file for integrity verification (string, optional)
- `metadata`: Embedded metadata from checkpoint (dict, optional)
  - `model_name`: Name of the model architecture
  - `training_config`: Training hyperparameters (if available)
  - `creation_date`: Timestamp of checkpoint creation

**Relationships**:
- Source checkpoint → Conversion → Target checkpoint (1:1)
- One checkpoint file contains many Parameters

**Validation Rules**:
- `file_path` must exist and be readable
- `format` must match file extension (.pdparams or .pth)
- `file_size_bytes` must be > 0
- For PaddlePaddle format: must be loadable by `paddle.load()`
- For PyTorch format: must be loadable by `torch.load()`

**State Transitions**:
```
[Created] → [Validated] → [Loaded] → [Processed]
```

---

### 2. Parameter

Represents a single named parameter (weight or bias tensor) within a checkpoint.

**Attributes**:
- `name`: Fully qualified parameter name (string)
  - Example: `"backbone.conv1.weight"`, `"encoder.layer.0.self_attn.q_proj.bias"`
- `tensor`: The actual parameter tensor data (framework-specific tensor type)
- `shape`: Dimensions of the tensor (tuple of integers)
  - Example: `(64, 3, 7, 7)` for Conv2D weights
- `dtype`: Data type of the tensor (string)
  - Example: `"float32"`, `"float16"`, `"bfloat16"`
- `device`: Device location (string)
  - Example: `"cpu"`, `"cuda:0"`
- `requires_grad`: Whether parameter requires gradient computation (boolean)
- `is_buffer`: Whether parameter is a buffer (e.g., BatchNorm running stats) vs trainable weight (boolean)

**Relationships**:
- Parameter belongs to one CheckpointFile
- Parameter has zero or one ParameterMapping (if mapped)

**Validation Rules**:
- `name` must be non-empty string
- `shape` must have at least 1 dimension (scalars stored as 0-D tensors)
- `dtype` must be a valid PyTorch/PaddlePaddle dtype
- Tensor data must match `shape` and `dtype` specifications

**Naming Conventions**:

| Framework | Layer Type | Weight Name | Bias Name | Additional Params |
|-----------|------------|-------------|-----------|-------------------|
| **PaddlePaddle** | Conv2D | `layer.w_0` | `layer.b_0` | - |
| **PaddlePaddle** | BatchNorm | `layer._scale` | `layer._offset` | `layer._mean`, `layer._variance` |
| **PaddlePaddle** | Linear | `layer.w_0` | `layer.b_0` | - |
| **PyTorch** | Conv2d | `layer.weight` | `layer.bias` | - |
| **PyTorch** | BatchNorm2d | `layer.weight` | `layer.bias` | `layer.running_mean`, `layer.running_var`, `layer.num_batches_tracked` |
| **PyTorch** | Linear | `layer.weight` | `layer.bias` | - |

---

### 3. ParameterMapping

Represents the mapping between a source parameter (PaddlePaddle) and target parameter (PyTorch).

**Attributes**:
- `source_name`: PaddlePaddle parameter name (string)
- `target_name`: PyTorch parameter name (string)
- `mapping_type`: How the mapping was determined (enum)
  - `MANUAL`: User-provided via JSON override
  - `RULE_BASED`: Auto-generated via naming convention rules
  - `FUZZY_MATCH`: Generated via string similarity matching
  - `IDENTITY`: Names are identical (no transformation needed)
- `confidence_score`: Mapping confidence 0.0-1.0 (float)
  - Manual: 1.0
  - Rule-based: 0.95
  - Fuzzy match: 0.5-0.9 (based on string similarity)
  - Identity: 1.0
- `transformation_applied`: Description of name transformation (string, optional)
  - Example: `"Replaced '.w_0' with '.weight'"`
- `shape_compatible`: Whether source and target shapes match (boolean)
- `notes`: Additional information or warnings (string, optional)

**Relationships**:
- ParameterMapping connects one source Parameter to one target Parameter (1:1)
- Many ParameterMappings belong to one ConversionSession

**Validation Rules**:
- `source_name` must exist in source checkpoint
- `target_name` must exist in target model structure (if validating)
- `confidence_score` must be between 0.0 and 1.0
- If `mapping_type` is MANUAL, `confidence_score` must be 1.0

**Lifecycle**:
```
[Generated] → [Validated] → [Applied] → [Verified]
```

---

### 4. ConversionSession

Represents a single execution of the weight conversion process.

**Attributes**:
- `session_id`: Unique identifier for this conversion (UUID string)
- `source_checkpoint`: Source CheckpointFile instance
- `target_checkpoint`: Target CheckpointFile instance (created during conversion)
- `start_time`: Conversion start timestamp (datetime)
- `end_time`: Conversion completion timestamp (datetime, nullable)
- `duration_seconds`: Total conversion time (float, computed)
- `status`: Current conversion status (enum)
  - `INITIALIZING`, `LOADING_SOURCE`, `GENERATING_MAPPING`, `CONVERTING`, `VALIDATING`, `SAVING`, `COMPLETED`, `FAILED`
- `config`: Conversion configuration settings (ConversionConfig instance)
- `statistics`: Conversion statistics (ConversionStatistics instance)
- `errors`: List of errors encountered (list of strings)
- `warnings`: List of warnings generated (list of strings)

**Relationships**:
- ConversionSession has one source CheckpointFile and one target CheckpointFile
- ConversionSession has one ConversionConfig
- ConversionSession has one ConversionStatistics
- ConversionSession has many ParameterMappings

**Validation Rules**:
- `session_id` must be unique across all sessions
- `start_time` must be before `end_time`
- If `status` is COMPLETED, `target_checkpoint` must exist
- If `status` is FAILED, `errors` list must be non-empty

**State Transitions**:
```
INITIALIZING → LOADING_SOURCE → GENERATING_MAPPING → CONVERTING → VALIDATING → SAVING → COMPLETED
             ↓                 ↓                     ↓            ↓            ↓
            FAILED           FAILED                FAILED       FAILED       FAILED
```

---

### 5. ConversionConfig

Represents configuration settings for a conversion session.

**Attributes**:
- `strict_mode`: Fail on shape/dtype mismatches vs skip them (boolean)
- `validate_values`: Perform numerical validation after conversion (boolean)
- `validation_tolerance`: Numerical tolerance for value comparison (float)
  - Default: 1e-5 (per constitution)
- `manual_mapping_file`: Path to JSON file with manual mappings (string, optional)
- `export_mapping`: Export generated mapping to file (boolean)
- `export_mapping_path`: Path to save mapping JSON (string, optional)
- `memory_efficient_mode`: Use chunked processing to reduce memory usage (boolean)
- `batch_size`: Number of parameters to process per chunk (integer, optional)
  - Default: None (process all at once)
  - Memory-efficient mode: 100 parameters per chunk
- `log_level`: Logging verbosity ('DEBUG', 'INFO', 'WARNING', 'ERROR') (enum)
- `output_metadata`: Additional metadata to embed in output checkpoint (dict, optional)

**Relationships**:
- ConversionConfig belongs to one ConversionSession

**Validation Rules**:
- `validation_tolerance` must be > 0
- If `export_mapping` is True, `export_mapping_path` must be specified
- If `manual_mapping_file` is specified, file must exist and be valid JSON
- `batch_size` must be > 0 if specified

**Default Configuration**:
```python
DEFAULT_CONFIG = ConversionConfig(
    strict_mode=False,
    validate_values=False,
    validation_tolerance=1e-5,
    manual_mapping_file=None,
    export_mapping=False,
    export_mapping_path=None,
    memory_efficient_mode=False,
    batch_size=None,
    log_level='INFO',
    output_metadata=None
)
```

---

### 6. ConversionStatistics

Tracks metrics and statistics for a conversion session.

**Attributes**:
- `total_parameters`: Total number of parameters in source checkpoint (integer)
- `converted_count`: Number of parameters successfully converted (integer)
- `skipped_count`: Number of parameters skipped (e.g., unmapped or shape mismatch) (integer)
- `failed_count`: Number of parameters that failed conversion (integer)
- `unmapped_source_keys`: List of source parameter names that were not mapped (list of strings)
- `unmapped_target_keys`: List of target parameter names that were not populated (list of strings)
- `shape_mismatches`: List of parameters with shape incompatibilities (list of ShapeMismatch)
- `dtype_conversions`: List of parameters with dtype changes (list of DtypeConversion)
- `total_source_size_bytes`: Total size of source parameters in bytes (integer)
- `total_target_size_bytes`: Total size of target parameters in bytes (integer)
- `compression_ratio`: Ratio of target size to source size (float)
- `peak_memory_usage_bytes`: Maximum memory used during conversion (integer)

**Relationships**:
- ConversionStatistics belongs to one ConversionSession
- ConversionStatistics contains many ShapeMismatch records
- ConversionStatistics contains many DtypeConversion records

**Validation Rules**:
- `total_parameters` = `converted_count` + `skipped_count` + `failed_count`
- All count fields must be non-negative
- `compression_ratio` = `total_target_size_bytes` / `total_source_size_bytes`

**Derived Metrics**:
- Conversion success rate: `converted_count / total_parameters`
- Mapping coverage: `1.0 - (len(unmapped_source_keys) / total_parameters)`
- Memory efficiency: `peak_memory_usage_bytes / total_source_size_bytes`

---

### 7. ShapeMismatch (Sub-entity)

Records a shape incompatibility between source and target parameters.

**Attributes**:
- `parameter_name`: Name of the parameter (string)
- `source_shape`: Shape in source checkpoint (tuple of integers)
- `target_shape`: Shape in target model (tuple of integers)
- `severity`: Impact level ('ERROR', 'WARNING') (enum)
  - ERROR: Shapes fundamentally incompatible (different total elements)
  - WARNING: Shapes compatible but require reshape (e.g., permutation)
- `suggested_fix`: Recommended action to resolve mismatch (string)
  - Example: `"Transpose axes (0,1,2,3) → (0,2,3,1)"`

**Relationships**:
- ShapeMismatch is part of ConversionStatistics

**Validation Rules**:
- `source_shape` and `target_shape` must be non-empty tuples
- If `severity` is WARNING, total elements must match:
  - `prod(source_shape) == prod(target_shape)`

---

### 8. DtypeConversion (Sub-entity)

Records a data type conversion applied during parameter transfer.

**Attributes**:
- `parameter_name`: Name of the parameter (string)
- `source_dtype`: Original data type (string)
- `target_dtype`: Converted data type (string)
- `precision_loss`: Whether conversion loses precision (boolean)
  - True for FP32 → FP16, False for FP16 → FP32
- `justification`: Reason for dtype conversion (string)
  - Example: `"Target model trained with mixed precision"`

**Relationships**:
- DtypeConversion is part of ConversionStatistics

**Validation Rules**:
- `source_dtype` and `target_dtype` must be valid PyTorch dtypes
- If `source_dtype` == `target_dtype`, record should not exist

---

## Data Relationships Diagram

```
ConversionSession
├── source_checkpoint: CheckpointFile
│   └── parameters: List[Parameter]
│       └── mapping: ParameterMapping (optional)
├── target_checkpoint: CheckpointFile
│   └── parameters: List[Parameter]
├── config: ConversionConfig
├── statistics: ConversionStatistics
│   ├── shape_mismatches: List[ShapeMismatch]
│   └── dtype_conversions: List[DtypeConversion]
└── parameter_mappings: List[ParameterMapping]
    ├── source_parameter: Parameter
    └── target_parameter: Parameter
```

---

## File Format Specifications

### 1. Manual Mapping JSON Schema

User-provided file for manual parameter name overrides.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "version": {
      "type": "string",
      "description": "Mapping file format version",
      "example": "1.0"
    },
    "model": {
      "type": "string",
      "description": "Target model architecture name",
      "example": "rtdetrv3_r50vd"
    },
    "mappings": {
      "type": "object",
      "description": "PaddlePaddle name → PyTorch name mappings",
      "patternProperties": {
        "^.*$": {
          "type": "string"
        }
      },
      "example": {
        "backbone.custom_layer.w_0": "backbone.custom_layer.weight",
        "neck.special_module._param": "neck.special_module.param"
      }
    }
  },
  "required": ["version", "mappings"]
}
```

**Example**:
```json
{
  "version": "1.0",
  "model": "rtdetrv3_r50vd",
  "mappings": {
    "backbone.res2a.conv1.w_0": "backbone.layer1.0.conv1.weight",
    "backbone.res2a.bn1._mean": "backbone.layer1.0.bn1.running_mean",
    "backbone.res2a.bn1._variance": "backbone.layer1.0.bn1.running_var"
  }
}
```

---

### 2. Exported Mapping JSON Schema

Tool-generated file documenting the parameter name mapping used during conversion.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "description": "Unique identifier for the conversion session"
    },
    "source_checkpoint": {
      "type": "string",
      "description": "Path to source .pdparams file"
    },
    "target_checkpoint": {
      "type": "string",
      "description": "Path to target .pth file"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "Conversion timestamp"
    },
    "mappings": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "source_name": {"type": "string"},
          "target_name": {"type": "string"},
          "mapping_type": {"enum": ["MANUAL", "RULE_BASED", "FUZZY_MATCH", "IDENTITY"]},
          "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
          "shape_compatible": {"type": "boolean"}
        },
        "required": ["source_name", "target_name", "mapping_type"]
      }
    },
    "unmapped_source": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Source parameters that were not mapped"
    },
    "unmapped_target": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Target parameters that were not populated"
    },
    "statistics": {
      "type": "object",
      "properties": {
        "total_parameters": {"type": "integer"},
        "mapped_count": {"type": "integer"},
        "unmapped_source_count": {"type": "integer"},
        "unmapped_target_count": {"type": "integer"}
      }
    }
  },
  "required": ["session_id", "source_checkpoint", "mappings"]
}
```

**Example**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_checkpoint": "pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams",
  "target_checkpoint": "pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth",
  "timestamp": "2025-10-16T14:30:00Z",
  "mappings": [
    {
      "source_name": "backbone.conv1.w_0",
      "target_name": "backbone.conv1.weight",
      "mapping_type": "RULE_BASED",
      "confidence_score": 0.95,
      "shape_compatible": true
    },
    {
      "source_name": "backbone.bn1._mean",
      "target_name": "backbone.bn1.running_mean",
      "mapping_type": "RULE_BASED",
      "confidence_score": 0.95,
      "shape_compatible": true
    }
  ],
  "unmapped_source": ["backbone.custom_module.extra_param"],
  "unmapped_target": ["backbone.new_module.weight"],
  "statistics": {
    "total_parameters": 315,
    "mapped_count": 312,
    "unmapped_source_count": 1,
    "unmapped_target_count": 2
  }
}
```

---

### 3. PyTorch Checkpoint Format

Output .pth file structure generated by the conversion tool.

```python
{
  'model': {
    # State dict with converted parameter tensors
    'backbone.conv1.weight': torch.Tensor([64, 3, 7, 7]),
    'backbone.bn1.running_mean': torch.Tensor([64]),
    'backbone.bn1.running_var': torch.Tensor([64]),
    # ... all converted parameters
  },
  'metadata': {
    'source': 'PaddlePaddle',
    'source_checkpoint': 'pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams',
    'conversion_timestamp': '2025-10-16T14:30:00Z',
    'conversion_tool_version': '1.0.0',
    'session_id': '550e8400-e29b-41d4-a716-446655440000',
    'conversion_stats': {
      'total': 315,
      'converted': 312,
      'skipped': 3,
      'shape_mismatches': []
    },
    'validation': {
      'numerical_validation_performed': True,
      'max_absolute_diff': 1.2e-6,
      'tolerance': 1e-5,
      'validation_passed': True
    }
  }
}
```

---

## Data Validation & Integrity

### Pre-Conversion Validation
1. **Source checkpoint integrity**: Verify file exists, is readable, and loadable
2. **Target model compatibility**: Confirm PyTorch model can be instantiated
3. **Manual mapping validity**: Validate JSON schema if provided
4. **Configuration sanity**: Check for conflicting or invalid config options

### During-Conversion Validation
1. **Shape compatibility**: Verify source and target shapes match (or can be reshaped)
2. **Dtype compatibility**: Check for precision loss in type conversions
3. **Value range**: Detect NaN, Inf, or out-of-range values in tensors
4. **Memory constraints**: Monitor memory usage and trigger chunked mode if needed

### Post-Conversion Validation
1. **Completeness**: Verify all mapped parameters were converted
2. **Numerical correctness**: Compare converted values against source (optional)
3. **Model loadability**: Ensure target checkpoint can be loaded by PyTorch
4. **Functional validation**: Run inference and compare outputs (optional)

---

## Implementation Notes

### Memory Management Strategy
- **Small models (<500MB)**: Load entire state dict into memory
- **Large models (500MB-2GB)**: Use chunked processing with batches of 100 parameters
- **Very large models (>2GB)**: Memory-mapped arrays with on-demand conversion

### Error Handling Strategy
- **Strict mode**: Fail fast on any error (shape mismatch, unmapped key, etc.)
- **Permissive mode**: Log warnings, skip problematic parameters, continue conversion
- **Partial success**: Generate valid checkpoint with subset of parameters + detailed report

### Extensibility Considerations
- **Plugin system**: Allow custom name mapping rules via configuration
- **Framework adapters**: Abstract checkpoint loading/saving for future framework support
- **Validation hooks**: Enable custom validation logic via callback functions

---

**Data Model Complete**: Ready for contract definition and implementation.

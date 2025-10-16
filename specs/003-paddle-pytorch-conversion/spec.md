# Feature Specification: Paddle to PyTorch Weight Conversion

**Feature Branch**: `003-paddle-pytorch-conversion`
**Created**: 2025-10-16
**Status**: Draft
**Input**: User description: "执行paddle版本到pytorch版本的权重转换, 目前有一个草稿式的转换脚本在 @tools/convert_weights.py paddle的权重在 @pretrained_models/paddle 目录下, 你还可以使用deepwiki工具或者perplexity工具去网上调研"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic Weight Conversion (Priority: P1)

Researchers and developers need to convert pre-trained RT-DETRv3 model weights from PaddlePaddle format (.pdparams) to PyTorch format (.pth) to use the models in PyTorch-based projects. This conversion should handle parameter name mapping and tensor format conversion automatically.

**Why this priority**: This is the core functionality required to enable PyTorch users to leverage pre-trained PaddlePaddle models. Without this, the PyTorch implementation cannot utilize existing trained weights.

**Independent Test**: Can be fully tested by providing a .pdparams file and verifying that the output .pth file contains correctly converted weights that can be loaded into a PyTorch model.

**Acceptance Scenarios**:

1. **Given** a valid .pdparams file (e.g., rtdetrv3_r50vd_6x_coco.pdparams), **When** user runs the conversion tool with the source file path and destination path, **Then** the system generates a valid .pth file with all parameters correctly converted
2. **Given** a converted .pth file, **When** user loads it into a PyTorch RT-DETRv3 model, **Then** all model layers successfully receive their weights without shape mismatches or missing parameters
3. **Given** a .pdparams file with 182MB size, **When** conversion completes, **Then** the output .pth file has similar size and contains the same number of parameters

---

### User Story 2 - Parameter Name Mapping Validation (Priority: P2)

Users need to understand how PaddlePaddle parameter names map to PyTorch parameter names, and identify any unmapped parameters. The tool should generate a mapping report to help debug conversion issues.

**Why this priority**: Name mapping is critical for successful conversion, but users may need to inspect or customize mappings for non-standard models. This transparency helps troubleshoot conversion failures.

**Independent Test**: Can be fully tested by running conversion with mapping export enabled and verifying that the mapping JSON file accurately reflects the parameter name transformations.

**Acceptance Scenarios**:

1. **Given** a conversion with mapping export enabled, **When** conversion completes, **Then** system generates a JSON file showing PaddlePaddle-to-PyTorch parameter name mappings
2. **Given** parameters with different naming conventions (e.g., "bn1._mean" in Paddle vs "bn1.running_mean" in PyTorch), **When** conversion runs, **Then** the mapping file shows the correct transformations applied
3. **Given** unmapped parameters exist, **When** conversion completes, **Then** the mapping file lists all unmapped parameters from both source and target models

---

### User Story 3 - Batch Conversion for Multiple Models (Priority: P3)

Users working with multiple model variants (r18vd, r34vd, r50vd) need to convert all pre-trained weights efficiently without repeating commands.

**Why this priority**: Improves workflow efficiency for users managing multiple model variants, but individual conversions remain functional without this feature.

**Independent Test**: Can be fully tested by providing a directory containing multiple .pdparams files and verifying all files are converted correctly.

**Acceptance Scenarios**:

1. **Given** a directory containing multiple .pdparams files, **When** user runs batch conversion, **Then** system converts all files and generates corresponding .pth files
2. **Given** batch conversion is running, **When** one file fails to convert, **Then** system continues converting remaining files and reports the failure

---

### User Story 4 - Shape Mismatch Handling (Priority: P2)

When parameter shapes don't match between PaddlePaddle and PyTorch models, users need clear error messages or optional shape transformation handling.

**Why this priority**: Shape mismatches can occur due to framework differences. Proper handling prevents silent failures and guides users to resolution.

**Independent Test**: Can be fully tested by intentionally providing mismatched model architectures and verifying error messages are clear and actionable.

**Acceptance Scenarios**:

1. **Given** a shape mismatch occurs during conversion, **When** strict mode is enabled, **Then** system halts conversion and reports the specific parameter and shape difference
2. **Given** a shape mismatch occurs during conversion, **When** strict mode is disabled, **Then** system skips the mismatched parameter, logs a warning, and continues conversion
3. **Given** conversion completes with skipped parameters, **When** user reviews the conversion report, **Then** all skipped parameters are listed with their shape differences

---

### Edge Cases

- What happens when a .pdparams file is corrupted or unreadable?
- How does the system handle parameter names with special characters or unicode?
- What if the PyTorch model architecture is not provided (only converting weights without validation)?
- How does the system handle different data types (float32, float16, bfloat16)?
- What if memory is insufficient to load large model weights (>1GB)?
- How does the system handle parameters with custom PaddlePaddle naming that don't follow conventions?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load .pdparams files from PaddlePaddle checkpoint format
- **FR-002**: System MUST convert source framework tensors to target framework tensor format with correct data representation
- **FR-003**: System MUST automatically map parameter names from PaddlePaddle conventions to PyTorch conventions
- **FR-004**: System MUST handle common naming pattern transformations:
  - `._mean` → `.running_mean` (BatchNorm)
  - `._variance` → `.running_var` (BatchNorm)
  - `.w_0` → `.weight` (convolution/linear layers)
  - `.b_0` → `.bias` (convolution/linear layers)
- **FR-005**: System MUST validate parameter shapes between source and target models when target model is provided
- **FR-006**: System MUST support both strict and non-strict conversion modes (strict fails on any error, non-strict skips mismatches)
- **FR-007**: System MUST save converted weights in PyTorch .pth format with metadata (source file, conversion statistics)
- **FR-008**: System MUST generate conversion statistics including total parameters, converted count, and skipped count
- **FR-009**: System MUST support optional manual parameter name mapping overrides via JSON configuration
- **FR-010**: System MUST export generated parameter name mappings to JSON file when requested
- **FR-011**: System MUST identify and report unmapped parameters from both source and target models
- **FR-012**: System MUST provide progress logging during conversion for transparency
- **FR-013**: System MUST accept user input for source file location, output destination, configuration settings, and conversion options
- **FR-014**: System MUST handle PaddlePaddle model loading errors gracefully with clear error messages
- **FR-015**: System MUST support conversion without requiring PyTorch model when validation is not needed

### Key Entities

- **PaddlePaddle Checkpoint**: Pre-trained model weights stored in .pdparams format, containing parameter tensors with PaddlePaddle-specific naming conventions
- **PyTorch Checkpoint**: Converted model weights stored in .pth format, containing state_dict with PyTorch-compatible parameter names and tensor formats
- **Parameter Name Mapping**: Dictionary mapping PaddlePaddle parameter names to corresponding PyTorch parameter names, supporting both automatic generation and manual overrides
- **Conversion Statistics**: Record tracking conversion progress including total parameters, successfully converted parameters, skipped parameters, and shape mismatches
- **Conversion Configuration**: Settings controlling conversion behavior including strict mode, manual mappings, and output options

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can convert a 182MB .pdparams file to .pth format in under 2 minutes on standard hardware
- **SC-002**: Converted weights achieve identical inference results (within 1e-6 numerical tolerance) compared to original PaddlePaddle model
- **SC-003**: Conversion tool successfully handles all three provided model variants (r18vd, r34vd, r50vd) without manual intervention
- **SC-004**: 95% of standard model parameters are automatically mapped without requiring manual mapping configuration
- **SC-005**: Clear error messages enable users to resolve conversion issues within 10 minutes for common problems
- **SC-006**: Conversion process consumes no more than 2x the source file size in peak memory usage

## Assumptions

- Users have both PaddlePaddle and PyTorch installed in their environment (PaddlePaddle for loading source weights)
- Model architectures between PaddlePaddle and PyTorch implementations are equivalent (same layer structure)
- Pre-trained weights are stored in standard PaddlePaddle checkpoint format (.pdparams)
- Users have basic command-line proficiency for running conversion scripts
- Converted models will be used for inference or fine-tuning, not necessarily for reproducing exact training behavior
- The existing draft script at tools/convert_weights.py provides a foundation for the conversion logic
- Target framework models follow standard naming conventions

## Dependencies

- PaddlePaddle framework must be installed to load .pdparams files
- PyTorch framework must be installed for saving .pth files
- Intermediate data format conversion capability is required for tensor transformation
- Access to pretrained_models/paddle directory containing source .pdparams files
- For validation mode, PyTorch model definition must be available to generate target state_dict

## Scope

### In Scope

- Converting RT-DETRv3 model weights from PaddlePaddle to PyTorch format
- Automatic parameter name mapping with common naming convention transformations
- Shape validation when target model is provided
- Conversion statistics and progress reporting
- Manual mapping override support
- Batch conversion for multiple model files
- Strict and non-strict conversion modes

### Out of Scope

- Converting model architecture definitions (only weights, not code)
- Training or fine-tuning the converted models
- Optimization or compression of converted weights
- Converting from PyTorch back to PaddlePaddle
- Converting models from other frameworks (TensorFlow, ONNX, etc.)
- Automatic fixing of shape mismatches (only detection and reporting)
- GPU acceleration for conversion process
- Web interface or GUI for conversion tool

# CLI Interface Contract

**Feature**: 003-paddle-pytorch-conversion
**Date**: 2025-10-16
**Interface Type**: Command-Line Interface

## Overview

This document defines the command-line interface contract for the `convert_weights.py` tool. The CLI provides access to all weight conversion functionality through a consistent, user-friendly interface.

## Command Structure

```bash
python tools/convert_weights.py [OPTIONS]
```

## Options

### Required Arguments

#### `--input, -i <PATH>`
**Type**: File path (string)
**Description**: Path to source PaddlePaddle checkpoint file (.pdparams)
**Validation**:
- File must exist
- File must have `.pdparams` extension
- File must be readable
- File must be valid PaddlePaddle checkpoint format

**Example**:
```bash
--input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams
```

---

#### `--output, -o <PATH>`
**Type**: File path (string)
**Description**: Path for output PyTorch checkpoint file (.pth)
**Validation**:
- Parent directory must exist or be creatable
- File path must have `.pth` or `.pt` extension
- If file exists, user confirmation required (or use `--force`)

**Example**:
```bash
--output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth
```

---

### Conditional Required Arguments

#### `--model-config, -c <PATH>`
**Type**: File path (string)
**Description**: Path to model configuration YAML file (required if `--no-validate` not set)
**Validation**:
- File must exist if validation is enabled
- File must be valid YAML format
- Must contain model architecture specification

**Example**:
```bash
--model-config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml
```

**When Required**: If `--no-validate` is NOT set (validation mode)
**When Optional**: If `--no-validate` IS set (no validation mode)

---

### Optional Arguments

#### `--manual-mapping, -m <PATH>`
**Type**: File path (string)
**Default**: None
**Description**: Path to JSON file with manual parameter name mapping overrides
**Validation**:
- File must exist
- Must be valid JSON
- Must conform to manual mapping schema (see data-model.md)

**Example**:
```bash
--manual-mapping configs/mappings/custom_r50vd_mapping.json
```

---

#### `--save-mapping, -s <PATH>`
**Type**: File path (string)
**Default**: None
**Description**: Export generated parameter name mapping to JSON file
**Validation**:
- Parent directory must exist or be creatable
- File path must have `.json` extension

**Example**:
```bash
--save-mapping outputs/mappings/r50vd_mapping_20251016.json
```

---

#### `--strict`
**Type**: Boolean flag
**Default**: False
**Description**: Enable strict mode (fail on any shape mismatch or unmapped parameter)
**Conflicts**: Cannot be used with `--permissive` (mutually exclusive)

**Example**:
```bash
--strict
```

---

#### `--permissive`
**Type**: Boolean flag
**Default**: True
**Description**: Enable permissive mode (skip mismatched parameters, continue conversion)
**Conflicts**: Cannot be used with `--strict` (mutually exclusive)

**Example**:
```bash
--permissive
```

---

#### `--no-validate`
**Type**: Boolean flag
**Default**: False
**Description**: Skip shape validation against target model (converts weights without model structure)
**Impact**: When set, `--model-config` becomes optional

**Example**:
```bash
--no-validate
```

---

#### `--validate-numerical`
**Type**: Boolean flag
**Default**: False
**Description**: Perform numerical validation after conversion (requires `--model-config`)
**Validation**:
- Requires `--model-config` to be set
- Cannot be used with `--no-validate`

**Example**:
```bash
--validate-numerical --tolerance 1e-5
```

---

#### `--tolerance <FLOAT>`
**Type**: Float
**Default**: 1e-5
**Description**: Numerical tolerance for validation (only used with `--validate-numerical`)
**Validation**:
- Must be positive number
- Typical range: 1e-7 to 1e-3

**Example**:
```bash
--validate-numerical --tolerance 1e-6
```

---

#### `--batch, -b <PATTERN>`
**Type**: Glob pattern (string)
**Description**: Batch convert multiple files matching glob pattern
**Conflicts**: Mutually exclusive with `--input` (use one or the other)
**Requires**: `--output-dir` must be specified when using batch mode

**Example**:
```bash
--batch "pretrained_models/paddle/*.pdparams" --output-dir pretrained_models/pytorch/
```

---

#### `--output-dir, -d <PATH>`
**Type**: Directory path (string)
**Default**: None
**Description**: Output directory for batch conversion (required when using `--batch`)
**Validation**:
- Directory must exist or be creatable
- Must have write permissions

**Example**:
```bash
--output-dir pretrained_models/pytorch/
```

---

#### `--memory-efficient`
**Type**: Boolean flag
**Default**: False
**Description**: Use chunked processing to reduce memory usage (slower but lower memory footprint)

**Example**:
```bash
--memory-efficient
```

---

#### `--force, -f`
**Type**: Boolean flag
**Default**: False
**Description**: Overwrite existing output files without confirmation

**Example**:
```bash
--force
```

---

#### `--log-level <LEVEL>`
**Type**: Enum (string)
**Choices**: `DEBUG`, `INFO`, `WARNING`, `ERROR`
**Default**: `INFO`
**Description**: Set logging verbosity level

**Example**:
```bash
--log-level DEBUG
```

---

#### `--quiet, -q`
**Type**: Boolean flag
**Default**: False
**Description**: Suppress all output except errors (overrides `--log-level`)

**Example**:
```bash
--quiet
```

---

#### `--version`
**Type**: Boolean flag
**Description**: Display tool version and exit

**Example**:
```bash
--version
```

**Output**:
```
Weight Conversion Tool v1.0.0
PyTorch: 2.1.0
PaddlePaddle: 2.5.1
```

---

#### `--help, -h`
**Type**: Boolean flag
**Description**: Display help message and exit

**Example**:
```bash
--help
```

---

## Usage Patterns

### Pattern 1: Basic Conversion with Validation
```bash
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --model-config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml
```

**Expected Output**:
```
[10/16 14:30:00] INFO: Loading PaddlePaddle checkpoint from pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams
[10/16 14:30:05] INFO: Loaded 315 parameters from PaddlePaddle checkpoint
[10/16 14:30:05] INFO: Generating parameter name mapping...
[10/16 14:30:06] INFO: Generated mapping for 312 parameters (3 unmapped)
[10/16 14:30:06] INFO: Converting parameters...
[10/16 14:31:45] INFO: Converted 312/312 parameters successfully
[10/16 14:31:45] INFO: Saving converted checkpoint to pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth
[10/16 14:31:50] INFO: Conversion completed successfully!
[10/16 14:31:50] INFO: Statistics: 312 converted, 0 skipped, 3 unmapped
```

**Exit Code**: 0 (success)

---

### Pattern 2: Strict Mode with Manual Mapping
```bash
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --model-config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml \
    --manual-mapping configs/mappings/r50vd_custom.json \
    --save-mapping outputs/final_mapping.json \
    --strict
```

**Expected Output**:
```
[10/16 14:30:00] INFO: Loading manual mapping from configs/mappings/r50vd_custom.json
[10/16 14:30:00] INFO: Loaded 5 manual mappings
[10/16 14:30:00] INFO: Loading PaddlePaddle checkpoint...
[10/16 14:30:05] INFO: Generating parameter name mapping...
[10/16 14:30:06] INFO: Applied 5 manual mappings
[10/16 14:30:06] INFO: Generated 310 rule-based mappings
[10/16 14:30:06] INFO: Total: 315 parameters mapped
[10/16 14:30:06] INFO: Converting parameters (strict mode enabled)...
[10/16 14:31:45] INFO: Converted 315/315 parameters successfully
[10/16 14:31:45] INFO: Saved mapping to outputs/final_mapping.json
[10/16 14:31:50] INFO: Conversion completed successfully!
```

**Exit Code**: 0 (success)

---

### Pattern 3: Batch Conversion
```bash
python tools/convert_weights.py \
    --batch "pretrained_models/paddle/*.pdparams" \
    --output-dir pretrained_models/pytorch/ \
    --model-config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml
```

**Expected Output**:
```
[10/16 14:30:00] INFO: Found 3 files matching pattern
[10/16 14:30:00] INFO: Converting rtdetrv3_r18vd_6x_coco.pdparams...
[10/16 14:30:45] INFO: Converted rtdetrv3_r18vd_6x_coco.pth (245 parameters)
[10/16 14:30:45] INFO: Converting rtdetrv3_r34vd_6x_coco.pdparams...
[10/16 14:31:35] INFO: Converted rtdetrv3_r34vd_6x_coco.pth (290 parameters)
[10/16 14:31:35] INFO: Converting rtdetrv3_r50vd_6x_coco.pdparams...
[10/16 14:32:40] INFO: Converted rtdetrv3_r50vd_6x_coco.pth (315 parameters)
[10/16 14:32:40] INFO: Batch conversion completed: 3/3 successful
```

**Exit Code**: 0 (success)

---

### Pattern 4: Numerical Validation
```bash
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --model-config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml \
    --validate-numerical \
    --tolerance 1e-6
```

**Expected Output**:
```
[10/16 14:30:00] INFO: Loading PaddlePaddle checkpoint...
[10/16 14:30:05] INFO: Converting parameters...
[10/16 14:31:45] INFO: Conversion complete
[10/16 14:31:45] INFO: Performing numerical validation (tolerance=1e-6)...
[10/16 14:31:50] INFO: Validating parameter values...
[10/16 14:31:55] INFO: Max absolute difference: 8.34e-7
[10/16 14:31:55] INFO: Numerical validation PASSED
[10/16 14:31:55] INFO: Saving converted checkpoint...
[10/16 14:32:00] INFO: Conversion completed successfully!
```

**Exit Code**: 0 (success)

---

### Pattern 5: No Validation Mode
```bash
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --no-validate
```

**Expected Output**:
```
[10/16 14:30:00] WARNING: Validation disabled - converting weights without shape checking
[10/16 14:30:00] INFO: Loading PaddlePaddle checkpoint...
[10/16 14:30:05] INFO: Converting parameters (no validation)...
[10/16 14:31:45] INFO: Converted 315 parameters
[10/16 14:31:45] WARNING: Shape validation was skipped - verify compatibility before use
[10/16 14:31:50] INFO: Conversion completed
```

**Exit Code**: 0 (success)

---

## Error Handling

### Error 1: Input File Not Found
**Command**:
```bash
python tools/convert_weights.py \
    --input non_existent.pdparams \
    --output output.pth
```

**Output**:
```
[10/16 14:30:00] ERROR: Input file not found: non_existent.pdparams
```

**Exit Code**: 1

---

### Error 2: Invalid PaddlePaddle Checkpoint
**Command**:
```bash
python tools/convert_weights.py \
    --input corrupted.pdparams \
    --output output.pth
```

**Output**:
```
[10/16 14:30:00] INFO: Loading PaddlePaddle checkpoint from corrupted.pdparams
[10/16 14:30:01] ERROR: Failed to load checkpoint: File is corrupted or not a valid PaddlePaddle checkpoint
```

**Exit Code**: 1

---

### Error 3: Shape Mismatch in Strict Mode
**Command**:
```bash
python tools/convert_weights.py \
    --input input.pdparams \
    --output output.pth \
    --model-config config.yml \
    --strict
```

**Output**:
```
[10/16 14:30:00] INFO: Converting parameters (strict mode enabled)...
[10/16 14:30:15] ERROR: Shape mismatch for parameter 'backbone.layer.weight'
[10/16 14:30:15] ERROR:   Source shape: (256, 128, 3, 3)
[10/16 14:30:15] ERROR:   Target shape: (256, 64, 3, 3)
[10/16 14:30:15] ERROR: Conversion failed in strict mode
```

**Exit Code**: 1

---

### Error 4: Missing Model Config in Validation Mode
**Command**:
```bash
python tools/convert_weights.py \
    --input input.pdparams \
    --output output.pth
```

**Output**:
```
[10/16 14:30:00] ERROR: --model-config is required when validation is enabled
[10/16 14:30:00] ERROR: Use --no-validate to skip validation, or provide --model-config
```

**Exit Code**: 1

---

### Error 5: Output File Exists Without --force
**Command**:
```bash
python tools/convert_weights.py \
    --input input.pdparams \
    --output existing_output.pth \
    --model-config config.yml
```

**Output**:
```
[10/16 14:30:00] WARNING: Output file already exists: existing_output.pth
[10/16 14:30:00] ERROR: Refusing to overwrite existing file (use --force to override)
```

**Exit Code**: 1

---

### Warning 1: Unmapped Parameters in Permissive Mode
**Command**:
```bash
python tools/convert_weights.py \
    --input input.pdparams \
    --output output.pth \
    --model-config config.yml \
    --permissive
```

**Output**:
```
[10/16 14:30:00] INFO: Converting parameters (permissive mode)...
[10/16 14:30:15] WARNING: Unmapped source parameter: backbone.custom_layer.extra_param
[10/16 14:30:15] WARNING: Unmapped target parameter: backbone.new_module.weight
[10/16 14:31:45] WARNING: Conversion completed with 2 unmapped parameters
[10/16 14:31:45] INFO: Statistics: 310 converted, 3 unmapped
```

**Exit Code**: 0 (success with warnings)

---

## Exit Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | Conversion completed successfully |
| 1 | Error | Conversion failed due to error |
| 2 | Invalid Arguments | Command-line argument validation failed |
| 3 | Validation Failed | Numerical validation failed |
| 130 | Interrupted | User interrupted conversion (Ctrl+C) |

---

## Environment Variables

### `PADDLE_CONV_LOG_LEVEL`
**Type**: String
**Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`
**Description**: Override default log level (can be overridden by `--log-level`)

**Example**:
```bash
export PADDLE_CONV_LOG_LEVEL=DEBUG
python tools/convert_weights.py --input input.pdparams --output output.pth
```

---

### `PADDLE_CONV_MEMORY_LIMIT`
**Type**: Integer (bytes)
**Description**: Maximum memory usage threshold (triggers memory-efficient mode)

**Example**:
```bash
export PADDLE_CONV_MEMORY_LIMIT=1073741824  # 1GB
python tools/convert_weights.py --input input.pdparams --output output.pth
```

---

## Progress Reporting

### Progress Bar (Interactive Mode)
When stdout is a TTY (interactive terminal), display progress bar:

```
Converting parameters: 78% |████████████████████░░░░░| 245/315 [00:32<00:08, 8.2 params/s]
```

### Log Messages (Non-Interactive Mode)
When stdout is redirected or not a TTY, use log messages:

```
[10/16 14:30:00] INFO: Converting parameters...
[10/16 14:30:10] INFO: Progress: 100/315 (32%)
[10/16 14:30:20] INFO: Progress: 200/315 (63%)
[10/16 14:30:30] INFO: Progress: 300/315 (95%)
[10/16 14:31:45] INFO: Converted 315/315 parameters successfully
```

---

## Compatibility

### Python Version Support
- **Minimum**: Python 3.8
- **Recommended**: Python 3.11
- **Tested**: Python 3.8, 3.10, 3.11

### Platform Support
- **Linux**: Full support (primary platform)
- **macOS**: Full support
- **Windows**: Full support (use forward slashes or raw strings for paths)

### Shell Support
- **Bash/Zsh**: Full support
- **PowerShell**: Full support (quote glob patterns)
- **CMD**: Basic support (no glob expansion)

---

**CLI Contract Complete**: Ready for implementation.

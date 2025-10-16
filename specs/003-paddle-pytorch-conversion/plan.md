# Implementation Plan: Paddle to PyTorch Weight Conversion

**Branch**: `003-paddle-pytorch-conversion` | **Date**: 2025-10-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-paddle-pytorch-conversion/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature enables automated conversion of RT-DETRv3 model weights from PaddlePaddle format (.pdparams) to PyTorch format (.pth). The conversion tool handles parameter name mapping (including framework-specific naming conventions), tensor format transformation, shape validation, and provides detailed conversion statistics. This is a critical utility for the RT-DETRv3 PyTorch migration project, enabling users to leverage pre-trained PaddlePaddle weights in the PyTorch implementation.

## Technical Context

**Language/Version**: Python 3.8+ (3.11 recommended per constitution)
**Primary Dependencies**: PyTorch ≥2.0, PaddlePaddle (for loading source weights), NumPy (tensor conversion intermediate)
**Storage**: File system (input .pdparams files, output .pth files, optional JSON mapping exports)
**Testing**: pytest with numerical equivalence validation (1e-5 tolerance per constitution)
**Target Platform**: Linux/Windows/macOS (cross-platform Python CLI tool)
**Project Type**: Single utility tool (CLI script)
**Performance Goals**: Convert 182MB model file in <2 minutes; memory usage ≤2x source file size
**Constraints**: <2 min conversion time for largest model (r50vd); must handle 95% automatic name mapping; strict numerical parity validation
**Scale/Scope**: 3 model variants (r18vd, r34vd, r50vd); ~15 functional requirements; supports batch processing

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Gate 1: Framework Parity First ✅
- **Status**: PASS
- **Justification**: Weight conversion is a prerequisite for framework parity validation. This tool enables loading PaddlePaddle pre-trained weights into PyTorch models for numerical equivalence testing.
- **Alignment**: Supports Principle I by providing the infrastructure to validate PyTorch model outputs against PaddlePaddle reference using identical weights.

### Gate 2: Modular Migration Strategy ✅
- **Status**: PASS
- **Justification**: Weight conversion is a standalone utility that operates independently of model architecture migration. It can be developed, tested, and validated in isolation.
- **Alignment**: Follows Principle II by creating a discrete, independently testable component.

### Gate 3: Validation-Driven Development ✅
- **Status**: PASS
- **Justification**: Conversion correctness will be validated by:
  1. Shape matching tests (converted tensors match expected PyTorch model shapes)
  2. Value preservation tests (converted tensors numerically match source tensors)
  3. End-to-end tests (PyTorch model loaded with converted weights produces same outputs as PaddlePaddle)
- **Alignment**: Strict adherence to Principle III with numerical validation at multiple levels.

### Gate 4: Reproducibility & Documentation ✅
- **Status**: PASS
- **Justification**: The conversion tool itself generates documentation:
  - Parameter name mapping export (JSON format)
  - Conversion statistics (converted/skipped/mismatched counts)
  - Clear logging of all transformations applied
- **Alignment**: Meets Principle IV documentation requirements for API mappings and behavioral differences.

### Gate 5: Performance Parity ⚠️
- **Status**: PASS (Modified Scope)
- **Justification**: Performance targets apply to conversion tool itself (not model inference):
  - Conversion speed: <2 min for 182MB file (specification SC-001)
  - Memory efficiency: ≤2x source file size (specification SC-006)
- **Alignment**: Performance goals are tool-specific and appropriate for a utility component.

### Gate 6: Configuration Compatibility ✅
- **Status**: PASS
- **Justification**: Weight conversion preserves parameter values exactly. The tool supports:
  - Manual mapping overrides via JSON (for non-standard naming)
  - Batch processing of multiple model files
  - Both strict and permissive conversion modes
- **Alignment**: Supports Principle VI by enabling flexible migration paths.

### Constitution Compliance Summary

**Overall Status**: ✅ **APPROVED TO PROCEED**

All constitution principles are satisfied:
- Supports framework parity validation (I)
- Standalone modular component (II)
- Validation-driven approach with numerical tests (III)
- Comprehensive documentation generation (IV)
- Appropriate performance targets for utility tool (V)
- Flexible configuration support (VI)

**No violations requiring justification.**

## Project Structure

### Documentation (this feature)

```
specs/003-paddle-pytorch-conversion/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file (in progress)
├── research.md          # Phase 0 output (to be generated)
├── data-model.md        # Phase 1 output (to be generated)
├── quickstart.md        # Phase 1 output (to be generated)
├── contracts/           # Phase 1 output (API schemas - if applicable)
│   └── cli-interface.md # Command-line interface contract
├── checklists/
│   └── requirements.md  # Specification quality checklist (complete)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created yet)
```

### Source Code (repository root)

```
# Single project structure (utility tool)
tools/
├── convert_weights.py           # Main conversion script (existing draft to be enhanced)
└── weight_conversion/           # Conversion utilities module (to be created)
    ├── __init__.py
    ├── converter.py             # WeightConverter class (enhanced from draft)
    ├── name_mapping.py          # Parameter name mapping logic
    ├── tensor_utils.py          # Tensor conversion utilities
    ├── validation.py            # Shape and value validation
    └── cli.py                   # Command-line interface

tests/
├── test_weight_conversion/      # Conversion tool tests
│   ├── __init__.py
│   ├── test_converter.py        # WeightConverter class tests
│   ├── test_name_mapping.py     # Name mapping tests
│   ├── test_tensor_utils.py     # Tensor conversion tests
│   ├── test_validation.py       # Validation tests
│   ├── test_cli.py              # CLI interface tests
│   └── fixtures/                # Test data
│       ├── sample_paddle.pdparams
│       └── expected_mappings.json
└── integration/
    └── test_full_conversion.py  # End-to-end conversion tests

pretrained_models/
├── paddle/                      # Source PaddlePaddle weights (existing)
│   ├── rtdetrv3_r18vd_6x_coco.pdparams
│   ├── rtdetrv3_r34vd_6x_coco.pdparams
│   └── rtdetrv3_r50vd_6x_coco.pdparams
└── pytorch/                     # Converted PyTorch weights (to be generated)
    ├── rtdetrv3_r18vd_6x_coco.pth
    ├── rtdetrv3_r34vd_6x_coco.pth
    └── rtdetrv3_r50vd_6x_coco.pth
```

**Structure Decision**: Single project structure is appropriate because this is a standalone CLI utility tool. The conversion logic is encapsulated in a dedicated `weight_conversion` module under `tools/`, with the existing `convert_weights.py` serving as the entry point. Test structure mirrors the source structure for clarity.

## Complexity Tracking

*No constitution violations requiring justification.*

This feature aligns with all constitution principles and requires no complexity budget allocation.

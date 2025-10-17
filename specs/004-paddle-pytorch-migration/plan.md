# Implementation Plan: RT-DETRv3 Paddle to PyTorch Migration Completion

**Branch**: `004-paddle-pytorch-migration` | **Date**: 2025-10-17 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-paddle-pytorch-migration/spec.md`

## Summary

Complete the migration of RT-DETRv3 component construction system from PaddlePaddle style to PyTorch, implementing registry-based component registration, dependency injection chain, and config-driven model building. The primary requirement is to ensure all 8 core components (RTDETRv3, ResNet, HybridEncoder, RTDETRTransformerv3, DINOv3Head, PPYOLOEHead, DINOv3Loss, and one additional component) are properly registered and support PaddlePaddle-style instantiation while maintaining 100% backward compatibility with existing direct instantiation code.

**Technical Approach**: Enhance the existing Registry class in `rtdetrv3_pytorch/models/__init__.py` to support `__inject__`, `__shared__`, and `__category__` annotations. Implement `from_config()` class methods in all components to enable dependency injection chain (backbone.out_shape → neck → transformer → head). Provide global `create()` function as the PaddlePaddle-style entry point for config-driven model building.

## Technical Context

**Language/Version**: Python 3.9+ (required, as specified in pyproject.toml)
**Primary Dependencies**:
  - PyTorch ≥2.5.1
  - torchvision ≥0.20.1
  - PaddlePaddle (for validation only)
  - pytest ≥8.4.2 (testing)
  - PyYAML ≥6.0 (config parsing)

**Storage**: N/A (model architecture only, no persistence layer)
**Testing**: pytest with markers for unit/integration/numerical tests
**Target Platform**: Linux/Windows/macOS (cross-platform Python)
**Project Type**: Single project (deep learning model library)
**Performance Goals**:
  - Registry lookup: O(1) constant time
  - Component instantiation overhead: <5ms per component
  - Config parsing: <100ms for typical YAML files

**Constraints**:
  - Must maintain 100% backward compatibility (FR-009, SC-004)
  - No breaking changes to existing component __init__ signatures
  - Registry system must be thread-safe for multi-GPU training
  - Dependency injection must not create circular dependencies

**Scale/Scope**:
  - 8 core components to migrate
  - 6 registry categories (ARCHITECTURE, BACKBONE, NECK, TRANSFORMER, HEAD, LOSS)
  - ~15 functional requirements to satisfy
  - Expected code additions: ~500 lines (registry enhancements + from_config methods)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Gate 1: Framework Parity First
✅ **PASS** - This feature enhances the component construction system to match PaddlePaddle's style, supporting the constitution's goal of framework parity. The migration guide documents (PADDLE_MIGRATION_SUMMARY.md, PADDLE_STYLE_MIGRATION.md) already establish the API mapping strategy.

### Gate 2: Modular Migration Strategy
✅ **PASS** - The feature follows modular approach: Registry system → Component annotations → Dependency injection → Config-driven building. Each phase can be validated independently.

### Gate 3: Validation-Driven Development (NON-NEGOTIABLE)
✅ **PASS** - Success criteria SC-001, SC-003, SC-004, SC-005 define specific validation targets. Existing validation script `verify_paddle_migration.py` will be enhanced to verify registration and injection functionality.

**Validation Plan**:
1. Unit tests for Registry.create() method with __inject__ and __shared__ support
2. Integration tests for dependency injection chain (backbone → neck → transformer → head)
3. Backward compatibility tests ensuring direct instantiation still works
4. Numerical equivalence tests to confirm registered components produce identical outputs

### Gate 4: Reproducibility & Documentation
✅ **PASS** - Requirements FR-001 through FR-015 explicitly document the mapping between PaddlePaddle and PyTorch registry systems. Success criterion SC-007 mandates 100% documentation completeness.

### Gate 5: Performance Parity
✅ **PASS** - Performance goals defined in Technical Context section. Registry overhead (<5ms per component) ensures minimal impact on overall training/inference speed.

### Gate 6: Configuration Compatibility
✅ **PASS** - Requirement FR-012 mandates support for nested config structures matching PaddlePaddle's YAML format. User Story 3 (Priority P2) focuses on YAML/dict-driven model building.

### Constitution Compliance Summary
**Status**: ✅ **ALL GATES PASSED**

No violations detected. This feature aligns with all constitution principles:
- Preserves framework parity by implementing PaddlePaddle's construction patterns
- Follows modular strategy (registry → injection → config)
- Includes comprehensive validation (unit, integration, backward compatibility)
- Fully documented with API mapping and examples
- Maintains performance with minimal overhead
- Supports PaddlePaddle config file compatibility

## Project Structure

### Documentation (this feature)

```
specs/004-paddle-pytorch-migration/
├── plan.md              # This file (/speckit.plan output)
├── research.md          # Phase 0 output - Registry pattern research
├── data-model.md        # Phase 1 output - Component metadata schema
├── quickstart.md        # Phase 1 output - Migration quickstart guide
├── contracts/           # Phase 1 output - Registry API contracts
│   ├── registry-api.md  # Registry class public API
│   ├── component-protocol.md  # Component annotation protocol
│   └── config-schema.yaml     # Configuration file schema
└── tasks.md             # Phase 2 output (/speckit.tasks - NOT created yet)
```

### Source Code (repository root)

```
rtdetrv3_pytorch/
├── models/
│   ├── __init__.py           # ✏️ MODIFY: Enhanced Registry with injection support
│   ├── rtdetrv3.py           # ✏️ MODIFY: Add from_config() class method
│   ├── backbones/
│   │   ├── __init__.py       # ✏️ MODIFY: Auto-register components
│   │   └── resnet.py         # ✏️ MODIFY: Add @register + from_config()
│   ├── necks/
│   │   ├── __init__.py       # ✏️ MODIFY: Auto-register
│   │   └── hybrid_encoder.py # ✏️ MODIFY: Add @register
│   ├── transformers/
│   │   ├── __init__.py       # ✏️ MODIFY: Auto-register
│   │   └── rtdetr_transformer.py # ✏️ MODIFY: Add @register
│   ├── heads/
│   │   ├── __init__.py       # ✏️ MODIFY: Auto-register
│   │   ├── detr_head.py      # ✏️ MODIFY: Add @register (DINOv3Head)
│   │   └── ppyoloe_head.py   # ✏️ MODIFY: Add @register (PPYOLOEHead)
│   └── losses/
│       ├── __init__.py       # ✏️ MODIFY: Auto-register + explicit import
│       └── detr_loss.py      # ✏️ MODIFY: Add @register (DINOv3Loss)
│
├── tests/
│   ├── unit/
│   │   └── test_registry.py  # ✨ NEW: Registry system unit tests
│   ├── integration/
│   │   └── test_config_driven_build.py  # ✨ NEW: Config-driven building tests
│   └── numerical/
│       └── test_registered_components.py  # ✨ NEW: Verify registered == direct
│
├── tools/
│   └── validate_migration.py  # ✏️ MODIFY: Enhanced verification script
│
└── configs/
    └── examples/
        └── rtdetrv3_r50_paddle_style.yml  # ✨ NEW: Example PaddlePaddle-style config

# Root level scripts
verify_paddle_migration.py     # ✏️ MODIFY: Add registration status reporting
test_registry_system.py        # ✏️ MODIFY: Add from_config() testing
```

**Structure Decision**: Single project structure (Option 1) is used as RT-DETRv3 is a model library without separate frontend/backend or mobile components. All enhancements are within the `rtdetrv3_pytorch/models/` directory, maintaining the existing modular organization (backbones, necks, transformers, heads, losses).

**Legend**:
- ✏️ MODIFY: Existing file requiring modifications
- ✨ NEW: New file to be created

## Complexity Tracking

*No constitution violations detected. This section is intentionally left empty.*

The feature adds functionality (registry enhancements, dependency injection) without introducing architectural complexity that violates constitution principles. All changes are additive and maintain backward compatibility.

## Phase 0: Research & Technical Decisions

**Status**: ⏳ In Progress

### Research Tasks

1. **Registry Pattern Best Practices**
   - Question: How do popular ML frameworks (TensorFlow, Keras, Detectron2) implement component registries?
   - Focus: Thread safety, performance optimization, error handling patterns

2. **Dependency Injection Patterns in Python**
   - Question: What are the standard patterns for dependency injection in Python dataclasses and classes?
   - Focus: Annotation-based injection, constructor injection, property injection

3. **PaddlePaddle `ppdet.core.workspace` API**
   - Question: What is the exact behavior of PaddlePaddle's `create()`, `register()`, and `from_config()` methods?
   - Focus: Parameter resolution order, global config handling, error messages

4. **YAML Configuration Schema Design**
   - Question: How to design a schema that supports both flat and nested component configs with type validation?
   - Focus: PyYAML parsing, schema validation libraries (e.g., cerberus, pydantic)

5. **Backward Compatibility Strategies**
   - Question: How to ensure decorators and metaclasses don't break existing instantiation?
   - Focus: Non-invasive decorator design, optional registry participation

**Output**: All findings will be consolidated in `research.md`

## Phase 1: Design Artifacts

**Status**: 🔜 Pending (Phase 0 completion)

### Planned Artifacts

1. **`data-model.md`**: Component metadata schema
   - Registry entry structure
   - Component annotation fields (__category__, __inject__, __shared__)
   - Configuration dictionary format
   - Dependency graph representation

2. **`contracts/registry-api.md`**: Registry class public API
   - `register(name=None)` decorator signature
   - `create(name, global_config=None, **kwargs)` method
   - `get(name)`, `list()`, `has(name)` methods
   - Error handling specifications

3. **`contracts/component-protocol.md`**: Component annotation protocol
   - Required attributes for registered components
   - `from_config(cls, cfg, global_config=None)` class method signature
   - Dependency injection resolution rules
   - Naming conventions

4. **`contracts/config-schema.yaml`**: YAML configuration schema
   - Top-level keys (architecture, global config)
   - Component config structure (type + parameters)
   - Nested component references
   - Default value handling

5. **`quickstart.md`**: Migration quickstart guide
   - Step 1: Add @register decorator to component
   - Step 2: Define __category__ and __inject__ attributes
   - Step 3: Implement from_config() class method
   - Step 4: Test with create() function
   - Step 5: Verify backward compatibility

**Post-Design Actions**:
- Run `.specify/scripts/bash/update-agent-context.sh claude` to update agent context
- Re-evaluate Constitution Check (expected to remain PASS)

## Notes

- This plan focuses on architectural enhancements to the component construction system
- No changes to model forward passes or loss computations (preserves numerical equivalence)
- The migration is non-breaking: all existing code using direct instantiation continues to work
- Validation emphasis: registry correctness, injection chain integrity, config compatibility

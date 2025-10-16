<!--
Sync Impact Report:
- Version: 0.0.0 → 1.0.0 (Initial constitution creation for framework migration project)
- Modified Principles: N/A (initial creation)
- Added Sections: All core principles and governance sections
- Removed Sections: N/A
- Templates Requiring Updates:
  ✅ plan-template.md: Aligned with framework migration workflow
  ✅ spec-template.md: Aligned with user scenarios for migration features
  ✅ tasks-template.md: Aligned with migration task types
- Follow-up TODOs: None
-->

# RT-DETRv3 PyTorch Migration Constitution

## Core Principles

### I. Framework Parity First

Every migrated component must maintain functional equivalence with the PaddlePaddle implementation. Mathematical correctness takes absolute priority over code style or performance optimization. All model outputs, intermediate activations, and loss values must be validated against the reference implementation within numerical tolerance (default: 1e-5 for FP32).

**Rationale**: The PaddlePaddle implementation is the validated reference (WACV 2025 Oral paper). Deviations risk invalidating published results and breaking downstream research.

### II. Modular Migration Strategy

Migration proceeds module by module following the dependency graph: backbone → neck → encoder → decoder → heads → loss → full model. Each module must pass standalone validation before integration. No monolithic rewrites.

**Rationale**: Modular approach enables incremental validation, easier debugging, and parallel development. Reduces risk of introducing correlated errors across components.

### III. Validation-Driven Development (NON-NEGOTIABLE)

For each migrated component:
1. Write numerical equivalence tests comparing PyTorch vs PaddlePaddle outputs
2. Tests must FAIL initially (proving they detect differences)
3. Implement migration
4. Tests must PASS (within tolerance) before component is considered complete

No exceptions. Validation infrastructure is as important as the migration code itself.

**Rationale**: Deep learning frameworks have subtle differences in operators, numerical stability, and default behaviors. Only systematic numerical testing can guarantee correctness.

### IV. Reproducibility & Documentation

Every migrated component must include:
- Mapping table: PaddlePaddle API → PyTorch API (e.g., `paddle.nn.Linear` → `torch.nn.Linear`)
- Notes on behavioral differences requiring workarounds
- Numerical tolerance values used in validation
- Random seed management for stochastic operations

**Rationale**: Framework migrations involve hundreds of API mappings and edge cases. Documentation prevents regression and enables knowledge transfer.

### V. Performance Parity

After correctness validation, performance must match or exceed the PaddlePaddle baseline:
- Training throughput: ≥95% of reference (measured in samples/sec on same hardware)
- Memory usage: ≤110% of reference
- Inference latency: ≤105% of reference (especially critical for real-time detection)

Performance regression must be investigated and justified before acceptance.

**Rationale**: RT-DETRv3 is a real-time detector. Significant performance loss would invalidate the "real-time" claim and limit adoption.

### VI. Configuration Compatibility

Maintain backward compatibility with existing PaddlePaddle config files where feasible. Provide automated config conversion tools (YAML → PyTorch format) to ease user transition. Breaking changes require migration guides.

**Rationale**: Users have invested time tuning hyperparameters on PaddlePaddle. Breaking configs without migration paths creates friction and slows adoption.

## Migration Workflow Constraints

### Pre-Migration Research Phase

Before migrating any component, MUST complete:
1. Catalog all PaddlePaddle APIs used in the component
2. Identify PyTorch equivalents and behavioral differences
3. Document known numerical pitfalls (e.g., padding modes, reduction methods)
4. Design validation strategy (reference inputs, expected outputs, tolerance)

### Implementation Phase Rules

- Use PyTorch native operations when available (avoid custom CUDA kernels unless necessary)
- Preserve original variable names and code structure where possible (aids review)
- Add inline comments for non-obvious API mappings
- Use PyTorch best practices: `torch.nn.Module` for layers, proper device management, autograd-compatible operations

### Validation Phase Requirements

- Numerical tests MUST use deterministic inputs (fixed random seeds)
- Test multiple input shapes and batch sizes
- Validate both forward pass and backward pass (gradient equivalence)
- Test edge cases: empty batches, single-item batches, maximum model capacity
- Compare on CPU first (easier debugging), then GPU

### Integration Testing

After component validation:
1. Integrate into larger subsystem (e.g., add encoder to backbone+neck)
2. Re-validate subsystem outputs against reference
3. Run end-to-end inference on COCO val samples
4. Compare mAP, latency, memory with PaddlePaddle baseline

## Quality Gates

### Gate 1: Component Completion
- [ ] All PaddlePaddle APIs mapped to PyTorch
- [ ] Numerical equivalence tests pass (forward + backward)
- [ ] Code review approved by ≥1 reviewer familiar with both frameworks
- [ ] Documentation complete (API mapping, behavioral notes)

### Gate 2: Subsystem Validation
- [ ] Component integrated into subsystem
- [ ] Subsystem-level numerical tests pass
- [ ] No performance regression >5% vs component in isolation

### Gate 3: Full Model Validation
- [ ] End-to-end model runs on COCO dataset
- [ ] mAP matches PaddlePaddle baseline (±0.1 AP tolerance)
- [ ] Training convergence matches reference (loss curves, epoch-wise mAP)
- [ ] Inference speed within ±5% of reference on same hardware

### Gate 4: Release Readiness
- [ ] All components pass validation
- [ ] Config conversion tools tested on ≥5 config files
- [ ] Documentation includes migration guide and troubleshooting
- [ ] Pre-trained weights converted and validated (checkpoint compatibility)

## Technology Standards

**Primary Framework**: PyTorch ≥2.0 (for `torch.compile` support)
**Python Version**: 3.8+ (minimum for compatibility), 3.11 recommended
**Dependencies**: torchvision, numpy, pycocotools, opencv-python, pyyaml
**Testing Framework**: pytest with `pytest-xdist` for parallel test execution
**GPU Support**: CUDA 11.8+ or ROCm 5.4+ (document multi-backend support)
**Hardware Validation**: NVIDIA T4 (reference GPU from paper), A100 (training)
**Numerical Precision**: FP32 for validation, FP16/BF16 for production (after validation)

## Governance

**Constitution Authority**: This document supersedes all other development practices for the RT-DETRv3 PyTorch migration project. When in conflict, constitution rules take precedence.

**Amendment Process**:
1. Propose change via issue/PR with rationale
2. Discuss impact on validation strategy and timeline
3. Require approval from ≥2 core contributors
4. Update constitution version (semantic versioning)
5. Announce breaking changes to all stakeholders

**Compliance Verification**:
- All PRs must reference relevant constitution principles in description
- Code reviews MUST verify adherence to validation requirements (Principle III)
- CI pipeline enforces automated numerical tests before merge
- Monthly audit of test coverage and validation completeness

**Complexity Budget**:
- Avoid introducing abstractions not present in PaddlePaddle version (preserve simplicity)
- New utilities (e.g., profiling, config conversion) must be independently documented
- If PyTorch requires >20% more lines of code for same functionality, investigate alternative approach

**Version**: 1.0.0 | **Ratified**: 2025-10-14 | **Last Amended**: 2025-10-14

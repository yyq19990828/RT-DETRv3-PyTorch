# Implementation Plan: PaddlePaddle to PyTorch Migration

**Branch**: `001-i-want-to` | **Date**: 2025-10-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `RT-DETRv3/specs/001-i-want-to/spec.md`

**Note**: This plan guides the migration of RT-DETRv3 from PaddlePaddle to PyTorch with focus on numerical equivalence, performance parity, and code structure preservation.

## Summary

Migrate RT-DETRv3 real-time object detection model from PaddlePaddle to PyTorch while maintaining:
- **Numerical Equivalence**: Model outputs match PaddlePaddle baseline within ±1e-4 (activations) and ±0.01 (predictions)
- **Performance Parity**: Training throughput ≥95%, inference latency ≤105%, memory ≤110% of baseline
- **Code Structure**: >80% similarity in module organization and naming conventions
- **Training Capability**: Full training pipeline with DDP, mixed precision, checkpoint conversion
- **Deployment**: Export to ONNX and TorchScript for production deployment

## Technical Context

**Language/Version**: Python 3.8+ (3.11 recommended for optimal performance)
**Primary Dependencies**: PyTorch 2.5.1, torchvision 0.20.1, MultiScaleDeformableAttention, pycocotools, opencv-python, pyyaml
**Storage**: File-based checkpoints (.pth format), COCO dataset (images + JSON annotations)
**Testing**: pytest with pytest-xdist for parallel execution, numerical equivalence tests vs PaddlePaddle
**Target Platform**: Linux (primary), NVIDIA GPUs with CUDA 11.8+ or CUDA 12.1+, T4 for inference, A100 for training
**Project Type**: Single project (deep learning library)
**Performance Goals**:
- Training: ≥95% of PaddlePaddle throughput (samples/sec)
- Inference: RT-DETRv3-R50 ≥108 FPS on T4 GPU with TensorRT FP16
- mAP: 53.4±0.5% on COCO val2017 (R50 variant)
**Constraints**:
- Memory: ≤110% of PaddlePaddle peak GPU memory
- Numerical: ±1e-4 tolerance for activations, ±0.01 for predictions
- Code similarity: >80% structure match with PaddlePaddle version
**Scale/Scope**:
- Model: ~42M parameters (R50), 4 variants (R18/R34/R50/R101)
- Training: 72 epochs on COCO train2017 (~118K images)
- Codebase: ~5-8K LOC (estimated, mirroring Paddle structure)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on RT-DETRv3 PyTorch Migration Constitution v1.0.0:

### Pre-Research Gates (Phase 0) ✅ PASSED

- [x] **Principle I - Framework Parity First**: Spec defines numerical tolerance (±1e-4 activations, ±0.01 predictions)
- [x] **Principle II - Modular Migration**: Plan follows dependency graph (backbone → neck → encoder → decoder → heads)
- [x] **Principle III - Validation-Driven**: Spec includes numerical equivalence testing (FR-013, SC-005)
- [x] **Principle IV - Documentation**: Research.md documents API mappings and behavioral differences
- [x] **Principle V - Performance Parity**: Spec defines performance targets (≥95% throughput, ≤105% latency)
- [x] **Principle VI - Config Compatibility**: Plan includes YAML config conversion utility

### Post-Design Gates (Phase 1) - TO BE VERIFIED

- [ ] **Gate 1 - Component Completion**: All PaddlePaddle APIs cataloged and mapped to PyTorch
- [ ] **Documentation**: API mapping tables generated for each component
- [ ] **Test Design**: Numerical equivalence test structure defined in data-model.md
- [ ] **Performance Baseline**: PaddlePaddle benchmarks measured and documented

**Status**: No constitution violations. All principles addressed in spec and research phase.

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```
rtdetrv3_pytorch/
├── models/
│   ├── backbones/              # ResNet-18/34/50/101 implementations
│   ├── necks/                  # HybridEncoder (FPN-PAN)
│   ├── transformers/           # RTDETRTransformerv3, attention ops
│   ├── heads/                  # DINOv3Head, PPYOLOEHead
│   ├── losses/                 # DINOv3Loss with multi-branch support
│   └── rtdetrv3.py             # Main model class
├── data/
│   ├── coco_dataset.py         # COCO dataset loader
│   ├── transforms.py           # Data augmentation
│   └── collate.py              # Batch collation
├── engine/
│   ├── trainer.py              # Training loop with DDP
│   ├── evaluator.py            # COCO evaluation
│   └── optimizer.py            # Optimizer and LR scheduler
├── utils/
│   ├── checkpoint.py           # Save/load checkpoints
│   ├── config.py               # YAML config parser
│   ├── distributed.py          # DDP utilities
│   └── logger.py               # Logging
├── tools/
│   ├── train.py                # Training entry point
│   ├── eval.py                 # Evaluation entry point
│   ├── infer.py                # Inference entry point
│   ├── export_onnx.py          # ONNX export
│   └── convert_weights.py      # Paddle → PyTorch conversion
├── configs/                     # YAML configuration files
└── tests/
    ├── unit/                    # Component tests
    ├── integration/             # System tests
    └── numerical/               # Equivalence vs Paddle
```

**Structure Decision**: Single project structure chosen (Option 1). Deep learning library with modular components mirroring PaddlePaddle organization to satisfy >80% similarity requirement (Constitution Principle V, FR-015). Tests organized by granularity (unit → integration → numerical) to support validation-driven development (Constitution Principle III).

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |

# Feature Specification: PaddlePaddle to PyTorch Migration

**Feature Branch**: `001-i-want-to`
**Created**: 2025-10-14
**Status**: Draft
**Input**: User description: "I want to migrate the padlle version to pytorch version, I need to create a new RT-DETRv3-pytorch. Migration requires ensuring consistency of structure. Reuse existing code as much as possible, modifying only the necessary parts, for example converting Paddle APIs to PyTorch APIs"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Core Model Execution (Priority: P1)

As a researcher or engineer, I want to load a migrated RT-DETRv3 PyTorch model and run inference on images to obtain the same detection results as the PaddlePaddle version, so that I can verify the migration maintains functional correctness.

**Why this priority**: This is the fundamental requirement - without a working model that produces correct outputs, no other functionality matters. This validates the core migration effort.

**Independent Test**: Can be fully tested by loading pre-trained weights into the PyTorch model, running inference on COCO validation images, and comparing detection outputs (bounding boxes, class predictions, confidence scores) against the PaddlePaddle baseline. Delivers a functional detector.

**Acceptance Scenarios**:

1. **Given** a COCO validation image and PaddlePaddle RT-DETRv3 checkpoint, **When** I load the checkpoint into PyTorch model and run inference, **Then** the detected bounding boxes match PaddlePaddle outputs within ±2 pixels and confidence scores match within ±0.01
2. **Given** a batch of images (batch size 1, 4, 8), **When** I run inference on PyTorch model, **Then** all batch sizes produce numerically equivalent results to PaddlePaddle (mAP difference <0.1%)
3. **Given** pre-trained weights from PaddlePaddle, **When** I convert and load them into PyTorch model, **Then** the model architecture accepts the weights without shape mismatches

---

### User Story 2 - Model Training (Priority: P2)

As a researcher, I want to train the RT-DETRv3 PyTorch model from scratch on COCO dataset to achieve the same accuracy (mAP) as reported in the paper, so that I can reproduce published results and fine-tune on custom datasets.

**Why this priority**: Training capability is essential for research reproducibility and customization, but it depends on having a working model architecture (P1). This validates the complete learning pipeline including loss functions and optimizers.

**Independent Test**: Can be fully tested by training the PyTorch model on COCO train2017 for the full schedule (6x epochs), evaluating on val2017, and comparing final mAP against the published PaddlePaddle baseline (±0.5% tolerance). Delivers a trainable research platform.

**Acceptance Scenarios**:

1. **Given** COCO train2017 dataset and default hyperparameters, **When** I train PyTorch RT-DETRv3-R50 for 72 epochs, **Then** final mAP on val2017 is 53.4 ± 0.5%
2. **Given** a training run in progress, **When** I monitor loss curves every epoch, **Then** loss convergence pattern matches PaddlePaddle baseline (same loss values ±5% at corresponding epochs)
3. **Given** identical data augmentation settings, **When** I train models with different random seeds, **Then** variance in final mAP is <0.3% (demonstrating stable training)
4. **Given** multi-GPU training setup (4 GPUs), **When** I train using distributed data parallel, **Then** training completes successfully and achieves same mAP as single-GPU training

---

### User Story 3 - Configuration and Deployment (Priority: P3)

As a practitioner, I want to use existing PaddlePaddle configuration files with the PyTorch version (with minimal modifications) and export the model for deployment, so that I can leverage existing knowledge and deploy to production systems.

**Why this priority**: Usability and deployment enable real-world adoption, but these depend on having a trained model (P1, P2). This reduces migration friction for existing users.

**Independent Test**: Can be fully tested by loading a PaddlePaddle YAML config, applying automated conversion (if available) or manual adjustment, training/evaluating with PyTorch, and exporting to ONNX/TorchScript for deployment. Delivers production-ready artifacts.

**Acceptance Scenarios**:

1. **Given** a PaddlePaddle config YAML (e.g., rtdetrv3_r50vd_6x_coco.yml), **When** I load it into PyTorch training script with documented adjustments, **Then** training proceeds without config parsing errors
2. **Given** a trained PyTorch checkpoint, **When** I export to ONNX format, **Then** ONNX model runs in ONNXRuntime and produces equivalent outputs (mAP difference <0.2%)
3. **Given** a trained PyTorch model, **When** I export to TorchScript, **Then** TorchScript model can be loaded in C++ libtorch environment and runs inference successfully
4. **Given** exported ONNX model, **When** I convert to TensorRT engine, **Then** inference speed matches or exceeds PaddlePaddle TensorRT baseline (e.g., ≥217 FPS for R18 on T4 GPU)

---

### Edge Cases

- **Empty batch handling**: What happens when inference is called with an empty image list or batch size 0?
- **Out-of-memory scenarios**: How does the system handle images larger than GPU memory capacity during training or inference?
- **Checkpoint version mismatches**: What happens if a user tries to load a PaddlePaddle checkpoint directly without conversion?
- **Missing configuration parameters**: How does the system behave when a required hyperparameter is omitted from config?
- **Numerical precision edge cases**: How does the system handle extreme values (very small/large coordinates, near-zero confidence scores) that might cause numerical instability?
- **Distributed training failures**: What happens when one GPU fails mid-training in a multi-GPU setup?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST produce numerically equivalent model outputs (bounding boxes, class predictions, confidence scores) to PaddlePaddle baseline for the same inputs and weights, within tolerance of ±1e-4 for intermediate activations and ±0.01 for final predictions
- **FR-002**: System MUST support all model variants documented in the PaddlePaddle version (RT-DETRv3-R18, R34, R50, R101) with same backbone architectures
- **FR-003**: System MUST accept pre-trained PaddlePaddle checkpoints after conversion and load weights correctly (all layers, no shape mismatches)
- **FR-004**: System MUST implement all three core components: CNN-based auxiliary branch, multi-group self-attention perturbation, and one-to-many dense supervision branch
- **FR-005**: System MUST support training on COCO dataset (train2017) and evaluation on COCO val2017 with standard metrics (mAP, AP50, AP75)
- **FR-006**: System MUST maintain training convergence characteristics matching PaddlePaddle (loss curves, learning rate schedules, epoch-wise mAP progression)
- **FR-007**: System MUST support multi-GPU distributed training with data parallel strategy (4 GPUs minimum as per paper setup)
- **FR-008**: System MUST preserve inference speed characteristics (e.g., RT-DETRv3-R50 achieves ≥108 FPS on T4 GPU with TensorRT FP16)
- **FR-009**: System MUST accept configuration files compatible with PaddlePaddle format (YAML-based) with documented necessary adjustments
- **FR-010**: System MUST export trained models to standard deployment formats: ONNX (opset 16+) and TorchScript
- **FR-011**: System MUST implement gradient accumulation and mixed precision training (FP16/BF16) for memory efficiency
- **FR-012**: System MUST provide checkpoint saving/loading functionality with optimizer states for training resumption
- **FR-013**: System MUST validate numerical correctness through automated testing comparing PyTorch outputs against PaddlePaddle reference outputs
- **FR-014**: System MUST handle dynamic input shapes during inference (support various image resolutions, not just fixed 640×640)
- **FR-015**: System MUST preserve module structure and naming conventions from PaddlePaddle version to maintain code readability and maintainability

### Assumptions

- **A-001**: PyTorch version is ≥2.0 (for `torch.compile` support and modern API compatibility)
- **A-002**: Users have access to COCO dataset (standard benchmark, publicly available)
- **A-003**: Target hardware includes NVIDIA GPUs with CUDA support (primary deployment target as per paper)
- **A-004**: Numerical tolerance of ±0.5% mAP is acceptable for research purposes (within experimental variance)
- **A-005**: Pre-trained PaddlePaddle checkpoints are available for conversion (provided in original release)
- **A-006**: Users have basic familiarity with PyTorch ecosystem (model training, checkpoint management)
- **A-007**: Configuration adjustments between frameworks are acceptable if clearly documented
- **A-008**: Deformable attention operators have PyTorch equivalents or can be adapted from existing libraries (e.g., MultiScaleDeformableAttention)

### Key Entities

- **Model Architecture**: The neural network structure comprising backbone (ResNet-18/34/50/101), neck (Hybrid Encoder with FPN-PAN), transformer encoder, multi-group decoder, detection heads (DINOv3Head for main branch, PPYOLOEHead for auxiliary branch)
- **Training Configuration**: Hyperparameters including learning rate schedule, batch size, optimizer settings, data augmentation pipeline, loss weights (for auxiliary branch, one-to-one matching, one-to-many matching)
- **Checkpoint**: Serialized model weights including all learnable parameters (backbone, encoder, decoder, heads), optimizer states, training metadata (epoch, iteration count)
- **Dataset Sample**: Input image with corresponding annotations (bounding boxes in COCO format: [x, y, width, height], class labels, image metadata)
- **Detection Output**: Model prediction consisting of bounding boxes (normalized coordinates), class probabilities (80 classes for COCO), confidence scores, organized per image in batch
- **Loss Components**: Multi-part loss function including classification loss (Varifocal Loss), bounding box regression loss (GIoU Loss + L1), auxiliary branch loss, denoising loss
- **Validation Metrics**: COCO evaluation metrics including mAP (mean Average Precision), AP50, AP75, AP_small, AP_medium, AP_large
- **Exported Model**: Deployment-ready artifact in ONNX or TorchScript format with frozen weights, supporting inference-only operations

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Migrated PyTorch models achieve published mAP benchmarks on COCO val2017 within ±0.5% tolerance (e.g., R50: 53.4±0.5% mAP, R18: 48.1±0.5% mAP)
- **SC-002**: Inference speed on NVIDIA T4 GPU matches or exceeds PaddlePaddle baseline for all model variants (±5% tolerance, e.g., R50: ≥108 FPS with TensorRT FP16)
- **SC-003**: Training time for 72 epochs on COCO train2017 using 4×A100 GPUs is within 110% of PaddlePaddle baseline duration
- **SC-004**: Memory consumption during training (peak GPU memory) is within 110% of PaddlePaddle baseline for same batch size
- **SC-005**: Numerical equivalence tests pass for all model components with >99% of test cases within tolerance (±1e-4 for activations, ±0.01 for predictions)
- **SC-006**: Exported ONNX models run successfully in ONNXRuntime with <0.2% mAP degradation compared to native PyTorch
- **SC-007**: Configuration migration process (PaddlePaddle YAML to PyTorch) documented with clear step-by-step instructions taking <15 minutes per config file
- **SC-008**: Pre-trained checkpoint conversion tool successfully converts all released PaddlePaddle weights (R18, R34, R50) without manual intervention
- **SC-009**: Training convergence stability demonstrated across 5 independent runs with different random seeds, achieving mAP standard deviation <0.3%
- **SC-010**: Code structure maintains >80% similarity in module organization and naming conventions compared to PaddlePaddle version (as measured by directory structure and class/function names)

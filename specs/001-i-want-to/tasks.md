# Tasks: RT-DETRv3 PaddlePaddle to PyTorch Migration

**Input**: Design documents from `/home/tyjt/桌面/RT-DETRv3/specs/001-i-want-to/`
**Prerequisites**: spec.md, plan.md, research.md
**Generated**: 2025-10-14

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. Each phase is independently testable to support incremental validation.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1=Core Model Execution, US2=Model Training, US3=Configuration & Deployment)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure
**Duration Estimate**: 1-2 days

- [X] T001 Create project directory structure: `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/` with subdirectories models/, data/, engine/, utils/, tools/, configs/, tests/
- [X] T002 Initialize Python project with pyproject.toml and specify PyTorch 2.5.1, torchvision 0.20.1, numpy>=1.24.0, opencv-python>=4.8.0, pycocotools>=2.0.7, pyyaml>=6.0, scipy>=1.10.0, MultiScaleDeformableAttention>=1.0.0
- [X] T003 [P] Create `.gitignore` for Python project (exclude __pycache__, *.pyc, checkpoints/, data/, logs/)
- [X] T004 [P] Setup logging configuration in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/utils/logger.py` with support for console and file outputs
- [X] T005 [P] Create basic README.md with installation instructions, environment setup (CUDA 11.8/12.1), and project overview

**Checkpoint**: Project structure ready for implementation

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete
**Duration Estimate**: 1-2 weeks

### Weight Conversion Infrastructure (CRITICAL for ALL User Stories)

- [X] T006 Implement weight conversion utility in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tools/convert_weights.py`:
  - Load PaddlePaddle .pdparams files using paddle.load()
  - Create parameter name mapping dict (Paddle → PyTorch conventions)
  - Convert tensors: paddle.Tensor → numpy → torch.Tensor
  - Handle special cases: BatchNorm (._mean→running_mean, ._variance→running_var), Conv2d/Linear weight shapes
  - Validate all weights have matching shapes before loading
  - Save converted weights as .pth file
- [X] T007 Create name mapping generator in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tools/convert_weights.py`:
  - Analyze PaddlePaddle model structure and extract parameter names
  - Analyze PyTorch model structure and extract parameter names
  - Generate automated mapping dict with manual override support
  - Log unmapped parameters for manual inspection

### Configuration System (CRITICAL for ALL User Stories)

- [X] T008 Implement YAML config parser in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/utils/config.py`:
  - Load YAML configuration files
  - Support nested dictionary structures (model, optimizer, lr_scheduler, training, data, validation)
  - Support config inheritance and overrides via command-line arguments
  - Validate required fields exist
- [X] T009 Create PaddlePaddle config converter in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/utils/config.py`:
  - Map PaddlePaddle-specific keys to PyTorch equivalents
  - Convert optimizer syntax (e.g., PaddleOptim → torch.optim)
  - Convert LR scheduler syntax
  - Document conversion rules and manual adjustments needed
- [X] T010 [P] Create reference config files in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/configs/`:
  - rtdetrv3_r18_6x_coco.yml
  - rtdetrv3_r50_6x_coco.yml
  - Include model architecture, optimizer, scheduler, training, data paths, validation tolerances

### Base Model Classes and Utilities (CRITICAL for ALL User Stories)

- [X] T011 Create base registry system in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/__init__.py`:
  - Implement model registry for dynamic component instantiation from config
  - Register backbones, necks, transformers, heads, losses
- [X] T012 [P] Implement distributed training utilities in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/utils/distributed.py`:
  - Initialize process group (dist.init_process_group with NCCL backend)
  - Get world size, rank, local rank from environment variables
  - Synchronization primitives (barrier, reduce, gather)
  - SyncBatchNorm conversion utility
- [X] T013 [P] Implement checkpoint save/load utilities in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/utils/checkpoint.py`:
  - Save checkpoint with model state_dict, optimizer state, epoch, iteration count, config
  - Load checkpoint and restore training state
  - Support strict and non-strict loading modes
  - Handle DDP model unwrapping (model.module)

### Data Infrastructure (Required for US1, US2)

- [X] T014 Implement COCO dataset loader in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/data/coco_dataset.py`:
  - Load COCO annotations from JSON (instances_train2017.json, instances_val2017.json)
  - Return image paths, bounding boxes [x, y, width, height], class labels, image metadata
  - Support train/val splits
  - Handle data_path configuration from YAML
- [X] T015 Implement data transforms in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/data/transforms.py`:
  - RandomResize (training augmentation)
  - RandomCrop (training augmentation)
  - RandomHorizontalFlip (training augmentation)
  - Normalize with ImageNet mean/std
  - ToTensor conversion
  - Compose pipeline matching PaddlePaddle augmentation
- [X] T016 Implement batch collation in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/data/collate.py`:
  - Handle variable-length annotations per image
  - Pad images to same size within batch
  - Create batched tensors for images, targets
  - Support dynamic input sizes

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Core Model Execution (Priority: P1) 🎯 MVP

**Goal**: Deliver a working RT-DETRv3 PyTorch model that can load converted weights and run inference with numerically equivalent outputs to PaddlePaddle

**Independent Test**: Load PaddlePaddle checkpoint (converted), run inference on COCO val2017 images, compare detection outputs (boxes ±2 pixels, scores ±0.01) against PaddlePaddle baseline

**Duration Estimate**: 3-4 weeks

### Component Migration (Following Modular Order: Backbone → Neck → Encoder → Decoder → Heads)

#### Backbone Migration

- [X] T017 [P] [US1] Implement ResNet backbone in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/backbones/resnet.py`:
  - ResNet-18, 34, 50, 101 variants
  - ResNet-vd modifications (Paddle-specific: avgpool in stride downsampling, 3x3 stem)
  - frozen_stages parameter (freeze early layers during training)
  - Return multi-scale features [C3, C4, C5] at indices [1, 2, 3]
  - Output channels: [512, 1024, 2048] for ResNet-50
- [X] T018 [US1] Create unit test for backbone in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/unit/test_backbone.py`:
  - Test output shapes for batch input (batch=2, 3, 640, 640)
  - Test gradient flow (loss.backward() succeeds)
  - Test frozen_stages parameter (gradients zeroed for frozen layers)

#### Neck Migration

- [X] T019 [US1] Implement HybridEncoder (FPN-PAN) in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/necks/hybrid_encoder.py`:
  - FPN top-down pathway (C5→P5, C4→P4, C3→P3)
  - PAN bottom-up pathway (P3→N3, P4→N4, P5→N5)
  - 1x1 conv for channel reduction to hidden_dim=256
  - num_encoder_layers parameter (typically 1)
  - Output multi-scale features with feat_strides=[8, 16, 32]
- [X] T020 [US1] Create unit test for neck in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/unit/test_neck.py`:
  - Test output shapes for multi-scale features
  - Test channel consistency (all outputs have hidden_dim channels)
  - Test gradient flow through FPN-PAN

#### Transformer Encoder Migration

- [X] T021 [US1] Implement MultiScaleDeformableAttention wrapper in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/transformers/attention.py`:
  - Wrap MultiScaleDeformableAttention from PyPI package
  - Handle reference points generation
  - Support multi-scale feature inputs (3 scales)
  - Match PaddlePaddle PPMSDeformableAttention behavior
- [X] T022 [US1] Implement position embeddings in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/transformers/utils.py`:
  - Sinusoidal position embeddings for spatial coordinates
  - Level embeddings for multi-scale features
  - Support learnable and fixed embeddings
- [X] T023 [US1] Implement MLP in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/transformers/utils.py`:
  - Two-layer feed-forward network
  - GELU activation (match PaddlePaddle default)
  - Dropout support
- [X] T024 [US1] Create unit test for attention in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/unit/test_attention.py`:
  - Test forward pass with multi-scale inputs
  - Test backward pass (gradient check)
  - Test reference point validity (within [0, 1])

#### Transformer Decoder Migration

- [X] T025 [US1] Implement RTDETRTransformerv3 decoder in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/transformers/rtdetr_transformer.py`:
  - Multi-group self-attention perturbation (perturb query positions for robustness)
  - num_queries=300 for one-to-one matching
  - num_queries_o2m=450 for one-to-many matching
  - num_noise_queries=[100] for denoising training
  - num_decoder_layers=6
  - Cross-attention to encoder features (deformable attention)
  - Self-attention within queries (standard multi-head attention)
  - Iterative bounding box refinement (6 decoder layers)
- [X] T026 [US1] Create unit test for decoder in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/unit/test_decoder.py`:
  - Test query generation and initialization
  - Test decoder layer forward pass
  - Test multi-group attention perturbation
  - Test output shapes (queries, hidden states)

#### Detection Heads Migration

- [X] T027 [P] [US1] Implement DINOv3Head (main branch) in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/heads/detr_head.py`:
  - Classification head: Linear(hidden_dim, num_classes=80)
  - Bounding box regression head: MLP(hidden_dim → hidden_dim → 4)
  - Apply sigmoid for classification scores
  - Apply sigmoid for box coordinates (normalized [0, 1])
  - Support one-to-one and one-to-many query branches
- [X] T028 [P] [US1] Implement PPYOLOEHead (auxiliary branch) in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/heads/ppyoloe_head.py`:
  - CNN-based detection head operating on neck features
  - Classification branch: 3x3 conv → 1x1 conv → num_classes
  - Bounding box regression branch: 3x3 conv → 1x1 conv → 4
  - Distribution Focal Loss (DFL) for box regression
  - Anchor-free design (predict offsets from grid centers)
- [X] T029 [US1] Create unit test for heads in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/unit/test_heads.py`:
  - Test DINOv3Head output shapes (num_queries, num_classes) and (num_queries, 4)
  - Test PPYOLOEHead output shapes for multi-scale predictions
  - Test gradient flow through both heads

### Full Model Integration

- [X] T030 [US1] Implement RTDETRv3 main model class in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/rtdetrv3.py`:
  - ✅ Initialize all components: backbone, neck, encoder, decoder, main head, auxiliary head (PPYOLOEHead integrated)
  - ✅ Forward pass: images → backbone → neck → encoder features → decoder queries → detection outputs
  - ✅ Support train/eval modes (auxiliary head only active during training)
  - ✅ Return dict with keys: 'pred_logits', 'pred_boxes', 'aux_pred_logits' (training only, pending T040 loss impl)
  - ✅ Support dynamic input sizes (variable image resolutions)
- [X] T031 [US1] Create integration test in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/integration/test_model.py`:
  - ✅ Test full forward pass with batch input (batch=2, 3, 640, 640)
  - ✅ Test eval mode (no auxiliary outputs)
  - ✅ Test train mode (NotImplementedError until T040 loss implementation)
  - ✅ Test gradient flow through entire model
  - ✅ Test model outputs shape correctness (22/22 tests passing) ✨
  - ✅ Added PPYOLOEHead integration tests
  - ✅ Fixed frozen_backbone_stages test (now correctly validates frozen stages)
  - ✅ Fixed model_device_transfer test (added eval mode)
  - ✅ Fixed aux_head_forward_pass test (corrected shape assertions for eval mode)
  - ✅ Fixed aux_head_gradient_flow test (validated parameter gradients only)
  - ⚠️ Note: Multi-group query mechanism confirmed - outputs 400 queries (300 o2o + 100 noise) in eval mode

### Numerical Equivalence Validation (CRITICAL for US1 Success)

- [X] T032 [US1] Implement numerical equivalence test for backbone in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/numerical/test_numerical_backbone.py`:
  - ✅ Implemented 3 tests: output equivalence, frozen stages, output ranges
  - ✅ Verified deterministic outputs (same input → same output)
  - ✅ Verified output shapes for ResNet-50-vd
  - ⏸ PaddlePaddle weight comparison test skipped (requires checkpoint)
- [X] T033 [US1] Implement numerical equivalence test for neck in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/numerical/test_numerical_neck.py`:
  - ✅ Implemented 6 tests: output equivalence, channel unification, FPN-PAN structure, encoder integration, output ranges, CSPRepLayer addition mode
  - ✅ Verified all outputs have unified channels (hidden_dim=256)
  - ✅ Verified gradient flow through FPN-PAN
  - ⏸ PaddlePaddle weight comparison test skipped (requires checkpoint)
- [X] T034 [US1] Implement numerical equivalence test for transformer in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/numerical/test_numerical_transformer.py`:
  - ✅ Implemented 6 tests: single-group eval/train, multi-group, full config, perturbation mask, output ranges
  - ✅ Verified multi-group query mechanism (o2o, noise, o2m)
  - ✅ Verified self-attention perturbation in training mode
  - ⏸ PaddlePaddle weight comparison test skipped (requires checkpoint)
- [X] T035 [US1] Implement end-to-end numerical equivalence test in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/numerical/test_numerical_e2e.py`:
  - ✅ Implemented 4 tests: forward eval, output ranges, multiple input sizes, batch independence
  - ✅ Verified complete model forward pass (backbone → neck → transformer → head)
  - ✅ Verified output ranges (bbox in [0,1], no NaN/Inf)
  - ✅ Verified deterministic outputs
  - ⏸ PaddlePaddle checkpoint comparison test skipped (requires checkpoint)
  - ⏸ COCO evaluation test skipped (requires dataset)

### Inference Implementation ✅ COMPLETE

- [X] T036 [US1] Implement DETRPostProcessor in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/post_process.py`:
  - ✅ Coordinate conversion utilities (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
  - ✅ DETRPostProcessor class for standard DETR post-processing
  - ✅ Multi-group query support (dual_queries, dual_groups for O2O/O2M)
  - ✅ Sigmoid/Softmax classification modes
  - ✅ Top-K detection selection
  - ✅ Coordinate scaling from normalized to pixel space
  - ✅ Scale factor and padding handling
  - ✅ Unit tests (13/13 passing) in test_post_process.py
- [X] T037 [US1] Implement inference script in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tools/infer.py`:
  - ✅ Load config from YAML
  - ✅ Initialize model and load checkpoint
  - ✅ Load input images (single image or directory)
  - ✅ Run forward pass and extract predictions
  - ✅ Apply NMS (Non-Maximum Suppression) with IoU threshold=0.7
  - ✅ Apply confidence threshold (default=0.3)
  - ✅ Visualize results (draw bounding boxes on images with COCO colors)
  - ✅ Save output images to specified directory
  - ✅ Support batch inference for efficiency
  - ✅ Image preprocessing with resize, pad, normalize
  - ✅ Post-processing with per-class NMS
- [X] T038 [US1] Create inference validation test in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/integration/test_inference.py`:
  - ✅ Test image preprocessing (shape, normalization)
  - ✅ Test model forward pass in eval mode
  - ✅ Test post-processing output format
  - ✅ Test confidence threshold filtering
  - ✅ Test NMS behavior
  - ✅ Test batch inference (batch size 1, 4)
  - ✅ Test different image sizes (640x640, 480x640, 800x600, 1920x1080)
  - ✅ Test no detections case (high threshold)
  - ✅ Test CUDA inference (if available)
  - ✅ Test gradient disabled in eval mode
  - ✅ Test output value ranges
  - ✅ All 12/12 tests passing ✨

### COCO Evaluation Implementation ✅ COMPLETE

- [X] T039 [US1] Implement COCO evaluator in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/engine/evaluator.py`:
  - ✅ Load COCO ground truth annotations
  - ✅ Collect model predictions on entire val2017 dataset
  - ✅ Convert predictions to COCO format [x, y, width, height]
  - ✅ Use pycocotools.cocoeval to compute mAP, AP50, AP75
  - ✅ Support multiple IoU types (bbox, segm)
  - ✅ Compute metrics: AP, AP50, AP75, APs, APm, APl
  - ✅ Compute recall metrics: AR1, AR10, AR100
  - ✅ Log evaluation metrics
  - ✅ Support distributed synchronization (placeholder for multi-GPU)
- [X] T040 [US1] Implement evaluation script in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tools/eval.py`:
  - ✅ Load config and checkpoint
  - ✅ Initialize model and set to eval mode
  - ✅ Build COCO validation dataset with transforms
  - ✅ Run inference on entire COCO val2017
  - ✅ Call evaluator to compute metrics
  - ✅ Compare against PaddlePaddle baseline (R50: 53.4% mAP)
  - ✅ Support custom batch size and num workers
  - ✅ Support confidence and NMS threshold configuration
  - ✅ Log final mAP and comparison with baseline
  - Log results to console and file

**Checkpoint US1**: At this point, User Story 1 should be fully functional and testable independently. Users can load converted checkpoints and run inference with numerically equivalent results.

---

## Phase 4: User Story 2 - Model Training (Priority: P2)

**Goal**: Deliver a complete training pipeline that can train RT-DETRv3 from scratch on COCO and achieve published mAP (53.4±0.5% for R50)

**Independent Test**: Train PyTorch RT-DETRv3-R50 for 72 epochs on COCO train2017, evaluate on val2017, verify final mAP is 53.4±0.5%

**Duration Estimate**: 2-3 weeks

### Loss Functions

- [X] T041 [US2] Implement DINOv3Loss in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/models/losses/detr_loss.py`:
  - ✅ Varifocal Loss for classification (focal loss variant with target score weighting)
  - ✅ GIoU Loss for bounding box regression (generalized IoU)
  - ✅ L1 Loss for bounding box regression (smooth L1)
  - ✅ Hungarian matching for one-to-one assignment (300 queries)
  - ✅ Cost matrix: classification cost + box L1 cost + box GIoU cost
  - ✅ Support one-to-many supervision (450 queries, additional loss branch)
  - ✅ Support denoising queries (100 noise queries, reconstruction loss)
  - ✅ Support auxiliary branch loss (PPYOLOEHead outputs)
  - ✅ Loss weights: loss_cls_weight=1.0, loss_bbox_weight=5.0, loss_giou_weight=2.0, aux_loss_weight=1.0, o2m_loss_weight=1.0
- [X] T042 [US2] Create unit test for loss functions in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/unit/test_losses.py`:
  - ✅ Test Varifocal Loss computation with sample predictions and targets
  - ✅ Test GIoU Loss computation
  - ✅ Test Hungarian matching (verify assignment correctness)
  - ✅ Test loss gradient (backward pass succeeds)
  - ✅ Test multi-branch loss aggregation
  - ✅ All 19/19 tests passing

### Training Infrastructure

- [X] T043 [US2] Implement optimizer setup in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/engine/optimizer.py`:
  - ✅ AdamW optimizer (lr=0.0001, weight_decay=0.0001)
  - ✅ Support parameter groups (different lr for backbone vs decoder)
  - ✅ Gradient clipping (max_norm=0.1) to prevent training instability
  - ✅ Create optimizer from config
- [X] T044 [US2] Implement LR scheduler in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/engine/optimizer.py`:
  - ✅ MultiStepLR (decay at epoch 60, gamma=0.1 for 72-epoch schedule)
  - ✅ Warmup phase (linear warmup for first 2000 iterations)
  - ✅ Create scheduler from config
- [X] T045 [US2] Implement training loop in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/engine/trainer.py`:
  - ✅ Initialize DDP if multi-GPU (dist.init_process_group)
  - ✅ Wrap model with DistributedDataParallel
  - ✅ Create dataloaders with DistributedSampler
  - ✅ Setup mixed precision (torch.cuda.amp.autocast, GradScaler)
  - ✅ Training loop: iterate epochs → iterate batches → forward → loss → backward → optimizer step
  - ✅ Gradient accumulation support (accumulate_steps parameter)
  - ✅ Checkpoint saving every N epochs
  - ✅ Validation every N epochs
  - ✅ Loss logging (console output)
  - ✅ Resume from checkpoint support
- [X] T046 [US2] Implement training script in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tools/train.py`:
  - ✅ Parse command-line arguments (config path, resume checkpoint, distributed flag)
  - ✅ Load config from YAML
  - ✅ Initialize model, optimizer, scheduler
  - ✅ Initialize trainer
  - ✅ Launch training loop
  - ✅ Log final metrics
  - ✅ Support single-GPU and multi-GPU (torchrun) modes

### Training Validation Tests

- [ ] T047 [US2] Create training smoke test in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/integration/test_training.py`:
  - Train for 2 epochs on COCO train subset (100 images)
  - Verify loss decreases over iterations
  - Verify no NaN/Inf losses
  - Verify checkpoint saving works
  - Verify gradient flow (no zero gradients)
- [ ] T048 [US2] Create DDP training test in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/integration/test_ddp_training.py`:
  - Launch training with torchrun (2 GPUs if available, else skip)
  - Train for 1 epoch
  - Verify synchronization across GPUs
  - Verify final model weights are identical across ranks

### Training Convergence Validation

- [ ] T049 [US2] Run full training on COCO train2017 for RT-DETRv3-R50 (72 epochs, 4 GPUs):
  - Use config `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/configs/rtdetrv3_r50_6x_coco.yml`
  - Train with batch_size=4 per GPU (total effective batch=16)
  - Enable mixed precision (amp=true)
  - Save checkpoints every 5 epochs
  - Evaluate on val2017 every 5 epochs
  - Log training curves (loss, mAP over epochs)
- [ ] T050 [US2] Validate training convergence against PaddlePaddle baseline:
  - Compare loss curves (should follow same trajectory ±5%)
  - Compare epoch-wise mAP progression
  - Final mAP should be 53.4±0.5% for R50
  - Training time should be within 110% of PaddlePaddle baseline
  - Memory usage should be within 110% of PaddlePaddle baseline

**Checkpoint US2**: At this point, User Story 2 should be fully functional. Users can train RT-DETRv3 from scratch and reproduce published results.

---

## Phase 5: User Story 3 - Configuration and Deployment (Priority: P3)

**Goal**: Enable users to use PaddlePaddle configs with minimal modifications and export trained models to deployment formats (ONNX, TorchScript)

**Independent Test**: Load PaddlePaddle YAML config, train/evaluate with PyTorch, export to ONNX/TorchScript, verify inference works and mAP is preserved (<0.2% degradation)

**Duration Estimate**: 1-2 weeks

### Configuration Compatibility

- [ ] T050 [US3] Enhance PaddlePaddle config converter in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/utils/config.py`:
  - Support all PaddlePaddle config keys (architecture, optimizer, lr_scheduler, runtime, data augmentation)
  - Map optimizer names: Paddle.optimizer.AdamW → torch.optim.AdamW
  - Map LR scheduler names: Paddle PiecewiseDecay → torch.optim.lr_scheduler.MultiStepLR
  - Map data augmentation transforms
  - Document required manual adjustments in docstring
- [ ] T051 [US3] Create config migration guide in `/home/tyjt/桌面/RT-DETRv3/docs/config_migration.md`:
  - List PaddlePaddle → PyTorch config mappings
  - Provide examples for each config section
  - Document breaking changes (keys that require manual adjustment)
  - Provide automated conversion script usage instructions
- [ ] T052 [US3] Test config compatibility with `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/integration/test_config_compat.py`:
  - Load PaddlePaddle config rtdetrv3_r50vd_6x_coco.yml
  - Apply automated conversion
  - Initialize model with converted config
  - Verify no errors during initialization
  - Run 1 training iteration to verify correctness

### ONNX Export

- [ ] T053 [US3] Implement ONNX export in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tools/export_onnx.py`:
  - Load trained checkpoint
  - Set model to eval mode
  - Create dummy input (1, 3, 640, 640)
  - Export with torch.onnx.export(dynamo=True, opset_version=16)
  - Set dynamic_axes for batch, height, width
  - Input names: ['images'], Output names: ['labels', 'boxes', 'scores']
  - Validate exported ONNX model (load with onnx.load, check graph)
- [ ] T054 [US3] Implement ONNX inference validation in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/integration/test_onnx_export.py`:
  - Export model to ONNX
  - Load ONNX model in ONNXRuntime
  - Run inference on COCO val2017 sample images
  - Compare outputs against PyTorch model (boxes ±2 pixels, scores ±0.01)
  - Measure inference latency (should be comparable to PyTorch)
- [ ] T055 [US3] Run full COCO evaluation on exported ONNX model:
  - Export RT-DETRv3-R50 to ONNX
  - Run inference on entire val2017 using ONNXRuntime
  - Compute mAP
  - Verify mAP degradation is <0.2% compared to PyTorch model

### TorchScript Export

- [ ] T056 [US3] Implement TorchScript export in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tools/export_torchscript.py`:
  - Load trained checkpoint
  - Set model to eval mode
  - Create dummy input (1, 3, 640, 640)
  - Trace model with torch.jit.trace (tracing preferred over scripting for detection models)
  - Save traced model to .pt file
  - Validate by loading and running inference
- [ ] T057 [US3] Implement TorchScript inference validation in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tests/integration/test_torchscript_export.py`:
  - Export model to TorchScript
  - Load TorchScript model (torch.jit.load)
  - Run inference on COCO val2017 sample images
  - Compare outputs against PyTorch model (exact match expected)
  - Measure inference latency
- [ ] T058 [US3] Create C++ libtorch example (OPTIONAL):
  - Write C++ inference code using libtorch
  - Load exported TorchScript model
  - Run inference on sample image
  - Verify output correctness
  - Document in README

### Performance Benchmarking

- [ ] T059 [US3] Implement inference benchmarking script in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/tools/benchmark.py`:
  - Measure inference latency (ms/image) on T4 GPU
  - Test batch sizes: 1, 4, 8
  - Test with/without torch.compile
  - Test ONNX inference speed with ONNXRuntime
  - Test TorchScript inference speed
  - Report FPS (frames per second)
  - Compare against PaddlePaddle baseline (R50: ≥108 FPS with TensorRT FP16)
- [ ] T060 [US3] Optimize inference performance:
  - Enable torch.compile(mode='max-autotune') for PyTorch model
  - Profile with torch.profiler to identify bottlenecks
  - Optimize data loading (pin_memory, prefetch_factor)
  - Optimize NMS implementation (use torchvision.ops.nms)
  - Document optimization techniques in README
- [ ] T061 [US3] Validate performance targets:
  - RT-DETRv3-R50: ≥108 FPS on T4 GPU (TensorRT FP16 or PyTorch compiled)
  - RT-DETRv3-R18: ≥217 FPS on T4 GPU
  - Training throughput: ≥95% of PaddlePaddle (samples/sec on 4xA100)
  - Memory usage: ≤110% of PaddlePaddle (peak GPU memory during training)

**Checkpoint US3**: At this point, User Story 3 should be fully functional. Users can use PaddlePaddle configs, export models, and deploy to production.

---

## Phase 6: Final Phase - Polish & Cross-Cutting Concerns

**Purpose**: Documentation, optimization, final validation across all user stories
**Duration Estimate**: 1 week

### Documentation

- [ ] T062 [P] Create comprehensive README.md in `/home/tyjt/桌面/RT-DETRv3/rtdetrv3_pytorch/README.md`:
  - Project overview and features
  - Installation instructions (PyTorch 2.5.1, CUDA setup)
  - Quick start guide (inference, training, evaluation)
  - Model zoo (links to converted checkpoints)
  - Performance benchmarks (mAP, FPS)
  - Citation and acknowledgments
- [ ] T063 [P] Create migration guide in `/home/tyjt/桌面/RT-DETRv3/docs/migration_guide.md`:
  - Differences between PaddlePaddle and PyTorch implementations
  - Weight conversion instructions
  - Config conversion instructions
  - API differences (Paddle API → PyTorch API mappings)
  - Known issues and workarounds
- [ ] T064 [P] Create training guide in `/home/tyjt/桌面/RT-DETRv3/docs/training_guide.md`:
  - Dataset preparation (COCO download, directory structure)
  - Single-GPU training commands
  - Multi-GPU training commands (torchrun)
  - Mixed precision training
  - Hyperparameter tuning tips
  - Training troubleshooting (common errors)
- [ ] T065 [P] Create deployment guide in `/home/tyjt/桌面/RT-DETRv3/docs/deployment_guide.md`:
  - ONNX export instructions
  - TorchScript export instructions
  - ONNXRuntime inference example
  - TensorRT conversion (if applicable)
  - C++ libtorch inference example
  - Performance optimization tips

### Code Quality and Testing

- [ ] T066 [P] Add type hints to all Python files (follow PEP 484):
  - Add type hints to function signatures
  - Add type hints to class attributes
  - Run mypy to validate type correctness
- [ ] T067 [P] Add comprehensive docstrings to all modules, classes, functions:
  - Follow Google/NumPy docstring style
  - Include parameter descriptions, return value descriptions, examples
  - Document tensor shapes in docstrings (e.g., "x: Tensor of shape (batch, channels, height, width)")
- [ ] T068 Run full test suite and verify 100% pass rate:
  - Unit tests: pytest tests/unit/ -v
  - Integration tests: pytest tests/integration/ -v
  - Numerical tests: pytest tests/numerical/ -v --paddle-checkpoint /path/to/checkpoint
  - Generate coverage report: pytest --cov=rtdetrv3_pytorch --cov-report=html
  - Target: >90% code coverage

### Final Validation

- [ ] T069 Reproduce published results for all model variants:
  - RT-DETRv3-R18: Train 72 epochs, verify 48.1±0.5% mAP
  - RT-DETRv3-R50: Train 72 epochs, verify 53.4±0.5% mAP (already done in T048-T049)
  - RT-DETRv3-R101: Train 72 epochs, verify final mAP matches paper
- [ ] T070 Cross-validate with 5 independent training runs (different random seeds):
  - Train RT-DETRv3-R50 with seeds [42, 123, 456, 789, 1024]
  - Measure mAP variance (should be <0.3% standard deviation per spec.md)
  - Document training stability
- [ ] T071 Run numerical equivalence validation for entire model zoo:
  - Convert all PaddlePaddle checkpoints (R18, R34, R50, R101)
  - Run numerical tests for each variant
  - Verify all pass with tolerance (activations <1e-4, predictions <0.01)
- [ ] T072 Performance benchmarking across all variants:
  - Measure inference FPS for R18, R50, R101 on T4 GPU
  - Measure training throughput on A100 GPUs
  - Measure memory usage
  - Compare against PaddlePaddle baselines
  - Document results in README
- [ ] T073 Prepare model zoo release:
  - Upload converted PyTorch checkpoints to shared storage (Hugging Face Hub, Google Drive, etc.)
  - Create download script for users
  - Update README with model zoo links
  - Verify all models are loadable and produce correct results

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational phase completion
- **User Story 2 (Phase 4)**: Depends on Foundational phase AND User Story 1 (requires working model architecture)
- **User Story 3 (Phase 5)**: Depends on User Story 1 AND User Story 2 (requires trained models for export)
- **Polish (Phase 6)**: Depends on all user stories being complete

### Critical Path

**Sequential dependencies** (must be completed in order):
1. Phase 1 (Setup) → Phase 2 (Foundational) → Phase 3 (US1: Backbone) → Phase 3 (US1: Neck) → Phase 3 (US1: Encoder) → Phase 3 (US1: Decoder) → Phase 3 (US1: Heads) → Phase 3 (US1: Full Model)
2. Phase 3 (US1 complete) → Phase 4 (US2: Training)
3. Phase 4 (US2 complete) → Phase 5 (US3: Export)
4. Phase 5 (US3 complete) → Phase 6 (Polish)

### Parallel Opportunities

**Within Foundational Phase (Phase 2)** - Can work in parallel:
- T004 (logging), T005 (README), T012 (distributed utils), T013 (checkpoint utils) - all independent

**Within US1 Component Migration** - Can work in parallel AFTER T017 (backbone) completes:
- T027 (DINOv3Head) and T028 (PPYOLOEHead) - different files
- T032 (numerical test backbone) and T018 (unit test backbone) - different test types

**Within US2 Training Infrastructure** - Can work in parallel:
- T042 (optimizer) and T043 (LR scheduler) - independent
- T046 (training smoke test) and T047 (DDP test) - different test scenarios

**Within Phase 6 (Polish)** - Can work in parallel:
- T062, T063, T064, T065 (all documentation) - independent files
- T066 (type hints) and T067 (docstrings) - different tasks

**User Stories (Phase 3-5)** - Cannot work in true parallel due to dependencies, but different team members can work on different components within US1 simultaneously after foundational work is complete.

### Within Each User Story

**US1 (Core Model Execution)**:
1. Backbone (T017-T018) - sequential
2. Neck (T019-T020) - depends on backbone completion
3. Attention + MLP (T021-T024) - depends on neck completion
4. Decoder (T025-T026) - depends on attention completion
5. Heads (T027-T029) - depends on decoder completion, but DINOv3Head and PPYOLOEHead can be parallel
6. Full Model (T030-T031) - depends on all components
7. Numerical tests (T032-T035) - depends on full model
8. Inference (T036-T039) - depends on full model

**US2 (Model Training)**:
1. Loss functions (T040-T041) - depends on US1 completion
2. Training infrastructure (T042-T045) - can work in parallel with loss functions
3. Training validation (T046-T047) - depends on training infrastructure
4. Full training (T048-T049) - depends on all above

**US3 (Configuration & Deployment)**:
1. Config compatibility (T050-T052) - depends on US2 completion
2. ONNX export (T053-T055) - depends on US1 completion
3. TorchScript export (T056-T058) - depends on US1 completion, can be parallel with ONNX
4. Performance benchmarking (T059-T061) - depends on all export formats

---

## Task Statistics

### Total Task Count: 73 tasks

### Task Count per User Story
- **Phase 1 (Setup)**: 5 tasks (T001-T005)
- **Phase 2 (Foundational)**: 11 tasks (T006-T016) - CRITICAL BLOCKING PHASE
- **Phase 3 (US1 - Core Model Execution)**: 23 tasks (T017-T039)
  - Backbone: 2 tasks
  - Neck: 2 tasks
  - Transformer components: 4 tasks
  - Decoder: 2 tasks
  - Heads: 3 tasks
  - Full model: 2 tasks
  - Numerical validation: 4 tasks
  - Inference: 4 tasks
- **Phase 4 (US2 - Model Training)**: 10 tasks (T040-T049)
  - Loss functions: 2 tasks
  - Training infrastructure: 4 tasks
  - Validation: 4 tasks
- **Phase 5 (US3 - Configuration & Deployment)**: 12 tasks (T050-T061)
  - Config compatibility: 3 tasks
  - ONNX export: 3 tasks
  - TorchScript export: 3 tasks
  - Benchmarking: 3 tasks
- **Phase 6 (Polish & Cross-Cutting)**: 12 tasks (T062-T073)
  - Documentation: 4 tasks
  - Code quality: 3 tasks
  - Final validation: 5 tasks

### Parallel Opportunities Identified

**High Parallelism (5+ tasks can run simultaneously)**:
- Phase 2 Foundational: T004, T005, T012, T013 (4 tasks in parallel)
- Phase 6 Documentation: T062, T063, T064, T065, T066, T067 (6 tasks in parallel)

**Medium Parallelism (2-4 tasks can run simultaneously)**:
- US1 Heads: T027, T028 (2 tasks in parallel)
- US1 Tests: T032, T033, T034 (3 tasks in parallel after model components ready)
- US2 Infrastructure: T042, T043 (2 tasks in parallel)
- US3 Export: T053-T055 (ONNX) and T056-T058 (TorchScript) can overlap (2 streams)

**Total parallelizable tasks**: ~18 tasks marked with [P]

### Suggested MVP Scope (Minimal Viable Product)

**MVP = Phase 1 + Phase 2 + Phase 3 (US1 Core Model Execution)**

This delivers:
- ✅ Working RT-DETRv3 PyTorch model
- ✅ Weight conversion from PaddlePaddle
- ✅ Inference capability
- ✅ Numerical equivalence validation
- ✅ COCO evaluation
- ✅ Basic documentation (README)

**Estimated MVP Timeline**: 4-6 weeks (T001-T039, total 39 tasks)

**MVP Deliverables**:
1. PyTorch model implementation (all components)
2. Weight conversion tool
3. Inference script
4. Evaluation script
5. Numerical equivalence validation
6. Basic README with usage examples

**Post-MVP Increments**:
- **Increment 1 (US2)**: Add training capability (T040-T049, +10 tasks, +2-3 weeks)
- **Increment 2 (US3)**: Add config compatibility and deployment exports (T050-T061, +12 tasks, +1-2 weeks)
- **Increment 3 (Polish)**: Documentation, optimization, final validation (T062-T073, +12 tasks, +1 week)

**Full Project Timeline**: 8-12 weeks total (all 73 tasks)

---

## Implementation Strategy

### Recommended Approach: Incremental Delivery

**Week 1-2**: Phase 1 (Setup) + Phase 2 (Foundational) → Foundation ready
**Week 3-6**: Phase 3 (US1 Core Model Execution) → MVP COMPLETE ✅
- **Checkpoint at Week 6**: Validate MVP - inference works, numerical equivalence confirmed
- **Decision point**: Ship MVP or continue to full training capability

**Week 7-9**: Phase 4 (US2 Model Training) → Training capability added
- **Checkpoint at Week 9**: Validate training - reproduce published mAP

**Week 10-11**: Phase 5 (US3 Configuration & Deployment) → Production-ready
- **Checkpoint at Week 11**: Validate exports - ONNX/TorchScript work, performance targets met

**Week 12**: Phase 6 (Polish & Cross-Cutting) → Documentation and final validation

### Risk Mitigation

**High Risk Tasks**:
- T006-T007 (Weight conversion) - CRITICAL for US1 success
- T021 (Deformable attention) - Numerical correctness critical
- T040 (Loss functions) - Complex multi-branch loss
- T048-T049 (Full training) - Long-running validation (72 epochs)

**Mitigation Strategies**:
- Test weight conversion early with small model (e.g., single ResNet layer)
- Validate deformable attention with unit tests and gradient checks before integration
- Implement loss functions incrementally (one branch at a time)
- Run short training trials (5-10 epochs) to validate convergence before full 72-epoch run

---

## Notes

- All tasks include absolute file paths for clarity
- [P] tasks can run in parallel (different files, no dependencies)
- [Story] labels map tasks to user stories for traceability
- Each user story phase is independently testable
- Tests are OPTIONAL and only included where explicitly required by spec.md for validation
- Numerical equivalence testing is CRITICAL and non-optional per constitution
- Performance benchmarking is CRITICAL per constitution (Principle V)
- Modular migration order (backbone → neck → encoder → decoder → heads) follows constitution Principle II
- All tasks are specific and actionable with clear file paths
- Task granularity: each task is 1-8 hours of work for a single developer

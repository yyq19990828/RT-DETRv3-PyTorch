# Research: PaddlePaddle to PyTorch Migration

**Date**: 2025-10-14
**Feature**: [spec.md](./spec.md)
**Purpose**: Technical research to resolve unknowns and guide implementation decisions

## Executive Summary

This research establishes the technical foundation for migrating RT-DETRv3 from PaddlePaddle to PyTorch. Key findings:

1. **PyTorch 2.x Ecosystem**: Mature support for detection models with torch.compile, DDP, and AMP
2. **Weight Conversion**: Manual state_dict mapping required; validated approaches exist
3. **Deformable Attention**: Multiple community implementations available (MultiScaleDeformableAttention, MMCV)
4. **Performance Parity**: PyTorch 2.x achieves comparable or better performance with proper optimization

---

## Decision 1: PyTorch Version and Core Dependencies

**Decision**: Use PyTorch 2.5.1 (latest stable as of 2025-10) with CUDA 11.8/12.1 support

**Rationale**:
- PyTorch 2.x provides `torch.compile` for performance optimization (5-20% speedup over PyTorch 1.x)
- Native support for mixed precision training via `torch.amp` (required per FR-011)
- Mature DDP implementation for multi-GPU training (required per FR-007)
- Full ONNX export support with `dynamo=True` flag (required per FR-010)
- Backward compatibility with CUDA 11.8+ ensures hardware compatibility per constitution

**Alternatives Considered**:
- **PyTorch 1.13**: Rejected - lacks `torch.compile`, older ONNX export, inferior performance
- **PyTorch 2.0.0**: Rejected - early 2.x release with known bugs in distributed training
- **PyTorch nightly**: Rejected - unstable for research reproduction

**Key Dependencies**:
```python
torch>=2.5.1
torchvision>=0.20.1  # Matches torch 2.5.1
numpy>=1.24.0
opencv-python>=4.8.0
pycocotools>=2.0.7
pyyaml>=6.0
scipy>=1.10.0
```

**Installation Command**:
```bash
# CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## Decision 2: Deformable Attention Implementation

**Decision**: Use `MultiScaleDeformableAttention` package from PyPI with fallback to MMCV implementation

**Rationale**:
- `MultiScaleDeformableAttention` (v1.0.0) provides standalone, CUDA-accelerated MSDA operator
- Lightweight dependency (no heavy framework overhead like MMDetection)
- Explicit PyTorch 2.x and CUDA 11.8+ compatibility
- Supports both forward and backward passes (training-ready)
- Direct conda/pip installation: `pip install MultiScaleDeformableAttention`
- MMCV fallback provides battle-tested alternative if primary package has issues

**Alternatives Considered**:
- **lucidrains/deformable-attention**: Pure PyTorch, flexible but slower (no CUDA acceleration)
- **MMCV only**: Requires entire MMDetection ecosystem (heavy dependencies, conflicts with constitution simplicity principle)
- **Custom CUDA kernel**: Rejected - high maintenance cost, violates constitution preference for native operations
- **FlexAttention**: Experimental API, not production-ready for MSDA

**Implementation Strategy**:
```python
# Primary approach
from MultiScaleDeformableAttention import MultiScaleDeformableAttention as MSDA

# Fallback if primary unavailable
try:
    from mmcv.ops import MultiScaleDeformableAttention as MSDA
except ImportError:
    # Use lucidrains pure PyTorch version for development
    from deformable_attention import MultiScaleDeformableAttention as MSDA
```

**Validation Requirements**:
- Numerical equivalence testing against PaddlePaddle `PPMSDeformableAttention`
- Gradient check for backward pass
- Performance benchmarking (should match or exceed PaddlePaddle speed)

---

## Decision 3: Weight Conversion Strategy

**Decision**: Manual state_dict mapping with automated conversion script

**Rationale**:
- PaddlePaddle and PyTorch use different parameter naming conventions (no direct compatibility)
- Weight shapes for Conv2d/Linear are identical in most cases, minimal transposition needed
- BatchNorm running statistics require explicit name mapping (`_mean` → `running_mean`)
- Manual mapping provides full control and transparency for validation
- ONNX intermediate path rejected due to potential precision loss and custom op incompatibility

**Key Challenges Identified**:

1. **Parameter Naming Differences**:
   - Paddle: `conv2d_1.w_0`, `bn_1._mean`, `bn_1._variance`
   - PyTorch: `features.0.weight`, `features.0.running_mean`, `features.0.running_var`

2. **Weight Shape Handling**:
   - Conv2d: Both use `[out_channels, in_channels, height, width]` - compatible
   - Linear: Both use `[out_features, in_features]` - compatible
   - BatchNorm: `weight` and `bias` compatible, running stats need renaming

3. **Optimizer States**:
   - Decision: Do NOT convert optimizer states (reinitialize PyTorch optimizers)
   - Rationale: Framework implementation differences make direct transfer risky

**Conversion Workflow**:

```python
import paddle
import torch
import numpy as np

def convert_paddle_to_torch(paddle_path, torch_model, name_map):
    """
    Convert PaddlePaddle checkpoint to PyTorch state_dict

    Args:
        paddle_path: Path to .pdparams file
        torch_model: PyTorch model instance
        name_map: Dict mapping Paddle keys to PyTorch keys
    """
    # Load Paddle weights
    paddle_state = paddle.load(paddle_path)
    torch_state = torch_model.state_dict()

    converted = {}
    for paddle_key, torch_key in name_map.items():
        paddle_param = paddle_state[paddle_key]

        # Convert to numpy then torch
        if isinstance(paddle_param, paddle.Tensor):
            paddle_param = paddle_param.numpy()

        # Handle shape transposition if needed (rare)
        if needs_transpose(paddle_key, torch_key):
            paddle_param = np.transpose(paddle_param, get_transpose_axes(paddle_key))

        # Convert to PyTorch tensor
        torch_tensor = torch.from_numpy(paddle_param)

        # Validate shape match
        assert torch_tensor.shape == torch_state[torch_key].shape, \
            f"Shape mismatch: {paddle_key} {torch_tensor.shape} vs {torch_key} {torch_state[torch_key].shape}"

        converted[torch_key] = torch_tensor

    # Load converted weights
    torch_model.load_state_dict(converted, strict=False)
    return torch_model

# Name mapping will be generated by analyzing both model structures
name_map = generate_name_mapping(paddle_model, torch_model)
```

**Validation Protocol** (per constitution Principle III):
1. Load Paddle checkpoint and run inference on COCO validation samples
2. Convert weights using mapping script
3. Load converted weights into PyTorch model
4. Run inference on same validation samples
5. Compare outputs: bounding boxes (±2 pixels), confidence scores (±0.01)
6. Validate numerical equivalence: intermediate activations (±1e-4 tolerance)

**Alternatives Considered**:
- **ONNX intermediate format**: Rejected - custom ops (deformable attention) may not export correctly, potential precision loss
- **Fully automated conversion**: Rejected - model-specific mappings require manual inspection for correctness
- **PaDiff tool**: Supplementary use for validation only, not for conversion

---

## Decision 4: Model Architecture Structure

**Decision**: Mirror PaddlePaddle's modular structure with PyTorch idioms

**Rationale**:
- Constitution Principle II mandates modular migration (backbone → neck → encoder → decoder → heads)
- Constitution Principle V requires >80% code structure similarity
- Familiar structure aids review and validation
- Enables incremental migration and testing

**Directory Structure**:
```
rtdetrv3_pytorch/
├── models/
│   ├── backbones/
│   │   ├── resnet.py              # ResNet-18/34/50/101 (from torchvision or custom)
│   │   └── resnet_vd.py           # ResNet-vd variant (Paddle-specific modifications)
│   ├── necks/
│   │   └── hybrid_encoder.py     # HybridEncoder with FPN-PAN
│   ├── transformers/
│   │   ├── rtdetr_transformer.py # RTDETRTransformerv3
│   │   ├── attention.py           # MultiScaleDeformableAttention wrapper
│   │   └── utils.py               # MLP, position embeddings
│   ├── heads/
│   │   ├── detr_head.py           # DINOv3Head (main detection head)
│   │   └── ppyoloe_head.py        # PPYOLOEHead (auxiliary branch)
│   ├── losses/
│   │   └── detr_loss.py           # DINOv3Loss with multi-branch support
│   └── rtdetrv3.py                # Main model class
├── data/
│   ├── coco_dataset.py            # COCO dataset loader
│   ├── transforms.py              # Data augmentation pipeline
│   └── collate.py                 # Batch collation
├── engine/
│   ├── trainer.py                 # Training loop with DDP support
│   ├── evaluator.py               # COCO evaluation
│   └── optimizer.py               # Optimizer and LR scheduler setup
├── utils/
│   ├── checkpoint.py              # Save/load checkpoints
│   ├── config.py                  # YAML config parser
│   ├── distributed.py             # DDP utilities
│   └── logger.py                  # Logging utilities
├── tools/
│   ├── train.py                   # Training entry point
│   ├── eval.py                    # Evaluation entry point
│   ├── infer.py                   # Inference entry point
│   ├── export_onnx.py             # ONNX export script
│   └── convert_weights.py         # Paddle → PyTorch weight conversion
├── configs/
│   ├── rtdetrv3_r18_6x_coco.yml
│   ├── rtdetrv3_r50_6x_coco.yml
│   └── ...
└── tests/
    ├── test_backbone.py           # Unit tests for backbone
    ├── test_encoder.py            # Unit tests for encoder
    ├── test_decoder.py            # Unit tests for decoder
    ├── test_heads.py              # Unit tests for heads
    ├── test_losses.py             # Unit tests for losses
    ├── test_numerical.py          # Numerical equivalence tests vs Paddle
    └── test_integration.py        # End-to-end tests
```

**PyTorch Idioms to Apply**:
- Use `torch.nn.Module` for all model components
- Implement `forward()` methods with clear input/output specifications
- Use `torch.nn.ModuleList` and `torch.nn.ModuleDict` for layer collections
- Apply `@torch.jit.unused` for training-only code branches
- Use `model.train()` / `model.eval()` for mode switching
- Leverage `torch.nn.Parameter` for learnable weights
- Use `register_buffer` for non-learnable tensors (anchors, masks)

---

## Decision 5: Training Infrastructure

**Decision**: Use PyTorch native DDP with torch.amp for mixed precision

**Rationale**:
- `torch.nn.parallel.DistributedDataParallel` is PyTorch's recommended multi-GPU solution
- `torch.cuda.amp` provides automatic mixed precision with minimal code changes
- Both are mature, well-documented, and widely used in production
- No external dependencies needed (FSDP/DeepSpeed overkill for this scale)

**Implementation Details**:

**Distributed Training Setup**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')  # NCCL for GPU, gloo for CPU
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model with DDP
model = RTDETRv3(config).cuda(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# SyncBatchNorm for better distributed training
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

**Mixed Precision Training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, targets in dataloader:
    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast():
        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = losses['total_loss']

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Training Command**:
```bash
# Single GPU
python tools/train.py -c configs/rtdetrv3_r50_6x_coco.yml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetrv3_r50_6x_coco.yml --distributed
```

**Alternatives Considered**:
- **FSDP (Fully Sharded Data Parallel)**: Overkill for models <1B parameters
- **DeepSpeed**: Unnecessary complexity, external dependency
- **Horovod**: Deprecated, DDP is now preferred

---

## Decision 6: Configuration System

**Decision**: YAML-based configuration with backward compatibility layer for PaddlePaddle configs

**Rationale**:
- Constitution Principle VI mandates configuration compatibility
- YAML is human-readable and widely used in both frameworks
- Provides migration path for existing PaddlePaddle users
- Enables configuration inheritance and overrides

**Configuration Structure**:
```yaml
# configs/rtdetrv3_r50_6x_coco.yml
model:
  type: RTDETRv3
  backbone:
    type: ResNet
    depth: 50
    variant: 'd'  # ResNet-vd
    frozen_stages: 1
    return_idx: [1, 2, 3]  # C3, C4, C5
  neck:
    type: HybridEncoder
    in_channels: [512, 1024, 2048]
    feat_strides: [8, 16, 32]
    hidden_dim: 256
    num_encoder_layers: 1
  transformer:
    type: RTDETRTransformerv3
    num_queries: 300
    num_queries_o2m: 450
    num_noise_queries: [100]
    num_decoder_layers: 6
    o2m_branch: true
  head:
    type: DINOv3Head
    num_classes: 80
  aux_head:
    type: PPYOLOEHead
    num_classes: 80

optimizer:
  type: AdamW
  lr: 0.0001
  weight_decay: 0.0001

lr_scheduler:
  type: MultiStepLR
  milestones: [60]
  gamma: 0.1

training:
  epochs: 72
  batch_size: 4  # Per GPU
  grad_clip: 0.1
  amp: true  # Enable mixed precision

data:
  dataset: COCO
  train_path: 'data/coco/train2017'
  val_path: 'data/coco/val2017'
  ann_file_train: 'data/coco/annotations/instances_train2017.json'
  ann_file_val: 'data/coco/annotations/instances_val2017.json'
  image_size: [640, 640]

# Numerical tolerance for validation
validation:
  activation_tolerance: 1e-4
  output_tolerance: 0.01
  map_tolerance: 0.005  # ±0.5%
```

**Config Conversion Utility**:
```python
def convert_paddle_config(paddle_yml):
    """
    Convert PaddlePaddle config to PyTorch config

    Key transformations:
    - PascalCase class names preserved
    - Paddle-specific keys mapped to PyTorch equivalents
    - Optimizer/LR scheduler syntax adapted
    """
    # Load Paddle YAML
    paddle_cfg = yaml.safe_load(open(paddle_yml))

    # Map keys
    torch_cfg = {
        'model': map_model_config(paddle_cfg),
        'optimizer': map_optimizer_config(paddle_cfg),
        # ... other mappings
    }

    return torch_cfg
```

---

## Decision 7: ONNX and TorchScript Export

**Decision**: Use `torch.onnx.export` with `dynamo=True` for ONNX; TorchScript via `torch.jit.trace`

**Rationale**:
- PyTorch 2.x recommends `dynamo=True` for ONNX export (cleaner graphs, better op coverage)
- TorchScript tracing handles detection models better than scripting (dynamic control flow)
- Both formats required per FR-010

**ONNX Export Implementation**:
```python
import torch.onnx

def export_onnx(model, save_path, input_shape=(1, 3, 640, 640)):
    model.eval()
    dummy_input = torch.randn(input_shape).cuda()

    # Export with dynamo for PyTorch 2.x
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=16,  # Or latest
        input_names=['images'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'labels': {0: 'batch', 1: 'num_detections'},
            'boxes': {0: 'batch', 1: 'num_detections'},
            'scores': {0: 'batch', 1: 'num_detections'}
        },
        dynamo=True  # Use torch.export-based exporter
    )
```

**TorchScript Export**:
```python
def export_torchscript(model, save_path, input_shape=(1, 3, 640, 640)):
    model.eval()
    dummy_input = torch.randn(input_shape).cuda()

    # Trace model (better for detection models than script)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(save_path)
```

**Validation**:
- ONNX: Load in ONNXRuntime, run inference, compare mAP (<0.2% degradation)
- TorchScript: Load in libtorch (C++), verify inference works

---

## Decision 8: Testing and Validation Strategy

**Decision**: Three-tier testing pyramid - Unit → Integration → Numerical Equivalence

**Rationale**:
- Constitution Principle III (Validation-Driven Development) is non-negotiable
- Modular testing aligns with modular migration (Principle II)
- Enables incremental validation at component and system levels

**Testing Levels**:

**1. Unit Tests** (per component):
```python
# tests/test_backbone.py
def test_resnet50_output_shapes():
    model = ResNet(depth=50, variant='d')
    x = torch.randn(2, 3, 640, 640)
    c3, c4, c5 = model(x)
    assert c3.shape == (2, 512, 80, 80)
    assert c4.shape == (2, 1024, 40, 40)
    assert c5.shape == (2, 2048, 20, 20)

def test_resnet50_gradient_flow():
    model = ResNet(depth=50, variant='d')
    x = torch.randn(2, 3, 640, 640, requires_grad=True)
    outputs = model(x)
    loss = sum(o.sum() for o in outputs)
    loss.backward()
    assert x.grad is not None  # Gradient flows to input
```

**2. Integration Tests** (subsystems):
```python
# tests/test_integration.py
def test_end_to_end_forward():
    """Test full model forward pass"""
    model = RTDETRv3(config).eval()
    images = torch.randn(2, 3, 640, 640)
    outputs = model(images)

    assert 'labels' in outputs
    assert 'boxes' in outputs
    assert outputs['labels'].shape[0] == 2  # Batch size
    assert outputs['boxes'].shape[-1] == 4  # [x, y, w, h]

def test_training_step():
    """Test full training iteration"""
    model = RTDETRv3(config).train()
    images = torch.randn(2, 3, 640, 640)
    targets = [...]  # Ground truth annotations

    outputs = model(images)
    losses = criterion(outputs, targets)

    assert 'loss_cls' in losses
    assert 'loss_bbox' in losses
    assert 'loss_giou' in losses
    losses['total_loss'].backward()  # Should not raise
```

**3. Numerical Equivalence Tests** (vs PaddlePaddle):
```python
# tests/test_numerical.py
def test_backbone_numerical_equivalence():
    """Compare PyTorch backbone outputs with PaddlePaddle"""
    # Load same weights into both models
    paddle_model = load_paddle_backbone()
    torch_model = load_torch_backbone_with_converted_weights()

    # Fixed random input
    torch.manual_seed(42)
    x_torch = torch.randn(1, 3, 640, 640)
    x_paddle = paddle.to_tensor(x_torch.numpy())

    # Forward pass
    out_torch = torch_model(x_torch)
    out_paddle = paddle_model(x_paddle)

    # Compare with tolerance
    for i, (ot, op) in enumerate(zip(out_torch, out_paddle)):
        diff = torch.abs(ot - torch.from_numpy(op.numpy())).max().item()
        assert diff < 1e-4, f"Output {i} differs by {diff}"

def test_full_model_map_equivalence():
    """Compare final mAP between PyTorch and PaddlePaddle"""
    # Run inference on COCO val2017
    paddle_map = run_paddle_evaluation()
    torch_map = run_torch_evaluation()

    map_diff = abs(paddle_map - torch_map)
    assert map_diff < 0.005, f"mAP differs by {map_diff:.4f}"
```

**Test Execution**:
```bash
# Unit tests (fast, run frequently)
pytest tests/test_*.py -v

# Integration tests
pytest tests/test_integration.py -v

# Numerical equivalence (slow, requires PaddlePaddle)
pytest tests/test_numerical.py -v --paddle-checkpoint /path/to/paddle.pdparams

# Full test suite with coverage
pytest tests/ --cov=rtdetrv3_pytorch --cov-report=html
```

---

## Decision 9: Performance Optimization

**Decision**: Enable `torch.compile` for inference; profile and optimize critical paths

**Rationale**:
- Constitution Principle V mandates performance parity (≥95% throughput, ≤105% latency)
- `torch.compile` provides 5-20% speedup with minimal code changes
- Deformable attention is performance bottleneck - ensure CUDA acceleration

**Optimization Strategy**:

**1. Enable torch.compile for Inference**:
```python
# After loading model
model = RTDETRv3(config)
model.load_state_dict(checkpoint)
model.eval()

# Compile for inference
model = torch.compile(model, mode='max-autotune')

# Warmup (compile time amortized over subsequent calls)
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 640, 640).cuda()
    for _ in range(10):
        model(dummy_input)
```

**2. Profile Critical Paths**:
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    outputs = model(images)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**3. Optimize Data Loading**:
```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,  # Tune based on CPU cores
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Reuse workers
    prefetch_factor=2  # Prefetch batches
)
```

**4. Memory Optimization**:
```python
# Gradient checkpointing for large models (if memory constrained)
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    x = checkpoint(self.encoder_layer1, x)
    x = checkpoint(self.encoder_layer2, x)
    return x
```

**Performance Targets** (from constitution):
- Training throughput: ≥95% of PaddlePaddle (samples/sec)
- Inference latency: ≤105% of PaddlePaddle (ms/image)
- Memory usage: ≤110% of PaddlePaddle (peak GPU memory)

**Benchmarking Protocol**:
1. Measure PaddlePaddle baseline on same hardware (T4 GPU for inference, A100 for training)
2. Implement PyTorch version with optimizations enabled
3. Run identical workloads (same batch size, image resolution, model variant)
4. Compare metrics and investigate if targets not met
5. Acceptable variance: ±5% due to framework differences

---

## Summary of Technical Decisions

| Category | Decision | Justification |
|----------|----------|---------------|
| **PyTorch Version** | 2.5.1 (stable) | torch.compile, modern APIs, CUDA 11.8+ support |
| **Deformable Attention** | MultiScaleDeformableAttention (PyPI) | Lightweight, CUDA-accelerated, training-ready |
| **Weight Conversion** | Manual state_dict mapping | Full control, transparent, validated approach |
| **Architecture** | Mirror Paddle structure | 80% similarity requirement, modular validation |
| **Distributed Training** | DDP + torch.amp | Native support, mature, minimal dependencies |
| **Config System** | YAML with compatibility layer | Backward compatible, human-readable |
| **Export** | ONNX (dynamo) + TorchScript (trace) | PyTorch 2.x best practices, dual format support |
| **Testing** | Unit + Integration + Numerical | Constitution-mandated validation pyramid |
| **Optimization** | torch.compile + profiling | Performance parity target (≥95% throughput) |

---

## Implementation Roadmap

**Phase 1: Foundation** (Week 1-2)
- Set up project structure
- Implement weight conversion script
- Create configuration system
- Establish testing framework

**Phase 2: Core Components** (Week 3-6)
- Migrate backbone (ResNet variants)
- Migrate neck (HybridEncoder)
- Migrate transformer encoder
- Validate numerical equivalence at each step

**Phase 3: Detection Components** (Week 7-9)
- Migrate decoder (multi-group queries)
- Migrate detection heads (DINOv3Head, PPYOLOEHead)
- Migrate loss functions (DINOv3Loss)
- Validate subsystem integration

**Phase 4: Training Infrastructure** (Week 10-11)
- Implement training loop with DDP
- Add mixed precision support
- Create optimizer and LR scheduler
- Validate training convergence

**Phase 5: Validation & Optimization** (Week 12-14)
- Full model training on COCO train2017
- Numerical equivalence validation (mAP within ±0.5%)
- Performance profiling and optimization
- Export to ONNX and TorchScript

**Phase 6: Documentation & Release** (Week 15-16)
- Write migration guide
- Create quickstart documentation
- Prepare model zoo (converted weights)
- Final validation and release

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Deformable attention numerical mismatch | High | Extensive unit testing, gradient checks, fallback to MMCV |
| Training instability (different optimizer behavior) | Medium | Match hyperparameters exactly, monitor loss curves closely |
| Performance regression | Medium | Profile early, enable torch.compile, optimize data loading |
| Config incompatibility | Low | Automated conversion tool, clear migration docs |
| Weight conversion errors | High | Automated validation script, visual inspection of mappings |

---

## Open Questions for Implementation

1. **ResNet-vd details**: Does PaddlePaddle's ResNet-vd variant differ significantly from standard ResNet-d? Need to inspect Paddle source code.
2. **Optimizer warmup**: Does PaddlePaddle use linear or cosine warmup? Check training logs.
3. **Data augmentation**: Exact augmentation pipeline details (RandomCrop parameters, color jitter strength). May need to extract from Paddle training code.
4. **Loss weights**: Exact coefficients for auxiliary branch loss, o2m loss, denoising loss. Check Paddle config files.

These will be resolved during implementation by inspecting PaddlePaddle source code and training configurations.

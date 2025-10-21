# Tests Directory Structure

**Date**: 2025-10-20
**Purpose**: RT-DETRv3 PyTorch migration test suite
**Package**: `rtdetrv3_pytorch.ppdet_pytorch`

本目录包含 RT-DETRv3 PyTorch 迁移项目的所有测试,组织结构与 `rtdetrv3_pytorch.ppdet_pytorch` 包结构对应。

---

## Directory Structure

```
rtdetrv3_pytorch/tests/
├── unit/                          # Unit tests (单元测试)
│   ├── core/                      # ppdet_pytorch.core module tests
│   │   ├── test_workspace.py      # Registration system tests
│   │   └── test_registry.py       # Registry implementation tests
│   ├── modeling/                  # ppdet_pytorch.modeling module tests
│   │   ├── test_backbone.py       # Backbone tests (ResNet, etc.)
│   │   ├── test_neck.py           # Neck tests (HybridEncoder)
│   │   ├── test_decoder.py        # Decoder tests (RTDETRTransformer)
│   │   ├── test_attention.py      # Attention mechanism tests
│   │   ├── test_heads.py          # Detection head tests (DINOv3Head, PPYOLOEHead)
│   │   ├── test_losses.py         # Loss function tests (DETRLoss, etc.)
│   │   └── test_post_process.py   # Post-processing tests (NMS, etc.)
│   ├── data/                      # ppdet_pytorch.data module tests
│   │   └── (future: test_transforms.py, test_dataset.py)
│   ├── engine/                    # ppdet_pytorch.engine module tests
│   │   └── (future: test_trainer.py, test_callbacks.py)
│   ├── optimizer/                 # ppdet_pytorch.optimizer module tests
│   │   └── (future: test_optimizer.py, test_ema.py)
│   └── metrics/                   # ppdet_pytorch.metrics module tests
│       └── (future: test_metrics.py, test_coco_utils.py)
├── integration/                   # Integration tests (集成测试)
│   ├── test_config_driven_build.py    # Config-driven model building
│   ├── test_registration.py           # Registration system integration
│   ├── test_dependency_injection.py   # Dependency injection (__inject__, __shared__)
│   ├── test_backward_compat.py        # Backward compatibility tests
│   ├── test_inference.py              # End-to-end inference tests
│   └── test_model.py                  # Model integration tests
├── numerical/                     # Numerical equivalence tests (数值验证)
│   ├── test_numerical_backbone.py     # Backbone Paddle vs PyTorch
│   ├── test_numerical_neck.py         # Neck Paddle vs PyTorch
│   ├── test_numerical_transformer.py  # Transformer Paddle vs PyTorch
│   ├── test_numerical_e2e.py          # End-to-end numerical equivalence
│   └── test_registered_components.py  # Registered components validation
├── weight_conversion/             # Weight conversion tests (权重转换测试)
│   └── (future: test_paddle_to_pytorch.py)
└── configs/                       # Test configuration files (测试配置文件)
    └── (future: test config YAML files)
```

---

## Test Categories

### 1. Unit Tests (单元测试)

**Purpose**: Test individual components in isolation
**Location**: `rtdetrv3_pytorch/tests/unit/`
**Run**: `pytest rtdetrv3_pytorch/tests/unit/ -v`

单元测试按照 `ppdet_pytorch` 包的模块结构组织:

- **core/**: 核心功能
  - Registration system (`@register`, `create()`, `global_config`)
  - Configuration loading and merging
  - Dependency injection (`__inject__`, `__shared__`)

- **modeling/**: 模型组件
  - Backbones: ResNet, ResNeXt, etc.
  - Necks: HybridEncoder
  - Transformers: RTDETRTransformerv3
  - Heads: DINOv3Head, PPYOLOEHead
  - Losses: DETRLoss, VFLLoss, GIoULoss
  - Post-processing: NMS, bbox conversion

- **data/**: 数据加载和增强 (future)
  - Transforms: Mosaic, Mixup, RandomCrop, etc.
  - Datasets: COCODataset, VOCDataset, etc.
  - DataLoader and collate functions

- **engine/**: 训练引擎 (future)
  - Trainer class
  - Callbacks: Checkpointer, LogPrinter, etc.
  - Training loop logic

- **optimizer/**: 优化器和调度器 (future)
  - OptimizerBuilder
  - LR schedulers: CosineDecay, LinearWarmup
  - EMA (Exponential Moving Average)

- **metrics/**: 评估指标 (future)
  - COCOMetric
  - mAP calculation utilities

### 2. Integration Tests (集成测试)

**Purpose**: Test component interactions and end-to-end workflows
**Location**: `rtdetrv3_pytorch/tests/integration/`
**Run**: `pytest rtdetrv3_pytorch/tests/integration/ -v`

集成测试验证:
- **Config-driven build**: 从 YAML 配置文件构建完整模型
- **Registration system**: `@register` + `create()` 工作流
- **Dependency injection**: `__inject__` 和 `__shared__` 机制
- **Backward compatibility**: 与旧版本 API 兼容性
- **End-to-end inference**: 完整的推理流程(数据加载→前向传播→后处理)
- **Model integration**: 多个组件协同工作

### 3. Numerical Tests (数值验证测试)

**Purpose**: Validate numerical equivalence with PaddlePaddle
**Location**: `rtdetrv3_pytorch/tests/numerical/`
**Run**: `pytest rtdetrv3_pytorch/tests/numerical/ -v`

数值验证测试确保:
- PyTorch 实现与 Paddle 数值等价 (tolerance: 1e-5)
- 各组件前向传播输出一致
- 损失计算结果一致
- 端到端训练/推理结果一致
- 注册组件的数值正确性

### 4. Weight Conversion Tests (权重转换测试) - Future

**Purpose**: Test Paddle→PyTorch checkpoint conversion
**Location**: `rtdetrv3_pytorch/tests/weight_conversion/`
**Run**: `pytest rtdetrv3_pytorch/tests/weight_conversion/ -v`

权重转换测试验证:
- Paddle `.pdparams` 到 PyTorch `.pth` 转换
- 参数名称映射正确性
- 张量形状和数值一致性
- 转换后的模型可加载并正常工作

---

## Running Tests

### Run All Tests
```bash
pytest rtdetrv3_pytorch/tests/ -v
```

### Run Specific Test Category
```bash
# Unit tests only
pytest rtdetrv3_pytorch/tests/unit/ -v

# Integration tests only
pytest rtdetrv3_pytorch/tests/integration/ -v

# Numerical validation only
pytest rtdetrv3_pytorch/tests/numerical/ -v
```

### Run Specific Module Tests
```bash
# Core module tests
pytest rtdetrv3_pytorch/tests/unit/core/ -v

# Modeling tests
pytest rtdetrv3_pytorch/tests/unit/modeling/ -v
```

### Run Single Test File
```bash
pytest rtdetrv3_pytorch/tests/unit/modeling/test_backbone.py -v
```

### Run with Coverage
```bash
pytest rtdetrv3_pytorch/tests/ --cov=rtdetrv3_pytorch.ppdet_pytorch --cov-report=html
```

### Run Specific Test Function
```bash
pytest rtdetrv3_pytorch/tests/unit/core/test_workspace.py::test_register_decorator -v
```

---

## Test Naming Conventions

- **Test files**: `test_<module_name>.py` (e.g., `test_backbone.py`)
- **Test functions**: `test_<functionality>` (e.g., `test_resnet50_forward`)
- **Test classes**: `Test<ComponentName>` (e.g., `TestResNet`)
- **Fixtures**: Use descriptive names with `_fixture` suffix (e.g., `model_config_fixture`)

---

## Test Organization Principles

### 1. Mirror Package Structure
单元测试目录镜像 `ppdet_pytorch` 包结构:
- `ppdet_pytorch/core/workspace.py` → `tests/unit/core/test_workspace.py`
- `ppdet_pytorch/modeling/backbones/resnet.py` → `tests/unit/modeling/test_backbone.py`
- `ppdet_pytorch/data/transform/operators.py` → `tests/unit/data/test_transforms.py`

### 2. Separation of Concerns
- **Unit tests**: Test single component in isolation
- **Integration tests**: Test multiple components together
- **Numerical tests**: Compare with Paddle baseline
- **Conversion tests**: Validate weight conversion process

### 3. Independence
- Each test should be runnable independently
- Tests should not depend on execution order
- Use fixtures for shared setup/teardown

---

## Test Dependencies

### Required Packages
```bash
# Install test dependencies
uv pip install pytest pytest-cov numpy torch

# Optional: For Paddle comparison
uv pip install paddlepaddle-gpu
```

### Environment
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: Optional (for GPU tests)
- **PaddlePaddle**: Optional (for numerical validation)

---

## Writing New Tests

### 1. Add Unit Test

Create test file in appropriate module directory:

```bash
# Example: Add optimizer tests
touch rtdetrv3_pytorch/tests/unit/optimizer/test_optimizer.py
```

```python
# tests/unit/optimizer/test_optimizer.py
import torch
from rtdetrv3_pytorch.ppdet_pytorch.optimizer import OptimizerBuilder

def test_adamw_creation():
    """Test AdamW optimizer creation through OptimizerBuilder"""
    model = torch.nn.Linear(10, 10)
    builder = OptimizerBuilder(
        optimizer={'type': 'AdamW', 'lr': 0.001, 'weight_decay': 0.0001}
    )
    optimizer = builder(model.parameters())

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == 0.001
```

### 2. Add Integration Test

```python
# tests/integration/test_config_driven_training.py
from rtdetrv3_pytorch.ppdet_pytorch.core.workspace import create, merge_config

def test_create_trainer_from_config():
    """Test creating Trainer from YAML config"""
    cfg = load_config('configs/rtdetrv3_r50vd_6x_coco.yml')
    merge_config(cfg)

    trainer = create('Trainer')
    assert trainer is not None
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'optimizer')
```

### 3. Add Numerical Test

```python
# tests/numerical/test_numerical_loss.py
import torch
import paddle
import numpy as np

def test_detr_loss_equivalence():
    """Compare DETRLoss output between Paddle and PyTorch"""
    # Setup identical inputs
    pred_logits_pt = torch.randn(2, 300, 80)
    pred_logits_pd = paddle.to_tensor(pred_logits_pt.numpy())

    # Compute losses
    loss_pt = detr_loss_pytorch(pred_logits_pt, ...)
    loss_pd = detr_loss_paddle(pred_logits_pd, ...)

    # Validate equivalence
    assert np.allclose(loss_pt.item(), loss_pd.item(), atol=1e-5)
```

---

## CI/CD Integration

Tests are organized to support tiered CI/CD workflows:

- **Tier 1 (Fast)**: Unit tests - Run on every commit (~1-2 minutes)
- **Tier 2 (Medium)**: Integration tests - Run on pull requests (~5-10 minutes)
- **Tier 3 (Slow)**: Numerical tests - Run nightly or before release (~30+ minutes)

Example GitHub Actions workflow:

```yaml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: uv pip install -e . pytest
      - name: Run unit tests
        run: pytest rtdetrv3_pytorch/tests/unit/ -v

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: uv pip install -e . pytest
      - name: Run integration tests
        run: pytest rtdetrv3_pytorch/tests/integration/ -v
```

---

## Debugging Tests

### Run with Verbose Output
```bash
pytest rtdetrv3_pytorch/tests/ -vv
```

### Run with Debugging
```bash
pytest rtdetrv3_pytorch/tests/ --pdb  # Drop into debugger on failure
```

### Run Specific Test with Print Statements
```bash
pytest rtdetrv3_pytorch/tests/unit/core/test_workspace.py::test_create_function -s
```

### Show Test Duration
```bash
pytest rtdetrv3_pytorch/tests/ --durations=10  # Show 10 slowest tests
```

---

## Migration Notes

**Original Structure** (Before Refactoring):
```
rtdetrv3_pytorch/tests/
├── unit/
│   ├── test_workspace.py
│   ├── test_backbone.py
│   ├── test_neck.py
│   └── ...
├── integration/
└── numerical/
```

**New Structure** (After Refactoring):
```
rtdetrv3_pytorch/tests/
├── unit/
│   ├── core/               # NEW: Organized by ppdet_pytorch module
│   │   ├── test_workspace.py
│   │   └── test_registry.py
│   ├── modeling/           # NEW: All modeling tests grouped
│   │   ├── test_backbone.py
│   │   ├── test_neck.py
│   │   └── ...
│   ├── data/               # NEW: Future data module tests
│   ├── engine/             # NEW: Future engine tests
│   ├── optimizer/          # NEW: Future optimizer tests
│   └── metrics/            # NEW: Future metrics tests
├── integration/            # Unchanged
└── numerical/              # Unchanged
```

**Key Changes**:
1. **Organized by module**: Unit tests now mirror `ppdet_pytorch` package structure
2. **Scalability**: Easy to add new tests for data, engine, optimizer, metrics modules
3. **Clarity**: Clear separation between different ppdet_pytorch sub-packages
4. **Consistency**: Test structure matches code structure

---

## Future Enhancements

### Planned Test Coverage

- [ ] **Data Module Tests**:
  - `test_transforms.py`: Mosaic, Mixup, RandomCrop, etc.
  - `test_dataset.py`: COCODataset, VOCDataset, LVISDataset
  - `test_reader.py`: DataLoader construction and collate functions

- [ ] **Engine Module Tests**:
  - `test_trainer.py`: Training loop, checkpointing, evaluation
  - `test_callbacks.py`: LogPrinter, Checkpointer, BestModelSaver

- [ ] **Optimizer Module Tests**:
  - `test_optimizer.py`: OptimizerBuilder, AdamW, SGD
  - `test_lr_scheduler.py`: CosineDecay, LinearWarmup
  - `test_ema.py`: Exponential Moving Average

- [ ] **Metrics Module Tests**:
  - `test_metrics.py`: COCOMetric, VOCMetric
  - `test_coco_utils.py`: mAP calculation utilities

### Performance Testing
- Benchmark tests for training speed
- Memory profiling tests
- GPU utilization tests

### End-to-End Tests
- Full training pipeline tests (multi-epoch)
- Distributed training tests (multi-GPU)
- Export and deployment tests (ONNX, TorchScript)

---

**Last Updated**: 2025-10-20
**Maintainer**: RT-DETRv3 PyTorch Migration Team
**Contact**: See `CLAUDE.md` for project guidelines

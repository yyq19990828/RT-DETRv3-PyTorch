# API Contracts for RT-DETRv3 PyTorch Migration

This directory contains the interface contracts (abstract base classes and type definitions) that all components must implement to ensure compatibility and maintainability.

## Overview

契约式设计 (Design by Contract) 确保了:
1. **接口一致性**: 所有组件遵循统一的接口规范
2. **可测试性**: 接口定义明确了输入/输出格式,便于单元测试
3. **可替换性**: 只要满足接口契约,组件可以自由替换实现
4. **文档作用**: 接口即文档,清晰描述了各组件的职责

## Files

### `model_interface.py`
定义模型相关的接口契约:
- `BaseDetector`: 所有检测模型的基类
- `RTDETRv3Interface`: RT-DETRv3 特定接口
- `BackboneInterface`: Backbone 网络接口
- `NeckInterface`: Neck (特征融合) 网络接口
- `TransformerInterface`: Transformer 模块接口
- `HeadInterface`: 检测头接口
- `LossInterface`: 损失函数接口

**Key Contracts**:
```python
# 模型必须实现 forward() 方法
def forward(
    images: Tensor,
    targets: Optional[List[Dict]]
) -> Union[Dict[str, Tensor], List[Dict]]:
    """
    - Training: 返回损失字典
    - Inference: 返回预测结果列表
    """
    pass
```

### `data_interface.py`
定义数据加载相关的接口契约:
- `BaseDataset`: 所有数据集的基类
- `COCODatasetInterface`: COCO 格式数据集接口
- `TransformInterface`: 数据增强基类
- `MosaicInterface`, `MixupInterface`: 特定增强接口
- `collate_batch()`: 批处理拼接函数
- `DataLoaderInterface`: 数据加载器封装

**Key Contracts**:
```python
# 数据集必须返回标准化的样本字典
def __getitem__(idx: int) -> Dict[str, Any]:
    """
    Returns:
        {
            'image': np.ndarray or Tensor,
            'gt_bbox': [N, 4] bounding boxes,
            'gt_class': [N] class labels,
            'im_id': int image ID
        }
    """
    pass
```

### `training_interface.py`
定义训练流程相关的接口契约:
- `TrainerInterface`: 训练引擎接口
- `OptimizerBuilderInterface`: 优化器构建器
- `LRSchedulerBuilderInterface`: 学习率调度器构建器
- `EMAInterface`: Exponential Moving Average 接口
- `EvaluatorInterface`: 模型评估器接口
- `CheckpointManagerInterface`: 检查点管理器接口

**Key Contracts**:
```python
# 训练器必须实现训练和评估方法
def train(self) -> None:
    """Main training loop."""
    pass

def evaluate(self, epoch: int) -> Dict[str, float]:
    """
    Returns:
        Metrics dict: {'mAP': 0.536, 'AP50': 0.715, ...}
    """
    pass
```

## Usage Guidelines

### 1. 实现新组件时

所有新组件必须继承对应的接口类:

```python
from contracts.model_interface import BackboneInterface

class ResNet(BackboneInterface):
    """ResNet backbone implementation."""

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 实现代码
        pass

    @property
    def out_channels(self) -> List[int]:
        return [256, 512, 1024, 2048]
```

### 2. 编写测试时

使用接口定义验证输入/输出格式:

```python
def test_backbone_interface():
    backbone = ResNet(depth=50)
    x = torch.randn(2, 3, 640, 640)

    # 验证输出符合接口定义
    features = backbone(x)
    assert isinstance(features, list)
    assert len(features) == 4
    assert features[0].shape == (2, 256, 160, 160)
```

### 3. 配置文件驱动

接口支持从配置文件实例化组件:

```yaml
# config.yml
ResNet:
  type: ResNet  # 在 global_config 中注册的类名
  depth: 50
  variant: d
  freeze_at: 0
```

```python
from ppdet.core.workspace import create

# 根据配置创建实例
backbone = create(config['ResNet'])
```

## Type Annotations

所有接口使用严格的类型注解,支持静态类型检查:

```python
from typing import Dict, List, Tuple, Optional
import torch

def process_predictions(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    """Type hints enable IDE autocomplete and mypy checking."""
    pass
```

运行类型检查:
```bash
mypy rtdetrv3_pytorch/
```

## Contract Validation

建议在每个组件的 `__init__()` 方法中验证参数:

```python
class DETRLoss(LossInterface):
    def __init__(self, num_classes: int, loss_coeff: Dict[str, float]):
        assert num_classes > 0, "num_classes must be positive"
        assert all(v > 0 for v in loss_coeff.values()), "All loss coefficients must be positive"
        self.num_classes = num_classes
        self.loss_coeff = loss_coeff
```

## Migration Strategy

迁移现有代码到新接口:

1. **识别组件类型**: 确定组件属于哪个接口类别 (Model, Data, Training)
2. **继承对应接口**: `class MyComponent(Interface):`
3. **实现必需方法**: 实现所有 `raise NotImplementedError` 的方法
4. **添加类型注解**: 使用接口定义的类型别名
5. **验证测试**: 运行单元测试确保符合接口契约

## Example: Implementing a New Backbone

```python
# rtdetrv3_pytorch/ppdet/modeling/backbones/my_backbone.py

from typing import List
import torch
import torch.nn as nn
from contracts.model_interface import BackboneInterface
from ppdet.core.workspace import register

@register
class MyBackbone(BackboneInterface):
    """Custom backbone implementation."""

    def __init__(self, depth: int = 50):
        super().__init__()
        self.depth = depth
        # ... 初始化层

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: [B, 3, H, W]

        Returns:
            [
                [B, 256, H/4, W/4],   # C2
                [B, 512, H/8, W/8],   # C3
                [B, 1024, H/16, W/16], # C4
                [B, 2048, H/32, W/32]  # C5
            ]
        """
        # 实现特征提取
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]

    @property
    def out_channels(self) -> List[int]:
        return [256, 512, 1024, 2048]
```

## References

- **Design by Contract**: https://en.wikipedia.org/wiki/Design_by_contract
- **Python Type Hints**: https://docs.python.org/3/library/typing.html
- **PyTorch nn.Module**: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- **COCO API**: https://github.com/cocodataset/cocoapi

---

**Last Updated**: 2025-10-20
**Status**: Phase 1 完成,接口定义已锁定

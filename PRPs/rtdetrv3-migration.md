name: "RT-DETRv3 PaddlePaddle to PyTorch Migration PRP"
description: |
  Complete migration of RT-DETRv3 from PaddlePaddle to PyTorch with full context and validation loops
  for AI agent implementation success.

---

## Goal
迁移完整的 RT-DETRv3 目标检测模型从 PaddlePaddle 框架到 PyTorch 框架，创建一个功能完整、性能对等的 PyTorch 实现，包括模型架构、训练/推理管道、配置系统和测试套件。

## Why
- **生态系统兼容性**: PyTorch 在研究和产业界有更广泛的应用和支持
- **性能优化**: 利用 PyTorch 的最新优化和部署工具（TorchScript、ONNX、TensorRT）
- **开发效率**: 与现有的 PyTorch 工作流程和工具链集成
- **社区支持**: 更活跃的开源社区和更丰富的第三方库支持
- **学习成本**: 团队已有 PyTorch 经验，减少学习成本

## What
创建一个完整的 RT-DETRv3 PyTorch 实现，包括：
- 完整的模型架构（Backbone、Neck、Head、Transformer）
- 训练和推理管道
- 数据加载和预处理
- 损失函数和优化器
- 配置系统
- 导出和部署支持
- 完整的测试套件

### Success Criteria
- [ ] 模型架构完全匹配 PaddlePaddle 版本
- [ ] 推理结果与 PaddlePaddle 版本数值一致（误差 < 1e-5）
- [ ] 支持 COCO 和 LVIS 数据集训练
- [ ] 支持多种骨干网络（ResNet-18/34/50/101）
- [ ] 支持模型导出为 ONNX 格式
- [ ] 通过所有单元测试和集成测试
- [ ] 性能基准测试通过（速度和精度）

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://pytorch.org/docs/stable/nn.html#transformer-layers
  why: PyTorch Transformer 实现模式和最佳实践
  
- url: https://pytorch.org/vision/stable/models.html
  why: PyTorch 视觉模型的标准实现模式
  
- file: /home/tyjt/桌面/migrate_paddle2pytorch/examples/rtdetrv2_pytorch/
  why: RT-DETRv2 PyTorch 实现参考模式，架构设计和工程实践
  
- file: /home/tyjt/桌面/migrate_paddle2pytorch/RT-DETRv3-paddle/ppdet/modeling/architectures/rtdetrv3.py
  why: RT-DETRv3 核心架构定义，需要完全迁移的源代码
  
- file: /home/tyjt/桌面/migrate_paddle2pytorch/RT-DETRv3-paddle/ppdet/modeling/transformers/rtdetr_transformerv3.py
  why: RT-DETRv3 Transformer 实现，包含关键的改进和优化
  
- url: https://github.com/lyuwenyu/RT-DETR
  why: 官方 RT-DETR 仓库，包含 PyTorch 实现参考
  
- url: https://github.com/huggingface/pytorch-image-models
  why: timm 库中的 Vision Transformer 实现模式
  
- doc: https://pytorch.org/docs/stable/torch.html#serialization
  why: PyTorch 模型保存和加载的标准做法
  
- file: /home/tyjt/桌面/migrate_paddle2pytorch/RT-DETRv3-paddle/configs/rtdetrv3/
  why: RT-DETRv3 配置文件结构，需要适配到 PyTorch 配置系统
```

### Current Codebase tree 
```bash
migrate_paddle2pytorch/
├── RT-DETRv3-paddle/          # 源 PaddlePaddle 实现
│   ├── ppdet/
│   │   ├── modeling/
│   │   │   ├── architectures/rtdetrv3.py    # 主架构
│   │   │   ├── transformers/rtdetr_transformerv3.py  # Transformer v3
│   │   │   ├── backbones/       # 骨干网络
│   │   │   ├── necks/          # 颈部网络
│   │   │   ├── heads/          # 检测头
│   │   │   └── losses/         # 损失函数
│   │   ├── data/               # 数据处理
│   │   └── core/               # 核心工具
│   ├── configs/rtdetrv3/       # 配置文件
│   └── tools/                  # 工具脚本
├── examples/
│   ├── rtdetrv2_pytorch/       # RT-DETRv2 PyTorch 参考实现
│   └── cpp_extension_example/  # C++ 扩展示例
└── PRPs/                       # 项目需求文档
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
migrate_paddle2pytorch/
├── examples/
│   └── rtdetrv3_pytorch/       # 新的 RT-DETRv3 PyTorch 实现
│       ├── src/
│       │   ├── core/
│       │   │   ├── __init__.py
│       │   │   ├── config.py   # 配置系统
│       │   │   └── workspace.py # 工作空间管理
│       │   ├── data/
│       │   │   ├── __init__.py
│       │   │   ├── dataset.py  # 数据集定义
│       │   │   ├── transforms.py # 数据变换
│       │   │   └── dataloader.py # 数据加载器
│       │   ├── nn/
│       │   │   ├── __init__.py
│       │   │   ├── backbone/
│       │   │   │   ├── __init__.py
│       │   │   │   └── resnet.py # ResNet 骨干网络
│       │   │   ├── neck/
│       │   │   │   ├── __init__.py
│       │   │   │   └── hybrid_encoder.py # 混合编码器
│       │   │   ├── transformer/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── rtdetr_transformerv3.py # RT-DETR Transformer v3
│       │   │   │   ├── layers.py # Transformer 层
│       │   │   │   └── utils.py # 工具函数
│       │   │   ├── head/
│       │   │   │   ├── __init__.py
│       │   │   │   └── rtdetr_head.py # RT-DETR 检测头
│       │   │   └── criterion/
│       │   │       ├── __init__.py
│       │   │       └── rtdetr_criterion.py # 损失函数
│       │   ├── zoo/
│       │   │   ├── __init__.py
│       │   │   └── rtdetrv3/
│       │   │       ├── __init__.py
│       │   │       └── rtdetrv3.py # RT-DETRv3 主模型
│       │   ├── solver/
│       │   │   ├── __init__.py
│       │   │   ├── trainer.py  # 训练器
│       │   │   └── evaluator.py # 评估器
│       │   └── optim/
│       │       ├── __init__.py
│       │       └── optimizer.py # 优化器配置
│       ├── configs/
│       │   ├── rtdetrv3_r18vd_6x_coco.yml
│       │   ├── rtdetrv3_r50vd_6x_coco.yml
│       │   └── rtdetrv3_r50vd_6x_lvis.yml
│       ├── tools/
│       │   ├── train.py        # 训练脚本
│       │   ├── eval.py         # 评估脚本
│       │   ├── infer.py        # 推理脚本
│       │   └── export.py       # 导出脚本
│       ├── tests/
│       │   ├── __init__.py
│       │   ├── test_model.py   # 模型测试
│       │   ├── test_transforms.py # 数据变换测试
│       │   └── test_integration.py # 集成测试
│       ├── requirements.txt
│       └── README.md
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: PaddlePaddle 到 PyTorch 的关键差异
# PaddlePaddle 使用 paddle.nn.functional 而 PyTorch 使用 torch.nn.functional
# PaddlePaddle 的 Tensor 形状约定可能与 PyTorch 不同

# 注意事项：
# 1. PaddlePaddle 的 LayerNorm 默认 epsilon=1e-5，PyTorch 默认 eps=1e-5
# 2. PaddlePaddle 的 MultiHeadAttention 实现细节与 PyTorch 不同
# 3. PaddlePaddle 的配置系统使用 @register 装饰器，需要适配到 PyTorch
# 4. 激活函数：PaddlePaddle 使用 F.silu，PyTorch 使用 F.silu 或 F.swish
# 5. 初始化：PaddlePaddle 使用 paddle.nn.initializer，PyTorch 使用 torch.nn.init

# 性能关键点：
# 1. PyTorch 的 scaled_dot_product_attention 可以提供更好的性能
# 2. 使用 torch.compile 可以显著提升推理速度
# 3. 确保使用 torch.jit.script 兼容的代码

# 配置系统迁移：
# PaddlePaddle 使用 __inject__ 进行依赖注入，PyTorch 需要手动管理
# 需要实现类似的注册机制来管理模型组件
```

## Implementation Blueprint

### Data models and structure
```python
# 核心数据结构和类型定义
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from dataclasses import dataclass
from omegaconf import DictConfig

@dataclass
class RTDETRConfig:
    # 模型配置
    backbone: str = "ResNet50"
    neck: str = "HybridEncoder"
    head: str = "RTDETRHead"
    num_classes: int = 80
    
    # 训练配置
    batch_size: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    
    # 数据配置
    input_size: Tuple[int, int] = (640, 640)
    num_queries: int = 300

class ModelOutput:
    """标准化的模型输出格式"""
    def __init__(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor):
        self.pred_logits = pred_logits
        self.pred_boxes = pred_boxes
```

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1: 创建项目结构和基础配置
CREATE examples/rtdetrv3_pytorch/:
  - 创建完整的目录结构
  - 设置 __init__.py 文件
  - 创建 requirements.txt
  - 创建基础的 README.md

Task 2: 实现配置系统
CREATE src/core/config.py:
  - 参考 rtdetrv2_pytorch/src/core/config.py 模式
  - 实现 YAML 配置加载
  - 支持配置继承和覆盖
  - 添加配置验证

CREATE src/core/workspace.py:
  - 实现组件注册机制
  - 支持依赖注入
  - 镜像 PaddlePaddle 的工作空间模式

Task 3: 实现骨干网络
CREATE src/nn/backbone/resnet.py:
  - 移植 RT-DETRv3-paddle/ppdet/modeling/backbones/resnet.py
  - 适配 PyTorch 的 ResNet 实现
  - 支持 ResNet-18/34/50/101 变种
  - 确保输出特征图匹配

Task 4: 实现颈部网络
CREATE src/nn/neck/hybrid_encoder.py:
  - 移植混合编码器实现
  - 包含 CNN 和 Transformer 的混合结构
  - 确保特征融合正确

Task 5: 实现 RT-DETR Transformer v3
CREATE src/nn/transformer/rtdetr_transformerv3.py:
  - 移植 RT-DETRv3-paddle/ppdet/modeling/transformers/rtdetr_transformerv3.py
  - 实现层次化密集正监督机制
  - 使用 PyTorch 的 nn.MultiheadAttention 或自定义实现
  - 确保前向传播逻辑完全匹配

CREATE src/nn/transformer/layers.py:
  - 实现 Transformer 编码器和解码器层
  - 包含位置编码和注意力机制
  - 支持缓存和推理优化

Task 6: 实现检测头
CREATE src/nn/head/rtdetr_head.py:
  - 移植检测头实现
  - 包含分类和边界框回归分支
  - 实现查询向量和输出投影

Task 7: 实现损失函数
CREATE src/nn/criterion/rtdetr_criterion.py:
  - 移植 RT-DETRv3 的损失函数
  - 包含分类损失、边界框损失和 IoU 损失
  - 实现匈牙利算法匹配

Task 8: 实现主模型
CREATE src/zoo/rtdetrv3/rtdetrv3.py:
  - 组合所有组件创建完整模型
  - 实现前向传播和后处理
  - 支持训练和推理模式

Task 9: 实现数据处理
CREATE src/data/dataset.py:
  - 支持 COCO 和 LVIS 数据集
  - 镜像 PaddlePaddle 的数据处理流程

CREATE src/data/transforms.py:
  - 实现数据增强和预处理
  - 确保与 PaddlePaddle 版本一致

Task 10: 实现训练和评估
CREATE src/solver/trainer.py:
  - 实现训练循环
  - 支持多 GPU 训练
  - 包含学习率调度和检查点保存

CREATE src/solver/evaluator.py:
  - 实现 COCO 评估指标
  - 支持验证和测试

Task 11: 创建工具脚本
CREATE tools/train.py:
  - 训练脚本
  - 支持配置文件和命令行参数

CREATE tools/eval.py:
  - 评估脚本
  - 支持单模型和多模型评估

CREATE tools/infer.py:
  - 推理脚本
  - 支持图像和视频推理

CREATE tools/export.py:
  - 模型导出脚本
  - 支持 ONNX 和 TorchScript 导出

Task 12: 创建配置文件
CREATE configs/:
  - 移植所有 RT-DETRv3 配置文件
  - 适配 PyTorch 配置格式
  - 确保超参数匹配

Task 13: 实现测试套件
CREATE tests/:
  - 单元测试所有组件
  - 集成测试完整流程
  - 数值精度验证测试
```

### Per task pseudocode as needed added to each task

```python
# Task 5: RT-DETR Transformer v3 实现关键伪代码
class RTDETRTransformerV3(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # CRITICAL: 使用 PyTorch 的 nn.TransformerEncoder 或自定义实现
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True  # 重要：PyTorch 2.0+ 建议使用 batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # PATTERN: 层次化密集正监督需要多个解码器分支
        self.decoder_layers = nn.ModuleList([
            RTDETRDecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)
        ])
        
        # GOTCHA: 确保位置编码与 PaddlePaddle 版本匹配
        self.pos_encoding = PositionEmbedding(d_model)
        
    def forward(self, features, query_embeddings, pos_embed=None):
        # CRITICAL: 特征图需要 flatten 并转置以匹配 Transformer 输入格式
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # [hw, bs, c]
        
        # PATTERN: 添加位置编码
        if pos_embed is not None:
            features = features + pos_embed
            
        # 编码器处理
        memory = self.encoder(features)
        
        # 解码器处理 - 层次化输出
        outputs = []
        for layer in self.decoder_layers:
            query_embeddings = layer(query_embeddings, memory)
            outputs.append(query_embeddings)
            
        return outputs  # 返回所有层的输出用于监督

# Task 7: 损失函数实现关键伪代码
class RTDETRCriterion(nn.Module):
    def __init__(self, num_classes, matcher_cost_class=1, matcher_cost_bbox=5, matcher_cost_giou=2):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou
        )
        
        # CRITICAL: 使用 focal loss 处理类别不平衡
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
    def forward(self, outputs, targets):
        # PATTERN: 对每个解码器层都计算损失（层次化监督）
        losses = {}
        for i, output in enumerate(outputs):
            # 匈牙利匹配
            indices = self.matcher(output, targets)
            
            # 计算分类损失
            cls_loss = self.focal_loss(output['pred_logits'], targets, indices)
            
            # 计算边界框损失
            bbox_loss = self.compute_bbox_loss(output['pred_boxes'], targets, indices)
            
            losses[f'loss_cls_{i}'] = cls_loss
            losses[f'loss_bbox_{i}'] = bbox_loss
            
        return losses

# Task 8: 主模型组装伪代码
class RTDETRv3(nn.Module):
    def __init__(self, backbone, neck, transformer, head, num_classes=80):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.transformer = transformer
        self.head = head
        
        # CRITICAL: 查询向量初始化
        self.query_embed = nn.Embedding(300, 256)  # 300 queries, 256 dim
        
    def forward(self, images, targets=None):
        # PATTERN: 特征提取 -> 特征融合 -> Transformer -> 检测头
        features = self.backbone(images)
        features = self.neck(features)
        
        # GOTCHA: 确保特征图尺寸正确传递给 Transformer
        query_embeddings = self.query_embed.weight.unsqueeze(0).repeat(images.size(0), 1, 1)
        
        transformer_outputs = self.transformer(features, query_embeddings)
        
        # 检测头处理
        outputs = []
        for transformer_output in transformer_outputs:
            pred_logits, pred_boxes = self.head(transformer_output)
            outputs.append({
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes
            })
            
        return outputs
```

### Integration Points
```yaml
CONFIGURATION:
  - adapt from: RT-DETRv3-paddle/configs/rtdetrv3/
  - pattern: "使用 OmegaConf 加载 YAML 配置"
  - integration: "与 src/core/config.py 集成"

DEPENDENCIES:
  - torch>=2.0.0: "利用最新的 PyTorch 特性"
  - torchvision>=0.15.0: "视觉变换和数据集支持"
  - timm>=0.9.0: "预训练模型和组件"
  - pycocotools: "COCO 数据集和评估"
  - omegaconf: "配置管理"

CHECKPOINTS:
  - convert from: "PaddlePaddle 预训练权重"
  - save to: "PyTorch 格式 (.pth)"
  - validation: "数值精度验证"

EXPORT:
  - formats: ["ONNX", "TorchScript", "TensorRT"]
  - optimization: "使用 torch.compile 优化"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# 运行这些命令首先 - 修复所有错误后再继续
ruff check src/ --fix           # 自动修复代码风格
mypy src/                       # 类型检查
python -m py_compile src/zoo/rtdetrv3/rtdetrv3.py  # 语法检查

# 预期：无错误。如果有错误，阅读错误信息并修复。
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns
```python
# CREATE tests/test_model.py 包含以下测试用例：
def test_rtdetrv3_forward():
    """测试模型前向传播"""
    model = RTDETRv3(backbone="ResNet50", num_classes=80)
    images = torch.randn(2, 3, 640, 640)
    outputs = model(images)
    
    assert len(outputs) == 6  # 6个解码器层
    assert outputs[0]['pred_logits'].shape == (2, 300, 80)
    assert outputs[0]['pred_boxes'].shape == (2, 300, 4)

def test_transformer_attention():
    """测试 Transformer 注意力机制"""
    transformer = RTDETRTransformerV3(d_model=256, nhead=8)
    features = torch.randn(2, 256, 20, 20)
    query_embeddings = torch.randn(2, 300, 256)
    
    outputs = transformer(features, query_embeddings)
    assert len(outputs) == 6
    assert outputs[0].shape == (2, 300, 256)

def test_criterion_loss():
    """测试损失函数计算"""
    criterion = RTDETRCriterion(num_classes=80)
    outputs = [{'pred_logits': torch.randn(2, 300, 80), 
                'pred_boxes': torch.randn(2, 300, 4)}]
    targets = [{'labels': torch.tensor([1, 2]), 
                'boxes': torch.tensor([[0.1, 0.1, 0.2, 0.2], 
                                      [0.3, 0.3, 0.4, 0.4]])}]
    
    losses = criterion(outputs, targets)
    assert 'loss_cls_0' in losses
    assert 'loss_bbox_0' in losses

def test_numerical_precision():
    """测试与 PaddlePaddle 版本的数值精度"""
    # 使用固定的随机种子
    torch.manual_seed(42)
    
    model = RTDETRv3.from_pretrained("rtdetrv3_r50vd_6x_coco")
    model.eval()
    
    # 使用与 PaddlePaddle 相同的输入
    images = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        outputs = model(images)
    
    # 与预期的 PaddlePaddle 输出比较
    # expected_logits = load_paddle_reference_output()
    # assert torch.allclose(outputs[0]['pred_logits'], expected_logits, atol=1e-5)
```

```bash
# 运行并迭代直到通过：
python -m pytest tests/test_model.py -v
# 如果失败：阅读错误，理解根本原因，修复代码，重新运行
```

### Level 3: Integration Test
```bash
# 训练测试
python tools/train.py --config configs/rtdetrv3_r50vd_6x_coco.yml --max_epochs 1

# 评估测试  
python tools/eval.py --config configs/rtdetrv3_r50vd_6x_coco.yml --checkpoint outputs/rtdetrv3_r50vd_6x_coco/best.pth

# 推理测试
python tools/infer.py --config configs/rtdetrv3_r50vd_6x_coco.yml --checkpoint outputs/rtdetrv3_r50vd_6x_coco/best.pth --image test_image.jpg

# 导出测试
python tools/export.py --config configs/rtdetrv3_r50vd_6x_coco.yml --checkpoint outputs/rtdetrv3_r50vd_6x_coco/best.pth --format onnx

# 预期：所有脚本成功运行，无错误
```

## Final validation Checklist
- [ ] 所有测试通过: `python -m pytest tests/ -v`
- [ ] 无代码风格错误: `ruff check src/`
- [ ] 无类型错误: `mypy src/`
- [ ] 训练脚本成功运行: `python tools/train.py --config configs/rtdetrv3_r50vd_6x_coco.yml --max_epochs 1`
- [ ] 评估脚本成功运行: `python tools/eval.py --config configs/rtdetrv3_r50vd_6x_coco.yml`
- [ ] 推理脚本成功运行: `python tools/infer.py --config configs/rtdetrv3_r50vd_6x_coco.yml --image test.jpg`
- [ ] 模型导出成功: `python tools/export.py --format onnx`
- [ ] 数值精度验证通过（与 PaddlePaddle 版本差异 < 1e-5）
- [ ] 性能基准测试通过
- [ ] 文档更新完成
- [ ] 所有配置文件格式正确

---

## Anti-Patterns to Avoid
- ❌ 不要直接复制粘贴 PaddlePaddle 代码而不适配 PyTorch 接口
- ❌ 不要忽略张量形状和维度顺序的差异
- ❌ 不要跳过数值精度验证
- ❌ 不要使用已弃用的 PyTorch API
- ❌ 不要忽略 PyTorch 的最佳实践（如 batch_first=True）
- ❌ 不要硬编码应该在配置文件中的值
- ❌ 不要在没有充分测试的情况下提交代码
- ❌ 不要忽略内存和计算效率
- ❌ 不要使用不必要的复杂实现

## Confidence Score: 9/10

这个 PRP 提供了完整的上下文、详细的实现计划和全面的验证策略。基于：
1. 完整的代码库分析和现有 PyTorch 参考实现
2. 详细的官方文档和社区实现研究
3. 清晰的任务分解和执行顺序
4. 全面的测试和验证策略
5. 丰富的上下文信息和已知陷阱

唯一的风险点是某些 PaddlePaddle 特定的实现细节可能需要额外的调研和实验。
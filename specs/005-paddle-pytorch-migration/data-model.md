# Data Model: RT-DETRv3 PyTorch Migration

**Date**: 2025-10-20
**Context**: RT-DETRv3 框架迁移的核心数据实体和关系定义

本文档定义了迁移项目中的关键数据实体、字段、关系和状态转换。

---

## 1. Model Architecture (模型架构)

### Entity: `RTDETRV3`
**Description**: RT-DETRv3 完整模型,集成 backbone、neck、transformer、head 和 auxiliary head。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `backbone` | `nn.Module` | 特征提取网络 (ResNet/ResNeXt) | 必须输出多尺度特征 [C3, C4, C5] | ResNet50 |
| `neck` | `nn.Module` | 特征融合网络 (HybridEncoder) | 输入多尺度,输出单尺度高级特征 | HybridEncoder |
| `transformer` | `nn.Module` | 查询-键-值变换器 (RTDETRTransformerv3) | 包含 encoder 和 decoder | RTDETRTransformerv3 |
| `detr_head` | `nn.Module` | 主检测头 (DINOv3Head) | 支持一对一和一对多分支 | DINOv3Head |
| `aux_o2m_head` | `nn.Module` (optional) | 辅助检测头 (PPYOLOEHead) | 仅训练时使用,可为 None | None |
| `num_classes` | `int` | 类别数量 | > 0 | 80 (COCO) |
| `num_queries` | `int` | Transformer query 数量 | > 0, 推荐 300 | 300 |
| `num_queries_one2many` | `int` | 一对多分支 query 数量 | > 0, 推荐 1500 | 1500 |

**Relationships**:
- **Composition**: `RTDETRV3` 包含 `backbone`, `neck`, `transformer`, `detr_head`, `aux_o2m_head`
- **Dependency**: `transformer` 依赖 `backbone` 和 `neck` 的输出特征
- **Conditional**: `aux_o2m_head` 仅在 `training=True` 时激活

**State Transitions**:
```
[Initialized] --forward()--> [Forward Pass] --loss()--> [Loss Computed]
                                 |
                                 +--inference()--> [Predictions]
```

**Code Reference**: `rtdetrv3_pytorch/models/rtdetrv3.py`

---

## 2. Backbone (特征提取器)

### Entity: `ResNet`
**Description**: ResNet 系列 backbone,提取多尺度图像特征。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `depth` | `int` | ResNet 深度 | 18, 34, 50, 101, 152 | 50 |
| `variant` | `str` | ResNet 变体 | 'd' (ResNetD 改进版) | 'd' |
| `freeze_at` | `int` | 冻结层数 | 0-5 | 0 (不冻结) |
| `return_idx` | `List[int]` | 返回特征层索引 | [1, 2, 3, 4] 对应 C2-C5 | [1, 2, 3, 4] |
| `num_stages` | `int` | Stage 数量 | 4 | 4 |
| `norm_type` | `str` | 归一化层类型 | 'bn', 'sync_bn' | 'bn' |

**Relationships**:
- **Used by**: `RTDETRV3.backbone`
- **Outputs**: `List[Tensor]` of shape `[B, C_i, H_i, W_i]` for i in `return_idx`

**Validation Rules**:
- `freeze_at <= num_stages`
- `max(return_idx) < num_stages`

**Code Reference**: `rtdetrv3_pytorch/models/backbones/resnet.py`

---

## 3. Neck (特征融合器)

### Entity: `HybridEncoder`
**Description**: 融合多尺度特征,输出高级语义特征。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `in_channels` | `List[int]` | 输入特征通道数 | 长度与 backbone 输出一致 | [512, 1024, 2048] |
| `feat_strides` | `List[int]` | 特征步长 | [8, 16, 32] | [8, 16, 32] |
| `hidden_dim` | `int` | 隐藏层维度 | > 0 | 256 |
| `use_encoder_idx` | `List[int]` | 使用 encoder 的特征索引 | [2] (仅最高层) | [2] |
| `num_encoder_layers` | `int` | Encoder 层数 | > 0 | 1 |
| `encoder_layer` | `nn.Module` | Encoder 层定义 | - | TransformerLayer |
| `expansion` | `float` | FFN 扩展比例 | > 0 | 1.0 |

**Relationships**:
- **Receives from**: `ResNet.outputs`
- **Outputs to**: `RTDETRTransformerv3.encoder`

**Code Reference**: `rtdetrv3_pytorch/models/necks/hybrid_encoder.py`

---

## 4. Transformer (查询-键-值变换器)

### Entity: `RTDETRTransformerv3`
**Description**: 包含 encoder 和 decoder 的 transformer 模块,支持多组自注意力扰动。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `num_queries` | `int` | Query embedding 数量 | > 0 | 300 |
| `num_queries_one2many` | `int` | 一对多分支 query 数量 | > 0 | 1500 |
| `num_decoder_layers` | `int` | Decoder 层数 | > 0 | 6 |
| `hidden_dim` | `int` | 隐藏层维度 | > 0 | 256 |
| `num_heads` | `int` | Multi-head attention 头数 | > 0, hidden_dim % num_heads == 0 | 8 |
| `dim_feedforward` | `int` | FFN 维度 | > 0 | 1024 |
| `dropout` | `float` | Dropout 比例 | [0, 1) | 0.0 |
| `num_groups` | `int` | 自注意力扰动分组数 | > 0 | 3 |
| `noise_scale` | `List[float]` | 每组扰动比例 | 长度 = num_groups | [0.0, 0.1, 0.1] |

**Relationships**:
- **Input from**: `HybridEncoder.outputs`
- **Outputs to**: `DINOv3Head`

**State Transitions**:
```
[Query Init] --encoder()--> [Encoded Features]
                |
                +--decoder()--> [Decoded Queries] --split()--> [One2One Queries]
                                                               [One2Many Queries]
                                                               [Denoising Queries]
```

**Code Reference**: `rtdetrv3_pytorch/models/transformers/rtdetr_transformer.py`

---

## 5. Detection Head (检测头)

### Entity: `DINOv3Head`
**Description**: 主检测头,支持一对一、一对多和去噪分支。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `num_classes` | `int` | 类别数量 | > 0 | 80 |
| `hidden_dim` | `int` | 隐藏层维度 | > 0 | 256 |
| `num_queries` | `int` | 一对一 query 数量 | > 0 | 300 |
| `num_queries_one2many` | `int` | 一对多 query 数量 | > 0 | 1500 |
| `num_decoder_layers` | `int` | Decoder 层数 | > 0 | 6 |

**Outputs**:
| Output | Type | Shape | Description |
|--------|------|-------|-------------|
| `pred_logits` | `Tensor` | `[B, N, num_classes]` | 分类 logits |
| `pred_boxes` | `Tensor` | `[B, N, 4]` | 边界框 (cx, cy, w, h 归一化) |
| `aux_outputs` | `List[Dict]` | - | 中间层输出 (用于深度监督) |

**Relationships**:
- **Input from**: `RTDETRTransformerv3.decoder_outputs`
- **Outputs to**: `DETRLoss` (训练) or `PostProcess` (推理)

**Code Reference**: `rtdetrv3_pytorch/models/heads/detr_head.py`

### Entity: `PPYOLOEHead`
**Description**: 辅助检测头,基于 CNN,仅训练时使用。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `num_classes` | `int` | 类别数量 | > 0 | 80 |
| `fpn_strides` | `List[int]` | FPN 特征步长 | [8, 16, 32] | [8, 16, 32] |
| `reg_max` | `int` | DFL 回归最大值 | > 0 | 16 |

**Relationships**:
- **Input from**: `HybridEncoder.outputs` (backbone 特征)
- **Outputs to**: `VFLLoss`, `DFLLoss`, `GIoULoss`

**Code Reference**: `rtdetrv3_pytorch/models/heads/ppyoloe_head.py`

---

## 6. Loss Functions (损失函数)

### Entity: `DETRLoss`
**Description**: 主检测分支的组合损失 (VFL + L1 + GIoU)。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `num_classes` | `int` | 类别数量 | > 0 | 80 |
| `loss_coeff` | `Dict[str, float]` | 损失权重 | 所有值 > 0 | {'class': 1, 'bbox': 5, 'giou': 2} |
| `aux_loss` | `bool` | 是否使用辅助损失 (深度监督) | - | True |
| `use_focal_loss` | `bool` | 是否使用 Focal Loss | - | True |
| `alpha` | `float` | Focal Loss alpha | [0, 1] | 0.75 |
| `gamma` | `float` | Focal Loss gamma | > 0 | 2.0 |

**Outputs**:
| Output | Type | Description |
|--------|------|-------------|
| `loss_class` | `Tensor` | 分类损失 |
| `loss_bbox` | `Tensor` | L1 边界框损失 |
| `loss_giou` | `Tensor` | GIoU 损失 |
| `loss` | `Tensor` | 总损失 (加权求和) |

**Relationships**:
- **Input from**: `DINOv3Head.outputs`, `ground_truth`
- **Used in**: `RTDETRV3.forward()` (训练模式)

**Code Reference**: `rtdetrv3_pytorch/models/losses/detr_loss.py`

---

## 7. Dataset (数据集)

### Entity: `COCODataset`
**Description**: COCO 格式数据集加载器,支持多种数据增强。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `dataset_dir` | `str` | 数据集根目录 | 路径存在 | 'dataset/coco' |
| `image_dir` | `str` | 图像目录 | 相对于 dataset_dir | 'train2017' |
| `anno_path` | `str` | 标注文件路径 | JSON 文件存在 | 'annotations/instances_train2017.json' |
| `transforms` | `List[Callable]` | 数据增强操作 | - | [] |
| `num_classes` | `int` | 类别数量 | > 0 | 80 |

**Sample Structure**:
```python
{
    'image': Tensor [3, H, W],        # RGB 图像
    'gt_bbox': Tensor [N, 4],         # 边界框 (x1, y1, x2, y2)
    'gt_class': Tensor [N],           # 类别 ID
    'gt_score': Tensor [N],           # 置信度 (默认全 1)
    'im_shape': Tensor [2],           # 原始图像尺寸 (H, W)
    'scale_factor': Tensor [2],       # 缩放因子
    'im_id': int                      # 图像 ID
}
```

**Relationships**:
- **Used by**: `DataLoader`
- **Applies**: `Transforms` (Mosaic, Mixup, RandomCrop, etc.)

**Code Reference**: `rtdetrv3_pytorch/dataset/coco_dataset.py`

---

## 8. Data Transforms (数据增强)

### Entity: `Mosaic`
**Description**: Mosaic 数据增强,将 4 张图像拼接为 1 张。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `target_size` | `Tuple[int, int]` | 输出图像尺寸 | > 0 | (640, 640) |
| `prob` | `float` | 应用概率 | [0, 1] | 1.0 |

**Behavior**:
1. 随机选择 3 张其他图像
2. 将 4 张图像按 2x2 网格拼接
3. 调整所有 bbox 坐标到新坐标系
4. 过滤超出边界的 bbox

**Code Reference**: `rtdetrv3_pytorch/dataset/transforms.py:Mosaic`

### Entity: `Mixup`
**Description**: Mixup 数据增强,混合两张图像。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `alpha` | `float` | Beta 分布参数 | > 0 | 1.5 |
| `prob` | `float` | 应用概率 | [0, 1] | 1.0 |

**Behavior**:
1. 随机选择另一张图像
2. 从 Beta(alpha, alpha) 分布采样混合比例 λ
3. 混合图像: `img = λ * img1 + (1-λ) * img2`
4. 合并 bbox 和 class 标注

**Code Reference**: `rtdetrv3_pytorch/dataset/transforms.py:Mixup`

---

## 9. Training Engine (训练引擎)

### Entity: `Trainer`
**Description**: 训练循环管理器,负责 epoch 迭代、优化器更新、学习率调度。

**Fields**:
| Field | Type | Description | Validation Rules | Default |
|-------|------|-------------|------------------|---------|
| `model` | `nn.Module` | 待训练模型 | - | - |
| `optimizer` | `Optimizer` | 优化器 | - | AdamW |
| `lr_scheduler` | `_LRScheduler` | 学习率调度器 | - | CosineAnnealing |
| `train_loader` | `DataLoader` | 训练数据加载器 | - | - |
| `val_loader` | `DataLoader` | 验证数据加载器 | - | - |
| `max_epochs` | `int` | 最大 epoch 数 | > 0 | 72 |
| `save_dir` | `str` | Checkpoint 保存目录 | - | 'output/' |
| `log_iter` | `int` | 日志打印间隔 | > 0 | 10 |
| `eval_epoch` | `int` | 评估间隔 (epoch) | > 0 | 1 |

**State Transitions**:
```
[Initialized] --train()--> [Training Epoch 1] --eval()--> [Evaluation]
                                 |                             |
                                 +--save_checkpoint()----------+
                                 |
                             [Training Epoch 2] ---> ...
                                 |
                             [Training Complete]
```

**Code Reference**: `rtdetrv3_pytorch/engine/trainer.py`

---

## 10. Checkpoint (模型检查点)

### Entity: `Checkpoint`
**Description**: 训练状态快照,包含模型参数、优化器状态等。

**Fields**:
| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| `epoch` | `int` | 当前 epoch | >= 0 |
| `model_state_dict` | `Dict[str, Tensor]` | 模型参数 | - |
| `optimizer_state_dict` | `Dict` | 优化器状态 | - |
| `lr_scheduler_state_dict` | `Dict` | 学习率调度器状态 | - |
| `ema_state_dict` | `Dict[str, Tensor]` (optional) | EMA 模型参数 | - |
| `best_metric` | `float` | 最佳评估指标 (mAP) | - |

**File Format**: `.pth` (PyTorch 原生格式)

**Code Reference**: `rtdetrv3_pytorch/utils/checkpoint.py`

---

## 11. Evaluation Metrics (评估指标)

### Entity: `COCOEvaluator`
**Description**: COCO 评估指标计算器。

**Outputs**:
| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `mAP` | `float` | Mean Average Precision @ IoU=0.50:0.95 | [0, 1] |
| `AP50` | `float` | AP @ IoU=0.50 | [0, 1] |
| `AP75` | `float` | AP @ IoU=0.75 | [0, 1] |
| `APs` | `float` | AP for small objects | [0, 1] |
| `APm` | `float` | AP for medium objects | [0, 1] |
| `APl` | `float` | AP for large objects | [0, 1] |

**Relationships**:
- **Input from**: `RTDETRV3.inference()` predictions
- **Used by**: `Trainer.evaluate()`

**Code Reference**: `rtdetrv3_pytorch/engine/evaluator.py`

---

## 12. Registration System (注册系统)

### Entity: `global_config`
**Description**: 全局注册表,存储所有可配置组件的类定义。

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `{class_name}` | `type` | 注册的类 (通过 `@register` 装饰) |
| `__shared__` | `Dict` | 共享配置参数 (如 num_classes) |

**Operations**:
- `register(cls)`: 注册类到 global_config
- `create(cfg)`: 根据配置创建实例
- `merge_config(cfg)`: 合并 YAML 配置到 global_config

**Code Reference**: `rtdetrv3_pytorch/ppdet/core/workspace.py`

---

## 13. Configuration (配置文件)

### Entity: `YAMLConfig`
**Description**: YAML 配置文件,定义模型、训练、数据等参数。

**Structure**:
```yaml
# Example: rtdetrv3_r50vd_6x_coco.yml

# Runtime
use_gpu: true
log_iter: 10
save_dir: output/rtdetrv3_r50vd_6x_coco

# Training
epoch: 72
LearningRate:
  base_lr: 0.0001
  schedulers:
    - !CosineDecay
      max_epochs: 72
    - !LinearWarmup
      start_factor: 0.001
      steps: 1000

OptimizerBuilder:
  optimizer:
    type: AdamW
    weight_decay: 0.0001

# Model
architecture: RTDETRV3
RTDETRV3:
  backbone: ResNet
  neck: HybridEncoder
  transformer: RTDETRTransformerv3
  detr_head: DINOv3Head
  aux_o2m_head: PPYOLOEHead

ResNet:
  depth: 50
  variant: d
  freeze_at: 0
  return_idx: [1, 2, 3]

# ... (更多配置)
```

**Validation Rules**:
- 所有类型 (`type` 字段) 必须在 `global_config` 中注册
- `__inject__` 依赖必须存在于配置中
- 数值参数必须满足各实体的验证规则

**Code Reference**: `rtdetrv3_pytorch/configs/rtdetrv3/`

---

## 14. Data Flow Diagram

```
COCO Dataset
    |
    v
[Transforms] (Mosaic, Mixup, RandomCrop, ...)
    |
    v
DataLoader (batch_size=32, collate_fn)
    |
    v
RTDETRV3 Model
    |
    +---> Backbone (ResNet) ---> [C3, C4, C5]
    |           |
    |           v
    |     HybridEncoder (Neck) ---> [Encoded Features]
    |           |
    |           v
    |     RTDETRTransformerv3 ---> [Queries]
    |           |
    |           v
    |     DINOv3Head ---> [Pred Logits, Pred Boxes]
    |
    +---> PPYOLOEHead (aux) ---> [Aux Predictions]
    |
    v
[Training] DETRLoss ---> [Total Loss]
    |
    v
Optimizer (AdamW) ---> [Parameter Update]
    |
    v
LR Scheduler ---> [Learning Rate Adjust]
    |
    v
[Validation] COCOEvaluator ---> [mAP, AP50, ...]
    |
    v
Checkpoint Save
```

---

## 15. State Management

### Training State Machine

```
[Start]
   |
   v
[Load Config] ---> [Build Model] ---> [Build DataLoader]
   |                    |                     |
   |                    v                     |
   |            [Load Pretrained]             |
   |                    |                     |
   +--------------------+---------------------+
                        |
                        v
                  [Training Loop]
                        |
        +---------------+---------------+
        |                               |
        v                               v
  [Forward Pass]                  [Evaluate]
        |                               |
        v                               v
  [Compute Loss]                  [Save Best]
        |                               |
        v                               |
  [Backward]                            |
        |                               |
        v                               |
  [Optimizer Step]                      |
        |                               |
        v                               |
  [LR Scheduler Step] ------------------+
        |
        v
  [Save Checkpoint]
        |
        v
  [Epoch Complete] ---> Continue or End
```

---

## 16. Validation Rules Summary

| Entity | Critical Validation |
|--------|---------------------|
| `RTDETRV3` | `num_classes > 0`, `num_queries > 0` |
| `ResNet` | `depth in [18,34,50,101,152]`, `freeze_at <= num_stages` |
| `HybridEncoder` | `len(in_channels) == len(feat_strides)` |
| `RTDETRTransformerv3` | `hidden_dim % num_heads == 0`, `len(noise_scale) == num_groups` |
| `DETRLoss` | `alpha in [0,1]`, `gamma > 0`, all `loss_coeff` > 0 |
| `COCODataset` | `dataset_dir` exists, `anno_path` is valid JSON |
| `Trainer` | `max_epochs > 0`, `log_iter > 0`, `eval_epoch > 0` |

---

**Last Updated**: 2025-10-20
**Status**: Phase 1 完成,准备生成 contracts

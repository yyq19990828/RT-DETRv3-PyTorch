# Research: Paddle to PyTorch Migration

**Date**: 2025-10-20
**Context**: RT-DETRv3 框架迁移关键技术点研究

本文档记录了从 PaddlePaddle 迁移到 PyTorch 过程中的关键技术决策、API映射和最佳实践。

---

## 1. 注册系统迁移 (Registration System)

### 问题陈述
PaddlePaddle 使用集中式注册模式 (`@register` 装饰器 + `global_config` 字典),支持配置文件驱动的组件实例化。PyTorch 原生不强制此模式,需要自行实现。

### 研究结果

#### Decision: 完全迁移 PaddlePaddle 的注册机制到 `ppdet/core/workspace.py`

**Rationale**:
1. **配置文件兼容性**: Paddle 的 YAML 配置文件依赖注册表查找组件,迁移注册系统可保持配置格式一致
2. **依赖注入支持**: PaddlePaddle 的 `__inject__` 和 `__shared__` 注解在注册系统中实现,这是核心功能
3. **代码对等性**: Constitution Principle I 要求框架对等,保留注册系统是最直接的方式

**Implementation Strategy**:
```python
# ppdet/core/workspace.py (模仿 Paddle)

global_config = {}  # 存储所有配置

def register(cls):
    """注册装饰器,将类注册到 global_config"""
    if cls.__name__ in global_config:
        raise ValueError(f"Duplicate registration: {cls.__name__}")
    global_config[cls.__name__] = cls
    return cls

def create(cfg_or_name, **kwargs):
    """工厂函数,根据配置或类名创建实例"""
    if isinstance(cfg_or_name, str):
        cls = global_config[cfg_or_name]
        return cls(**kwargs)
    elif isinstance(cfg_or_name, dict):
        cfg = cfg_or_name.copy()
        name = cfg.pop('type')
        cls = global_config[name]

        # 处理依赖注入 (__inject__)
        if hasattr(cls, '__inject__'):
            for key in cls.__inject__:
                if key in cfg and isinstance(cfg[key], dict):
                    cfg[key] = create(cfg[key])

        # 处理共享配置 (__shared__)
        if hasattr(cls, '__shared__'):
            for key in cls.__shared__:
                if key in global_config.get('__shared__', {}):
                    cfg[key] = global_config['__shared__'][key]

        return cls(**cfg, **kwargs)
```

**Usage Example**:
```python
# 在模型定义中
from ppdet.core.workspace import register

@register
class ResNet(nn.Module):
    __inject__ = ['norm_layer']  # 声明需要注入的依赖

    def __init__(self, depth, norm_layer='BatchNorm2d'):
        super().__init__()
        self.norm_layer = create(norm_layer) if isinstance(norm_layer, dict) else norm_layer

# 在配置文件中
# rtdetrv3_r50vd_6x_coco.yml
ResNet:
  type: ResNet
  depth: 50
  norm_layer:
    type: BatchNorm2d
    num_features: 2048
```

**Alternatives Considered**:
- ❌ **直接使用类构造函数**: 会破坏配置文件格式,需要重写所有 YAML 配置
- ❌ **使用 MMDetection 的注册系统**: 引入额外依赖,且 API 与 Paddle 不完全兼容
- ✅ **自实现简化版 Paddle 注册系统**: 保持对等性,无额外依赖,完全控制

**Numerical Equivalence Impact**: 无影响 (纯粹的元编程,不涉及计算)

---

## 2. 优化器迁移 (Optimizer)

### 问题陈述
PaddlePaddle 的 `AdamW` 使用 `paddle.regularizer.L2Decay` 实现权重衰减,需要映射到 PyTorch 的对等 API。

### 研究结果

#### Decision: 使用 `torch.optim.AdamW` 的原生 `weight_decay` 参数

**Rationale**:
1. **数值等价**: PyTorch 的 `AdamW` 实现解耦权重衰减 (decoupled weight decay),与 Paddle 的 `L2Decay` 数值等价
2. **API 简洁性**: 避免自定义正则化器,使用 PyTorch 原生 API
3. **性能优化**: PyTorch 的 AdamW 包含 CUDA 优化

**API Mapping**:
```python
# PaddlePaddle
import paddle
optimizer = paddle.optimizer.AdamW(
    learning_rate=0.0001,
    parameters=model.parameters(),
    weight_decay=paddle.regularizer.L2Decay(0.01),
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)

# PyTorch Equivalent
import torch.optim as optim
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.01,  # 直接传入标量值
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Critical Parameters for Numerical Equivalence**:
| Parameter | PaddlePaddle | PyTorch |
|-----------|-------------|---------|
| Learning Rate | `learning_rate` | `lr` |
| Weight Decay | `L2Decay(coeff)` | `weight_decay=coeff` |
| Beta1 | `beta1` | `betas[0]` |
| Beta2 | `beta2` | `betas[1]` |
| Epsilon | `epsilon` | `eps` |

**Validation Strategy**:
```python
# 测试代码 (验证数值等价性)
import torch
import paddle
import numpy as np

# 设置相同的随机种子
torch.manual_seed(42)
paddle.seed(42)

# 创建相同的模型参数
param_pt = torch.randn(10, 10, requires_grad=True)
param_pd = paddle.to_tensor(param_pt.detach().numpy(), stop_gradient=False)

# 创建优化器
opt_pt = torch.optim.AdamW([param_pt], lr=0.01, weight_decay=0.001)
opt_pd = paddle.optimizer.AdamW(
    learning_rate=0.01,
    parameters=[param_pd],
    weight_decay=paddle.regularizer.L2Decay(0.001)
)

# 进行一步优化
loss_pt = param_pt.sum()
loss_pt.backward()
opt_pt.step()

loss_pd = param_pd.sum()
loss_pd.backward()
opt_pd.step()

# 检查参数更新是否一致 (tolerance: 1e-6)
assert np.allclose(param_pt.detach().numpy(), param_pd.numpy(), atol=1e-6)
```

**Alternatives Considered**:
- ❌ **自定义 L2Decay 正则化器**: 增加复杂性,PyTorch 原生支持已足够
- ✅ **使用 torch.optim.AdamW**: 数值等价,API 清晰,推荐方案

---

## 3. 学习率调度器迁移 (Learning Rate Scheduler)

### 问题陈述
PaddlePaddle 支持嵌套式学习率调度器 (如 `LinearWarmup(CosineDecay(...))`),PyTorch 需要使用 `SequentialLR` 或自定义实现。

### 研究结果

#### Decision: 使用 `torch.optim.lr_scheduler.SequentialLR` 组合 `LinearLR` 和 `CosineAnnealingLR`

**Rationale**:
1. **PyTorch 2.0+ 原生支持**: `SequentialLR` 是 PyTorch 官方推荐的组合调度器方式
2. **数值等价**: 通过正确设置参数,可以完全复现 Paddle 的调度曲线
3. **可读性**: 明确分离 warmup 和主调度阶段

**API Mapping**:
```python
# PaddlePaddle (嵌套式)
from paddle.optimizer.lr import LinearWarmup, CosineAnnealingDecay

lr = LinearWarmup(
    learning_rate=CosineAnnealingDecay(
        learning_rate=0.0001,
        T_max=72,  # epochs
        eta_min=0
    ),
    warmup_steps=1000,
    start_lr=0.0,
    end_lr=0.0001
)

# PyTorch Equivalent (使用 SequentialLR)
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.0,  # start_lr / base_lr = 0.0 / 0.0001
    end_factor=1.0,
    total_iters=1000
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=72,  # 总 epoch 数
    eta_min=0
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[1000]  # warmup 结束的 iteration
)

# 在训练循环中
for epoch in range(epochs):
    for batch in dataloader:
        # ... 训练代码 ...
        scheduler.step()  # 每个 iteration 调用一次
```

**Critical Parameters**:
| Concept | PaddlePaddle | PyTorch |
|---------|--------------|---------|
| Warmup 起始 LR | `start_lr` | `start_factor = start_lr / base_lr` |
| Warmup 结束 LR | `end_lr` | `end_factor = end_lr / base_lr` |
| Warmup 步数 | `warmup_steps` | `total_iters` |
| Cosine 周期 | `T_max` (epochs) | `T_max` (iterations or epochs) |
| Cosine 最小 LR | `eta_min` | `eta_min` |

**Important Notes**:
1. **Step 单位对齐**: PaddlePaddle 的调度器可能按 epoch 或 iteration 计步,PyTorch 需要显式控制 `scheduler.step()` 的调用时机
2. **Last Epoch 处理**: PyTorch 的 `last_epoch` 参数用于恢复训练,需要正确设置以匹配 checkpoint

**Validation Code**:
```python
# 验证调度曲线一致性
import torch
import paddle
import matplotlib.pyplot as plt

# PaddlePaddle
lr_pd = paddle.optimizer.lr.LinearWarmup(
    paddle.optimizer.lr.CosineAnnealingDecay(0.0001, T_max=1000),
    warmup_steps=100,
    start_lr=0.0,
    end_lr=0.0001
)

# PyTorch
model = torch.nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.0001)
warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.0, end_factor=1.0, total_iters=100)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=900)
lr_pt = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[100])

# 采样 LR 值
lrs_pd = [lr_pd() for _ in range(1000)]
lrs_pt = []
for _ in range(1000):
    lrs_pt.append(opt.param_groups[0]['lr'])
    opt.step()
    lr_pt.step()

# 绘制对比图
plt.plot(lrs_pd, label='PaddlePaddle')
plt.plot(lrs_pt, label='PyTorch', linestyle='--')
plt.legend()
plt.savefig('lr_schedule_comparison.png')

# 数值检查 (tolerance: 1e-7)
assert np.allclose(lrs_pd, lrs_pt, atol=1e-7)
```

**Alternatives Considered**:
- ❌ **自定义 LambdaLR**: 灵活但增加代码复杂度,且不易与 PyTorch 生态集成
- ❌ **使用第三方库 (如 timm)**: 引入额外依赖
- ✅ **SequentialLR 组合**: 官方推荐,代码清晰,数值等价

---

## 4. 数据增强迁移 (Data Augmentation)

### 问题陈述
PaddlePaddle 在 `ppdet/data/transform/` 实现了多种检测专用数据增强 (Mosaic, Mixup, RandomCrop),需要找到 PyTorch 等价实现。

### 研究结果

#### Decision: 保留 Paddle 的数据增强实现,仅替换底层 Tensor 操作

**Rationale**:
1. **数值一致性优先**: 数据增强对训练结果影响显著,完全复现 Paddle 的实现可保证数值等价
2. **Paddle 实现已优化**: PaddlePaddle 的数据增强针对目标检测任务专门优化 (如 Mosaic 的边界框处理)
3. **避免第三方依赖**: Albumentations 等库的实现细节可能与 Paddle 不同,引入差异

**Migration Strategy**:
- **保留**: Paddle 的数据增强逻辑 (如 bbox 坐标变换、标签重分配)
- **替换**: 底层张量操作从 `paddle.to_tensor()` 改为 `torch.tensor()`
- **验证**: 对每个增强操作编写单元测试,对比 Paddle 和 PyTorch 版本的输出

**API Mapping**:

| Augmentation | Paddle Implementation | PyTorch Strategy |
|--------------|----------------------|------------------|
| Mosaic | `ppdet/data/transform/operators.py:Mosaic` | 保留逻辑,替换 `paddle.*` 为 `torch.*` |
| Mixup | `ppdet/data/transform/operators.py:Mixup` | 保留逻辑,替换张量操作 |
| RandomCrop | `ppdet/data/transform/operators.py:RandomCrop` | 保留逻辑,使用 `torchvision.transforms.functional.crop` 作为底层实现 |
| RandomFlip | `ppdet/data/transform/operators.py:RandomFlipImage` | `torchvision.transforms.RandomHorizontalFlip` + 自定义 bbox 翻转 |
| Resize | `ppdet/data/transform/operators.py:Resize` | `torchvision.transforms.functional.resize` + bbox 缩放 |

**Example Migration (Mosaic)**:
```python
# PaddlePaddle 原始实现 (简化)
import paddle
import numpy as np

class Mosaic:
    def __call__(self, samples):
        # ... 逻辑代码 ...
        mosaic_img = paddle.concat([img1, img2, img3, img4], axis=...)
        return mosaic_img

# PyTorch 迁移版本
import torch
import numpy as np

class Mosaic:
    def __call__(self, samples):
        # ... 保持相同的逻辑代码 ...
        # 仅替换 Tensor 操作
        mosaic_img = torch.cat([img1, img2, img3, img4], dim=...)
        return mosaic_img
```

**Critical Migration Points**:
1. **坐标系统**: 确保 bbox 坐标格式一致 (x1y1x2y2 vs xywh)
2. **随机种子**: 使用相同的 `torch.manual_seed()` 和 `np.random.seed()` 保证可复现性
3. **边界情况**: 处理空 bbox、超出边界的 bbox 等边缘情况

**Validation Strategy**:
```python
# 单元测试 (验证数值等价性)
import torch
import paddle
import numpy as np

def test_mosaic_equivalence():
    # 准备相同的输入数据
    np.random.seed(42)
    images = [np.random.rand(640, 640, 3) for _ in range(4)]
    bboxes = [np.random.rand(10, 4) * 640 for _ in range(4)]

    # PaddlePaddle 版本
    mosaic_pd = MosaicPaddle()
    result_pd = mosaic_pd({'image': images, 'gt_bbox': bboxes})

    # PyTorch 版本
    np.random.seed(42)  # 重置种子
    mosaic_pt = MosaicPyTorch()
    result_pt = mosaic_pt({'image': images, 'gt_bbox': bboxes})

    # 验证输出一致 (tolerance: 1e-5)
    assert np.allclose(result_pd['image'], result_pt['image'], atol=1e-5)
    assert np.allclose(result_pd['gt_bbox'], result_pt['gt_bbox'], atol=1e-5)
```

**Alternatives Considered**:
- ❌ **使用 Albumentations**: 实现细节不同,难以保证数值等价
- ❌ **使用 torchvision.transforms**: 缺少目标检测专用增强 (如 Mosaic)
- ✅ **保留 Paddle 实现并迁移**: 最大化数值一致性,推荐方案

---

## 5. DataLoader 和 Batch Collation 迁移

### 问题陈述
PaddlePaddle 的 `paddle.io.DataLoader` 与 PyTorch 的 `torch.utils.data.DataLoader` API 相似,但在 `collate_fn` 的默认行为上有差异。

### 研究结果

#### Decision: 使用 PyTorch 的 `DataLoader` + 自定义 `collate_fn`

**Rationale**:
1. **API 高度相似**: 两个框架的 DataLoader 接口几乎一致,迁移成本低
2. **自定义 collate_fn 必须**: 目标检测任务中,每张图像的 bbox 数量不同,需要自定义 batch 拼接逻辑
3. **性能优化**: PyTorch 的 DataLoader 支持 `pin_memory` 和多进程加载,性能更优

**API Mapping**:
```python
# PaddlePaddle
import paddle
from paddle.io import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_batch,
    use_shared_memory=True  # 共享内存加速
)

# PyTorch Equivalent
import torch
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_batch,
    pin_memory=True  # 等价于 use_shared_memory
)
```

**Custom Collate Function** (目标检测专用):
```python
import torch

def collate_batch(batch):
    """
    自定义 collate 函数,处理变长 bbox 列表

    Args:
        batch: List of samples, each sample is a dict:
            {
                'image': Tensor [3, H, W],
                'gt_bbox': Tensor [N, 4],  # N 是可变的
                'gt_class': Tensor [N],
                'im_id': int
            }

    Returns:
        batched_data: Dict of batched tensors
    """
    images = torch.stack([sample['image'] for sample in batch], dim=0)  # [B, 3, H, W]

    # bbox 和 class 保持为列表 (因为长度不一致)
    gt_bboxes = [sample['gt_bbox'] for sample in batch]  # List of [N_i, 4]
    gt_classes = [sample['gt_class'] for sample in batch]  # List of [N_i]
    im_ids = torch.tensor([sample['im_id'] for sample in batch])

    return {
        'image': images,
        'gt_bbox': gt_bboxes,  # 保持为列表,损失函数内部处理
        'gt_class': gt_classes,
        'im_id': im_ids
    }
```

**Critical Differences**:

| Feature | PaddlePaddle | PyTorch | Notes |
|---------|--------------|---------|-------|
| Shared Memory | `use_shared_memory=True` | `pin_memory=True` | 功能等价,加速 GPU 传输 |
| Default collate | 支持变长列表 | 要求固定形状 | PyTorch 需要自定义 `collate_fn` |
| Random Seed | `worker_init_fn` | `worker_init_fn` | 两者都支持,确保多进程可复现 |

**Worker Init Function** (确保多进程随机种子):
```python
import numpy as np
import random
import torch

def worker_init_fn(worker_id):
    """为每个 DataLoader worker 设置不同的随机种子"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 在 DataLoader 中使用
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn  # 确保可复现性
)
```

**Validation Strategy**:
```python
# 验证 DataLoader 输出一致性
import torch
import paddle
import numpy as np

# 设置相同的随机种子
torch.manual_seed(42)
paddle.seed(42)
np.random.seed(42)

# PaddlePaddle
loader_pd = paddle.io.DataLoader(dataset_pd, batch_size=2, shuffle=False, num_workers=0)
batch_pd = next(iter(loader_pd))

# PyTorch
loader_pt = torch.utils.data.DataLoader(dataset_pt, batch_size=2, shuffle=False, num_workers=0)
batch_pt = next(iter(loader_pt))

# 验证图像一致
assert np.allclose(batch_pd['image'].numpy(), batch_pt['image'].numpy(), atol=1e-5)
# 验证 bbox 一致 (逐个检查,因为是列表)
for bbox_pd, bbox_pt in zip(batch_pd['gt_bbox'], batch_pt['gt_bbox']):
    assert np.allclose(bbox_pd.numpy(), bbox_pt.numpy(), atol=1e-5)
```

**Alternatives Considered**:
- ❌ **Padding bbox 到固定长度**: 浪费内存,且需要额外掩码处理
- ✅ **保持 bbox 为列表,在损失函数中处理**: 灵活,符合 PyTorch 生态习惯

---

## 6. 其他关键 API 映射

### 6.1 BatchNorm 和 SyncBatchNorm

| PaddlePaddle | PyTorch | Notes |
|--------------|---------|-------|
| `paddle.nn.BatchNorm2D` | `torch.nn.BatchNorm2d` | 默认参数一致 |
| `paddle.nn.SyncBatchNorm` | `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` | PyTorch 使用转换函数 |

**Migration Example**:
```python
# PaddlePaddle
import paddle
model = paddle.nn.SyncBatchNorm(num_features=256)

# PyTorch
import torch
model = torch.nn.BatchNorm2d(num_features=256)
# 分布式训练时转换为 SyncBatchNorm
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

### 6.2 Checkpoint 保存和加载

| PaddlePaddle | PyTorch | Notes |
|--------------|---------|-------|
| `paddle.save(state_dict, path)` | `torch.save(state_dict, path)` | 功能一致 |
| `paddle.load(path)` | `torch.load(path, map_location='cpu')` | PyTorch 需要指定 `map_location` |

**Critical Note**: Paddle 的 `.pdparams` 文件无法直接被 PyTorch 加载,需要权重转换脚本:
```python
import paddle
import torch
import numpy as np

def convert_paddle_to_pytorch(paddle_path, pytorch_path):
    """转换 Paddle checkpoint 到 PyTorch 格式"""
    state_dict_pd = paddle.load(paddle_path)
    state_dict_pt = {}

    for key, value in state_dict_pd.items():
        # 转换 Tensor 为 NumPy,再转为 PyTorch
        if isinstance(value, paddle.Tensor):
            state_dict_pt[key] = torch.from_numpy(value.numpy())
        else:
            state_dict_pt[key] = value

    torch.save(state_dict_pt, pytorch_path)
```

### 6.3 分布式训练

| Feature | PaddlePaddle | PyTorch |
|---------|--------------|---------|
| 初始化 | `paddle.distributed.init_parallel_env()` | `torch.distributed.init_process_group()` |
| 包装模型 | `paddle.DataParallel(model)` | `torch.nn.parallel.DistributedDataParallel(model)` |
| Rank 获取 | `paddle.distributed.get_rank()` | `torch.distributed.get_rank()` |

**Migration Example**:
```python
# PaddlePaddle
import paddle
paddle.distributed.init_parallel_env()
model = paddle.DataParallel(model)

# PyTorch
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
model = torch.nn.parallel.DistributedDataParallel(
    model.to(local_rank),
    device_ids=[local_rank]
)
```

---

## 7. 数值验证清单 (Numerical Validation Checklist)

根据 Constitution Principle III (Validation-Driven Development),以下是迁移后必须验证的检查点:

### 7.1 组件级验证
- [ ] Backbone 前向传播输出一致 (tolerance: 1e-5)
- [ ] Neck 前向传播输出一致
- [ ] Transformer 前向传播输出一致
- [ ] Head 前向传播输出一致
- [ ] Loss 计算结果一致
- [ ] 后处理 (NMS) 输出一致

### 7.2 训练流程验证
- [ ] 优化器参数更新一致 (单步)
- [ ] 学习率调度曲线一致
- [ ] 数据增强输出一致 (固定种子)
- [ ] DataLoader batch 输出一致
- [ ] 梯度反向传播一致
- [ ] EMA 更新一致

### 7.3 端到端验证
- [ ] 单 epoch 训练 loss 曲线一致
- [ ] 5 epoch 训练后 COCO mAP 差异 < 0.5%
- [ ] 推理速度差异 < 5% (同硬件)
- [ ] 内存占用差异 < 10%

### 7.4 边界情况验证
- [ ] 空 batch 处理
- [ ] 单样本 batch 处理
- [ ] 极大/极小图像尺寸处理
- [ ] 超多/超少 bbox 处理

---

## 8. 已知风险和缓解策略

### Risk 1: 浮点数精度累积误差
**描述**: 长时间训练后,微小的数值差异可能累积导致精度偏差。
**缓解**:
- 使用相同的随机种子
- 每个 epoch 保存 checkpoint 并对比中间结果
- 使用混合精度训练时,确保两个框架的 AMP 策略一致

### Risk 2: 数据增强的随机性
**描述**: 即使设置相同种子,不同框架的随机数生成器可能产生不同序列。
**缓解**:
- 优先使用 NumPy 的随机数生成器 (两框架共享)
- 编写单元测试锁定增强输出
- 必要时禁用随机增强进行对比实验

### Risk 3: 硬件和库版本差异
**描述**: CUDA、cuDNN 版本差异可能导致数值结果不同。
**缓解**:
- 使用 Docker 容器固定环境
- 文档记录验证时的硬件和软件版本
- 在多台机器上重复验证

---

## 9. 实施优先级 (Implementation Priority)

基于 Constitution 的模块化迁移策略 (Principle II),推荐以下顺序:

1. ✅ **Phase 0 (已完成)**: Backbone → Neck → Transformer → Head → Loss
2. **Phase 1 (当前)**: 注册系统 → 优化器 → 学习率调度器
3. **Phase 2**: 数据增强 → DataLoader → 训练引擎
4. **Phase 3**: 分布式训练 → EMA → Checkpoint 管理
5. **Phase 4**: 端到端验证 → 性能优化 → 文档完善

---

## 10. 参考资源

- **PaddlePaddle 官方文档**: https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html
- **PyTorch 官方文档**: https://pytorch.org/docs/stable/index.html
- **Perplexity 研究结果**: 见本文档第 1-5 节
- **RT-DETRv3 技术报告**: `/home/tyjt/桌面/RT-DETRv3/tech-report.md`
- **Constitution**: `/home/tyjt/桌面/RT-DETRv3/.specify/memory/constitution.md`

---

**Last Updated**: 2025-10-20
**Status**: 完成 Phase 0 研究,准备进入 Phase 1 设计

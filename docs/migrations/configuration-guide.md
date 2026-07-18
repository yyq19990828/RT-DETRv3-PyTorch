# RT-DETRv3 配置迁移指南

本文只描述本仓库已声明支持的 RT-DETRv3 R18/R34/R50 训练、评估、推理和权重转换配置，不把 PaddleDetection 其他架构的 YAML 字段自动视为可用。每个结论分为“直接支持”“需要改写”和“尚未支持”；字段名称相似仍需以当前测试和调用点为准。

## 配置加载合同

- `_BASE_` 以当前 YAML 所在目录为基准递归解析，列表中的 base 依次合并，子配置最后覆盖 base。
- `load_config()` 每次保留已注册组件的 schema，但清除上一份 YAML 的运行时值和命名配置块；同一进程连续加载 R18 → R50 → R18 不会继承前一模型字段。
- 需要在一份已加载配置上追加测试或命令级 override 时，先 `load_config()`，再调用 `merge_config()`；不要依赖第二次 `load_config()` 做增量合并。
- 解析失败发生在替换当前 workspace 之前，因此格式错误的下一份 YAML 不会清空仍在使用的配置。
- `create()` 接受注册类、注册名、全局命名配置块，或包含 `name`/`type` 的配置映射。当前优先级和注入规则见[注册与配置迁移](registry-and-configuration.md)。

**已验证（2026-07-19）**：同一 Python 进程连续构建 R18、R50、R18，backbone depth 为 `18/50/18`，参数量为 `22,942,893 / 45,483,573 / 22,942,893`；workspace 定向测试同时覆盖失败解析保留、shared、inject、from_config 和显式参数冲突。

## 字段支持矩阵

| Paddle 配置范围 | PyTorch 当前状态 | 迁移要求 |
|---|---|---|
| `_BASE_`、`architecture`、`num_classes` | 直接支持 | 保持相对路径；`num_classes` 通过 `__shared__` 注入 reader/head/post-process 等组件 |
| `RTDETRV3`、`ResNet`、`HybridEncoder`、`RTDETRTransformerv3`、`DINOv3Head`、`PPYOLOEHead`、`DETRPostProcess` | 当前三种 COCO 变体直接支持 | 保留已注册类名和当前 schema 字段；新字段先确认构造函数而不是直接复制 |
| `TrainDataset`、`EvalDataset`、`TestDataset` | 支持当前 COCO/ImageFolder 路径 | `dataset_dir` 改为本机或仓库相对路径；不要提交工作站绝对路径 |
| `TrainReader`、`EvalReader`、`TestReader` | 变换字段基本保留，构建形式需要改写 | 每个 reader 块添加 `name`；当前 Infer 严格复用 `TestReader.sample_transforms` |
| Decode/Resize/NormalizeImage/Permute 和当前训练增强 | 当前 RT-DETRv3 路径支持 | 插值、`keep_ratio`、归一化和 collate 语义仍需按输入验证，不能只按同名推断等价 |
| `worker_num`、batch size、shuffle、drop_last | 支持 | `worker_num` 映射为 PyTorch DataLoader worker 数；DDP sampler 与 worker RNG 使用当前 PyTorch 合同 |
| `epoch`、`log_iter`、`save_dir`、`snapshot_epoch` | 支持 | checkpoint 频率和 ETA 语义见[训练与数值验证](training-and-validation.md) |
| `find_unused_parameters`、`norm_type: sync_bn` | 支持当前 DDP 训练路径 | 只在分布式初始化后生效；CPU/单进程不能证明 SyncBN 路径通过 |
| `use_ema`、`ema_decay`、`ema_decay_type` | 支持 PyTorch 自有训练状态 | EMA checkpoint 可由 Eval/Infer 的 `--use-ema` 选择；不导入 Paddle optimizer/EMA 状态 |
| `pretrain_weights` 的 Paddle URL/`.pdparams` | 需要改写 | 先转换为 PyTorch `.pth`，再使用本地路径；Paddle 与转换依赖只在 `dev` extra |
| `LearningRate.schedulers` 的 `!PiecewiseDecay`、`!LinearWarmup` | 需要改写 | 删除 Paddle YAML tag，改为带 `name` 的普通映射；当前 milestone 是全局 optimizer step 语义 |
| `LearningRate.base_lr` | 当前配置需要改写 | 仓库使用顶层 `base_lr` 创建 optimizer，scheduler 从 optimizer parameter groups 读取 LR |
| `OptimizerBuilder.optimizer.type: AdamW`、weight decay、clip norm | 当前路径支持 | PyTorch 参数组、LR multiplier、decay 与执行顺序以当前训练测试为准，不声称 optimizer state 或逐元素更新跨框架一致 |
| `use_gpu` | Train 支持；Eval/Infer 使用各自 `--device` | Train 在字段缺失时自动检测 CUDA；需要强制设备时显式设 `use_gpu`。旧的仓库内 `use_cuda` 没有调用方，已移除 |
| `use_xpu`、`use_mlu`、`use_npu` | 尚未支持 | Train 入口只发出警告并继续使用 CPU/CUDA；不要把这些字段当作后端选择成功 |

## 当前三变体必须保留的改写

### Reader 命名块

Paddle reader 块依赖 `create('TrainReader')` 的全局命名约定；当前 PyTorch 统一用配置映射构建，所以仓库配置显式包含：

```yaml
TrainReader:
  name: TrainReader
  sample_transforms: [...]
```

EvalReader/TestReader 同理。省略 `name` 后把任意 reader dict 直接传给 `create()` 会显式报错，不应通过恢复第二套 Registry 绕过。

### Scheduler 普通映射

```yaml
base_lr: 0.0001
LearningRate:
  schedulers:
    - name: PiecewiseDecay
      gamma: 0.1
      milestones: [60]
      use_warmup: true
    - name: LinearWarmup
      start_factor: 0.001
      steps: 1000
```

上述数值是当前 PyTorch 训练策略，不是 Paddle 原配置 `base_lr=0.0004`、warmup 2,000 step、milestone 100、`gamma=1.0` 的逐项复制。用户已决定训练优化不要求完全对齐；差异必须保留在迁移报告，不能误标为框架 bug。

### 权重路径

配置中的 Paddle 预训练 URL 不能由核心 PyTorch runtime 直接加载。转换步骤使用 `uv sync --extra dev` 环境，训练/评估/推理只接收转换并审核过的 PyTorch state dict。模型参数转换不包含 Paddle optimizer、scheduler 或 scaler 状态。

## Override 边界

- Train 沿用 `-o key=value` 并用 YAML 解析值，支持点分嵌套字段。
- Eval/Infer 当前使用 `-o/--override key=value` 的轻量解析器；它支持布尔、null、数值和逗号列表，但不是 Paddle ArgsParser 的完整语法。
- Convert 的配置参数用于构建目标 state dict，不继承 Train 的所有 override。
- 对列表、字符串转义或复杂嵌套结构，优先写一份派生 YAML，而不是假设四个 CLI 的命令行解析完全相同。统一 CLI contract 是 M5 的后续计划项。

## 尚未声明支持

- Paddle slim/OFA/prune/distill、SSOD、MOT 和非 RT-DETRv3 数据配置。
- Paddle Fleet 专用参数、VisualDL 配置和 Paddle 自定义设备后端。
- 任意 Paddle YAML tag 或序列化 Python 对象的跨框架复用。
- 只因类名、字段名或 tensor shape 相同而推断的语义等价。

添加新配置时，至少验证：递归加载后的实际值、`create()` 构建、一个真实 batch、模型前向、对应 CLI 错误路径，以及不会污染同一进程随后加载的另一份配置。

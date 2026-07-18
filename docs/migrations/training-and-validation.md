# 训练与数值验证

本文整合历史迁移研究中与训练流程有关的可复用经验。所有框架 API 映射都应视为待验证假设，直到完成固定输入的单步或端到端对比。

## 迁移顺序

1. 固定配置、环境、权重、输入和预处理。
2. 验证 dataset 单样本和 transform 输出。
3. 验证 collate 后的 batch 字段、shape、dtype 和 padding。
4. 逐模块对比 backbone → neck → transformer → head。
5. 比较 loss 分项、梯度和一次优化器更新。
6. 验证调度器、EMA、AMP 和 checkpoint 恢复。
7. 最后才做完整 epoch、COCO AP 和性能对比。

## 优化器

Paddle 与 PyTorch 都有 AdamW，但同名 API 不自动保证等价。需要核对：

- weight decay 是解耦衰减还是加入梯度的 L2 正则。
- 哪些参数组被排除 decay（bias、BatchNorm、特定名称匹配）。
- beta、epsilon、gradient clipping 的默认值和执行顺序。
- `zero_grad(set_to_none=...)` 与 Paddle 清梯度语义。

最小验收是在同一初始参数和手工设置的相同梯度上执行一步，比较每个参数组的更新值。

## 学习率调度器

历史方案使用 `SequentialLR(LinearLR, CosineAnnealingLR)` 表达 Paddle 的 warmup + cosine。实际对齐的关键不是类名，而是：

- warmup 和 cosine 是按 iteration 还是 epoch 计步。
- `scheduler.step()` 在 `optimizer.step()` 之前还是之后。
- milestone/T_max 是全局 step 还是排除 warmup 后的 step。
- 从 checkpoint 恢复时 `last_epoch`/当前 iteration 是否连续。

应导出整条 LR 曲线并逐 step 对比，不只检查起点与终点。

## 数据增强与 DataLoader

- 优先保留 Paddle 的 bbox/标签处理逻辑，只替换底层图像或 tensor 操作。
- 不同框架的随机数生成器即使使用相同 seed 也不保证同序列。对照测试宜把随机参数预先生成为共享 NumPy 输入，或直接注入确定参数。
- 必须显式验证 RGB/BGR、HWC/CHW、xyxy/xywh、归一化、插值、padding 和空 bbox。
- 目标检测样本的 bbox 数可变，需要自定义 `collate_fn`；不要依赖 PyTorch 默认 collate。
- Paddle `use_shared_memory` 与 PyTorch `pin_memory` 不是完全相同的机制。兼容参数可以映射为性能提示，但不能在文档中宣称语义等价。
- 多 worker 验证需要记录 `worker_init_fn`、sampler epoch、shuffle 和 drop-last 设置。

## 已验证的 PyTorch 迁移陷阱

以下结论在 2026-07-18 的 R18 最小训练链中已通过 PyTorch 单元或集成测试；它们仍不是 Paddle/PyTorch 数值等价证据。

- Paddle 中按索引取行的逻辑在 PyTorch 中应根据语义使用 `index_select` 或高级索引；不能把 Paddle 的 `gather` 调用形式直接套给 `torch.gather`。批内 DETR 匹配要先逐样本选取，再拼接。
- `Tensor.split(n)` 在 PyTorch 中的 `n` 是每块大小，不是“分成 n 块”。bbox 的 `x1/y1/x2/y2` 若需四个单元分量，应使用 `split(1, dim=-1)` 或 `unbind` 并检查输出数量。
- VFL/focal loss 的动态权重依赖模型预测时，应先计算 `reduction="none"` 的 BCE，再与动态权重相乘。把可导权重直接传给 PyTorch BCE 的 `weight=` 参数会触发不支持的权重求导路径。
- PyTorch `cross_entropy` 默认把第 2 维当作类别轴。DFL 输入如果是 `[N, 4, C]`，应先展平为 `[N*4, C]`，不能直接沿用 Paddle 支持的任意 `axis` 心智模型。
- `one_hot` 返回整型 tensor；与 bbox/score 浮点权重相乘前要转为目标浮点 dtype。GT class 在进入 `one_hot`/`scatter` 前则必须是 `int64`/`long`。
- 总 loss 的累加初值应从现有 loss tensor 创建标量零值，以保留 device/dtype 并避免把标量 loss 意外扩展为 shape `[1]`。

**已验证的 M1 边界**：CPU/float32、seed `2026`、两张合成 COCO 图像、固定 `96×96`、batch size 2、R18 缩减 query/decoder 配置。完整 loss 键、有限梯度、裁剪、一次参数更新和 5 step 训练均通过。未验证空 GT、AMP、EMA、DDP、checkpoint 恢复或 Paddle 对齐。

## BatchNorm、AMP 与分布式训练

- 通常在包装 DDP **之前** 转换 SyncBatchNorm。
- 先在单进程 CPU/float32 建立基线，再逐个开启 CUDA、TF32、AMP、SyncBN 和 DDP。
- AMP 需要记录 autocast dtype、GradScaler 状态、溢出跳步和梯度裁剪是在 unscale 前还是之后。
- DDP 需要检查 sampler 分片、loss reduce 方式、unused parameters、梯度累积时 `no_sync()` 和 rank-0 checkpoint 写入。

## Checkpoint 恢复

一个可恢复训练的 PyTorch checkpoint 至少应明确处理：

- model 与 EMA model 权重。
- optimizer 与 scheduler 状态。
- AMP scaler。
- epoch、global step 和 sampler epoch。
- Python、NumPy、PyTorch CPU/CUDA RNG 状态。
- 训练配置版本和 Git 提交。

验收时应比较“不中断连续训练”与“保存后恢复”在下一步的 LR、loss 和参数更新，而不只检查能否读取文件。

## 容差与证据

- FP32 参数复制可从 `atol=1e-6` 开始，模块激活可从 `atol=1e-5, rtol=1e-4` 开始，但必须根据算子和精度调整。
- 最终 boxes/scores 容差必须与预处理、图像尺寸和后处理关联；不使用无单位的单一数字。
- COCO 精度目标用 AP 点表达，发布门槛暂定为与 Paddle 基线的绝对差不超过 `0.5 AP`，并记录 AP50/AP75/APs/APm/APl。
- 所有数值报告必须附环境、权重 checksum、数据集版本、seed、dtype、命令和容差。

## 当前缺口

当前已有合成 COCO batch 的 PyTorch 前向/反向基线，但尚未使用真实 COCO 数据、官方 checkpoint 或 Paddle 进程做激活、loss、梯度、优化器更新和 AP 对齐。这些是 [`ROADMAP.md`](../../ROADMAP.md) M2–M4 的主要缺口。

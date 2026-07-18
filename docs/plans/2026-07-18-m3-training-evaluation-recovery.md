# M3——训练、评估与恢复验收计划

- 状态：`completed`
- 创建日期：`2026-07-18`
- 最后更新：`2026-07-18`
- 负责人：`maintainer`
- 对应路线图：[`ROADMAP.md` Milestone 3](../../ROADMAP.md)
- 前置计划：[`M2——官方 checkpoint 转换与分层数值对齐`](2026-07-18-m2-official-checkpoint-alignment.md)

## 背景

M1 已打通合成数据最小训练链，M2 已完成三个官方 checkpoint 的转换与受控数值对齐。M3 要回答的是训练库是否能在真实配置下稳定训练、评估和恢复，而不是继续追求 AdamW 更新逐元素一致。

### 当前基线

- **已验证**：隐藏 GPU 的默认测试 `165 passed, 8 skipped`，其中 5 项因 CUDA 不可见而跳过、3 项因未设置官方 Paddle checkpoint 环境变量而跳过；5 项 CUDA 用例所在的定向文件另行 `8 passed`。R18/R34/R50 官方数值回归在 M2 分别通过。
- **已验证**：受控场景的 loss 与整体梯度方向接近。
- **已验证**：ResNet stage 的 `lr_mult_list` 已转为 PyTorch optimizer 参数组，并有实际参数更新倍率测试。
- **已验证**：schema v1 checkpoint 可恢复下一步 LR/loss/参数、EMA/scaler 与 RNG 状态。
- **观察到**：本机有 2 张 NVIDIA GeForce RTX 3090（各 24 GiB），驱动 `595.71.05`，Python `3.12.11`、PyTorch `2.5.1+cu121`、CUDA runtime `12.1`、cuDNN `9.1.3`。
- **已验证**：本机 COCO 2017 包含 train2017 `118287` 图/`860001` 标注和 val2017 `5000` 图/`36781` 标注；annotation SHA-256 分别为 `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d` 和 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。
- **已验证**：真实 COCO 短训练已在单卡 FP32、单卡 AMP 和 2-GPU DDP AMP 下产生有效 optimizer 更新及 schema v1 checkpoint。
- **已验证**：安装后的 Eval CLI 已用官方 R18 转换权重和 4 张真实 val2017 图像跑通 EvalReader、模型后处理和 COCO API；小样本指标不作为 val2017 AP 结论。
- **已验证**：R18 的双卡 AMP train2017 完整 1 epoch 与同一 checkpoint 的 val2017 已完成；本次结果只证明 M3 可运行性，不作为 M4 精度对齐结论。
- **已验证**：`accumulate_steps=2` 的真实双卡 AMP+EMA 短训练完成，非边界微批使用 DDP `no_sync()`，optimizer/scheduler/global-step/EMA 仅在累积边界推进。

## 目标与非目标

### 目标

- 明确 optimizer 参数分组、学习率倍率、weight decay 和梯度裁剪契约，并用参数级测试锁定。
- 导出并核对 warmup/decay 的逐 step LR 轨迹与调用顺序。
- 使用 R18 和真实 COCO 完成至少 1 epoch 训练及 val2017 评估，记录环境、命令、数据版本和指标。
- 使 checkpoint 明确保存并恢复 model、EMA、optimizer、scheduler、scaler、epoch/global-step 和 RNG 状态。
- 比较连续训练与中断恢复后的下一步 LR、loss 和参数更新。
- 在具备至少 2 张 GPU 的环境中验收 DDP sampler、SyncBN、loss reduce、unused parameters 和 rank-0 写入。

### 非目标

- 不要求 Paddle/PyTorch AdamW 更新逐元素完全一致。
- 不在 M3 声称最终 COCO AP 达到发布门槛；多 seed 精度结论属于 M4。
- 不为缺失的真实数据或 GPU 伪造验收结果；无资源项必须保持 planned 并记录前置条件。

## 实施步骤

### 阶段 1：优化器与 LR 语义

- [x] 审计 Paddle optimizer 配置、当前 `OptimizerBuilder`、Trainer 调用顺序和可训练参数集合。
- [x] 为 ResNet `lr_mult_list` 建立显式 PyTorch param groups，保证每个可训练参数恰好出现一次。
- [x] 锁定 bias/BatchNorm/no-decay 规则、梯度裁剪位置和 EMA 更新顺序。
- [x] 生成 warmup + decay 的逐 step LR 轨迹，并验证保存/恢复后的连续性。

### 阶段 2：单机训练与评估

- [x] 核验本机 COCO train2017/val2017 路径、annotation checksum/样本数和磁盘前置条件。
- [x] 用 R18 官方转换权重完成 1 epoch 训练，记录有限 loss、LR、吞吐和 checkpoint。
- [x] 使用同一 checkpoint 完成 val2017 评估，记录 AP/AP50/AP75/APs/APm/APl；本阶段只验证可运行性，不设 M4 精度门槛。
- [x] 分别验证 float32 与 AMP 的有限 loss/梯度、scaler 溢出和跳步记录。

### 阶段 3：恢复一致性

- [x] 定义完整训练 checkpoint schema 和向后兼容边界。
- [x] 固定 seed 与数据顺序，对比连续训练和保存后恢复的紧接一步。
- [x] 验证 model/EMA、optimizer、scheduler、scaler、epoch/global-step、sampler 和 RNG 状态。

### 阶段 4：DDP

- [x] 在 2 张 GPU 上运行短训练，验证 DistributedSampler、SyncBN、同步梯度/scaler、跨 rank 日志 loss 均值和 rank-0 checkpoint。
- [x] 验证梯度累积的 `no_sync()` 边界以及只有 rank 0 写应用日志/checkpoint。
- [x] 记录硬件、驱动、CUDA/cuDNN、world size、batch 和启动命令。

## 风险与回退

- 风险：完整 COCO epoch 耗时且可能在随机 800 尺寸 batch 达到显存峰值。缓解：先用相同数据增强完成 batch 2/8 容量烟测，正式任务保留独立日志和 checkpoint。
- 风险：修改 optimizer 分组改变 M1 行为。缓解：先写参数归属与实际更新倍率测试，再接入 Trainer；保留原全局 LR 作为明确回退。
- 风险：调度器以 epoch/iteration 计步混淆。缓解：以全局 step 表格验收，不以类名或最终 LR 猜测。
- 回退：优化器、checkpoint schema、AMP 和 DDP 分成独立变更；任一阶段失败可回退实现而保留测试与差异报告。

## 验收

- [x] 所有可训练参数恰好属于一个 optimizer group，ResNet stage 的实际 LR 倍率有直接更新证据。
- [x] LR 轨迹、裁剪和 EMA 顺序有确定性回归。
- [x] R18 真实 COCO 1 epoch 与 val2017 命令成功，环境和指标完整记录。
- [x] 连续训练与恢复训练的下一步达到计划中明确的 LR/loss/参数容差。
- [x] AMP 与 2-GPU DDP 短训练结果有真实 COCO 证据；梯度累积和跨 rank 日志 loss 均值均按阶段 4 的独立项完成。
- [x] 默认核心测试保持通过，测试与训练产生的临时产物按仓库规则清理。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-18 | 先验收参数组和 LR 轨迹，再启动真实 epoch | 当前已知最大优化器缺口是 ResNet `lr_mult_list=0.1`，先消除配置语义不确定性 |
| 2026-07-18 | 不要求 AdamW 逐元素一致 | 训练验收以参数组、梯度方向、收敛和指标为准 |
| 2026-07-18 | 真实数据/GPU 缺失时不降级结论 | 合成烟雾测试不能替代 COCO 或 DDP 证据 |
| 2026-07-18 | 保留当前 PyTorch schedule，只修复可执行语义 | 用户不要求训练优化逐项完全对齐；基础 LR/warmup/decay 差异在迁移报告显式保留 |
| 2026-07-18 | COCO 缺失时转入可独立验证的 AMP/DDP 前置工作 | 本机具备 2×RTX 3090，但不使用合成数据替代真实 COCO epoch/AP 结论 |
| 2026-07-18 | 对 GradScaler 溢出跳步不立即中止训练 | 跳步时同步不推进 scheduler/global-step/EMA，使状态与真实 optimizer 更新保持一致 |
| 2026-07-18 | 完整 epoch 使用 2-GPU AMP、每卡 batch 8 | 短训练观察到双卡合计约 24.6 images/s，每卡峰值约 21.3 GiB |
| 2026-07-18 | 梯度累积按实际窗口平均 loss | 最后不足 `accumulate_steps` 的窗口不能按固定分母缩小梯度；scheduler/global-step/EMA 只按成功 optimizer update 计数 |
| 2026-07-18 | 保留长跑的空原始配置并另存重建快照 | 原 checkpoint 的空 config 是历史缺陷；覆盖它会混淆原始证据，修复后的新运行已直接验证非空配置 |

## 实测命令与证据

完整训练使用 Python `3.12.11`、PyTorch `2.5.1+cu121`、CUDA runtime `12.1`、cuDNN `9.1.3`、driver `595.71.05` 和 2×RTX 3090。COCO 路径通过环境变量提供，文档不固化工作站路径：

```bash
COCO_ROOT=/path/to/coco2017
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 \
  .venv/bin/rtdetrv3-train --ddp --amp --enable_ce True \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  -o TrainDataset.dataset_dir="$COCO_ROOT" TrainReader.batch_size=8 \
     worker_num=4 epoch=1 log_iter=50 use_ema=False \
     pretrain_weights=pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth \
     save_dir=output/m3-r18-coco-one-epoch
```

- **观察到**：2026-07-18 18:35:36 至 19:49:15 完成 `7329` 个数据 batch；日志中的单 rank 吞吐约 `10.7–15.3 images/s`，峰值 reserved/allocated 显存为 `21346/16952 MiB`。
- **已验证**：`model_final.pth` 为 `276140051` 字节，SHA-256 为 `8ace1a5a6461427a1ab145a5d07263d082a70a39c0302cc00a93521834cee3e5`；记录 epoch `1`、global-step/scheduler step `7319`、GradScaler `512`、sampler epoch `1` 和最终 loss `11.35697078704834`。`7329 - 7319 = 10` 次 AMP 跳步与日志一致；648 个 model tensor 全部有限，optimizer 有 420 个参数状态。
- **限制**：该长跑发生在配置序列化修复前，原 `config.yaml` 和 checkpoint `config` 为空。`output/m3-r18-coco-one-epoch/config.reconstructed.yaml` 依据上面的 config、CLI flags 和 overrides 重建，SHA-256 为 `91c70aacf86331a72eeadae459b4f7ee5bc1d6b224973e880cda61869da99450`；它是重建证据，不冒充原始快照。修复后的真实 DDP 运行已直接生成 90 项配置和 5868 字节 YAML。

同一 checkpoint 的完整 val2017 命令为：

```bash
COCO_ROOT=/path/to/coco2017
CUDA_VISIBLE_DEVICES=0 .venv/bin/rtdetrv3-eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint output/m3-r18-coco-one-epoch/model_final.pth \
  --anno_file "$COCO_ROOT/annotations/instances_val2017.json" \
  --image_dir "$COCO_ROOT/val2017" \
  --batch_size 16 --num_workers 4 --device cuda
```

- **已验证**：5000 图、313 batch 的推理和 COCO API 汇总以退出码 0 完成。bbox 指标为 AP `0.468`、AP50 `0.643`、AP75 `0.504`、APs `0.302`、APm `0.501`、APl `0.624`；AR1/AR10/AR100 为 `0.364/0.616/0.686`，ARs/ARm/ARl 为 `0.494/0.729/0.852`。
- **限制**：这是“官方转换权重再训练 1 epoch”的单次 M3 可运行性结果，没有相同训练 schedule、seed 集合和 Paddle 对照，不能声称训练精度已对齐。

真实梯度累积验收使用 256 图临时 annotation、world size 2、每卡 batch 8、AMP、EMA 和 `accumulate_steps=2`。每 rank 得到 15 个微批，scheduler 正确构建为 8 step；5 次 AMP 跳步后 checkpoint 的 global-step、scheduler 和 EMA step 均为 `3`，scaler 为 `2048`，447 个浮点 tensor 的 EMA 值与即时 model 不同。应用 logger 的每条 warning、完成消息和 checkpoint 消息只出现一次；PyTorch 自身的 DDP warning 仍可能带 rank 前缀。

补齐 loss reduce 后又用 64 图临时 annotation、world size 2、每卡 batch 4、FP32、EMA 和 `accumulate_steps=2` 运行 7 个微批。状态 loss 在不参与反向的 detached tensor 上做 world-size mean，真实 NCCL collective 正常结束；checkpoint 的 model/scheduler/EMA step 均为 `4`、配置 90 项，完成和保存日志各一次。单元测试用本地 loss `1`、远端 loss `3` 直接验收上报均值为 `2` 且不携带梯度。两次实测的临时数据与 checkpoint 均已清理。

## 完成记录

M3 四个阶段的功能与实测项均已完成：optimizer/LR、完整 train2017 1 epoch、val2017、schema v1 恢复、AMP/EMA、2-GPU DDP、跨 rank 日志 loss 均值、梯度累积 `no_sync()` 和 rank-0 应用日志/checkpoint 都有直接证据。最终回归为隐藏 GPU 的默认测试 `165 passed, 8 skipped` 和 CUDA 定向文件 `8 passed`；本阶段不声称 M4 的标准 schedule、多 seed 或 Paddle AP 门槛已经完成。

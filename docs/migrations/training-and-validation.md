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

当前 RT-DETRv3 配置使用 `regularizer: false` 和 AdamW `weight_decay: 1e-4`，没有配置 bias/BatchNorm 排除规则。已验证自动生成的基础组和 stage LR 倍率组都继承该 decay；这一结论不应泛化到配置了显式 `param_groups` 的其他模型。训练单步顺序已用回归锁定为 gradient clip → optimizer step → scheduler step → EMA update；AMP 路径会在 clip 前先 unscale。

## 学习率调度器

历史方案使用 PyTorch scheduler 组合表达 Paddle 的 warmup + decay。实际对齐的关键不是类名，而是：

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

### 可复现随机性边界

- 训练入口的 seed 应是**全局 base seed**。当前 PyTorch 实现用 `base + rank` 初始化每个进程的 Python、NumPy、PyTorch CPU/CUDA RNG，同时把共同的 base seed 交给 `DistributedSampler`；不能把 rank-specific seed 直接传给 sampler，否则各 rank 会生成不同全局排列而破坏分片关系。
- 单卡 `RandomSampler` 需要独立 `torch.Generator`，并在每个 epoch 用 `base + epoch` 重置。分布式 sampler 则调用 `set_epoch(epoch)`，由 PyTorch 使用共同的 base seed 和 epoch 生成排列。
- DataLoader worker generator 当前使用 `base + epoch * world_size + rank`。PyTorch worker 启动逻辑据此初始化 worker 内的 Python、NumPy 和 PyTorch RNG；定向测试验证同 seed/epoch 可重放、不同 epoch 会变化。
- checkpoint 同时记录 RNG state 与下一 sampler epoch。DDP 中每个 rank 的 process RNG 不同，因此周期 checkpoint 必须集合 `rng_state_by_rank`，恢复时按当前 rank 选取；把 rank 0 RNG 同时恢复到所有 rank 会破坏随机性边界。当前验收边界是相同 world size 的 epoch checkpoint 恢复；不能把它外推为任意 mid-epoch 数据游标恢复。
- 相同数值 seed 不保证 Paddle 与 PyTorch 产生同一随机增强序列。跨框架逐样本对齐仍应注入共享随机参数或固定增强，而不是只传相同 seed。

**已验证（2026-07-18）**：19 个 seed/sampler/worker/checkpoint/训练链定向测试通过；真实 2-GPU、32 图训练以 base seed 17 完成 8 次更新，checkpoint 记录 `seed=17`、`sampler_epoch=1` 与完整有效配置。修复 rank 0 RNG 复用问题后，另一次真实 2-GPU checkpoint 烟测写入 2 份不同的 rank RNG，SHA-256 分别为 `d7b3b2a9dc4bf089cee4d62c8d7e2d19e76f7f0cad58cc79d1773b5751c2c58e` 和 `0bd2daed27d16a78e0af552aa3968970fccd5625ada5b1236fd06744432502a3`。该烟测的唯一 AMP update 被初始 loss scale 跳过，所以它只验证分布式 checkpoint collective/RNG，不作优化器更新证据。这些结果只证明 PyTorch 内部可复现协议，不是多 seed AP 稳定性结论。

## 已验证的 PyTorch 迁移陷阱

以下结论在 2026-07-18 的 R18 最小训练链中已通过 PyTorch 单元或集成测试；它们仍不是 Paddle/PyTorch 数值等价证据。

- Paddle 中按索引取行的逻辑在 PyTorch 中应根据语义使用 `index_select` 或高级索引；不能把 Paddle 的 `gather` 调用形式直接套给 `torch.gather`。批内 DETR 匹配要先逐样本选取，再拼接。
- `Tensor.split(n)` 在 PyTorch 中的 `n` 是每块大小，不是“分成 n 块”。bbox 的 `x1/y1/x2/y2` 若需四个单元分量，应使用 `split(1, dim=-1)` 或 `unbind` 并检查输出数量。
- VFL/focal loss 的动态权重依赖模型预测时，应先计算 `reduction="none"` 的 BCE，再与动态权重相乘。把可导权重直接传给 PyTorch BCE 的 `weight=` 参数会触发不支持的权重求导路径。
- PyTorch `cross_entropy` 默认把第 2 维当作类别轴。DFL 输入如果是 `[N, 4, C]`，应先展平为 `[N*4, C]`，不能直接沿用 Paddle 支持的任意 `axis` 心智模型。
- `one_hot` 返回整型 tensor；与 bbox/score 浮点权重相乘前要转为目标浮点 dtype。GT class 在进入 `one_hot`/`scatter` 前则必须是 `int64`/`long`。
- 总 loss 的累加初值应从现有 loss tensor 创建标量零值，以保留 device/dtype 并避免把标量 loss 意外扩展为 shape `[1]`。
- 在 CUDA 上训练时，matcher 返回的索引、动态 anchor/stride、assigner 的 batch index 和空标注输出都必须跟随被索引 tensor 或 feature 的 device。仅在最终算子前临时 `.to(device)` 会遗漏其他路径；应在 `arange/full/zeros/tensor` 的创建点指定 device。
- PPYOLOE 辅助头输出 sigmoid probability 后再计算 BCE。PyTorch 禁止 `binary_cross_entropy` 在 CUDA autocast 区域中执行；在不改变 probability 接口的前提下，应将该 BCE 局部切换为 FP32，不能直接替换成接收 logits 的损失而不同步修改上游。

**已验证的 M1 边界**：CPU/float32、seed `2026`、两张合成 COCO 图像、固定 `96×96`、batch size 2、R18 缩减 query/decoder 配置。完整 loss 键、有限梯度、裁剪、一次参数更新和 5 step 训练均通过。未验证空 GT、AMP、EMA、DDP、checkpoint 恢复或 Paddle 对齐。

## 官方权重训练态对齐证据（2026-07-18）

- **已验证**：使用同一官方 R18 checkpoint、CPU/float32、PyTorch 单线程、seed `2026` 和两条固定 GT，将 eval transformer 的四个输出作为叶子 tensor 输入训练 head。12 个 loss 分项和总 loss 通过 `rtol=1e-4, atol=1e-5`，四个输出 tensor 的梯度也通过同一容差。这把 head/loss 的反向语义与 denoising RNG、backbone 和 transformer 隔离开来。
- **已验证**：在 `96×96` 确定性输入上，把两侧 transformer 都缩减为两组、每组 20 个 query、4 个 denoising query，并关闭 label/box noise。修复多分组 attention mask 后，完整训练前向产生的 30 个 loss 分项全部通过上述容差；总 loss 为 Paddle `68.8954163`、PyTorch `68.8954239`，绝对差 `7.63e-6`。
- **已验证**：PyTorch BatchNorm 收到 NCHW → NLC 重排产生的非连续 grad-output 时，input-gradient 会偏离相同数值的 contiguous 参考。R18 的 backbone、neck 和 transformer 投影统一使用 layout-stable BatchNorm 后，384 个可训练参数的整体梯度相对 L2 误差为 `0.00434`，余弦相似度为 `0.999991`，有效元素符号分歧率为 `0.1304%`；可选 numerical 用例固定验收 `relative_l2 < 0.01`、`cosine > 0.9999`、符号分歧率 `< 0.5%`。
- **已验证**：同一流程扩展到 R34/R50 后，R34 的 462 个参数和 R50 的 445 个可训练参数均通过上述整体梯度门槛。R50 配置的 `freeze_norm=True` 还要求冻结 BN 在 `model.train()` 后继续使用 running/global statistics；仅冻结 `Parameter.requires_grad` 不等价。
- **已验证**：ResNet `ConvNormLayer` 现在保留 `lr_mult_list` 元数据，`OptimizerBuilder` 在没有显式 `param_groups` 时按倍率生成参数组。单测确认所有可训练参数恰好出现一次，且在相同梯度的 SGD 探针中 `0.1×` stage 的实际更新量是基础组的十分之一。
- **已验证**：warmup 期间不改变参数组 LR 比例；piecewise milestone 按全局 step 解释，不再被 `SequentialLR` 额外后移一段 warmup 长度。
- **观察到**：当前 PyTorch `optimizer_6x.yml` 使用 `base_lr=1e-4`、warmup 1,000 step、epoch 60 衰减为 `0.1×`；Paddle 参考配置是 `base_lr=4e-4`、warmup 2,000 step、milestone 100 与 `gamma=1.0`。这是已知训练策略差异，本阶段保留 PyTorch 策略，不声称两者完全对齐。
- **决策**：迁移不以 AdamW 每个元素逐位一致为目标。M2 验收梯度整体方向和误差；参数组、LR multiplier、weight decay、裁剪/EMA 顺序、训练收敛和 AP 由 M3/M4 验收。
- **限制**：上述训练证据使用确定性缩减 query/noise 配置和合成输入，不代表官方完整训练配置、随机 denoising、真实 COCO 或训练收敛已经对齐。

## BatchNorm、AMP 与分布式训练

- 通常在包装 DDP **之前** 转换 SyncBatchNorm。
- 先在单进程 CPU/float32 建立基线，再逐个开启 CUDA、TF32、AMP、SyncBN 和 DDP。
- AMP 需要记录 autocast dtype、GradScaler 状态、溢出跳步和梯度裁剪是在 unscale 前还是之后。
- DDP 需要检查 sampler 分片、loss reduce 方式、unused parameters、梯度累积时 `no_sync()` 和 rank-0 checkpoint 写入。

梯度累积应按每个窗口的实际微批数平均 loss；最后一个不足配置步数的窗口不能继续除以固定 `accumulate_steps`。DDP 的非边界微批需要把 forward 和 backward 都放在 `no_sync()` 中，边界微批才触发梯度同步。optimizer、scheduler、global-step 和 EMA 都只按成功的边界更新推进；学习率调度器的 `steps_per_epoch` 也应使用微批数除以累积步数后的向上取整值。

GradScaler 检测到溢出时，`scaler.step()` 会跳过 optimizer 更新并降低 scale。这是动态缩放的正常探测机制，不应仅因 gradient norm 为 Inf 立即中止。但跳步时也不应推进 scheduler、global step 或 EMA；否则 checkpoint 记录的训练状态会领先于真实参数更新。

## 真实 COCO 短训练证据（2026-07-18）

- **环境**：Python `3.12.11`、PyTorch `2.5.1+cu121`、CUDA runtime `12.1`、cuDNN `9.1.3`、driver `595.71.05`、2×RTX 3090 24 GiB。
- **数据**：COCO train2017 `118287` 图/`860001` 标注，val2017 `5000` 图/`36781` 标注。annotation SHA-256 为 `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d` 和 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。
- **FP32 已验证**：R18 官方转换权重、单卡、batch 2、8 图子集的 4 个 batch 完成前向、反向和 checkpoint；首个记录 loss `27.2693`，峰值 reserved memory `5130 MiB`。
- **AMP 已验证**：单卡 batch 8 的 64 图子集完成 7 个数据 batch，checkpoint 记录 `global_step=3`、scaler `scale=4096`；峰值 reserved memory `14894 MiB`。初始高 scale 导致的跳步被显式记录，不被冒充为 optimizer update。
- **DDP AMP 已验证**：world size 2、每卡 batch 8 的 128 图子集完成 7 个 batch，checkpoint 记录 `global_step=2`、scaler `scale=2048`。已观察 DistributedSampler 分片、SyncBN 转换、两 rank 一致的溢出跳步、rank-0 checkpoint 及正常 NCCL 退出。
- **DDP 累积与 EMA 已验证**：256 图临时子集、world size 2、每卡 batch 8、AMP、EMA、`accumulate_steps=2`。每 rank 的 15 个微批被折算为 8 个 scheduler 窗口；5 次 AMP 跳步后 model/scheduler/EMA step 均为 `3`，scaler 为 `2048`，447 个浮点 tensor 的 EMA 与即时 model 不同。单元测试直接锁定 `no_sync()` 只包住非边界微批，并验证最后单微批窗口的梯度归一化。
- **DDP loss reduce 已验证**：训练反向继续使用各 rank 本地 loss，只对 detached 状态值做 world-size mean。64 图临时子集的真实双卡 FP32+EMA 累积训练完成 7 个微批和 4 次更新，checkpoint 的 model/scheduler/EMA step 均为 `4`；单元测试直接验收本地 `1`、远端 `3` 上报为 `2` 且不带梯度。
- **Eval CLI 已验证**：官方 R18 转换权重和 4 张 val2017 子集在 CPU 上完成 config 加载、EvalReader、模型后处理、COCO JSON 转换和 COCO API 统计。子集 AP `0.448` 只证明链路可执行，因样本仅 4 张，不能与官方 val2017 基线比较。
- **完整 epoch 已验证**：同一 R18/world-size-2/每卡 batch-8 配置完成 train2017 1 epoch。7329 个数据 batch 中 7319 次 optimizer 更新成功、10 次被 GradScaler 跳过；checkpoint SHA-256 为 `8ace1a5a6461427a1ab145a5d07263d082a70a39c0302cc00a93521834cee3e5`，648 个 model tensor 全部有限。
- **完整 val2017 已验证**：同一 checkpoint 在 5000 图/313 batch 上退出码为 0，bbox AP/AP50/AP75/APs/APm/APl 为 `0.468/0.643/0.504/0.302/0.501/0.624`。该单次结果没有 Paddle 同 schedule 对照，只是 M3 可运行性证据。
- **M4 初始化协议已验证**：官方 R18 正式训练依赖 ImageNet `ResNet18_vd_pretrained`，不能沿用 M3 的完整检测 checkpoint。转换后的 115 个 backbone tensor 完整加载；真实 2-GPU AMP+EMA 烟测完成 1 次有效更新并成功用 Eval CLI 的 `--use-ema` 读取保存的 EMA 权重。小样本 AP 不作精度证据。
- **M4 同权重推理 gate 已验证**：同一官方 R18 checkpoint、val2017、CPU/FP32、batch 16 条件下，Paddle/PyTorch 独立复算 AP 分别为 `0.480477300367/0.480477134768`，绝对差 `1.65599e-7`；score `>=0.3` 的 `53780` 个 prediction 在同类、坐标 L∞ `<=1px` 下全部匹配。对不同设备或框架的 prediction 不应只按 score 排名强行对齐；临界 score 交换会被误认为大坐标偏差，应按同图、同类、score 阈值和明确像素容差匹配。
- **checkpoint 频率应进入容量预算**：启用 EMA 的 R18 checkpoint 实测约 368 MB。Trainer 现已把 `snapshot_epoch` 传给 Checkpointer，并保证 final epoch 无论间隔都保存；M4 三 seed 使用间隔 3，在恢复窗口与磁盘占用之间做显式权衡。

可复现命令使用环境变量，避免在文档中固化工作站路径：

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

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/rtdetrv3-eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint output/m3-r18-coco-one-epoch/model_final.pth \
  --anno_file "$COCO_ROOT/annotations/instances_val2017.json" \
  --image_dir "$COCO_ROOT/val2017" \
  --batch_size 16 --num_workers 4 --device cuda
```

完整长跑发生在配置序列化修复前，原 `config.yaml` 与 checkpoint `config` 为空；仓库保留明确标注的 `config.reconstructed.yaml`，不能把它称为原始快照。根因是配置对象为 `dict` 子类但同时具有空 `__dict__`，旧转换顺序误选了后者。当前转换器优先处理 `Mapping`，并已由真实 DDP 运行验证 90 项 checkpoint config 和 5868 字节 YAML。

## Checkpoint 恢复

一个可恢复训练的 PyTorch checkpoint 至少应明确处理：

- model 与 EMA model 权重。
- optimizer 与 scheduler 状态。
- AMP scaler。
- epoch、global step 和 sampler epoch。
- Python、NumPy、PyTorch CPU/CUDA RNG 状态。
- 训练配置版本和 Git 提交。

验收时应比较“不中断连续训练”与“保存后恢复”在下一步的 LR、loss 和参数更新，而不只检查能否读取文件。

**已验证（2026-07-18）**：训练 checkpoint schema v1 使用统一的 `model`/`ema`/`optimizer`/`scheduler`/`scaler`、`epoch`/`global_step`/`sampler_epoch`、单进程 `rng_state` 和可选 `rng_state_by_rank` 字段，并通过同目录临时文件原子发布。加载器兼容早期回调使用的 `model_state_dict`/`optimizer_state_dict`/`scheduler_state_dict` 别名，也对旧的单 RNG checkpoint 保持回退。固定数据的中断恢复测试已确认下一步 LR、loss 和参数与连续训练一致，同时恢复 EMA step/decay、GradScaler 和 Python/NumPy/PyTorch RNG；分布式单测与真实双卡烟测另行验证按 rank 恢复和集合写入。

由于完整训练状态包含 Python/NumPy RNG 和配置对象，该路径使用 pickle 能力加载；只能对可信的自有 checkpoint 使用恢复入口。不可信或只含模型权重的文件应走限定权重加载/转换路径。

## 容差与证据

- FP32 参数复制可从 `atol=1e-6` 开始，模块激活可从 `atol=1e-5, rtol=1e-4` 开始，但必须根据算子和精度调整。
- 最终 boxes/scores 容差必须与预处理、图像尺寸和后处理关联；不使用无单位的单一数字。
- COCO 精度目标用 AP 点表达，发布门槛暂定为与 Paddle 基线的绝对差不超过 `0.5 AP`，并记录 AP50/AP75/APs/APm/APl。
- 所有数值报告必须附环境、权重 checksum、数据集版本、seed、dtype、命令和容差。

## 当前缺口

官方 R18/R34/R50 已完成 checkpoint 参数转换、eval 分层激活、受控训练 loss、head/loss 输出梯度和完整模型整体梯度方向对齐。PyTorch 已表达 ResNet LR multiplier、锁定 piecewise 的全局 step 语义，并验证自有 checkpoint 恢复、真实 COCO 完整 epoch/val2017、AMP、EMA、2-GPU DDP 和梯度累积。标准 schedule、多 seed、R18/R34/R50 的 Paddle AP 对照和性能基准仍是 [`ROADMAP.md`](../../ROADMAP.md) M4 及后续里程碑的主要缺口。

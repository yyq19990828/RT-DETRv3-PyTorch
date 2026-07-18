# M4——COCO 精度与稳定性对齐计划

- 状态：`deferred`
- 创建日期：`2026-07-18`
- 最后更新：`2026-07-19`
- 负责人：`maintainer`
- 对应路线图：[`ROADMAP.md` Milestone 4](../../ROADMAP.md)
- 前置计划：[`M3——训练、评估与恢复验收`](2026-07-18-m3-training-evaluation-recovery.md)

## 背景

M3 已证明 R18 能在真实 COCO 上完成完整 epoch、val2017、恢复、AMP/EMA 和 DDP，但“官方转换权重再训练 1 epoch”的 AP `0.468` 只属于可运行性证据。M4 要建立同权重推理精度、完整训练 schedule 和多 seed 稳定性证据，不能把类名、shape、短训或单次 AP 当作 Paddle/PyTorch 精度对齐。

## 暂缓说明（2026-07-19）

R18 官方同权重完整 val2017 gate 已通过，但完整 `3 seeds × 72 epoch` 预计占用约 11 天双卡时间，当前不继续执行。已经启动的 seed 0 运行已在首个 3-epoch checkpoint 原子落盘后停止，并完成 checkpoint 结构与可恢复状态审计；该结果属于训练/恢复探针，不填写最终 AP、均值或方差。R34/R50 长训同时暂缓，恢复本计划需要新的明确决策和时间预算。

### 当前基线

- **官方发布参考**：固定 Paddle 子模块 revision `349e7d99a5065e7b684118912e6a74178d4f4625` 的模型库记录 R18/R34/R50 val2017 AP 为 `48.1/49.9/53.4`，AP50 为 `66.2/67.7/71.7`。发布表是来源参考，不替代本机直接评估。
- **权重**：R18 Paddle checkpoint SHA-256 为 `f32dbd008bd7e5311c877d522f6d8c9e349795978c889f53823588b5e5d74a5f`；对应转换 PyTorch checkpoint SHA-256 为 `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547`。
- **训练初始化**：官方 R18 训练从 `ResNet18_vd_pretrained` 启动，不能用随机初始化或已训练好的 RT-DETRv3 检测权重代替。源 checkpoint SHA-256 为 `68d7632cb67ad2c658fe67ab5837d8eb65466a7bc1574badc74860059ef5e7f0`；目标感知转换得到 115 个完整 backbone tensor，0 unexpected、0 缺失 backbone key。
- **数据**：COCO train2017 `118287` 图/`860001` 标注，val2017 `5000` 图/`36781` 标注；annotation SHA-256 分别为 `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d` 和 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。
- **环境**：Python `3.12.11`、PyTorch `2.5.1+cu121`、CUDA runtime `12.1`、cuDNN `9.1.3`、driver `595.71.05`、2×RTX 3090 24 GiB。当前 dev extra 的 Paddle `3.3.0` 是 CPU 构建，Paddle GPU 评估不可直接使用。
- **训练策略差异**：两侧都是 72 epoch，但 Paddle 参考配置为 base LR `4e-4`、warmup 2000 step、milestone 100、gamma `1.0`；当前 PyTorch 配置为 base LR `1e-4`、warmup 1000 step、milestone 60、gamma `0.1`。项目已决定不要求 AdamW/训练优化逐元素完全对齐，这些差异必须进入报告而不能隐去。
- **成本估算**：M3 双卡实测一个 epoch 约 `1:13:39`。按线性外推，72 epoch 单 seed 约 `88.4` 小时，3 seed 约 `11.1` 天；这是计划估算，不是已完成基准。

## 目标与非目标

### 目标

- 使用同一官方 R18 checkpoint、相同 val2017、640×640 预处理、eval mode 和 FP32，分别得到 Paddle 与转换后 PyTorch 的完整 COCO 指标。
- 将“绝对差不超过 `0.5 AP`”明确解释为 COCO 百分点不超过 `0.5`，即日志中 `[0, 1]` 标度的差不超过 `0.005`；同时记录 AP50/AP75/APs/APm/APl。
- 若同权重完整 AP 超差，先比较预测 JSON 和第一个分歧的中间激活，再修改实现；不通过放宽最终指标掩盖差异。
- 固定 R18 的 72 epoch PyTorch 训练协议、seed、数据 checksum、命令、配置快照、日志和 checkpoint，完成至少一个 seed 后再扩展到 3 seed。
- R18 达标后按同样证据结构验证 R34 和 R50。
- 维护 [`docs/reports/accuracy-validation.md`](../reports/accuracy-validation.md)，明确区分官方发布参考、本机直接观察、推断、训练策略差异和未完成项。

### 非目标

- 不要求 Paddle/PyTorch AdamW 更新逐元素一致，也不为追求形式一致而无证据地改写当前 PyTorch 优化策略。
- 不在 R18 尚未定位同权重 AP 差异时并行启动 R34/R50 长训。
- 不把官方模型库表格、4 图子集、M3 单 epoch 或单 seed 结果冒充多 seed 稳定性结论。
- 不覆盖 R101、LVIS、性能基准、导出或部署；这些属于后续里程碑。

## 实施步骤

### 阶段 1：同权重完整 val2017 基线

- [x] 用官方转换 R18 PyTorch checkpoint 完成 val2017，记录完整指标、耗时、命令和日志：两次 CUDA 运行的 AP 为 `0.480/0.481`，保留预测独立复算为 `0.480502167075`；CPU 运行复算为 `0.480477134768`，耗时 `1233.57s`。CPU/CUDA 在 score `>=0.5` 时分别选出 `26243/26240` 个预测，同类且坐标 L∞ 差 `<=1px` 的匹配为 `26218`，匹配框坐标差中位数 `0.002892px`。CUDA/CPU prediction SHA-256 分别为 `7bfd1a4cf0e32561ef1d74d1aa39e617ceb1b2a6a0d76dea34f3dddce015f054` / `bb29e359cb521dacc32235f3eafe727d0693d8eb5d76d7fb5885ac9fb0b07b40`。
- [x] 用同一来源 R18 Paddle checkpoint 完成 CPU/FP32 val2017；5000 图总耗时 `6120.77s`，prediction SHA-256 为 `50ad6ceebb052f8ff7e826fa9109dc60e058404203338c29905b2fe3da0ce873`。
- [x] 比较本机双框架指标与官方发布 `48.1/66.2` 参考；Paddle/PyTorch CPU 精确 AP 分别为 `0.480477300367/0.480477134768`，绝对差 `1.65599e-7`。
- [x] AP 差未超过 `0.5` 百分点；score `>=0.3` 的 `53780/53780` 个 prediction 在同类、坐标 L∞ `<=1px` 下全部匹配，不触发激活差异定位回退。

### 阶段 2：可复现训练协议

- [x] 为 Train CLI 增加显式 seed，验证 Python/NumPy/PyTorch、DistributedSampler 和 DataLoader worker 的可复现边界；真实双卡烟测确认 base seed、配置快照、sampler epoch 和 checkpoint 状态能落盘，且周期 checkpoint 会集合不同的 per-rank RNG state。
- [x] 冻结 R18 72 epoch 的设备、world size、每卡 batch、AMP、EMA、LR、warmup、milestone、checkpoint/eval 频率和恢复命令，详见下表。
- [x] 在正式启动前做容量、首个有效 optimizer update、恢复和磁盘预算烟测；使用真正的 ImageNet backbone 预训练权重完成 2-GPU AMP+EMA 有效更新，并从保存的 EMA 权重跑通 Eval CLI。
- [x] 保存原始有效配置；真实双卡烟测的 checkpoint 已记录 90 项有效配置，包括 base seed、初始权重、schedule 和保存目录。

### 已冻结的 R18 PyTorch 训练协议

| 项目 | 固定值 |
|---|---|
| 初始化 | 转换后的官方 `ResNet18_vd_pretrained`，只填充 115 个 backbone tensor |
| 数据/输入 | COCO train2017，原训练增强，随机多尺度 `480–800` |
| 设备 | 单机 2×RTX 3090，DDP + SyncBatchNorm |
| batch | 每 rank 8，global batch 16，`accumulate_steps=1` |
| 精度/EMA | CUDA AMP；EMA `0.9999` exponential；最终用 `rtdetrv3-eval --use-ema`，即时模型只作诊断对照 |
| schedule | 72 epoch；base LR `1e-4`；warmup 1000 update；milestone 60；gamma `0.1` |
| seed | `0/1/2`，CLI 显式传入；rank process seed 为 `base + rank` |
| 保存/恢复 | `snapshot_epoch=3`，每 3 epoch 原子保存且 final 强制保存；恢复使用 `--resume`，保留 optimizer/scheduler/scaler/EMA/RNG/sampler epoch |
| 评估 | 每个 seed 的最终 EMA checkpoint 完整 val2017；三次完成后计算均值与标准差 |

该协议刻意保留当前 PyTorch schedule，不声称与 Paddle AdamW 更新逐元素一致。正式命令必须显式覆盖 `pretrain_weights=pretrained_models/pytorch/ResNet18_vd_pretrained.pth`，不能沿用 M3 为可运行性而使用的完整检测 checkpoint。

### 社区分片执行

完整实验暂缓本机执行，但仓库提供 [`scripts/run_stability_experiment.py`](../../scripts/run_stability_experiment.py)，便于在 GitHub Issue 中由不同贡献者各自认领一个 `model + seed`。脚本一次只启动一个任务，默认固定 2 GPU、每 rank batch 8、AMP、EMA、72 epoch、每 3 epoch保存，并在成功训练后自动评估最终 EMA checkpoint。

先按 [`weight-conversion.md`](../migrations/weight-conversion.md#r18-vd-backbone-预训练权重2026-07-18) 准备官方 ImageNet backbone 转换权重，然后验证命令；`--dry-run` 不创建文件或启动训练：

```bash
uv sync --extra dev
export COCO_ROOT=/path/to/coco2017
.venv/bin/python scripts/run_stability_experiment.py \
  --model r18 --seed 1 --coco-root "$COCO_ROOT" --dry-run
```

确认无误后移除 `--dry-run`。结果写入 `output/stability/r18-seed-1/`：

- `metadata.json`：commit/dirty 状态、环境、GPU、完整协议、命令和输入 checksum；预训练权重同时记录整个文件 SHA 与忽略转换元数据的有序 tensor SHA；
- `train.log` 与 `eval.log`：原始训练和评估日志；
- `result.json`：退出码、最终 checkpoint checksum 与六项 COCO AP；
- `model_final.pth` 和 `eval/`：checkpoint 与 COCO 预测产物，不建议直接作为 Issue 附件上传。

中断后应保持全部参数不变，并显式传入同一目录中的周期 checkpoint：

```bash
.venv/bin/python scripts/run_stability_experiment.py \
  --model r18 --seed 1 --coco-root "$COCO_ROOT" \
  --resume output/stability/r18-seed-1/epoch_3.pth
```

Issue 第一批只认领 R18 的 seed `0/1/2`。R34/R50 的 CLI 入口已经保留，但仍受“R18 三次完成后再展开”的阶段 gate 约束，并需要各自的官方 backbone 转换权重。转换文件的 metadata 含时间戳，整个文件 SHA 可能不同；合并统计应核对 `pretrain_tensor_sha256`，并同时核对相同 commit、config/annotation checksum、world size、每 rank batch、epoch、AMP/EMA 和评估方式。任何改参结果只能记为观察，不能混入均值与标准差。

建议 Issue 用下列清单认领，避免重复计算：

- [ ] R18 / seed 0
- [ ] R18 / seed 1
- [ ] R18 / seed 2

完成者应在评论中给出 commit、认领项、`metadata.json`、`result.json`、`train.log`、`eval.log`，并说明训练是否中断/恢复。checkpoint 只报告 `result.json` 中的 SHA-256；若需要共享大文件，应使用外部制品存储，不直接上传到 Issue。maintainer 审核证据前保持清单未完成，审核通过后再把对应结果写入准确率报告。

### 阶段 3：R18 标准 schedule 与稳定性

- [x] 提供单 `model + seed` 社区执行脚本，自动记录环境、协议、文件/权重 tensor checksum、日志、最终 EMA 指标和恢复命令；定向测试 `8 passed`。
- [ ] 完成 seed 0 的 72 epoch 训练与 val2017，记录 checkpoint checksum 和全部 COCO 指标。（deferred；3-epoch probe 已停止并审计）
- [ ] 对 seed 1、2 重复相同协议，报告均值、标准差和单 seed 离群情况。（deferred）
- [ ] 仅在 3 seed 证据完成后判断 R18 是否满足精度与稳定性门槛。（deferred）

### 阶段 4：R34/R50 与报告

- [ ] 在 R18 门槛通过后依次运行 R34、R50 的同权重完整 val 和训练验收。（deferred）
- [ ] 更新模型表、权重 checksum、可复现命令和局限说明。
- [ ] 完成准确率报告并把可复用结论推广到 `docs/migrations/`。

## 风险与回退

- 风险：当前 Paddle 是 CPU 构建，全量 val 可能耗时过长。缓解：先用固定小样本测量单图耗时；若外推不可接受，单独准备与当前 CUDA 兼容的 Paddle GPU 环境，不修改核心 PyTorch runtime 依赖。
- 风险：72 epoch × 3 seed 占用约 11 天双卡时间，且中途可能出现 AMP 或硬件故障。缓解：先完成同权重 AP gate；正式训练按 epoch 原子保存并做恢复探针，保留独立日志和配置快照。
- 风险：当前 PyTorch 与 Paddle schedule 明确不同，训练后 AP 差异无法直接归因于框架。缓解：报告中分开记录“同权重推理等价”和“各自训练策略结果”；必要时增加同 schedule 对照实验，但不静默改变主配置。
- 风险：多 seed 实际仍共享错误的数据顺序或 worker RNG。缓解：为 seed 传播和 DistributedSampler epoch 写直接测试，并记录每次运行的有效配置。
- 回退：任何超差先回到官方 checkpoint 的同图输出/激活对比；不通过重复长训碰运气。

## 验收

- [x] R18 同权重 Paddle/PyTorch 完整 val2017 的 AP 差不超过 `0.5` 百分点，并记录六项 AP 指标。
- [ ] R18 72 epoch 至少 3 seed 的配置、命令、日志、checkpoint、checksum、均值和标准差完整可获取。
- [ ] R34/R50 只在 R18 通过后验收，且每个声称支持的变体都有同结构证据。
- [ ] 默认核心测试和 CUDA 定向测试保持通过；所有测试/探针临时产物按仓库规则清理。
- [ ] `ROADMAP.md`、计划索引、准确率报告和迁移经验保持一致，不含工作站绝对路径。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-18 | 先做官方同权重完整 val，再启动 72 epoch | 同权重推理是低成本精度 gate；若它失败，长训结果没有清晰归因 |
| 2026-07-18 | `0.5 AP` 表示 0.5 个 COCO 百分点 | 避免把日志中的 `0.005` 与模型表中的 `0.5` 混为不同门槛 |
| 2026-07-18 | 训练优化不要求逐元素完全对齐 | 优化策略服务于收敛和 AP，但必须公开记录两侧 schedule 差异 |
| 2026-07-18 | Paddle 先做 CPU 耗时探针 | 当前 dev 环境没有 Paddle CUDA，不能假设全量 CPU 评估成本可接受 |
| 2026-07-18 | 正式训练使用官方 ImageNet R18-vd backbone 初始化 | 与官方训练起点一致；完整检测 checkpoint 只用于同权重 eval，不用于 72 epoch 训练 |
| 2026-07-18 | 最终训练 AP 评估 EMA checkpoint | R18 配置默认启用 EMA；即时模型和 EMA 已分开保存，Eval CLI 必须显式选择 |
| 2026-07-18 | 正式训练每 3 epoch 保存 | EMA checkpoint 实测约 368 MB；三 seed 每 epoch保存约需 79.5 GB，每 3 epoch 可降至约 26.5 GB，同时把最大恢复窗口限制在 3 epoch |
| 2026-07-18 | DDP checkpoint 按 rank 保存 RNG | rank process 使用 `base + rank`；只保存 rank 0 RNG 会让所有 rank 恢复成同一随机状态 |
| 2026-07-18 | R18 同权重 gate 通过 | 同 CPU/FP32 精确 AP 差仅 `1.65599e-7`，且 score `>=0.3` 的预测全部匹配；可进入 72 epoch 长训 |
| 2026-07-19 | 暂缓完整稳定性实验 | `3 seeds × 72 epoch` 约需 11 天双卡时间；当前只保留 3-epoch checkpoint 探针，不继续 R34/R50 长训 |
| 2026-07-19 | 用 GitHub Issue 分片征集完整运行 | 每位贡献者只运行一个 `model + seed`，脚本统一记录可复现证据；社区结果仍必须经过协议和 checksum 审核 |

## 完成记录

M4 已完成 R18 官方同权重 CPU/FP32 完整 val2017 gate，精确 AP 差 `1.65599e-7`，prediction 高置信度匹配也通过；显式 seed、官方 backbone 初始化、EMA Eval CLI、DDP per-rank RNG checkpoint 和社区分片执行入口已验证。默认全量回归为 `190 passed, 8 skipped`，CUDA 定向回归为 `8 passed`。seed 0 在 3 epoch 内完成 `21972` 次有效 update、15 次 AMP 自恢复跳步，随后原子保存 `epoch_3.pth` 并按计划主动停止；checkpoint 为 `368376901` 字节，SHA-256 `08d4977d9fa0ac963dce8c364895070e3a961aaadf0d7ac0ea5da7a97372116c`，model/optimizer/EMA 无非有限 tensor，EMA/scheduler step 均为 `21972`，且包含 2 份不同的 rank RNG state。完整 72 epoch、多 seed 与 R34/R50 长训仍为 deferred；该 3-epoch 结果不产生训练精度或稳定性结论。

# RT-DETRv3 PyTorch Migration Roadmap

**Status**: Active
**Last updated**: 2026-07-19
**Current evidence snapshot**: [`docs/plans/2026-07-18-migration-status.md`](docs/plans/2026-07-18-migration-status.md)
**Latest completed execution plan**: [`M3——训练、评估与恢复验收计划`](docs/plans/2026-07-18-m3-training-evaluation-recovery.md)
**Current execution plan**: [`M5——配置、CLI 与导出边界计划`](docs/plans/2026-07-19-m5-cli-export-boundaries.md)

本路线图以未完成的迁移大纲为主，并保留已完成里程碑的验收摘要。“完成”必须有当前代码、可复现命令和实际验收结果，不以历史 `specs/` 勾选状态为准。

## 目标

建立一个可安装、可训练、可评估、可恢复、可转换官方 Paddle 权重，并能在同一数据与硬件上给出数值、精度和性能对齐证据的 RT-DETRv3 PyTorch 训练库。

## Milestone 1 — 打通当前 API 的最小训练链（P0）

- [x] 使用当前 `workspace` 和 `configs/rtdetrv3/*.yml` 构建 R18 完整模型，不依赖 `tests/legacy/` API。
- [x] 使用最小 COCO fixture 构建 Dataset/DataLoader，验证一个 batch 的字段、shape、dtype、bbox 和 padding。
- [x] 完成一次训练态前向，输出所有 loss 分项且无 NaN/Inf。
- [x] 完成一次反向与 optimizer step，验证关键组件有有限梯度且参数实际更新。
- [x] 将 M1 所需的 backbone、head、loss、post-process、配置构建和模型集成场景按当前 API 重写回活跃测试集。
- [x] 在 CPU 上运行 5 iteration 的短训练烟雾测试。

**Exit criteria**: 一条可在 CI/开发机重复的 config → data → model → loss → backward → optimizer 链路，关键回归不再依赖旧 Registry 或旧 builder。

**验收记录**：2026-07-18 本机 CPU/float32 验证通过，全量测试 `127 passed, 1 skipped`。完整环境、override 和首末 step 数据见 M1 执行计划。

**完成提交**：`f95f4e0`。

## Milestone 2 — 官方权重转换与分层数值对齐（P0）

- [x] 下载并记录 R18/R34/R50 官方 Paddle checkpoint 的来源、checksum 和配置。
- [x] 对每个变体导出名称映射和未映射清单，审核 Linear 转置、BatchNorm 状态和特殊 head 参数。
- [x] 使转换后的权重以受控的 missing/unexpected key 集合加载到完整 PyTorch 模型。
- [x] 在 CPU/float32 上按 backbone → neck → transformer → head 比较第一个分歧激活。
- [x] 对齐预测、loss 分项和整体梯度方向；记录优化器差异，不要求 AdamW 更新逐元素一致。
- [x] 实现并测试批量转换、失败隔离、转换汇总与可选低内存模式。

**完成证据**：R18/R34/R50 已完成官方来源/SHA-256、共 2,041 个 tensor 的目标感知转换与逐值校验、受控加载，以及单线程 CPU/float32 的分层 eval、确定性缩减训练 loss 和整体梯度方向验收。批量 CLI 已覆盖目录/glob、自动输出命名、失败后继续、JSON 汇总和原子发布；R18 低内存严格转换为 571/571，峰值 RSS 观测为 `925,780 KiB`。R50 后处理保留 2/300 个 top-k 离散边界记录。当时移交给 M3 的 optimizer LR multiplier 缺口已在 M3 阶段 1 修复。

**Exit criteria**: 三个官方变体均有可重复转换命令、映射报告、加载结果和分层数值报告。

**验收记录**：2026-07-18 本机验证通过；默认测试 `144 passed, 3 skipped`，完整证据见 M2 执行计划。

## Milestone 3 — 训练、评估与恢复可用（P0）

- [x] 验证 R18 完整 1 epoch 训练和 COCO val2017 评估。
- [x] 核对 optimizer 参数组、ResNet `lr_mult_list=0.1`、weight decay 规则、warmup/decay step 单位、梯度裁剪与 EMA 顺序。
- [x] 在真实 COCO 短训练中验证 AMP 与 float32 的 loss/梯度有限性，记录 scaler 溢出与跳步。
- [x] 验证 checkpoint 恢复包含 model、EMA、optimizer、scheduler、scaler、epoch/global-step 和 RNG 状态。
- [x] 对比连续训练与中断恢复后紧接的 LR、loss 和参数更新。
- [x] 在 2 GPU 上验证 DDP sampler、SyncBN、同步梯度/scaler 和 rank-0 checkpoint 写入。
- [x] 实现并验证梯度累积与 DDP `no_sync()` 边界。

**完成证据**：R18 以 2-GPU AMP、每卡 batch 8 完成 train2017 1 epoch，checkpoint 记录 7319 次有效更新并通过完整 val2017，bbox AP/AP50/AP75 为 `0.468/0.643/0.504`。真实双卡短训练另行验证 `accumulate_steps=2`、DDP `no_sync()`、跨 rank 日志 loss 均值、EMA、非整窗口和 rank-0 应用日志/checkpoint。该结论只关闭训练库可用性，不代替 M4 的标准 schedule、多 seed 和 Paddle AP 对照。

**Exit criteria**: Train/Eval CLI 能用真实配置稳定运行，并有恢复一致性与 DDP 集成测试。

**验收记录**：2026-07-18 本机验证通过；隐藏 GPU 的默认测试 `165 passed, 8 skipped`，CUDA 定向文件另行 `8 passed`。

## Milestone 4 — COCO 精度与稳定性对齐（P1）

**执行计划**：[`M4——COCO 精度与稳定性对齐计划`](docs/plans/2026-07-18-m4-coco-accuracy-stability.md)。同一官方 R18 checkpoint 的 Paddle/PyTorch 完整 val2017 gate 已完成；72 epoch、多 seed 与 R34/R50 长训因时间成本暂缓，恢复前需要新的明确决策。

**当前进度**：R18 官方同权重 CPU/FP32 完整 val2017 gate 已通过：Paddle/PyTorch 精确 AP 分别为 `0.480477300367/0.480477134768`，绝对差 `1.65599e-7`；score `>=0.3` 的 `53780` 个 prediction 全部匹配。显式 seed、官方 ImageNet R18-vd backbone 初始化、2-GPU AMP+EMA 协议、EMA Eval CLI 和 DDP per-rank RNG checkpoint 已通过定向测试与真实双卡烟测。seed 0 已在 3 epoch 边界原子保存并主动停止；checkpoint 的 model/optimizer/EMA tensor 全部有限，恢复状态完整，但该探针不作标准 schedule、单 seed 精度或多 seed 稳定性证据。仓库已提供单 `model + seed` 的社区执行脚本，长训通过 GitHub Issue #3 分片认领，结果需按 commit、协议和 checksum 审核后才能进入统计。

- [ ] 对 R18 完成标准训练 schedule，保存环境、命令、配置、日志和 checkpoint。（deferred）
- [x] 与对应 Paddle 基线比较 AP/AP50/AP75/APs/APm/APl；R18 同设备主 AP 绝对差 `1.65599e-7`，通过 `0.5 AP` 目标。
- [ ] 在 R18 通过后依次验证 R34 和 R50，不在未定位数值差异时同时展开多变体训练。（deferred）
- [ ] 至少使用 3 个 seed 记录均值和方差；发布验收扩展到 5 个 seed。（deferred）
- [x] 生成 `docs/reports/accuracy-validation.md`，明确区分训练误差和框架实现缺陷。

**Exit criteria**: 每个声称支持的模型变体都有可复现的精度报告和可获取的权重。

## Milestone 5 — 配置、CLI 与导出边界（P1）

**执行计划**：[`M5——配置、CLI 与导出边界计划`](docs/plans/2026-07-19-m5-cli-export-boundaries.md)。第一阶段已恢复 Infer eager 基线：当前入口复用 TestReader、batch dict、模型内置 `bbox/bbox_num` 后处理和 Eval checkpoint 加载规则；官方 R18 已完成 CPU/FP32 真实 COCO 单图与 batch 4 验证。该证据不代表 ONNX/TorchScript 或全部 Paddle Infer 参数已经支持。

- [ ] 为 `workspace` 补充 shared/inject/from_config/显式参数冲突和全局状态隔离测试。
- [ ] 明确哪些 Paddle YAML 字段直接兼容、哪些映射、哪些不支持，补充配置迁移指南。
- [ ] 为 Train/Eval/Infer/Convert 编写 CLI contract 测试，对 Paddle 参数的兼容差异做显式文档化。
  - [x] Infer 已覆盖参数校验、当前/历史参数拼写、TestReader 预处理、batch dict、`bbox/bbox_num`、阈值、JSON 和官方 R18 真实推理。
  - [ ] Train/Eval/Convert 的统一错误路径、Paddle 参数差异和端到端合同仍待补齐。
- [ ] 完成 ONNX 导出与 ONNXRuntime 回归，记录不支持的动态控制流/算子。
- [ ] 完成 TorchScript 导出与重新加载回归。
- [ ] 验证动态输入尺寸、batch 1/4/8 和空预测等边界。

**Exit criteria**: 所有面向用户的入口都有功能测试，支持边界和框架差异有文档。

## Milestone 6 — 性能、质量与发布（P2）

- [ ] 在同一硬件、驱动、CUDA/cuDNN、batch 和精度下建立 Paddle/PyTorch 基准。
- [ ] 记录训练吞吐、推理延迟、峰值显存、DataLoader 占比和关键算子 profile。
- [ ] 目标训练吞吐不低于 Paddle 的 95%，峰值显存不超过 110%；无法达成时记录可定位瓶颈。
- [ ] 引入统一 lint/format/type-check 命令，清理当前 mypy 和 API 注解缺口。
- [ ] 生成覆盖率报告，将已迁移核心模块的有效覆盖率提升到 90% 目标。
- [ ] 建立 Python 3.9–3.12 与 CPU/主要 CUDA 组合的 CI 矩阵。
- [ ] 发布模型库、checksum、配置、许可说明和最终验证报告。

**Exit criteria**: 安装、测试、训练、评估、导出和模型获取都有可重复发布流程。

## 依赖顺序

```text
M1 最小训练链
 ├──> M2 权重/数值对齐 ──> M4 精度对齐
 └──> M3 训练/评估/恢复 ─┘
M1–M3 ──> M5 CLI/导出
M4–M5 ──> M6 性能与发布
```

## 不作为当前阻塞的延伸项

- TensorRT 引擎专项优化。
- C++ libtorch 完整示例。
- 剪枝、量化、NAS 或新模型架构。
- Paddle 中与 RT-DETRv3 训练/评估无关的所有检测任务分支。

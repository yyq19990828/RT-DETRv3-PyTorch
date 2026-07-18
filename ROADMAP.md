# RT-DETRv3 PyTorch Migration Roadmap

**Status**: Active
**Last updated**: 2026-07-18
**Current evidence snapshot**: [`docs/plans/2026-07-18-migration-status.md`](docs/plans/2026-07-18-migration-status.md)

本路线图只列未完成的迁移大纲。“完成”必须有当前代码、可复现命令和实际验收结果，不以历史 `specs/` 勾选状态为准。

## 目标

建立一个可安装、可训练、可评估、可恢复、可转换官方 Paddle 权重，并能在同一数据与硬件上给出数值、精度和性能对齐证据的 RT-DETRv3 PyTorch 训练库。

## Milestone 1 — 打通当前 API 的最小训练链（P0）

- [ ] 使用当前 `workspace` 和 `configs/rtdetrv3/*.yml` 构建 R18 完整模型，不依赖 `tests/legacy/` API。
- [ ] 使用最小 COCO fixture 构建 Dataset/DataLoader，验证一个 batch 的字段、shape、dtype、bbox 和 padding。
- [ ] 完成一次训练态前向，输出所有 loss 分项且无 NaN/Inf。
- [ ] 完成一次反向与 optimizer step，验证关键组件有有限梯度且参数实际更新。
- [ ] 将 backbone、head、loss、post-process、配置构建和模型集成的历史用例按当前 API 重写回活跃测试集。
- [ ] 在 CPU 或单 GPU 上运行 5–10 iteration 的短训练烟雾测试。

**Exit criteria**: 一条可在 CI/开发机重复的 config → data → model → loss → backward → optimizer 链路，关键回归不再依赖旧 Registry 或旧 builder。

## Milestone 2 — 官方权重转换与分层数值对齐（P0）

- [ ] 下载并记录 R18/R34/R50 官方 Paddle checkpoint 的来源、checksum 和配置。
- [ ] 对每个变体导出名称映射和未映射清单，审核 Linear 转置、BatchNorm 状态和特殊 head 参数。
- [ ] 使转换后的权重以受控的 missing/unexpected key 集合加载到完整 PyTorch 模型。
- [ ] 在 CPU/float32 上按 backbone → neck → transformer → head 比较第一个分歧激活。
- [ ] 对齐预测、loss 分项、梯度和一次 AdamW 参数更新。
- [ ] 实现并测试批量转换、失败隔离、转换汇总与可选低内存模式。

**Exit criteria**: 三个官方变体均有可重复转换命令、映射报告、加载结果和分层数值报告。

## Milestone 3 — 训练、评估与恢复可用（P0）

- [ ] 验证 R18 完整 1 epoch 训练和 COCO val2017 评估。
- [ ] 核对 optimizer 参数组、weight decay 排除、warmup/cosine step 单位、梯度裁剪与 EMA 顺序。
- [ ] 验证 AMP 与 float32 的 loss/梯度有限性，记录 scaler 溢出与跳步。
- [ ] 验证 checkpoint 恢复包含 model、EMA、optimizer、scheduler、scaler、epoch/global-step 和 RNG 状态。
- [ ] 对比连续训练与中断恢复后紧接的 LR、loss 和参数更新。
- [ ] 在至少 2 GPU 上验证 DDP sampler、SyncBN、loss reduce、梯度累积和 rank-0 写入。

**Exit criteria**: Train/Eval CLI 能用真实配置稳定运行，并有恢复一致性与 DDP 集成测试。

## Milestone 4 — COCO 精度与稳定性对齐（P1）

- [ ] 对 R18 完成标准训练 schedule，保存环境、命令、配置、日志和 checkpoint。
- [ ] 与对应 Paddle 基线比较 AP/AP50/AP75/APs/APm/APl，目标绝对差不超过 `0.5 AP`。
- [ ] 在 R18 通过后依次验证 R34 和 R50，不在未定位数值差异时同时展开多变体训练。
- [ ] 至少使用 3 个 seed 记录均值和方差；发布验收扩展到 5 个 seed。
- [ ] 生成 `docs/reports/accuracy-validation.md`，明确区分训练误差和框架实现缺陷。

**Exit criteria**: 每个声称支持的模型变体都有可复现的精度报告和可获取的权重。

## Milestone 5 — 配置、CLI 与导出边界（P1）

- [ ] 为 `workspace` 补充 shared/inject/from_config/显式参数冲突和全局状态隔离测试。
- [ ] 明确哪些 Paddle YAML 字段直接兼容、哪些映射、哪些不支持，补充配置迁移指南。
- [ ] 为 Train/Eval/Infer/Convert 编写 CLI contract 测试，对 Paddle 参数的兼容差异做显式文档化。
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

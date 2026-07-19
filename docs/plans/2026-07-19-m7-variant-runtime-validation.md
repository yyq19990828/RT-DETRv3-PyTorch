# M7——公开模型多变体运行时验收计划

- 状态：`completed`
- 创建日期：`2026-07-19`
- 最后更新：`2026-07-19`
- 负责人：`Codex / repository maintainers`

## 背景

`v0.1.0` 已发布 R18、R34、R50 三个检测权重，并完成 11 个 Release asset 的公开 checksum 回读。R18 另有 Models CLI 下载、真实 COCO Infer、完整 val2017 和小样本 Eval 证据；R34/R50 目前只有转换、分层数值、同图跨框架可视化和 Release 整体回读证据。公开文件可下载不等于对应配置能完成用户侧 eager 推理与评估，因此需要补齐逐变体运行时验收。

M4 的标准 schedule、多 seed 和 R34/R50 长训继续按维护者决策 deferred，不进入本计划。

## 目标与非目标

### 目标

- 使用 manifest 固定的 `v0.1.0` URL，通过 Models CLI 分别下载并校验 R34/R50。
- 在 CPU/FP32 下使用真实 COCO 图片完成 R34/R50 单图 Infer，验收可视化和 JSON 输出。
- 使用相同四图 COCO 子集完成 R34/R50 Eval 链路冒烟，记录环境、输入、命令和输出规模。
- 将验证范围和仍未覆盖的导出/GPU/完整 AP 边界写入迁移文档与独立报告。

### 非目标

- 不执行 72 epoch 标准训练、多 seed 稳定性或 R34/R50 长训。
- 不用四图 Eval 指标替代完整 val2017 AP。
- 不在本计划中扩展 R34/R50 ONNX/TorchScript、GPU provider、TensorRT、LVIS 或性能优化。
- 不修改或覆盖已发布的 `v0.1.0` asset。

## 实施步骤

- [x] 修正发布后计划索引和迁移局限中的陈旧状态。
- [x] 从固定公开 URL 经 Models CLI 下载并校验 R34/R50。
- [x] 对同一真实 COCO 图片运行 R34/R50 CPU/FP32 Infer。
- [x] 对同一四图 COCO 子集运行 R34/R50 CPU/FP32 Eval。
- [x] 审计变体运行结果；未发现需要实现修复的故障。
- [x] 记录实际证据、更新 ROADMAP，并清理全部临时下载和测试输出。

## 依赖

- `v0.1.0` GitHub Release 保持公开且固定 asset 不变。
- 仓库 UV `.venv` 可运行 PyTorch CPU 路径。
- 本机 COCO val2017 图片和 `instances_val2017.json` 可用。

## 风险与回退

- 风险：公开下载或 GitHub CDN 暂时失败会被误判为模型问题。缓解：先核对 HTTP/size/SHA-256，再进入模型构建与权重加载。
- 风险：四图 AP 波动被错误外推为正式精度。缓解：只将退出码、输出结构和候选规模作为链路证据；完整 AP 继续单独标注未验证。
- 风险：同时运行多个 CPU 模型造成资源竞争。缓解：下载可以并行，模型 Infer/Eval 默认顺序执行并固定 `num_workers=0`。
- 回退：本计划优先只增加验证文档；若需实现修复，按变体故障最小范围提交，不改变已发布 tag 或资产。

## 验收

- [x] R34/R50 下载文件大小和 SHA-256 与 manifest 一致。
- [x] 两个变体都严格加载 checkpoint，单图 Infer 生成可解码图片和非空 `detections.json`。
- [x] 两个变体都完成四图 Eval 并生成非空 `bbox.json`。
- [x] 证据明确区分公开回读、运行时冒烟和正式精度。
- [x] 临时 checkpoint、COCO 子集、预测、日志和缓存已清理。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | M7 先补公开 R34/R50 eager 运行时矩阵，不恢复 M4 长训 | 这是发布后最直接的用户可见证据缺口，且符合维护者对长训时间成本的决定 |
| 2026-07-19 | 使用 CPU/FP32 和统一四图子集作为首轮协议 | 降低设备差异并验证配置、加载、预处理、前向、后处理和 COCO metric 全链路；不把小样本指标当精度结论 |

## 完成记录

已完成。Python `3.12.11`、PyTorch `2.5.1+cu121`、CPU/FP32 下，Models CLI 从 `v0.1.0` 固定 URL 下载的 R34/R50 分别为 `137,170,947/182,510,207` 字节，SHA-256 与 manifest 一致。两者在 COCO `000000000139.jpg` 上分别生成 `31/28` 条阈值 `0.3` 检测和可解码图片；同一四图、43 annotation 子集的 Eval 均处理两个 batch 并写出 1,200 条候选。checkpoint 严格加载，只有既有合同允许重建的两个派生 buffer；未发现代码缺陷，因此没有行为实现改动。完整环境、输入 checksum、命令和限制见[多变体运行时报告](../reports/variant-runtime-validation.md)。临时下载与输出已清理。

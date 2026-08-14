# 模型文档

本目录按面向用户和验证驱动使用的模型族组织专属文档。跨模型复用的配置、训练、权重转换和排错经验仍保存在 [`docs/migrations`](../migrations/README.md)，这里不重复这些公共合同。

每个模型文档域都维护四个入口：`README.md` 描述当前用户合同，`validation-report.md` 记录验证方法、环境、结论和限制，`metrics.md` 保存逐变体 checkpoint、精度、数值对齐与部署指标，`evidence-index.md` 按验证能力域组织结论。一个文档域可以包含多个运行时 family；DEIM 的两个 decoder profile 统一在 `deim/` 中维护。机器日志中的临时路径、重复输出和无结论排错过程不会原样进入正式文档。

## 当前支持状态

| 模型族 | 变体 | 模型级验收 | 本项目权重发布 |
|---|---|---|---|
| RT-DETRv3 | R18/R34/R50 | 已完成 | `v0.1.0` 已发布 |
| D-FINE | N/S/M/L/X | 已完成 | 未发布，使用上游资产 |
| DEIM | D-FINE N/S/M/L/X；RT-DETRv2 S/M/M*/L/X | 已完成 | 未发布，使用上游资产 |
| RT-DETRv4 | S/M/L/X | 已完成 | 未发布，使用上游资产 |
| DEIMv2 | DINOv3 X/L/M/S；HGNetv2 N/Pico/Femto/Atto | 已完成 | 未发布，使用上游资产 |

“模型级验收已完成”表示固定官方 checkpoint 的构建、数值、完整 COCO、reduced train/resume 和部署矩阵达到各报告预注册门槛，不表示完整 schedule、多 seed、性能或权重发布。2026-08-14 跨模型最终审计的环境、测试计数和打包记录保存在[集成计划完成记录](../plans/2026-08-12-dfine-deim-rtdetrv4-integration.md#完成记录)，不作为会随代码变化的模型总览内容重复维护。

## 模型入口

- [RT-DETRv3](rtdetrv3/README.md)：已发布模型，包含配置支持、CLI/导出边界、已知限制和 `v0.1.0` 验证证据入口。
- [D-FINE](dfine/README.md)：N/S/M/L/X 的 checkpoint、数值、COCO、训练恢复和部署合同；集成与打包已验收，尚未发布权重。
- [DEIM](deim/README.md)：统一记录 D-FINE 与受限 RT-DETRv2 decoder profile 的十个变体、独立资产、数值和部署边界；不称 DEIMv1，也不包含独立 DEIMv2 上游。
- [RT-DETRv4](rtdetrv4/README.md)：S/M/L/X 的 checkpoint、真实 DINOv3 reduced train、COCO 和 student-only 部署合同；模型级与打包验收已完成，尚未发布权重。
- [DEIMv2](deimv2/README.md)：DINOv3 X/L/M/S 与 HGNetv2 N/Pico/Femto/Atto 共八个变体的 checkpoint、上游数值对齐、完整 val2017、reduced train/resume 和部署合同；vendored DINOv3 前向代码的许可边界在 NOTICE 单列。

目录状态必须明确区分“已发布”、“已完成模型级验收但未发布”和“计划中”。新增模型文档域时应创建同级目录并维护 `README.md`、`validation-report.md`、`metrics.md` 和 `evidence-index.md`；同一上游的多个 profile 优先在一个文档域内分节，不按 CLI family 机械拆目录。

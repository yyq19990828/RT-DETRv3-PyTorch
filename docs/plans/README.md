# 计划文档

本目录用于保存可执行的开发、迁移和验证计划。计划应记录目标、范围、阶段、验收方式和已知风险，不在这里堆放临时命令或无结论的排错记录。

## 文件约定

- 命名：`YYYY-MM-DD-<topic>.md`。
- 状态：`draft`、`in-progress`、`deferred`、`blocked`、`completed`或 `cancelled`。
- 计划中的命令以仓库根目录为工作目录。
- 完成后补充实际验收结果；可复用的迁移结论应沉淀到 `docs/migrations/`。

新计划可从 [`TEMPLATE.md`](TEMPLATE.md) 复制。

## 当前活动计划

- [D-FINE、DEIM 与 RT-DETRv4 集成计划](2026-08-12-dfine-deim-rtdetrv4-integration.md)：in-progress；Task 1-23 已完成，模型、Models CLI、打包、许可和文档任务均已通过；下一步为 F2-F4 并行最终验证及其后的 F1 合规审计。模型合同见 [`docs/models`](../models/README.md)。
- [M4——COCO 精度与稳定性对齐计划](2026-07-18-m4-coco-accuracy-stability.md)：活动但 deferred；同权重完整 val2017 gate 已通过，本机 72 epoch、多 seed 与 R34/R50 长训于 2026-07-19 暂缓。

## 已完成记录

- [RT-DETRv3 v0.1.0 已完成计划归档](../archive/rtdetrv3-v0.1.0/plans/README.md)：M1–M3、M5–M12 和 2026-07-18 迁移进度快照。
- [仓库迁移路线图](../../ROADMAP.md)：未完成工作的唯一顶层大纲。

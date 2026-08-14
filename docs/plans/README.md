# 计划文档

本目录用于保存可执行的开发、迁移和验证计划。计划应记录目标、范围、阶段、验收方式和已知风险，不在这里堆放临时命令或无结论的排错记录。

## 文件约定

- 命名：`YYYY-MM-DD-<topic>.md`。
- 状态：`draft`、`in-progress`、`deferred`、`blocked`、`completed`或 `cancelled`。
- 计划中的命令以仓库根目录为工作目录。
- 完成后补充实际验收结果；可复用的迁移结论应沉淀到 `docs/migrations/`。

新计划可从 [`TEMPLATE.md`](TEMPLATE.md) 复制。

## 当前活动计划

- [M4——COCO 精度与稳定性对齐计划](2026-07-18-m4-coco-accuracy-stability.md)：活动但 deferred；同权重完整 val2017 gate 已通过，本机 72 epoch、多 seed 与 R34/R50 长训于 2026-07-19 暂缓。

## 已完成记录

- [包与 CLI 重命名计划](2026-08-15-rename-to-detrs.md)：2026-08-15 完成 `ppdet_pytorch` → `detrs` 包名/项目名切换,六个 `rtdetrv3-*` 命令收敛为单 `detrs` 入口 + 子命令;TorchScript 跨设备修复与 CI 恢复另见后续提交。
- [DEIMv2 集成计划](../archive/2026-08-14-deimv2-integration.md)：2026-08-15 维护者接受,归档;8 个 COCO 变体的实现、数值对齐、完整 val2017、reduced train/resume、导出与打包验收完成。
- [D-FINE、DEIM 与 RT-DETRv4 集成计划](../archive/2026-08-12-dfine-deim-rtdetrv4-integration.md)：2026-08-15 维护者接受,归档;全部实现阶段与最终审计通过,模型报告、逐变体指标和证据索引见 [`docs/models`](../models/README.md)。

- [DEIM 模型文档目录合并计划](../archive/2026-08-14-consolidate-deim-documentation.md)：2026-08-14 将两个 decoder profile 合并到唯一 `docs/models/deim/` 文档域。
- [仓库文档优化与整理计划](../archive/2026-08-14-documentation-reorganization.md)：2026-08-14 完成入口收敛、历史归档和通用文档 CI 门禁。
- [RT-DETRv3 v0.1.0 已完成计划归档](../archive/rtdetrv3-v0.1.0/plans/README.md)：M1–M3、M5–M12 和 2026-07-18 迁移进度快照。
- [仓库迁移路线图](../../ROADMAP.md)：未完成工作的唯一顶层大纲。

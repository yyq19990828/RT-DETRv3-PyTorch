# 计划文档

本目录用于保存可执行的开发、迁移和验证计划。计划应记录目标、范围、阶段、验收方式和已知风险，不在这里堆放临时命令或无结论的排错记录。

## 文件约定

- 命名：`YYYY-MM-DD-<topic>.md`。
- 状态：`draft`、`in-progress`、`blocked`、`completed`或 `cancelled`。
- 计划中的命令以仓库根目录为工作目录。
- 完成后补充实际验收结果；可复用的迁移结论应沉淀到 `docs/migrations/`。

新计划可从 [`TEMPLATE.md`](TEMPLATE.md) 复制。

## 当前文档

- [M1——R18 最小训练链迁移计划](2026-07-18-m1-minimal-training-chain.md)：已完成的 M1 计划，覆盖 config、最小 COCO batch、loss、backward、optimizer 和短训练验收证据。
- [2026-07-18 迁移进度快照](2026-07-18-migration-status.md)：对历史 `specs/` 记录与当前代码的核验结果。
- [仓库迁移路线图](../../ROADMAP.md)：未完成工作的唯一顶层大纲。

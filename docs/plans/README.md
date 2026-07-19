# 计划文档

本目录用于保存可执行的开发、迁移和验证计划。计划应记录目标、范围、阶段、验收方式和已知风险，不在这里堆放临时命令或无结论的排错记录。

## 文件约定

- 命名：`YYYY-MM-DD-<topic>.md`。
- 状态：`draft`、`in-progress`、`deferred`、`blocked`、`completed`或 `cancelled`。
- 计划中的命令以仓库根目录为工作目录。
- 完成后补充实际验收结果；可复用的迁移结论应沉淀到 `docs/migrations/`。

新计划可从 [`TEMPLATE.md`](TEMPLATE.md) 复制。

## 当前文档

- [M6——性能、质量与发布计划](2026-07-19-m6-performance-quality-release.md)：进行中；Ruff 已覆盖全部活跃 Python 文件，Mypy 已覆盖整个 package 和纳入门禁的仓库脚本；Python 3.9–3.12 CPU CI、本机 CUDA/性能证据、wheel/sdist 发布候选验证、R18/R34/R50 COCO 可视化对比、四产物 Models CLI 和 11-asset checksum/严格回读预检已完成。直接维护范围本机覆盖率达到 90.80%，门禁提高到 50.5%/90%；权重公开发布及真实 URL 回读尚未完成。
- [M5——配置、CLI 与导出边界计划](2026-07-19-m5-cli-export-boundaries.md)：已完成；覆盖 workspace/config、五个 CLI 合同、官方 R18 eager 推理，以及 ONNX/TorchScript 固定高宽和动态 batch 边界。
- [M4——COCO 精度与稳定性对齐计划](2026-07-18-m4-coco-accuracy-stability.md)：同权重完整 val2017 gate 已通过；本机长训于 2026-07-19 暂缓，已提供 GitHub 社区分片执行脚本。
- [M3——训练、评估与恢复验收计划](2026-07-18-m3-training-evaluation-recovery.md)：已完成，覆盖 optimizer/LR、完整 COCO epoch/val、恢复、AMP/EMA、DDP、loss reduce 与梯度累积。
- [M2——官方 checkpoint 转换与分层数值对齐计划](2026-07-18-m2-official-checkpoint-alignment.md)：已完成的 M2 计划，覆盖三变体官方权重、分层数值、批量失败隔离和受控内存转换。
- [M1——R18 最小训练链迁移计划](2026-07-18-m1-minimal-training-chain.md)：已完成的 M1 计划，覆盖 config、最小 COCO batch、loss、backward、optimizer 和短训练验收证据。
- [2026-07-18 迁移进度快照](2026-07-18-migration-status.md)：对历史 `specs/` 记录与当前代码的核验结果。
- [仓库迁移路线图](../../ROADMAP.md)：未完成工作的唯一顶层大纲。

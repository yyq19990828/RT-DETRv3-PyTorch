# 历史规格整合记录

**整合日期**：2026-07-18
**原始范围**：仓库根目录原 `specs/001`–`specs/005`

本文记录原 `specs/` 目录的信息如何整合到当前文档体系。原文档是多轮设计快照，其路径、API 和勾选状态彼此有冲突，因此没有原样搬运到活跃文档中。

## 来源与去向

| 原记录 | 历史勾选 | 有效信息 | 当前去向 |
|---|---:|---|---|
| `001-i-want-to` | 46 完成 / 28 未完成 | 初始迁移目标、分层验证、训练/导出/性能风险 | [`ROADMAP.md`](../../ROADMAP.md)、[`training-and-validation.md`](training-and-validation.md) |
| `002-specify-scripts-bash` | 无真实任务 | 未填写的通用模板，不含项目经验 | 不保留；新计划使用 [`docs/plans/TEMPLATE.md`](../plans/TEMPLATE.md) |
| `003-paddle-pytorch-conversion` | 58 完成 / 20 未完成 | 权重映射、张量转换、严格/宽松模式、分层校验 | [`weight-conversion.md`](weight-conversion.md)、[`ROADMAP.md`](../../ROADMAP.md) |
| `004-paddle-pytorch-migration` | `tasks.md` 称 87/91；另一状态文档仅称 30/90 | Registry、依赖注入、YAML 配置和导入时注册的经验 | [`registry-and-configuration.md`](registry-and-configuration.md) |
| `005-paddle-pytorch-migration` | 71 完成 / 37 未完成 | 包结构、数据/引擎迁移、工具、端到端验证大纲 | [`2026-07-18-migration-status.md`](../plans/2026-07-18-migration-status.md)、[`ROADMAP.md`](../../ROADMAP.md) |

> 上表的数字只是原任务文件中的勾选数，不是当前完成率。

## 主要冲突与处理

### 包结构

历史方案先后使用过 `rtdetrv3_pytorch.models`、双层
`rtdetrv3_pytorch/ppdet_pytorch` 和仿 Paddle `ppdet` 的表达。当前以已安装并经 wheel 验证的 `src/ppdet_pytorch` 为唯一有效结构。

### 注册系统

`004` 曾把分类 Registry 及“直接构造向后兼容”标为完成；
`005` 又要求移除分类 Registry，改为 Paddle 风格的统一
`core.workspace`。当前代码采用后者，前者的测试已保留到 `tests/legacy/`，不应再把旧勾选当作现状证据。

### 数值等价

历史文档多次把结构对应、形状通过或随机输入确定性写成“数值一致”。当前文档统一改为分层证据模型：参数值、中间激活、loss/梯度/优化器单步、真实数据集指标必须分别验证。

### API 等价假设

原研究对 AdamW、DataLoader 共享内存、随机数和调度器给出过过强的“等价”结论。这些结论现在只作为待测假设，必须用方程、step 单位和固定输入的单步实验验证。

## 现状证据优先级

1. 当前代码、锁文件和子模块提交。
2. 在当前环境实际运行的测试、构建和基准数据。
3. `docs/plans/` 中标有日期的进度快照。
4. `docs/migrations/` 中明确标记的历史快照。
5. 已删除的原 `specs/` 勾选状态。

原始内容仍可从 Git 历史恢复；日常开发不应重建顶层 `specs/` 目录。

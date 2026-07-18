# RT-DETRv3 PyTorch 迁移进度快照

- 状态：`in-progress`
- 快照日期：`2026-07-18`
- 下一步大纲：[`ROADMAP.md`](../../ROADMAP.md)
- 历史规格去向：[`spec-history.md`](../migrations/spec-history.md)

## 已验证基线

| 范围 | 当前证据 | 状态 |
|---|---|---|
| 仓库与包结构 | `src/ppdet_pytorch`、根 `configs/`、`tests/`、`tools/`、`docs/`；wheel 构建成功 | 已完成 |
| Paddle 参考代码 | `third-party/RT-DETRv3-paddle` 子模块，固定提交 `349e7d99a5065e7b684118912e6a74178d4f4625` | 已完成 |
| 环境与依赖 | Python 3.12 `.venv`；`uv sync --extra dev`；Paddle 仅在 `dev` extra | 已完成 |
| 配置 | R18 主配置可递归加载并通过当前 `workspace` 构建完整模型 | M1 已验证 |
| 最小训练链 | 合成 COCO batch → R18 → 30 个 loss 项 → backward → gradient clip → AdamW/LR；5 step 有限 | M1 已验证 |
| CLI | 四个 console entry point 和四个 `tools/` 兼容入口的 `--help` 通过 | 部分完成 |
| 权重转换 | 单文件转换、映射导出、严格/宽松形状处理的单元和集成测试通过 | 部分完成 |
| 活跃测试 | `127 passed, 1 skipped`；M1 场景已进入活跃测试，其他历史 API 用例在 `tests/legacy/` | 基线稳定 |

上述结果只证明当前测试范围通过，不证明完整训练、COCO 精度或 Paddle 数值对齐已完成。

## 当前阶段

### 已完成

- Python `src-layout` 和可安装包元数据。
- Paddle 源码从 vendored 副本改为固定子模块。
- 核心运行时与 Paddle 顶层导入解耦，Paddle 相关依赖转入 `dev` extra。
- 基础权重转换引擎、张量转换、名称映射、会话元数据和回归测试。
- README、历史报告、计划与迁移经验的目录规则。
- R18 原始配置构建、最小 COCO batch 合同、缩减配置训练前向/反向/优化器单步和 5-step CPU 烟雾测试。
- M1 所需的 workspace、backbone、head/loss、post-process 和完整模型场景已按当前 API 重写。

### 部分完成

- 当前 Trainer 已覆盖最小训练 epoch 内循环，但真实 COCO、完整 epoch、Eval/Infer、checkpoint 恢复、AMP/EMA/DDP 仍待验收。
- Train/Eval/Infer CLI 可导入并解析参数，但尚未使用真实 COCO 数据和转换权重做端到端验收。
- 权重转换已在小型 fixture 上验证，但尚未对 R18/R34/R50 官方权重全量验证。
- `workspace` 的 M1 配置合同已锁定；shared/inject 全部冲突矩阵和多配置进程隔离留待 M5。

### 未完成

- 真实 COCO 数据和官方 checkpoint 下的完整 epoch、checkpoint 恢复、AMP/EMA/DDP 组合验收。
- Paddle/PyTorch 分层激活、loss、梯度、优化器更新与 COCO AP 对齐。
- ONNX/TorchScript 导出与部署时回归。
- 同硬件下的训练吞吐、推理延迟和峰值显存基准。
- 转换后模型权重、验证报告和模型库发布。

## 历史任务的重分类

- 历史上标记“模块迁移完成”的任务，现在只能记为“代码存在”，直到它有当前 API 测试或端到端证据。
- M1 直接所需的历史场景已重写；R34/R50/R101、冻结边界、空 GT、Eval/Infer 与 numerical 场景依然不计为已迁移。
- 工具脚本路径与 CLI 帮助已修复的历史任务记为“入口完成、功能待验收”。
- 权重转换的基础功能记为“小型 fixture 已验证”，不等于官方模型库已验证。
- 历史数值测试未进入真正框架对比的，不计入数值等价完成度。

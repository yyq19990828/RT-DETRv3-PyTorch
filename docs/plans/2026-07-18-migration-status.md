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
| 配置 | `configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml` 可递归加载，`architecture=RTDETRV3` | 部分完成 |
| CLI | 四个 console entry point 和四个 `tools/` 兼容入口的 `--help` 通过 | 部分完成 |
| 权重转换 | 单文件转换、映射导出、严格/宽松形状处理的单元和集成测试通过 | 部分完成 |
| 活跃测试 | `109 passed, 1 skipped`；历史 API 用例在 `tests/legacy/` | 基线稳定 |

上述结果只证明当前测试范围通过，不证明完整训练、COCO 精度或 Paddle 数值对齐已完成。

## 当前阶段

### 已完成

- Python `src-layout` 和可安装包元数据。
- Paddle 源码从 vendored 副本改为固定子模块。
- 核心运行时与 Paddle 顶层导入解耦，Paddle 相关依赖转入 `dev` extra。
- 基础权重转换引擎、张量转换、名称映射、会话元数据和回归测试。
- README、历史报告、计划与迁移经验的目录规则。

### 部分完成

- 模型、数据、引擎、优化器和指标模块已存在，但活跃测试尚未覆盖完整的配置构建链和真实数据流。
- Train/Eval/Infer CLI 可导入并解析参数，但尚未使用真实 COCO 数据和转换权重做端到端验收。
- 权重转换已在小型 fixture 上验证，但尚未对 R18/R34/R50 官方权重全量验证。
- Paddle 风格 `workspace` 已存在，但历史 Registry/构建器测试已与当前 API 分叉，需要按当前语义重写。

### 未完成

- 真实数据 batch 的完整模型构建、前向、loss、反向和优化器单步。
- 短训练、完整 epoch、checkpoint 恢复、AMP/EMA/DDP 组合验收。
- Paddle/PyTorch 分层激活、loss、梯度、优化器更新与 COCO AP 对齐。
- ONNX/TorchScript 导出与部署时回归。
- 同硬件下的训练吞吐、推理延迟和峰值显存基准。
- 转换后模型权重、验证报告和模型库发布。

## 历史任务的重分类

- 历史上标记“模块迁移完成”的任务，现在只能记为“代码存在”，直到它有当前 API 测试或端到端证据。
- 工具脚本路径与 CLI 帮助已修复的历史任务记为“入口完成、功能待验收”。
- 权重转换的基础功能记为“小型 fixture 已验证”，不等于官方模型库已验证。
- 历史数值测试未进入真正框架对比的，不计入数值等价完成度。

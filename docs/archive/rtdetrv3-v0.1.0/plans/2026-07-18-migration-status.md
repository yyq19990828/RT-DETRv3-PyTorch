# RT-DETRv3 PyTorch 迁移进度快照

- 状态：`in-progress`
- 快照日期：`2026-07-18`
- 下一步大纲：[`ROADMAP.md`](../../../../ROADMAP.md)
- 快照创建时的执行计划：[`M3——训练、评估与恢复验收`](2026-07-18-m3-training-evaluation-recovery.md)
- 历史规格去向：[`spec-history.md`](../migration/spec-history.md)

> 本文是 2026-07-18 迁移过程中的历史快照，不代表当前仓库状态；当前结论以 [`ROADMAP.md`](../../../../ROADMAP.md) 和对应里程碑计划为准。当前合同见 [`docs/models/rtdetrv3`](../../../models/rtdetrv3/README.md)。

## 已验证基线

| 范围 | 当前证据 | 状态 |
|---|---|---|
| 仓库与包结构 | `src/ppdet_pytorch`、根 `configs/`、`tests/`、`tools/`、`docs/`；wheel 构建成功 | 已完成 |
| Paddle 参考代码 | `third-party/RT-DETRv3-paddle` 子模块，固定提交 `349e7d99a5065e7b684118912e6a74178d4f4625` | 已完成 |
| 环境与依赖 | Python 3.12 `.venv`；`uv sync --extra dev`；Paddle 仅在 `dev` extra | 已完成 |
| 配置 | R18 主配置可递归加载并通过当前 `workspace` 构建完整模型 | M1 已验证 |
| 最小训练链 | 合成 COCO batch → R18 → 30 个 loss 项 → backward → gradient clip → AdamW/LR；5 step 有限 | M1 已验证 |
| CLI | 四个 console entry point 和四个 `tools/` 兼容入口的 `--help` 通过 | 部分完成 |
| 权重转换 | R18/R34/R50 共 2,041 个官方 tensor 全部转换并逐 tensor 精确校验，目标感知 Linear 转置，受控 missing key 加载 | M2 三变体已验证 |
| 数值对齐 | 三变体 eval 分层、受控 loss 及整体梯度方向通过；R50 记录 2/300 个后处理 top-k 边界候选差异 | M2 三变体已验证 |
| 活跃测试 | `153 passed, 3 skipped`；三变体可选 numerical 回归独立运行均为 `1 passed, 2 deselected, 2 warnings` | 基线稳定 |

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
- M1 完成提交为 `f95f4e0`。
- 建立 R18/R34/R50 官方 checkpoint manifest；三个源文件的字节数与 SHA-256 已实际验证。
- conversion CLI 默认根据 `--config` 构建目标模型，使用实际 `torch.nn.Linear` 模块集合决定转置。
- R18 方阵 Linear 漏转置问题已定位并修正；官方权重的 eval 分层对齐已加入可选 numerical 用例。
- 多分组训练 attention mask 的布尔语义已修正，并有交叉组阻断单元测试。
- 官方 R18 的确定性缩减训练前向已完成 30 个 loss 分项对齐；head/loss 输出梯度通过。
- 非连续 BatchNorm grad-output 的反向语义已修正并加入核心单测；官方 R18 完整模型的整体梯度方向通过可选 numerical 验收。
- R34/R50 已复用同一转换、分层 eval、loss 和整体梯度方向用例通过；R50 冻结 BN 的全局统计语义已修正并加入单测。
- R50 后处理 300 个候选中观察到 2 个 top-k 离散边界差异；全部 score、298 个稳定候选坐标和 `bbox_num` 通过，未将其误写为完全逐候选一致。
- conversion CLI 已支持同 config/架构 checkpoint 的目录/glob 批量发现、失败隔离、自动命名、mapping/JSON 汇总和非零失败退出码。
- checkpoint 使用同目录临时文件原子发布；模拟写入失败时已有输出保持不变且临时文件被清理。
- 官方 R18 低内存严格转换完成 571/571，峰值 RSS 观测为 `925,780 KiB`；该模式降低中间驻留但不等于流式 checkpoint。
- ResNet `lr_mult_list` 已转为显式 optimizer groups，piecewise milestone 已锁定为全局 step，梯度裁剪 → optimizer → scheduler → EMA 顺序已有回归。
- 训练 checkpoint schema v1 已覆盖 model/EMA/optimizer/scheduler/scaler、epoch/global-step/sampler 和 RNG；连续训练与中断恢复的紧接一步已验证。

### 部分完成（快照当时状态）

- 快照创建时 Trainer 只覆盖最小训练 epoch 内循环和可恢复 checkpoint；后续状态不得从本段推断。
- 快照创建时尚未定位真实 COCO；后续数据发现与实测证据见 M3 计划。
- 快照创建时 Train/Eval/Infer CLI 只完成导入和参数解析；后续端到端状态见当前路线图。
- M2 的三变体单文件/批量转换、确定性训练 loss、整体梯度方向和受控内存路径已完成；不要求 AdamW 逐元素一致。当前 PyTorch schedule 与 Paddle 的数值策略差异作为已知边界保留。
- `workspace` 的 M1 配置合同已锁定；shared/inject 全部冲突矩阵和多配置进程隔离留待 M5。

### 未完成（快照当时状态）

- 真实 COCO 数据和官方 checkpoint 下的完整 epoch、AMP/EMA/DDP 组合验收。
- 训练收敛与 COCO AP 验收。
- ONNX/TorchScript 导出与部署时回归。
- 同硬件下的训练吞吐、推理延迟和峰值显存基准。
- 转换后模型权重、验证报告和模型库发布。

## 历史任务的重分类

- 历史上标记“模块迁移完成”的任务，现在只能记为“代码存在”，直到它有当前 API 测试或端到端证据。
- M1 直接所需的历史场景已重写；R34/R50 官方转换与 numerical 已在 M2 重写验证，R101、空 GT 和完整 Eval/Infer 仍不计为已迁移。
- 工具脚本路径与 CLI 帮助已修复的历史任务记为“入口完成、功能待验收”。
- 权重转换的基础、批量与三变体官方模型已分别验证；这仍不等于已发布稳定模型库。
- 历史数值测试未进入真正框架对比的，不计入数值等价完成度。

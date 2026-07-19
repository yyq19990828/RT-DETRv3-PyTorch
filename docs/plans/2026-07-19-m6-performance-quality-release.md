# M6——性能、质量与发布计划

- 状态：`in-progress`
- 创建日期：`2026-07-19`
- 最后更新：`2026-07-19`
- 负责人：`Codex / repository maintainers`

## 背景

M1–M5 已完成当前 RT-DETRv3 PyTorch 训练、转换、评估、恢复、Infer 和导出边界，M4 的完整长训稳定性实验按维护者决策转为社区协作，不阻塞本阶段。仓库尚无 CI workflow、统一质量命令、框架性能 runner 或发布清单；直接声明“可发布”没有证据。

2026-07-19 的 M6 基线审计观察到：174 个 Git 跟踪 Python 文件中，Ruff `0.15.22` 认为 128 个需要重新格式化；全仓默认 Ruff lint 有 293 项，Mypy 全包有 123 项；默认活跃测试为 `237 passed, 8 skipped`，全包语句覆盖率为 45%。这些数字是当前审计快照，不是质量目标完成证据。

## 目标与非目标

### 目标

- 使用 Ruff 统一 Python 格式化和基础 lint，使用 Mypy 负责类型检查，并提供单一可执行质量命令。
- 以明确记录的渐进范围建立可通过门禁，再扩展到全部活跃 Python 代码；不通过大规模无审计自动修复伪造“零告警”。
- 建立同硬件、同 batch、同精度的 Paddle/PyTorch 训练和推理性能协议与机器可读报告。
- 建立 Python 3.9–3.12 CPU CI、受控 CUDA 验证、覆盖率报告和 wheel 安装 smoke。
- 汇总模型、checksum、配置、许可和可复现命令，形成发布候选验收记录。

### 非目标

- 不恢复 M4 的 72 epoch、多 seed 本地长训；社区结果仍按 Issue #3 审核。
- 不在 M6 第一批质量改动中一次性格式化全部历史继承文件或静默忽略现有问题。
- 不把不同硬件、provider、数据管线或精度的测量混进同一性能比值。
- 不引入 TensorRT、量化、剪枝或 C++ 部署。

## 实施步骤

- [x] 建立 Ruff/Mypy 统一质量命令；第一批覆盖 `cli`、`conversion`、`core`、`deploy`、`scripts` 及对应单测，并记录未覆盖范围。
- [x] 将 Ruff 格式与基础 lint 扩展到全部活跃 Python 文件，删除 Ruff 临时范围清单。
- [ ] 将 Mypy 类型门禁逐批扩展到其余活跃模块，最后删除 Mypy 临时范围清单。
- [ ] 生成模块级覆盖率报告，区分已迁移核心合同与尚未支持的继承代码，逐步建立可执行阈值。
- [x] 建立 Python 3.9–3.12 CPU CI，覆盖安装、质量、核心测试、导出和 wheel smoke。
- [ ] 为 CUDA 增加独立受控 job 或自托管验证证据，不把 CUDA wheel 安装等同于 GPU 验证。
- [ ] 编写 Paddle/PyTorch 训练和推理 benchmark runner，固定 warmup、采样、同步、batch、dtype、设备和内存口径。
- [ ] 在同一硬件执行 R18 基准并定位吞吐或峰值显存未达目标的第一处瓶颈，再决定是否扩展 R34/R50。
- [ ] 生成发布清单和最终验证报告，包含模型来源、checksum、配置、许可、环境、命令与已知限制。

## 风险与回退

- 风险：全仓格式化会把行为改动淹没在机械 diff 中。缓解：按已迁移模块分批提交，每批运行 Ruff、定向测试和默认全量测试。
- 风险：Mypy 的第三方 stub 缺口与真实类型错误混杂。缓解：先记录并隔离缺失 stub，再逐模块清理真实错误；不使用全局 `# type: ignore` 掩盖问题。
- 风险：CI 的 CUDA wheel、Paddle nightly 和 GPU runner 不稳定。缓解：CPU 核心矩阵不依赖 Paddle；Paddle/CUDA 使用独立可选 job 和固定环境证据。
- 风险：异步 CUDA 或不同 DataLoader 会夸大性能结论。缓解：runner 强制 warmup、同步和环境记录，并分别报告 model-only 与 end-to-end。
- 回退：质量配置、脚本和 CI 都可独立回退，不改变模型 checkpoint、训练状态或数值合同。

## 验收

- [x] `uv run --extra quality python scripts/check_quality.py` 通过，并明确输出 Ruff format、Ruff lint 和 Mypy 三个步骤。
- [x] 第一批范围经 Ruff 实际格式化后，定向测试与默认全量测试通过。
- [x] Ruff 全活跃范围经格式化、lint 和默认全量测试验证通过。
- [ ] 覆盖率报告与排除规则有文档，不把未支持继承模块排除后仍声称全包覆盖率。
- [x] Python 3.9–3.12 CPU CI 均能从干净环境安装并通过其声明的测试。
- [ ] Paddle/PyTorch 性能报告记录硬件、软件、命令、warmup、采样、batch、dtype、吞吐、延迟和峰值内存。
- [x] wheel 安装后五个 CLI help、Infer、Eval 和 Export smoke 可重复。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | Ruff 同时负责格式化和基础 lint，移除 Black | 维护者明确指定 Ruff；单一工具减少格式规则分叉 |
| 2026-07-19 | 第一批质量门禁只覆盖 M1–M5 直接维护面 | 全仓当前有 128 个格式文件和 293 个 lint 项，一次性机械改写不利于审计 |
| 2026-07-19 | Mypy 与 Ruff 分工，不把类型检查并入 lint | Ruff 不替代静态类型语义；当前全包 123 项需要渐进清理 |
| 2026-07-19 | 全包 45% 只作为覆盖率基线 | 大量继承但未支持模块拉低数字，后续必须同时报告全包和已迁移核心范围 |
| 2026-07-19 | Python 3.9/3.10 使用兼容的 ONNX Runtime 上界 | ONNX Runtime 新版已分别停止提供 3.9/3.10 wheel；统一无上界会使干净矩阵无法安装 |
| 2026-07-19 | CI 核心矩阵不安装 `dev` extra | `test` extra 覆盖 Pytest、ONNX 与 ONNX Runtime，但不引入 Paddle；Paddle 对齐继续使用独立环境 |
| 2026-07-19 | 锁文件与 CI 统一使用 UV 0.11.29.x | UV 0.7 与 0.11 的锁文件修订语义不同；固定版本范围避免本地通过而托管 `--locked` 拒绝 |

## 完成记录

进行中。第一批已移除 Black，锁定 Ruff `0.15.22`，增加独立 `quality` extra 和 `scripts/check_quality.py`。第二批将 Ruff format/lint 扩展到仓库根目录；排除规则仅保留只读 `third-party/`、历史 `tests/legacy/` 和生成目录，158 个活跃 Python 文件通过，默认全量测试为 `246 passed, 3 skipped, 6 warnings in 9.87s`。Mypy 仍只对 6 个 source file/目录通过，初始全包 123 项尚未清完。

同批增加不含 Paddle 的 `test` extra 和 GitHub Actions workflow。四个 UV 隔离 CPU 环境已在隐藏 GPU 后本地验证：Python 3.9/3.10/3.11/3.12 均完成锁文件安装，分别为 `211 passed, 7 skipped, 17 deselected`；Python 3.9 使用 ONNX Runtime `1.19.2`，3.10 使用 `1.20.1`，3.11/3.12 使用 `1.27.0`。wheel 重装后五个 CLI `--help` 全部通过，Infer/Eval/Export 定向 smoke 为 `34 passed`。

GitHub Actions [run 29670978523](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29670978523) 在提交 `b2ffcff` 上完成托管验收：Python 3.9/3.10/3.11/3.12 四个 CPU jobs 均为 `211 passed, 7 skipped, 17 deselected`；质量 job 为 158 个 Ruff 文件和 6 个 Mypy source 通过；wheel job 成功构建并完成 `34 passed` smoke。首次 run 在测试前因 UV 0.7/0.11 锁文件修订差异失败，统一 `required-version` 并用 UV `0.11.29` 重建锁文件后关闭。覆盖率阈值、完整 Mypy、CUDA CI、性能和发布候选仍未完成。

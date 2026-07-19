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
- [x] 将 Mypy 类型门禁逐批扩展到其余活跃模块，最后删除 Mypy 临时范围清单。
  - [x] 将门禁扩展到完整 `cli`、`conversion`、`deploy` 和 3 个质量/稳定性脚本，17 个 source file 通过。
  - [x] 将完整 `optimizer` 目录加入门禁，累计 22 个 source file 通过。
  - [x] 将完整 `metrics` 目录加入门禁，累计 27 个 source file 通过。
  - [x] 将完整 `utils` 目录加入门禁，累计 36 个 source file 通过。
  - [x] 将完整 `core` 目录加入门禁，累计 41 个 source file 通过。
  - [x] 将完整 `engine` 目录加入门禁，累计 47 个 source file 通过。
  - [x] 将完整 `modeling` 目录加入门禁，累计 84 个 source file 通过。
  - [x] 清理 `data` 的 87 项错误，将整个 `src/ppdet_pytorch` 收敛为单一 Mypy 目标；含 3 个脚本共 103 个 source file 通过。
- [x] 生成模块级覆盖率报告，区分全包与直接维护范围，建立可执行的初始回退阈值。
- [x] 建立 Python 3.9–3.12 CPU CI，覆盖安装、质量、核心测试、导出和 wheel smoke。
- [x] 为 CUDA 增加独立受控 job 或自托管验证证据，不把 CUDA wheel 安装等同于 GPU 验证。
- [x] 编写 Paddle/PyTorch 训练和推理 benchmark runner，固定 warmup、采样、同步、batch、dtype、设备和内存口径。
- [x] 在同一硬件执行 R18 基准并定位吞吐或峰值显存未达目标的第一处瓶颈，再决定是否扩展 R34/R50。
- [x] 生成发布清单和发布候选验证报告，包含模型来源、checksum、配置、许可、环境、命令与已知限制。
- [x] 为 R18/R34/R50 补充官方 Paddle 权重与转换后 PyTorch 权重的 COCO 同图统一渲染及机器可读差异证据。
- [x] 增加 manifest 驱动的 Models CLI，支持发布状态列表、本地 size/SHA-256 校验和发布后的 HTTPS 原子下载。
- [ ] 由维护者确认 tag 后对外发布 wheel/sdist、三个检测权重、R18-vd backbone 初始化权重和 `SHA256SUMS`，并从公开 URL 回读验收。

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
- [x] 覆盖率报告与排除规则有文档，不把未支持继承模块排除后仍声称全包覆盖率。
- [x] Python 3.9–3.12 CPU CI 均能从干净环境安装并通过其声明的测试。
- [x] Paddle/PyTorch 性能报告记录硬件、软件、命令、warmup、采样、batch、dtype、吞吐、延迟和峰值内存。
- [x] wheel 安装后公开 CLI help、Infer、Eval 和 Export smoke 可重复；Models CLI 安装后合同纳入本轮验收。
- [x] 发布候选清单、法律文件、wheel/sdist 内容和本地模型文件 checksum 通过自动检查。
- [x] R18/R34/R50 同权重 COCO 单图对比使用统一渲染器，并保留原始预测和逐项匹配误差。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | Ruff 同时负责格式化和基础 lint，移除 Black | 维护者明确指定 Ruff；单一工具减少格式规则分叉 |
| 2026-07-19 | 第一批质量门禁只覆盖 M1–M5 直接维护面 | 全仓当前有 128 个格式文件和 293 个 lint 项，一次性机械改写不利于审计 |
| 2026-07-19 | Mypy 与 Ruff 分工，不把类型检查并入 lint | Ruff 不替代静态类型语义；初始全包 123 项快照需要通过当前重新审计和渐进清理校正 |
| 2026-07-19 | Mypy 以完整低错误目录为最小扩展单位 | `cli` 和 `conversion` 合计只有 4 项，可在不进行大范围继承代码改写的前提下审核并入门禁 |
| 2026-07-19 | 同时执行全包 42% 和直接维护范围 65% 的回退下限 | 干净 `test` extra 实测分别为 42.48% 和 65.56%；双范围能防止用排除低覆盖模块的方式制造虚假达标 |
| 2026-07-19 | 新增 metrics 活跃测试后将全包下限提高到 43% | 新测试使隐藏 GPU 的本地全包实测达到 43.94%，应将真实新增覆盖转换为回退约束 |
| 2026-07-19 | 新增 utils 活跃测试后将全包下限提高到 44% | 显式隐藏 GPU 的本地 CPU 实测达到 44.33%，新增覆盖应继续转成可执行回退约束 |
| 2026-07-19 | modeling 批次后将全包下限提高到 45% | 3 个实际边界回归使隐藏 GPU 的本地 CPU 实测达到 45.05%，新增覆盖应继续转成可执行回退约束 |
| 2026-07-19 | data 批次后完成全包 Mypy，并将覆盖率下限提高到 47% | 8 个 data 边界回归使隐藏 GPU 的本地 CPU 实测达到 47.09%；质量命令已无需逐模块范围清单 |
| 2026-07-19 | 新增 core schema 活跃测试后将直接维护范围下限提高到 66% | 显式隐藏 GPU 的本地 CPU 实测达到 66.89%；Typeguard 4 合法值/错误值路径已成为可执行回退证据 |
| 2026-07-19 | PyYAML stub 只进入 `quality`/`dev`，当时将 Paddle nightly 固定为 3.3.0.dev20251015 | 类型 stub 不属于核心运行时；宽松 Paddle nightly 约束曾解析到只有 aarch64 wheel 的版本。该 CPU nightly 决策后续已被稳定 GPU 构建取代 |
| 2026-07-19 | Python 3.9/3.10 使用兼容的 ONNX Runtime 上界 | ONNX Runtime 新版已分别停止提供 3.9/3.10 wheel；统一无上界会使干净矩阵无法安装 |
| 2026-07-19 | CI 核心矩阵不安装 `dev` extra | `test` extra 覆盖 Pytest、ONNX 与 ONNX Runtime，但不引入 Paddle；Paddle 对齐继续使用独立环境 |
| 2026-07-19 | 锁文件与 CI 统一使用 UV 0.11.29.x | UV 0.7 与 0.11 的锁文件修订语义不同；固定版本范围避免本地通过而托管 `--locked` 拒绝 |
| 2026-07-19 | `dev` extra 改用 PaddlePaddle GPU 3.3.0/cu118 | cu126 与当前 PyTorch cu121 的 `nvidia-nvtx-cu12` 强制版本冲突；cu118 可与 PyTorch cu121 并存，且同一 wheel 已验证 CUDA 执行和 CPU fallback |
| 2026-07-19 | 性能比值只作观测，不追求训练优化完全对齐 | R18 吞吐已超过 Paddle；CUDA 训练峰值显存约高 16% 的差距已记录，不值得在没有真实使用需求时为了指标完全一致而专项优化 |
| 2026-07-19 | GitHub Releases 作权重主托管，Hugging Face Model Hub 作可选镜像 | 四个权重均远低于 Release 的 2 GiB 单文件限制，可直接绑定源码 tag；Hub 用于 model card 和发现性，不把权重写入 `main` 历史 |
| 2026-07-19 | 权重下载严格由 manifest 发布状态和固定 HTTPS URL 驱动 | 未发布时显式失败比猜测 latest URL 更可审计；下载只在 size/SHA-256 通过后原子替换目标 |
| 2026-07-19 | Models CLI 与转换验证核心补测后将覆盖率下限提高到 48%/71% | 隐藏 GPU 的本地 CPU 实测达到全包 48.10%、直接维护范围 71.85%；新增回归同时修复额外输出字段和非有限值误判 |
| 2026-07-19 | 转换器与 YAML 配置补测后将覆盖率下限提高到 49%/80% | 16 个纯 CPU 回归使本地全包达到 49.41%、直接维护范围达到 80.67%，并修复配置对象共享状态和 mapping 静默覆盖 |
| 2026-07-19 | 用户可见边界补测后将覆盖率下限提高到 50%/84% | 17 个纯 CPU 回归使本地全包达到 50.08%、直接维护范围达到 84.93%，并拒绝布局误判、Infer 静默漏图和非法 Export 输入 shape |
| 2026-07-19 | 通用 Python 依赖锁定到官方 PyPI，保留 PyTorch/Paddle 专用索引 | 清华镜像对 Python 3.9 的 `zipp==3.23.0` artifact 返回 HTTP 403；重建锁文件没有改变包名或版本，干净 3.9 安装和回归通过 |
| 2026-07-19 | R18-vd backbone 与三个检测权重共用发布合同 | 官方训练复现依赖该初始化权重；manifest 现在为四个产物声明唯一 CLI alias，Models CLI 和发布检查共同覆盖 list/verify/download |
| 2026-07-19 | 四产物下载补测后将直接维护范围下限提高到 85% | 重复 alias、路径逃逸和 backbone 下载回归使本地直接维护范围达到 85.11%；新增发布边界应转化为回退约束 |
| 2026-07-19 | PyTorch 2.5.1/cu121 改用官方专用索引 | 南京镜像在 GitHub 六个 job 中均超过 45 分钟未完成 744 MiB Torch wheel；官方源空缓存完整安装为 2m21s，包版本不变 |

## 完成记录

进行中。第一批已移除 Black，锁定 Ruff `0.15.22`，增加独立 `quality` extra 和 `scripts/check_quality.py`。第二批将 Ruff format/lint 扩展到仓库根目录；排除规则仅保留只读 `third-party/`、历史 `tests/legacy/` 和生成目录，158 个活跃 Python 文件通过，默认全量测试为 `246 passed, 3 skipped, 6 warnings in 9.87s`。该批当时的 Mypy 门禁为 6 个 source file/目录，123 项全包数字为后续已校正的历史快照。

同批增加不含 Paddle 的 `test` extra 和 GitHub Actions workflow。四个 UV 隔离 CPU 环境已在隐藏 GPU 后本地验证：Python 3.9/3.10/3.11/3.12 均完成锁文件安装，分别为 `211 passed, 7 skipped, 17 deselected`；Python 3.9 使用 ONNX Runtime `1.19.2`，3.10 使用 `1.20.1`，3.11/3.12 使用 `1.27.0`。wheel 重装后五个 CLI `--help` 全部通过，Infer/Eval/Export 定向 smoke 为 `34 passed`。

GitHub Actions [run 29670978523](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29670978523) 在提交 `b2ffcff` 上完成托管验收：Python 3.9/3.10/3.11/3.12 四个 CPU jobs 均为 `211 passed, 7 skipped, 17 deselected`；质量 job 为 158 个 Ruff 文件和 6 个 Mypy source 通过；wheel job 成功构建并完成 `34 passed` smoke。首次 run 在测试前因 UV 0.7/0.11 锁文件修订差异失败，统一 `required-version` 并用 UV `0.11.29` 重建锁文件后关闭。该次 run 尚未包含后续的覆盖率门禁。

2026-07-19 覆盖率阶段以 Python `3.12.11` 和不含 Paddle 的独立 `test` extra 执行活跃测试，结果为 `216 passed, 7 skipped, 17 deselected, 6 warnings in 11.68s`。全包为 `5,605/13,195` 语句（`42.48%`），`cli/conversion/core/deploy` 为 `1,169/1,783`（`65.56%`）；`scripts/check_coverage.py` 强制 42%/65% 双下限，Python 3.12 CPU job 执行该门禁，临时 coverage 产物会自动清理。已安装 `dev` extra 的本机 `.venv` 观测为 `43.11%`，不作为 CI 下限。逐模块结果和限制见 [`docs/reports/coverage-validation.md`](../reports/coverage-validation.md)。90% 提升目标、完整 Mypy、CUDA CI、性能和发布候选仍未完成。

GitHub Actions [run 29671674073](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29671674073) 在提交 `19bcb60` 上通过全部 6 个 jobs。其中 Python 3.12 CPU coverage job 为 `216 passed, 7 skipped, 17 deselected`，全包 `5,606/13,195`（`42.49%`）和直接维护范围 `1,169/1,783`（`65.56%`）均通过门禁；其余 Python 3.9–3.11 CPU、Ruff/Mypy 质量和 wheel smoke jobs 也均通过。

Mypy 第二批将完整 `cli`、`conversion` 目录并入既有 `deploy` 和脚本门禁，共 17 个 source file 通过。清理的 4 项包含两个实际边界：训练 CLI 对未实现的半监督 teacher/student 权重显式失败，转换验证器对 Paddle/PyTorch 输出结构不一致显式失败；定向测试为 `112 passed`。同日全包重新审计为 235 项、39 个文件，历史 123 项仅保留为初始快照，不代表当前待办数量。

GitHub Actions [run 29672051076](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29672051076) 在提交 `6750f62` 上通过全部 6 个 jobs：质量 job 为 160 个 Ruff 文件和 17 个 Mypy source file 通过；Python 3.12 CPU 为 `217 passed, 7 skipped, 17 deselected`，覆盖率为全包 `42.50%` 和直接维护范围 `65.60%`；其余 Python 3.9–3.11 CPU 与 wheel smoke 也均通过。

Mypy 第三批仅为 `optimizer` 中的 LR multiplier 分组字典和已访问参数名列表补充可证明的元素类型，不改动优化器方程、参数组或更新顺序。完整 `optimizer` 并入后，统一门禁为 22 个 source file，定向测试为 `13 passed`；全包待办降为 233 项、38 个文件。

GitHub Actions [run 29672324668](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29672324668) 在提交 `9979863` 上通过全部 6 个 jobs：质量 job 为 160 个 Ruff 文件和 22 个 Mypy source file 通过；Python 3.12 CPU 为 `217 passed, 7 skipped, 17 deselected`，覆盖率维持全包 `42.50%` 和直接维护范围 `65.60%`；其余 Python 3.9–3.11 CPU 与 wheel smoke 也均通过。

Mypy 第四批清理 `metrics` 的 5 项类型错误，并新增 4 个活跃回归，覆盖 AP 积分、零 padding 裁剪、DetectionMAP classwise 累积和 COCO prediction-only JSON。完整 `metrics` 并入后，统一门禁为 27 个 source file，定向测试为 `17 passed`，默认全量为 `257 passed, 3 skipped`；全包 Mypy 待办降为 228 项、35 个文件。新测试使隐藏 GPU 的本地全包覆盖率提高到 `43.94%`，因此回退下限从 42% 同步提高到 43%。

GitHub Actions [run 29672789563](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29672789563) 在提交 `e22c2b4` 上通过全部 6 个 jobs：质量 job 为 161 个 Ruff 文件和 27 个 Mypy source file 通过；Python 3.12 CPU 为 `221 passed, 7 skipped, 17 deselected`，全包覆盖率 `43.95%`、直接维护范围 `65.60%`，其余 Python 3.9–3.11 CPU 与 wheel smoke 也均通过。

Mypy 第五批清理 `utils` 的 21 项类型错误，并新增 4 个活跃回归，覆盖 Paddle 风格嵌套 CLI override、自定义 argparse namespace、配置文件父目录创建、Pillow 默认字体回退和 checkpoint 目录路径。字体回归实际发现并修复了 NumPy `float32` RGB 传入 Pillow 会抛 `TypeError` 的问题。完整 `utils` 并入后，统一门禁为 36 个 source file，定向测试为 `25 passed`，默认全量为 `261 passed, 3 skipped`；全包 Mypy 待办降为 204 项、30 个文件。显式隐藏 GPU 的本地非 Paddle CPU 覆盖率为 `44.33%`，因此回退下限从 43% 提高到 44%。同批增加 `types-PyYAML` 质量 stub，并把 Paddle nightly 固定为已有迁移证据使用的 `3.3.0.dev20251015`；UV 0.11.29 的 `uv sync --extra dev --locked` 已在 Linux x86_64 重新通过。

GitHub Actions [run 29673364179](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29673364179) 在提交 `7ee602b` 上通过全部 6 个 jobs：质量 job 为 162 个 Ruff 文件和 36 个 Mypy source file 通过；Python 3.12 CPU 为 `225 passed, 7 skipped, 17 deselected`，全包覆盖率 `44.34%`、直接维护范围 `65.60%`，其余 Python 3.9–3.11 CPU 与 wheel smoke 也均通过。该 run 同时证明 required-environment 与固定 Paddle nightly 没有破坏不含 Paddle 的跨版本锁解析。

Mypy 第六批清理 `core` 的 19 项类型错误，并修复 Typeguard 4 调用签名仍按旧版传入参数路径、导致合法带注解值也被误判的问题。新增活跃回归同时覆盖合法 `int` 通过和错误 `str` 被拒绝；完整 `core` 并入后，统一门禁为 41 个 source file，定向测试为 `12 passed`，默认全量为 `262 passed, 3 skipped`。全包 Mypy 待办降为 185 项、28 个文件，仅余 `data` 87、`modeling` 73 和 `engine` 25。显式隐藏 GPU 的本地非 Paddle CPU 覆盖率为全包 `5,882/13,212`（`44.52%`）和直接维护范围 `1,202/1,797`（`66.89%`），因此直接维护范围回退下限从 65% 提高到 66%。

GitHub Actions [run 29673733080](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29673733080) 在提交 `230c6c3` 上通过全部 6 个 jobs：质量 job 为 162 个 Ruff 文件和 41 个 Mypy source file 通过；Python 3.9–3.12 CPU jobs 均为 `226 passed, 7 skipped, 17 deselected`，其中 Python 3.12 全包覆盖率为 `5,883/13,212`（`44.53%`），直接维护范围为 `1,202/1,797`（`66.89%`）；wheel smoke 为 `34 passed`。托管环境仍比本地多覆盖 1 条 `data` 语句，两个环境均通过新的 44%/66% 双门禁。

Mypy 第七批清理 `engine` 的 25 项类型错误，并修复四条实际运行边界：Trainer 在干净进程未触发 DataSet/Reader 注册、norm 重建误用 Paddle 的 `epsilon`/`set_state_dict` API、参数量日志对 Python float 调用 NumPy 方法，以及缺失的 FairMOT eval 图片收集方法。4 个新增活跃回归覆盖干净子进程注册、参数量日志、确定性 MOT 图片排序和 BatchNorm2d/LayerNorm/GroupNorm 状态重建；完整 `engine` 并入后，统一门禁为 47 个 source file，engine 单测为 `20 passed`，训练集成测试为 `12 passed`，默认全量为 `266 passed, 3 skipped`。全包 Mypy 待办降为 160 项、26 个文件，仅余 `data` 87 和 `modeling` 73。显式隐藏 GPU 的本地非 Paddle CPU 覆盖率为全包 `5,945/13,264`（`44.82%`）和直接维护范围 `1,202/1,797`（`66.89%`）；现有 44%/66% 门禁通过，但证据不足以把全包下限提高到 45%。

GitHub Actions [run 29674294425](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29674294425) 在提交 `7daad87` 上通过全部 6 个 jobs：质量 job 为 162 个 Ruff 文件和 47 个 Mypy source file 通过；Python 3.9–3.12 CPU jobs 均为 `230 passed, 7 skipped, 17 deselected`，其中 Python 3.12 全包覆盖率为 `5,946/13,264`（`44.83%`），直接维护范围为 `1,202/1,797`（`66.89%`）；wheel smoke 为 `34 passed`。托管环境仍比本地多覆盖 1 条 `data` 语句，两个环境均通过 44%/66% 双门禁。

Mypy 第八批清理 `modeling` 的 73 项类型错误，并修复三条实际运行边界：PyTorch `Tensor.max(dim)` 的返回元组被误作 Softmax 分数、无 neck 配置仍访问 `neck.out_shape`，以及空标注批次把缺失 denoising tensor 当作必有值。新增 3 个活跃回归覆盖 Softmax mask/实际 `bbox_num`、无 neck/aux head 构建和空标注无去噪回退；完整 `modeling` 并入后，统一门禁为 84 个 source file，建模与训练定向测试为 `60 passed, 5 skipped`，隐藏 GPU 的默认全量为 `264 passed, 8 skipped`。全包 Mypy 仅余 `data` 的 87 项。本机非 Paddle CPU 覆盖率为全包 `5,990/13,296`（`45.05%`）和直接维护范围 `1,202/1,797`（`66.89%`），因此全包回退下限从 44% 提高到 45%。

GitHub Actions [run 29674832957](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29674832957) 在提交 `c309c42` 上通过全部 6 个 jobs：质量 job 为 162 个 Ruff 文件和 84 个 Mypy source file 通过；Python 3.9–3.12 CPU jobs 均为 `233 passed, 7 skipped, 17 deselected`，其中 Python 3.12 全包覆盖率为 `5,991/13,296`（`45.06%`），直接维护范围为 `1,202/1,797`（`66.89%`）；wheel smoke 为 `34 passed`。托管环境仍比本地多覆盖 1 条 `data` 语句，两个环境均通过新的 45%/66% 双门禁。

Mypy 第九批清理 `data` 的 87 项类型错误，并删除逐模块临时范围清单；统一质量命令现对 100 个 package source 和 3 个质量/稳定性脚本执行 Mypy，本机实测 103 个 source file 无错误。审计同时修复了无 annotation 的 ImageFolder 错误构造 `COCO(None)`、SSOD loader 首次迭代未初始化、SSOD 固定 resize 返回未定义 `index`、VOC 必需 XML 字段缺失和 Pillow affine 新 API 等边界。8 个新回归覆盖这些数据加载/变换路径；data 与训练链定向为 `35 passed`，隐藏 GPU 的默认全量为 `272 passed, 8 skipped`。非 Paddle CPU 覆盖率为全包 `6,284/13,345`（`47.09%`）和直接维护范围 `1,202/1,799`（`66.81%`），因此全包回退下限从 45% 提高到 47%。本段为本机观测；托管证据如下。

GitHub Actions [run 29675617264](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29675617264) 在提交 `28ec38d` 上通过全部 6 个 jobs：质量 job 为 163 个 Ruff 文件和 103 个 Mypy source file 通过；Python 3.9–3.12 CPU jobs 均为 `241 passed, 7 skipped, 17 deselected`，其中 Python 3.12 全包覆盖率为 `6,285/13,345`（`47.10%`），直接维护范围为 `1,202/1,799`（`66.81%`）；wheel smoke 为 `34 passed`。托管环境比本地多覆盖 1 条 `data` 语句，两个环境均通过 47%/66% 双门禁。

M6 性能阶段在干净提交 `39e12b3` 上增加隔离框架进程的 JSON benchmark runner，统一质量命令现对 104 个 source file 执行 Mypy，Ruff 覆盖 165 个活跃 Python 文件。Paddle GPU 3.3.0/cu118 在同一 `.venv` 中通过 CUDA 与 CPU R18 前向；Paddle 标记测试为 `31 passed, 3 skipped`，安装 `dev` extra 的默认全量为 `286 passed, 3 skipped`。同一 RTX 3090/CPU 上的 R18 model-only 短采样表明，PyTorch 的 CPU/CUDA 推理吞吐分别为 Paddle 的 `2.061×`/`1.904×`，CPU/CUDA 训练 step 为 `1.528×`/`1.195×`；CUDA 训练 allocated 峰值显存约为 Paddle 的 `116%`，已定位到训练专属路径但不为完全对齐开启专项优化。完整协议、原始 JSON 和局限见 [`docs/reports/performance-validation.md`](../reports/performance-validation.md)。end-to-end DataLoader/profile、90% 覆盖率和发布验收仍未完成。

GitHub Actions [run 29676369588](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29676369588) 在提交 `39e12b3` 上通过全部 6 个 jobs：质量 job 为 165 个 Ruff 文件和 104 个 Mypy source file 通过；Python 3.9–3.12 CPU jobs 均为 `250 passed, 7 skipped, 17 deselected`，其中 Python 3.12 全包覆盖率为 `6,285/13,345`（`47.10%`），直接维护范围为 `1,202/1,799`（`66.81%`）；wheel smoke 为 `34 passed`。

发布加固阶段在提交 `dc09cd8` 增加 Apache-2.0/NOTICE、完整 checkpoint 产物清单、包内 config fallback 和 `scripts/check_release.py`。本地 `--require-models` 对 4 个 manifest 条目的 12 个文件/报告完成大小、SHA-256 和 mapping 数检查；干净 wheel 在仓库外安装后五个 CLI、包内 R18 config 和 `22,942,893` 参数模型构建通过。GitHub Actions [run 29678063506](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29678063506) 全部 6 个 jobs 通过：Ruff 167 文件、Mypy 105 source file、Python 3.9–3.12 均为 `254 passed, 7 skipped, 17 deselected`、Python 3.12 覆盖率 `47.14%/67.00%`，wheel smoke `34 passed`。详细证据见[`docs/reports/release-validation.md`](../reports/release-validation.md)。

可视化阶段先用官方 Paddle R18 权重、转换后 PyTorch R18 权重和 COCO `000000000139.jpg` 在 CPU/FP32 下分别生成原始预测，再使用同一脚本渲染；`score >= 0.3` 的两侧 30 个预测全部匹配，最大 score/框差为 `1.37e-6`/`9.16e-5 px`。随后按相同协议补齐 R34/R50，分别为 `31/31`、`28/28` 匹配，最大 score 差为 `3.78e-6`、`3.10e-6`，最大框差均为 `1.22e-4 px`。这些单图证据不替代完整 val2017 AP；只有 R18 已完成完整 val2017 门禁。详见[`docs/reports/prediction-visualization.md`](../reports/prediction-visualization.md)。对外 tag、Release assets 和公开 URL 回读尚未执行，因此 M6 仍为进行中。

GitHub Actions [run 29678700952](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29678700952) 在可视化提交 `118fd43` 上通过全部 6 个 jobs：质量 job 为 169 个 Ruff 文件和 106 个 Mypy source file；Python 3.9–3.12 均为 `255 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为 `47.14%/67.00%`；发布归档检查和 wheel smoke `34 passed`。

Models CLI 加固阶段新增 manifest 驱动的 `list/verify/download`：未发布状态没有 URL 时显式失败，发布状态只接受 HTTPS，下载写入目标同目录临时文件并在 size/SHA-256 通过后原子替换，不匹配的既有文件默认保留。纯 CPU 转换验证核心测试同时修复 PyTorch 额外输出字段未被比较、任一侧 NaN/Inf 可能被错误接受的边界。本机隐藏 GPU 的非 Paddle 测试为 `272 passed, 5 skipped, 34 deselected`，全包覆盖率 `6,511/13,535`（`48.10%`），直接维护范围 `1,429/1,989`（`71.85%`），双门禁提高到 48%/71%。

GitHub Actions [run 29679546423](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29679546423) 在提交 `e1a306b` 上通过全部 6 个 jobs：质量 job 为 Ruff 172 个文件和 Mypy 107 个 source file；Python 3.9–3.12 均为 `272 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为全包 `6,512/13,535`（`48.11%`）和直接维护范围 `1,429/1,989`（`71.85%`）；wheel 安装后六个 CLI、包内配置和 `34 passed` smoke 均通过。

转换器/YAML 配置批次把 Paddle-only 集成用例中的核心编排合同拆为 16 个纯 CPU 回归，覆盖 checkpoint 元数据、shape/transpose 传递、严格/宽松失败、内存释放、完整 session metadata、mapping 导出、批量失败隔离和 YAML 构造/序列化。回归修复 `Callable` 可变默认参数跨实例共享，以及批量转换在同名 mapping 已存在时未遵守 `--force` 的问题。本机非 Paddle CPU 为 `288 passed, 5 skipped, 34 deselected`，全包 `6,689/13,538`（`49.41%`），直接维护范围 `1,607/1,992`（`80.67%`），门槛提高到 49%/80%。

GitHub Actions [run 29680140237](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29680140237) 在提交 `3fbd4ec` 上通过全部 6 个 jobs：质量 job 为 Ruff 174 个文件和 Mypy 107 个 source file；Python 3.9–3.12 均为 `288 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为全包 `6,690/13,538`（`49.42%`）和直接维护范围 `1,607/1,992`（`80.67%`）；wheel 安装后六个 CLI 和 `34 passed` smoke 均通过。

用户可见边界批次将等元素数 shape 从“可自动 reshape”收紧为必须显式验证，只保留二维反转的已知 transpose 候选；Infer 对 `bbox_num` 的 rank、整数类型、非负性、总行数和 batch group 数执行完整检查，并用实际临时图片验证可视化与 JSON 主流程；Export 配置输入必须为整数且为正数的 `[3, H, W]`。本机隐藏 GPU 的定向测试为 `46 passed`，含 Paddle 的默认全量为 `336 passed, 8 skipped`，非 Paddle CPU 全量为 `305 passed, 5 skipped, 34 deselected`；全包 `6,790/13,557`（`50.08%`），直接维护范围 `1,708/2,011`（`84.93%`），门槛提高到 50%/84%。托管复验待本批提交后补录。

该批首次 [GitHub Actions run 29681330920](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29681330920) 的 Python 3.9 job 在测试前失败：锁文件中的清华 PyPI 镜像对 `zipp==3.23.0` 下载返回 HTTP 403，属于依赖源可用性问题，不是测试失败。通用依赖默认索引已改为官方 PyPI 并重建锁文件；包名和版本集合未变化，PyTorch cu121 与 Paddle cu118 的显式索引未改变。干净 Python `3.9.23` 环境完成 `uv sync --python 3.9 --locked --extra test`，确认 `zipp 3.23.0`、`torch 2.5.1+cu121` 可导入，非 Paddle 回归为 `305 passed, 7 skipped, 17 deselected`。临时环境已在验证后删除；新的托管复验将在修复提交后补录。

四产物发布合同批次消除了 manifest、发布检查器和 Models CLI 的范围分歧：三个检测权重使用 `r18/r34/r50`，R18-vd 训练初始化权重使用 `r18-backbone`，四个唯一 alias 全部由 manifest 声明。CLI 同时读取 `models/pretraining`，拒绝重复或非法 alias、路径逃逸、状态/URL 不一致，并为 backbone 复用现有 HTTPS 临时下载、size/SHA-256 校验和原子替换。14 项发布定向回归、Ruff `174` 文件、Mypy `107` source file、四个真实本地权重校验和 `scripts/check_release.py --require-models` 均通过；临时 wheel/sdist 在仓库外从包内 manifest 列出四个 alias。隐藏 GPU 的非 Paddle CPU 回归为 `308 passed, 5 skipped, 34 deselected`，全包 `6,802/13,567`（`50.14%`），直接维护范围 `1,720/2,021`（`85.11%`），门槛调整为 50%/85%。所有临时发布产物在验证后清理；托管证据待提交后补录。

依赖源复验进一步定位了托管安装停滞：run `29681801681` 的六个 job 均已完成 NVIDIA CUDA 依赖和 torchvision 下载，只有南京 PyTorch 镜像的 `torch 2.5.1+cu121` 在 45 分钟后仍未完成，随后因新提交被取消。PyTorch 专用索引已切换为官方 `download.pytorch.org/whl/cu121`，重新锁定不改变包名或版本。独立空 UV 缓存、临时 Python `3.9.23` 环境从零准备 52 个包耗时 `2m21s`，版本/CUDA 检查通过，非 Paddle 回归为 `308 passed, 7 skipped, 17 deselected`；临时环境和缓存已清理。托管复验待本批提交后补录。

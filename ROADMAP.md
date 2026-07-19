# RT-DETRv3 PyTorch Migration Roadmap

**Status**: Active
**Last updated**: 2026-07-19
**Current evidence snapshot**: [`docs/plans/2026-07-18-migration-status.md`](docs/plans/2026-07-18-migration-status.md)
**Latest completed execution plan**: [`M10——TorchScript CUDA/CPU 推理计划`](docs/plans/2026-07-19-m10-torchscript-cuda-inference.md)
**Current execution plan**: [`M11——ONNX Runtime CUDA/CPU 推理计划`](docs/plans/2026-07-19-m11-onnx-runtime-cuda-inference.md)；M4 长训保持 deferred

本路线图以未完成的迁移大纲为主，并保留已完成里程碑的验收摘要。“完成”必须有当前代码、可复现命令和实际验收结果，不以历史 `specs/` 勾选状态为准。

## 目标

建立一个可安装、可训练、可评估、可恢复、可转换官方 Paddle 权重，并能在同一数据与硬件上给出数值、精度和性能对齐证据的 RT-DETRv3 PyTorch 训练库。

## Milestone 1 — 打通当前 API 的最小训练链（P0）

- [x] 使用当前 `workspace` 和 `configs/rtdetrv3/*.yml` 构建 R18 完整模型，不依赖 `tests/legacy/` API。
- [x] 使用最小 COCO fixture 构建 Dataset/DataLoader，验证一个 batch 的字段、shape、dtype、bbox 和 padding。
- [x] 完成一次训练态前向，输出所有 loss 分项且无 NaN/Inf。
- [x] 完成一次反向与 optimizer step，验证关键组件有有限梯度且参数实际更新。
- [x] 将 M1 所需的 backbone、head、loss、post-process、配置构建和模型集成场景按当前 API 重写回活跃测试集。
- [x] 在 CPU 上运行 5 iteration 的短训练烟雾测试。

**Exit criteria**: 一条可在 CI/开发机重复的 config → data → model → loss → backward → optimizer 链路，关键回归不再依赖旧 Registry 或旧 builder。

**验收记录**：2026-07-18 本机 CPU/float32 验证通过，全量测试 `127 passed, 1 skipped`。完整环境、override 和首末 step 数据见 M1 执行计划。

**完成提交**：`f95f4e0`。

## Milestone 2 — 官方权重转换与分层数值对齐（P0）

- [x] 下载并记录 R18/R34/R50 官方 Paddle checkpoint 的来源、checksum 和配置。
- [x] 对每个变体导出名称映射和未映射清单，审核 Linear 转置、BatchNorm 状态和特殊 head 参数。
- [x] 使转换后的权重以受控的 missing/unexpected key 集合加载到完整 PyTorch 模型。
- [x] 在 CPU/float32 上按 backbone → neck → transformer → head 比较第一个分歧激活。
- [x] 对齐预测、loss 分项和整体梯度方向；记录优化器差异，不要求 AdamW 更新逐元素一致。
- [x] 实现并测试批量转换、失败隔离、转换汇总与可选低内存模式。

**完成证据**：R18/R34/R50 已完成官方来源/SHA-256、共 2,041 个 tensor 的目标感知转换与逐值校验、受控加载，以及单线程 CPU/float32 的分层 eval、确定性缩减训练 loss 和整体梯度方向验收。批量 CLI 已覆盖目录/glob、自动输出命名、失败后继续、JSON 汇总和原子发布；R18 低内存严格转换为 571/571，峰值 RSS 观测为 `925,780 KiB`。R50 后处理保留 2/300 个 top-k 离散边界记录。当时移交给 M3 的 optimizer LR multiplier 缺口已在 M3 阶段 1 修复。

**Exit criteria**: 三个官方变体均有可重复转换命令、映射报告、加载结果和分层数值报告。

**验收记录**：2026-07-18 本机验证通过；默认测试 `144 passed, 3 skipped`，完整证据见 M2 执行计划。

## Milestone 3 — 训练、评估与恢复可用（P0）

- [x] 验证 R18 完整 1 epoch 训练和 COCO val2017 评估。
- [x] 核对 optimizer 参数组、ResNet `lr_mult_list=0.1`、weight decay 规则、warmup/decay step 单位、梯度裁剪与 EMA 顺序。
- [x] 在真实 COCO 短训练中验证 AMP 与 float32 的 loss/梯度有限性，记录 scaler 溢出与跳步。
- [x] 验证 checkpoint 恢复包含 model、EMA、optimizer、scheduler、scaler、epoch/global-step 和 RNG 状态。
- [x] 对比连续训练与中断恢复后紧接的 LR、loss 和参数更新。
- [x] 在 2 GPU 上验证 DDP sampler、SyncBN、同步梯度/scaler 和 rank-0 checkpoint 写入。
- [x] 实现并验证梯度累积与 DDP `no_sync()` 边界。

**完成证据**：R18 以 2-GPU AMP、每卡 batch 8 完成 train2017 1 epoch，checkpoint 记录 7319 次有效更新并通过完整 val2017，bbox AP/AP50/AP75 为 `0.468/0.643/0.504`。真实双卡短训练另行验证 `accumulate_steps=2`、DDP `no_sync()`、跨 rank 日志 loss 均值、EMA、非整窗口和 rank-0 应用日志/checkpoint。该结论只关闭训练库可用性，不代替 M4 的标准 schedule、多 seed 和 Paddle AP 对照。

**Exit criteria**: Train/Eval CLI 能用真实配置稳定运行，并有恢复一致性与 DDP 集成测试。

**验收记录**：2026-07-18 本机验证通过；隐藏 GPU 的默认测试 `165 passed, 8 skipped`，CUDA 定向文件另行 `8 passed`。

## Milestone 4 — COCO 精度与稳定性对齐（P1）

**执行计划**：[`M4——COCO 精度与稳定性对齐计划`](docs/plans/2026-07-18-m4-coco-accuracy-stability.md)。同一官方 R18 checkpoint 的 Paddle/PyTorch 完整 val2017 gate 已完成；72 epoch、多 seed 与 R34/R50 长训因时间成本暂缓，恢复前需要新的明确决策。

**当前进度**：R18 官方同权重 CPU/FP32 完整 val2017 gate 已通过：Paddle/PyTorch 精确 AP 分别为 `0.480477300367/0.480477134768`，绝对差 `1.65599e-7`；score `>=0.3` 的 `53780` 个 prediction 全部匹配。显式 seed、官方 ImageNet R18-vd backbone 初始化、2-GPU AMP+EMA 协议、EMA Eval CLI 和 DDP per-rank RNG checkpoint 已通过定向测试与真实双卡烟测。seed 0 已在 3 epoch 边界原子保存并主动停止；checkpoint 的 model/optimizer/EMA tensor 全部有限，恢复状态完整，但该探针不作标准 schedule、单 seed 精度或多 seed 稳定性证据。仓库已提供单 `model + seed` 的社区执行脚本，长训通过 GitHub Issue #3 分片认领，结果需按 commit、协议和 checksum 审核后才能进入统计。

- [ ] 对 R18 完成标准训练 schedule，保存环境、命令、配置、日志和 checkpoint。（deferred）
- [x] 与对应 Paddle 基线比较 AP/AP50/AP75/APs/APm/APl；R18 同设备主 AP 绝对差 `1.65599e-7`，通过 `0.5 AP` 目标。
- [ ] 在 R18 通过后依次验证 R34 和 R50，不在未定位数值差异时同时展开多变体训练。（deferred）
- [ ] 至少使用 3 个 seed 记录均值和方差；发布验收扩展到 5 个 seed。（deferred）
- [x] 生成 `docs/reports/accuracy-validation.md`，明确区分训练误差和框架实现缺陷。

**Exit criteria**: 每个声称支持的模型变体都有可复现的精度报告和可获取的权重。

## Milestone 5 — 配置、CLI 与导出边界（P1）

**执行计划**：[`M5——配置、CLI 与导出边界计划`](docs/plans/2026-07-19-m5-cli-export-boundaries.md)。Infer eager 基线复用 TestReader、batch dict、模型内置 `bbox/bbox_num` 后处理和 Eval checkpoint 加载规则；官方 R18 已完成 CPU/FP32 真实 COCO 单图、batch 4 和 608/640 输入验证。workspace 冲突优先级、连续配置隔离、RT-DETRv3 Paddle YAML 支持矩阵和五个公开 CLI contract 均已有活跃测试与指南。ONNX opset 17/ONNX Runtime CPU 和 traced TorchScript 已验证固定高宽、动态 batch 1/4/8；这不代表单产物动态高宽、全部 Paddle 参数、其他模型或其他 provider 已支持。

- [x] 为 `workspace` 补充 shared/inject/from_config/显式参数冲突和全局状态隔离测试。
- [x] 明确哪些 Paddle YAML 字段直接兼容、哪些映射、哪些不支持，补充配置迁移指南。
- [x] 为 Train/Eval/Infer/Convert 编写 CLI contract 测试，对 Paddle 参数的兼容差异做显式文档化。
  - [x] Infer 已覆盖参数校验、当前/历史参数拼写、TestReader 预处理、batch dict、`bbox/bbox_num`、阈值、JSON 和官方 R18 真实推理。
  - [x] Train/Eval/Convert 已覆盖 help、主参数、main wiring 和错误路径；既有 M2–M4 真实运行作为端到端证据。
- [x] 完成 ONNX 导出与 ONNXRuntime 回归，记录不支持的动态控制流/算子。
- [x] 完成 TorchScript 导出与重新加载回归。
- [x] 验证输入尺寸、动态 batch 1/4/8 和空阈值结果边界；单产物空间 shape 明确不动态。

**Exit criteria**: 所有面向用户的入口都有功能测试，支持边界和框架差异有文档。

**验收记录**：2026-07-19 本机 CPU/FP32 验证通过；最终默认测试 `237 passed, 8 skipped`，wheel 包含五个 console entry point。详细版本、容差和导出限制见 M5 计划与 CLI/导出迁移文档。

## Milestone 6 — 性能、质量与发布（P2）

**执行计划**：[`M6——性能、质量与发布计划`](docs/plans/2026-07-19-m6-performance-quality-release.md)。2026-07-19 初始快照为全包 45% 语句覆盖率、128 个待 Ruff 格式化文件、293 项默认 Ruff lint 和 123 项 Mypy 全包错误。当前 Ruff/Mypy 已扩展到全部活跃 Python 范围和纳入门禁的仓库脚本；最新托管非 Paddle CPU 覆盖率为全包 51.42%、直接维护范围 90.45%，回退下限保持 50.5%/90%，直接维护范围的 90% 目标已有托管证据。Python 3.9–3.12 CPU CI、本机 CUDA 运行和 R18 同机 CPU/CUDA model-only 与真实 COCO 端到端性能证据均已通过。PyTorch 四个 model-only workload 吞吐均高于 Paddle，CUDA 训练峰值 allocated 显存约高 16%；COCO 端到端推理吞吐为 Paddle 的 1.579×，可见 input-pipeline stall 占 29.68%，按维护者决策只记录差异而不追求完全对齐。发布候选的许可、清单、wheel/sdist、包外安装和模型 checksum 已验证，11 个上传资产已支持单命令原子组装、拒绝覆盖和失败清理；三个检测权重与 R18-vd backbone 初始化权重已纳入统一发布合同，R18/R34/R50 的 Paddle 原权重/PyTorch 转换权重均完成 COCO 同图统一渲染和机器可读差异报告。维护者已确认并发布 `v0.1.0`；11 个固定 tag 资产已通过匿名公开回读，公开下载的 R18 权重也已通过 CPU Infer/Eval 链路冒烟。详见[性能报告](docs/reports/performance-validation.md)、[发布报告](docs/reports/release-validation.md)和[预测可视化报告](docs/reports/prediction-visualization.md)。

- [x] 在同一硬件、驱动、batch 和精度下建立 Paddle/PyTorch 基准；两个官方 wheel 的 CUDA/cuDNN 版本不同，已分别记录而不声称完全同运行时。
- [x] 记录训练吞吐、推理延迟、峰值显存、DataLoader 占比和关键算子 profile。
  - [x] 完成 model-only 训练吞吐、推理延迟和峰值内存证据。
  - [x] 使用真实 COCO val2017 补充 R18 CUDA 端到端推理、可见 input-pipeline stall 和双框架单次算子 profile。
- [x] 评估训练吞吐不低于 Paddle 的 95% 和峰值显存不超过 110% 目标：吞吐通过，CUDA 训练显存约为 116%，已缩小到训练专属路径并记录不专项优化的决策。
- [x] 引入统一 lint/format/type-check 命令，清理当前 mypy 和 API 注解缺口。
  - [x] 首批 Ruff format/lint 覆盖 `cli`、`conversion`、`core`、`deploy`、`scripts` 及对应单测；Mypy 首批 6 个 source file/目录通过。
  - [x] Ruff format/lint 扩展到全部活跃 Python 文件并移除临时范围清单。
  - [x] Mypy 扩展到完整 `cli`、`conversion`、`deploy` 和 3 个质量/稳定性脚本，17 个 source file 通过。
  - [x] Mypy 继续扩展到完整 `optimizer`，累计 22 个 source file 通过。
  - [x] Mypy 扩展到完整 `metrics`，累计 27 个 source file 通过。
  - [x] Mypy 扩展到完整 `utils`，累计 36 个 source file 通过。
  - [x] Mypy 扩展到完整 `core`，累计 41 个 source file 通过。
  - [x] Mypy 扩展到完整 `engine`，累计 47 个 source file 通过。
  - [x] Mypy 扩展到完整 `modeling`，累计 84 个 source file 通过。
  - [x] Mypy 扩展到完整 `data`，删除临时范围清单；全部 100 个 package source 与 3 个脚本通过。
- [x] 生成覆盖率报告，将定义明确的直接维护范围有效覆盖率提升到 90% 目标。
  - [x] 记录全包和逐模块基线，建立并将全包回退下限从 42% 逐步提高到 50%，`cli/conversion/core/deploy` 提高到 85%。
  - [x] 为 Convert/Export 编排和转换输出对齐路径补测，本机直接维护范围达到 90.80%，阈值提高至 90%。
- [x] 建立 Python 3.9–3.12 与 CPU/主要 CUDA 组合的 CI 矩阵。
  - [x] 增加 Python 3.9–3.12 非 Paddle CPU workflow，并在本机 UV 隔离环境验证锁文件安装和测试。
  - [x] GitHub 托管 Python 3.9–3.12 CPU jobs、质量门禁和 wheel smoke 通过。
  - [x] 补充受控 CUDA job 或自托管验证证据。
- [x] 生成模型清单、checksum、配置、许可说明和发布候选验证报告。
- [x] 对 R18/R34/R50 转换权重生成同一 COCO 图片、统一渲染器和机器可读匹配证据。
- [x] 增加 manifest 驱动的 Models CLI，支持发布状态、本地 checksum 校验和发布后 HTTPS 原子下载。
- [x] 将四份 mapping report 的 size/SHA-256 固化到 manifest，并完成 11-asset 扁平 Release 目录的严格回读预演。
- [x] 将四权重、四报告、wheel/sdist 和 `SHA256SUMS` 收敛为单命令原子暂存，并用真实 11-asset 目录严格回读。
- [x] 确认 `v0.1.0`，从本地 annotated tag 的 detached 工作树重建并校验实际 11-asset 上传目录。
- [x] 以固定 tag 公开发布三个检测权重、R18-vd backbone 初始化权重、四份 mapping report 和 `SHA256SUMS`，并从公开 URL 回读验收。

**Exit criteria**: 安装、测试、训练、评估、导出和模型获取都有可重复发布流程。

**验收记录**：2026-07-19，annotated tag `v0.1.0` 指向 `c0317ef`，GitHub Release 的 11 个公开资产完成匿名下载与双重 checksum 回读；公开 R18 权重完成 Models CLI 下载、单图 Infer 和四图 COCO Eval 冒烟。提交 `80d2a80` 的最终 GitHub Actions 6 个 job 全部通过。

## Milestone 7 — 公开模型多变体运行时验收（P1）

**执行计划**：[`M7——公开模型多变体运行时验收计划`](docs/plans/2026-07-19-m7-variant-runtime-validation.md)。本阶段只补发布后三个检测模型的用户侧 eager 运行证据，不恢复 M4 长训，也不把小样本 Eval 指标作为正式 AP。

- [x] 使用公开 R18 asset 完成 Models CLI 下载、真实 COCO 单图 Infer 和四图 Eval 冒烟。
- [x] 使用公开 R34 asset 完成相同 CPU/FP32 运行时验收。
- [x] 使用公开 R50 asset 完成相同 CPU/FP32 运行时验收。
- [x] 更新发布后局限、运行矩阵和可复现证据，清理全部测试产物。

**Exit criteria**: 三个已发布检测权重均能从固定 tag URL 下载并通过 checksum，使用各自配置严格加载，完成真实图片 Infer 和同一 COCO 小样本 Eval；报告明确不外推为 R34/R50 完整 AP、训练收敛或导出支持。

**验收记录**：2026-07-19，R34/R50 公开下载的 size/SHA-256 与 manifest 一致；同一 COCO 单图分别生成 `31/28` 条检测和可解码图片，同一四图子集 Eval 均写出 1,200 条候选。未发现变体专属实现故障；完整证据见[公开模型多变体运行时报告](docs/reports/variant-runtime-validation.md)。

## Milestone 8 — R34/R50 多变体导出验收（P1）

**执行计划**：[`M8——R34/R50 多变体导出验收计划`](docs/plans/2026-07-19-m8-variant-export-validation.md)。复用 M5 已建立的 tensor-only 导出和严格输出合同，只扩展另两个已发布检测变体。

- [x] 导出 R34 的 ONNX opset 17 和 traced TorchScript，完成 CPU 重载与输出回归。
- [x] 导出 R50 的 ONNX opset 17 和 traced TorchScript，完成 CPU 重载与输出回归。
- [x] 对两变体验证固定 640 下 batch 1/4/8 和真实 COCO 输入；继续明确空间 shape 非动态。
- [x] 修复近似并列候选行序导致的验证误判，记录产物、误差、警告和限制，并清理全部中间产物。

**Exit criteria**: R34/R50 两种格式均通过 checker/reload，并在 CPU/FP32、640×640 的 batch 1/4/8 和真实输入上满足 `bbox_num`/分组严格一致、每图全部候选按类别/score/box 一对一匹配、score `<=2e-5`、坐标 `<=0.02 px` 的合同。

**验收记录**：2026-07-19，R34/R50 ONNX 最大 score/box 误差分别为 `9.4771e-6/0.011780 px` 和 `1.8962e-5/0.005615 px`；TorchScript 本次逐值为 0。ONNX 近似并列低分候选最多每图重排 2 行，但 300/300 候选均完成同图唯一匹配。详见[多变体导出验证报告](docs/reports/variant-export-validation.md)。

## Milestone 9 — 导出产物端到端推理（P1）

**执行计划**：[`M9——导出产物端到端推理计划`](docs/plans/2026-07-19-m9-exported-inference.md)。在不复制预处理和展示逻辑的前提下，让 Infer CLI 直接消费 M5/M8 已验证的 ONNX/TorchScript tensor 合同。

- [x] Infer 模型源互斥接受 checkpoint、ONNX 和 TorchScript，保留原 checkpoint 用法。
- [x] 导出后端复用同一 TestReader、batch、阈值、JSON、类别映射和可视化路径。
- [x] 对 CPU-only、EMA 和固定尺寸边界做显式参数校验。
- [x] 使用 R18 真实 COCO 图片完成 eager/ONNX/TorchScript 用户侧输出对照并记录限制。

**Exit criteria**: 安装后的 `rtdetrv3-infer` 可使用三种模型源；R18 真实图片三后端在同一预处理和阈值下满足 M8 每图全候选数值合同，并均生成可解码可视化与机器可读 JSON。

**验收记录**：2026-07-19，R18 同一真实图在 checkpoint/ONNX/TorchScript 下均输出 30 条阈值后检测；ONNX 相对 eager 的最大 score/框误差为 `1.49e-6/9.16e-5 px`，TorchScript 为 0，三份渲染图字节一致。640 产物与 608 预处理的负例在执行前明确失败。提交 `545578a` 的 [GitHub Actions run 29689593612](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29689593612) 六个 job 全部通过。详见[导出产物推理报告](docs/reports/exported-inference-validation.md)。

## Milestone 10 — TorchScript CUDA/CPU 推理（P1）

**执行计划**：[`M10——TorchScript CUDA/CPU 推理计划`](docs/plans/2026-07-19-m10-torchscript-cuda-inference.md)。M10 当时保持 ONNX Runtime CPU provider 边界，只把 PyTorch 自身可执行的 TorchScript module 扩展到 CUDA/CPU 双设备。

- [x] TorchScript 在 CUDA 可用时默认 GPU，无 CUDA时自动回退 CPU，并支持显式设备。
- [x] ONNX 继续拒绝非 CPU provider，不把 PyTorch CUDA 可用性外推给 ONNX Runtime。
- [x] 使用 R18 四张真实 COCO 图片、batch 4 对比 eager CUDA、TorchScript CUDA 和 TorchScript CPU。
- [x] 记录设备、数值误差、固定尺寸、JSON/可视化和剩余 provider 边界。

**Exit criteria**: TorchScript Infer 在真实 CUDA 与 CPU fallback 上均完成四图 batch，并与 eager CUDA 满足记录的每图输出合同；M10 验收时 ONNX 的 CPU-only 限制保持显式。

**验收记录**：2026-07-19，实现提交 `85b956d`。四条 eager/TorchScript × CUDA/CPU 路径均输出 `[30,1,25,2]` 条检测；TorchScript CUDA 相对 eager CUDA 最大 score/box 误差为 `2.79218e-4/0.00872803 px`，TorchScript CPU 相对 eager CPU 为 `1.90735e-6/9.15527e-5 px`，同设备渲染全部字节一致。跨设备两条近似候选换序单独记录，不全局放宽同设备门槛。本地非 Paddle 全仓 `353 passed, 7 skipped, 34 deselected`，覆盖率 `51.45%/90.46%`，Ruff/Mypy 通过。提交 `f8b7439` 的 [GitHub Actions run 29690660612](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29690660612) 六个 job 全绿；Python 3.9–3.12 均为 `353 passed, 9 skipped, 17 deselected`，托管覆盖率 `51.46%/90.46%`，wheel smoke `60 passed`。详见[TorchScript 设备验证报告](docs/reports/torchscript-device-validation.md)。

## Milestone 11 — ONNX Runtime CUDA/CPU 推理（P1）

**执行计划**：[`M11——ONNX Runtime CUDA/CPU 推理计划`](docs/plans/2026-07-19-m11-onnx-runtime-cuda-inference.md)。保持 ONNX 默认 CPU 和托管 CPU CI，只为显式设备选择增加 CUDA provider，并隔离 CPU/GPU ORT distributions。

- [x] ONNX 默认 CPU，显式 `--device cuda[:id]` 选择 `CUDAExecutionProvider` 并保留 CPU 算子回退。
- [x] provider 缺失或 session 完全静默降级时明确失败并给出 GPU extra 安装指引。
- [x] CPU `export`/`test` 与 GPU `dev`/`export-gpu` extras 可分别安装，冲突组合由 UV 拒绝。
- [x] 使用 R18 四图 batch 4 完成 ONNX/eager × CUDA/CPU 的同设备数值、JSON 与可视化对照。
- [ ] 更新 provider 支持矩阵、复现命令和未覆盖边界，清理全部中间产物。

**Exit criteria**: ONNX Infer 可在显式 CUDA 和 CPU fallback 上完成真实图片 batch 推理；两个 provider 分别与同设备 eager 满足记录的每图输出合同，默认 CPU 用户和 Python 3.9–3.12 CPU CI 不回归。

## 依赖顺序

```text
M1 最小训练链
 ├──> M2 权重/数值对齐 ──> M4 精度对齐
 └──> M3 训练/评估/恢复 ─┘
M1–M3 ──> M5 CLI/导出
M4–M5 ──> M6 性能与发布 ──> M7 公开模型运行时矩阵 ──> M8 多变体导出 ──> M9 导出产物推理 ──> M10 TorchScript CUDA ──> M11 ONNX Runtime CUDA
```

## 不作为当前阻塞的延伸项

- TensorRT 引擎专项优化。
- C++ libtorch 完整示例。
- 剪枝、量化、NAS 或新模型架构。
- Paddle 中与 RT-DETRv3 训练/评估无关的所有检测任务分支。

# M12——R34/R50 导出后端设备矩阵计划

> 历史计划快照（2026-07-19，M12）：本文保存已完成执行记录，不代表当前仓库状态。当前合同见 [`docs/models/rtdetrv3`](../../../models/rtdetrv3/README.md)。

- 状态：`completed`
- 创建日期：`2026-07-19`
- 最后更新：`2026-07-19`
- 负责人：`Codex / repository maintainers`

## 背景

M8 已验证 R34/R50 的 ONNX 与 TorchScript 在 CPU 上可导出和重载，M10/M11 又分别验证了 R18 的 TorchScript 与 ONNX CUDA Infer。当前公开模型矩阵仍缺 R34/R50 在 CUDA 上的用户侧证据，因此不能把 R18 的设备结论直接外推给更深变体。本阶段复用现有 CLI 与四图协议，关闭两个导出后端的多变体设备缺口。

## 目标与非目标

### 目标

- 从 `v0.1.0` R34/R50 转换 checkpoint 分别导出固定 640、动态 batch 的 ONNX 与 TorchScript。
- 对每个变体运行 eager、ONNX、TorchScript × CUDA、CPU 六条 Infer 路径，保持同一 checkpoint、预处理、输入、batch、阈值、eval 模式、FP32 和设备内参考。
- 严格检查每图检测数、类别、score、box、JSON 与可视化；近似并列候选允许一对一配对后记录换序，不允许遗漏、跨图匹配或超出门槛。
- 记录环境、权重与临时导出 checksum、实际 provider、数值误差、可视化差异、警告和未覆盖边界。

### 非目标

- 不恢复 M4 标准 schedule、多 seed 或 R34/R50 完整 val2017 长训。
- 不引入 I/O Binding、TensorRT、动态高宽、AMP/FP16、量化、C++/mobile 或性能排名。
- 不把 CPU/CUDA 跨设备差异作为同设备后端合同，也不要求可视化文件逐字节一致。
- 不发布临时 ONNX/TorchScript 产物，也不改动 `v0.1.0` Release。

## 实施步骤

- [x] 核对公开 R34/R50 checkpoint checksum、GPU/ORT provider 与四图输入 checksum。
- [x] 导出两变体的 ONNX/TorchScript，记录 checker、重载、动态 batch 和固定空间尺寸证据。
- [x] 运行两变体的 eager/ONNX/TorchScript × CUDA/CPU 六路径四图 batch 4 Infer。
- [x] 按图、类别和数值门槛一对一比较 JSON，并检查全部渲染图可解码、尺寸一致。
- [x] 更新支持矩阵、报告和 ROADMAP，执行门禁并清理临时导出、输出、日志与测试缓存。

## 依赖

- 仓库 UV `.venv` 的 GPU `dev` 环境、PyTorch CUDA 与 ONNX Runtime `CUDAExecutionProvider`/`CPUExecutionProvider`。
- `v0.1.0` R34/R50 转换 checkpoint 及 `configs/checkpoints/rtdetrv3_coco.yml` 中的 size/SHA-256 真值。
- COCO val2017 固定四图 `139/285/632/724` 与 annotation checksum。
- M8 的全候选一对一匹配规则，以及 M10/M11 的同设备 CUDA/CPU 误差合同。

## 风险与回退

- 风险：更深 backbone 可能累积比 R18 更大的 CUDA 舍入误差。缓解：先沿用既有门槛；失败时比较同设备第一个分歧输出或中间激活，不为通过而事后放宽。
- 风险：高显存占用或多个 runtime 同进程保留显存。缓解：每条 CLI 使用独立进程，batch 固定为 4，并串行运行变体。
- 风险：近似并列 top-k 候选跨后端换序。缓解：按图、同类别、score/box 构造一对一匹配，换序单独计数但不忽略任何候选。
- 回退：若某个变体或后端失败，保留已验证矩阵项，明确记录失败边界；不改变 M8 已验证的 CPU 导出合同。

## 验收

- [x] 两个 checkpoint 与既有 SHA-256 真值一致；ONNX session 回读实际 CUDA/CPU providers，TorchScript 映射到请求设备。
- [x] 每个变体的六条路径均完成四图 batch 4，并生成合法 JSON 与四张可解码图片。
- [x] 执行预注册 ONNX 门槛：CPU 延续 `score <= 2e-5`、`box <= 0.02 px` 并通过；CUDA 的 R18 `1e-3/0.03 px` 外推门槛未通过，实际偏差与 A/B 诊断已明确记录，不改写成已通过。
- [x] TorchScript CPU 使用 `score <= 2e-5`、`box <= 0.02 px`；TorchScript CUDA 使用 `score <= 5e-4`、`box <= 0.02 px`。两变体四项均与同设备 eager 逐值一致。
- [x] Ruff/Mypy、相关回归和非 Paddle 全仓门禁通过；临时产物清理，`.venv` 保留。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | ONNX 与 TorchScript 一次补齐 R34/R50 CUDA/CPU 矩阵 | 两个后端存在同层级的 R18-only 证据缺口，分拆会留下已知空洞并重复四图协议 |
| 2026-07-19 | 每个设备使用同设备 eager 参考 | CPU/CUDA 算法和 TF32 舍入不同，跨设备逐值比较不能证明导出后端正确性 |
| 2026-07-19 | 初始门槛继承 M8/M10/M11 | 先使用已有验证合同；若更深变体失败，应定位原因而不是按结果反向设门槛 |
| 2026-07-19 | 不把 `2e-3/0.05 px` 观测包络升级为全局门槛 | R34/R50 ONNX CUDA 超出 R18 门槛；TF32 与 cuDNN 算法 A/B 未给出更可靠设置，应保留失败证据和 provider 局限 |

## 完成记录

2026-07-19 完成，证据提交 `fc3a6f8`。R34/R50 各六条路径均运行；TorchScript CUDA/CPU 逐值一致，ONNX CPU 通过既有门槛，ONNX CUDA 的 R18 门槛外推失败及 TF32/cuDNN A/B 已记录。非 Paddle 本地全仓 `358 passed, 7 skipped, 34 deselected`，覆盖率 `51.48%/90.50%`，Ruff/Mypy、构建、发布检查通过，临时产物已清理。[GitHub Actions run 29693029694](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29693029694) 六个 job 全绿；Python 3.9–3.12 均为 `358 passed, 9 skipped, 17 deselected`，托管全包/直接维护范围为 `7,079/13,748 (51.49%)` 和 `1,991/2,200 (90.50%)`，Ruff `174` 个文件、Mypy `107` 个 source file、发布检查和 `65 passed` wheel smoke 同时通过。

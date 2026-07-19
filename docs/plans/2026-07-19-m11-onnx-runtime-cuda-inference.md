# M11——ONNX Runtime CUDA/CPU 推理计划

- 状态：`completed`
- 创建日期：`2026-07-19`
- 最后更新：`2026-07-19`
- 负责人：`Codex / repository maintainers`

## 背景

M9 已让 Infer CLI 直接消费 ONNX，M10 又验证了 TorchScript CUDA/CPU 双设备；但当前 ONNX 路径仍固定为 `CPUExecutionProvider`。开发机具备 PyTorch `2.5.1+cu121`、cuDNN 9 和 RTX 3090，而当前 CPU `onnxruntime` wheel 不提供 CUDA provider。下一步应补齐显式 ONNX CUDA 执行能力，同时避免把 GPU wheel 加入核心依赖、默认 CPU 用户和托管 CPU CI。

## 目标与非目标

### 目标

- ONNX Infer 保持默认 CPU，并接受显式 `--device cuda[:id]`；CUDA session 同时注册 CPU provider 作为不支持算子的回退。
- 当 CUDA provider 未安装或 session 静默回退时明确失败，不把仅有 PyTorch CUDA 误报为 ONNX CUDA 可用。
- 保留 CPU `export`/`test` extra，新增 GPU 导出 extra；开发环境使用含 CPU provider 的 GPU ORT wheel，并用 UV 声明互斥组合。
- 使用官方 R18、四张真实 COCO 图片和 batch 4，分别比较 ONNX CUDA/eager CUDA 与 ONNX CPU/eager CPU。
- 记录版本、设备、provider、输入 checksum、数值误差、JSON/可视化结果和局限。

### 非目标

- 不引入 ONNX Runtime I/O Binding、TensorRT、C++、量化或混合精度。
- 不做吞吐、延迟或显存优劣结论；NumPy 输入仍由 ORT 从 CPU 复制到所选 provider。
- 不改变 ONNX 图格式、opset、动态 batch 或固定空间尺寸合同。
- 不扩展 R34/R50 CUDA 验证，也不恢复 M4 标准 schedule 和多 seed 长训。

## 实施步骤

- [x] 增加 ONNX CPU/CUDA 参数、provider 选择、device id 和静默回退测试。
- [x] 实现可复用 ONNX session 的 CUDA/CPU provider 合同，并保持 ONNX 预处理 tensor 位于 CPU。
- [x] 将 CPU/GPU ORT wheel 隔离到明确的 UV extras，验证 Python 3.9–3.12 lock 与 CPU CI 安装。
- [x] 导出 R18 ONNX，运行四图 batch 的 eager/ONNX × CUDA/CPU 对照。
- [x] 更新 ROADMAP、CLI/limitations 和独立报告，执行本地门禁并清理全部中间产物；推送后确认托管 CI。

## 依赖

- 仓库 UV `.venv`、PyTorch `2.5.1+cu121`、cuDNN 9 和可用 NVIDIA GPU。
- `onnxruntime-gpu` 的 CUDA 12/cuDNN 9 wheel；GPU wheel 自带 CPU provider。
- `v0.1.0` R18 checkpoint、COCO val2017 四图与 annotation。
- M8 的每图候选匹配合同和 M9 的 ONNX Infer runner。

## 风险与回退

- 风险：CPU/GPU ORT distributions 安装同名 Python 模块并相互覆盖。缓解：默认 CPU extras 与 GPU extras 显式互斥，任何环境只安装一个 distribution。
- 风险：ORT 1.27 起 PyPI GPU wheel 切换到 CUDA 13，与当前 PyTorch CUDA 12 不兼容。缓解：CUDA 12 extra 显式限制 `<1.27`，升级 PyTorch/CUDA 主版本时再单独调整。
- 风险：provider 在全局列表中可见但 CUDA/cuDNN 初始化失败后静默回退 CPU。缓解：session 创建后重新读取实际 provider，缺少 CUDA 时失败。
- 风险：GPU provider 的舍入导致 top-k 近似候选重排。缓解：以同设备 eager 为主参考，按每图类别/score/box 一对一匹配并单独记录重排。
- 回退：默认设备仍为 CPU；用户无需 GPU 路径时继续安装 `export`，核心包和 CPU CI 不改变。

## 验收

- [x] 默认 ONNX 走 CPU；显式 `cuda`/`cuda:N` 将正确 device id 交给 `CUDAExecutionProvider`。
- [x] 缺少 CUDA provider 或实际 session 回退时给出带 GPU extra 指引的错误；显式 CPU 始终可用。
- [x] R18 四图 batch 4 的 ONNX CUDA/CPU 均生成四张可解码图片和 JSON，并分别与同设备 eager 满足记录的数值合同。
- [x] 非 Paddle 全仓、覆盖率门禁、Ruff/Mypy 和托管 CPU CI 通过；中间产物已清理。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | ONNX 默认保持 CPU，CUDA 必须显式选择 | 默认安装和 CPU-only 环境应继续无歧义工作，不能仅凭 PyTorch CUDA 自动推断 ORT provider |
| 2026-07-19 | CUDA session 同时注册 CPU provider，但拒绝整个 session 静默降级 | 单个不支持算子回退是 ORT 正常合同；用户显式请求 CUDA 时，完全落到 CPU 则是配置错误 |
| 2026-07-19 | CPU/GPU ORT 使用互斥 extras | 两个 distribution 提供同名模块和共享库，共装结果不可靠 |
| 2026-07-19 | CUDA 12 环境将 GPU ORT 限制为 `<1.27` | 实装 1.27.0 需要 `libcudart.so.13`；官方 1.26 release 明确 1.27 将移除 CUDA 12 支持 |
| 2026-07-19 | 本阶段保留 CPU NumPy feed，不加入 I/O Binding | 先验证 provider 功能和数值边界；设备直连输入属于后续性能工作 |
| 2026-07-19 | CUDA provider 显式保留 `use_tf32=1` | 默认模式相对 eager CUDA 为 `6.07e-4/0.02386 px` 且无重排；关闭 TF32 虽贴近 ONNX CPU，却相对默认 eager CUDA 放大到 `0.00738/1.42 px` 并产生两条重排 |

## 完成记录

2026-07-19 完成。实现提交 `dc97927`；GPU session 实际使用 CUDA/CPU providers，R18 四图在 CUDA/CPU 两条路径均满足本计划记录的同设备合同。本地非 Paddle 全仓 `358 passed, 9 skipped, 17 deselected`，覆盖率 `51.48%/90.50%`，Ruff/Mypy、构建与发布检查通过，临时产物已清理。文档提交 `983821f` 的 [GitHub Actions run 29692163999](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29692163999) 六个 job 全绿；Python 3.9–3.12 均为 `358 passed, 9 skipped, 17 deselected`，托管全包/直接维护范围为 `7,079/13,748 (51.49%)` 和 `1,991/2,200 (90.50%)`，Ruff `174` 个文件、Mypy `107` 个 source file、发布检查和 `65 passed` wheel smoke 同时通过。

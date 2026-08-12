# M10——TorchScript CUDA/CPU 推理计划

- 状态：`completed`
- 创建日期：`2026-07-19`
- 最后更新：`2026-07-19`
- 负责人：`Codex / repository maintainers`

> 历史计划快照（2026-07-19，M10）：本文的 ONNX CPU-only 边界后来由 [M11](2026-07-19-m11-onnx-runtime-cuda-inference.md) 扩展。当前合同见 [`docs/models/rtdetrv3`](../../../models/rtdetrv3/README.md)。

## 背景

M9 已让 Infer CLI 直接消费 ONNX/TorchScript，但为避免混淆 provider，两个导出后端都暂时限制为 CPU。当前开发机的 PyTorch CUDA 和 RTX 3090 可用，TorchScript module 可由 PyTorch 自身映射到 CUDA；ONNX Runtime 则仍只有 `CPUExecutionProvider`。继续把两者绑定为同一设备边界，会无意义地放弃已经具备的 TorchScript CUDA 路径。

## 目标与非目标

### 目标

- 将 Infer 的设备合同拆分为：ONNX 继续只接受 CPU，TorchScript 与 checkpoint 一样默认选择可用 CUDA并支持显式 CPU fallback。
- 保持 TorchScript 只加载一次、固定空间尺寸预检和现有 TestReader/JSON/可视化链路不变。
- 使用官方 R18、四张真实 COCO 图片和 batch 4，对比 eager CUDA、TorchScript CUDA 与 TorchScript CPU 的阈值后检测。
- 记录 GPU/驱动/CUDA/cuDNN、输入 checksum、命令、数值误差和未覆盖边界。

### 非目标

- 不安装或声明 ONNX Runtime CUDA provider，不新增 TensorRT/C++/量化/混合精度路径。
- 不做吞吐或显存优劣结论；本轮只验证功能与数值合同。
- 不扩展动态高宽，不改变 checkpoint、权重或 `v0.1.0` Release assets。
- 不恢复 M4 标准 schedule、多 seed 或多变体长训。

## 实施步骤

- [x] 修改 Infer 的默认设备和后端专属校验，保持 ONNX 非 CPU 失败。
- [x] 补 checkpoint/TorchScript CUDA 默认、显式 CPU 与 ONNX 拒绝 CUDA 的参数回归。
- [x] 导出 R18 TorchScript，运行四图 batch 的 eager CUDA、TorchScript CUDA 和 TorchScript CPU。
- [x] 按每图类别/score/box 一对一合同比较三个结果，验证 JSON 与可视化。
- [x] 更新 ROADMAP、CLI/limitations 和独立报告，执行本地门禁并清理全部临时产物；托管 CI 待推送后确认。

## 依赖

- 仓库 UV `.venv` 中的 PyTorch `2.5.1+cu121` 和可用 NVIDIA GPU。
- `v0.1.0` R18 checkpoint、COCO val2017 四图与完整 annotation。
- M8 的每图候选匹配合同、M9 的导出产物 Infer runner 和固定尺寸元数据。

## 风险与回退

- 风险：traced module 中的常量或设备绑定阻止 CUDA 重载。缓解：使用 `torch.jit.load(..., map_location=device)` 后以真实四图执行，不以参数位于 CUDA 代替前向证据。
- 风险：CPU/CUDA 算子舍入导致阈值边界或 top-k 重排。缓解：比较同一预处理后的每图完整阈值后候选，记录实际误差和第一个分歧，不静默放宽门槛。
- 风险：GPU 可用机器默认改为 CUDA影响既有 CPU 用户。缓解：无 CUDA 时自动回退 CPU，任何机器都可显式 `--device cpu`。
- 回退：只修改 TorchScript 分支的设备选择；ONNX runner、导出格式和 checkpoint 加载均不改变。

## 验收

- [x] `--onnx-model --device cuda` 继续在加载前失败。
- [x] CUDA 可用时 TorchScript 默认 `cuda`，显式 `--device cpu` 可用；无 CUDA时默认 CPU。
- [x] R18 四图 batch 4 的 TorchScript CUDA/CPU 均生成四张可解码图片和 JSON，并分别与同设备 eager 满足记录的数值合同。
- [x] 非 Paddle 全仓、覆盖率门禁、Ruff/Mypy 和托管 CI 通过；中间产物已清理。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | 分离 ONNX provider 与 TorchScript device 合同 | ONNX Runtime provider 列表只有 CPU；TorchScript 由已验证的 PyTorch CUDA runtime 执行，两者不是同一能力 |
| 2026-07-19 | 使用四图 batch 4，不做性能排名 | 同时覆盖真实输入、动态 batch 和 CUDA 功能；单次命令耗时不足以形成稳定性能结论 |
| 2026-07-19 | 固定导出缓存使用 non-persistent buffer | CPU trace 的普通 Tensor 属性和显式 device 转换会被固化；buffer 可随 `map_location` 迁移且不改变 checkpoint schema |
| 2026-07-19 | 同设备为主数值参考，跨设备单独记录 | CPU/CUDA 的 top-k 近似项可发生合理漂移，不能把同设备门槛无依据地全局放宽 |

## 完成记录

已完成。实现提交 `85b956d`；R18 四图阈值后检测数为 `[30,1,25,2]`。TorchScript CUDA 相对 eager CUDA 的最大 score/box 误差为 `2.79218e-4/0.00872803 px`，TorchScript CPU 相对 eager CPU 为 `1.90735e-6/9.15527e-5 px`，两组同设备渲染均逐字节一致。跨设备对照有两条近似候选换序，按独立观测记录，不修改 M8 默认门槛。定向回归 `56 passed`；隐藏 GPU 的本地非 Paddle 全仓 `353 passed, 7 skipped, 34 deselected`，覆盖率为 `51.45%/90.46%`；Ruff `174` 个文件、Mypy `107` 个 source file 通过。提交 `f8b7439` 的 [GitHub Actions run 29690660612](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29690660612) 六个 job 全绿；Python 3.9–3.12 均为 `353 passed, 9 skipped, 17 deselected`，托管覆盖率 `51.46%/90.46%`，wheel smoke `60 passed`。临时产物已清理，完整环境、checksum、命令与限制见[TorchScript 设备验证报告](../reports/torchscript-device-validation.md)。

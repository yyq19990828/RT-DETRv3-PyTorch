# M9——导出产物端到端推理计划

- 状态：`completed`
- 创建日期：`2026-07-19`
- 最后更新：`2026-07-19`
- 负责人：`Codex / repository maintainers`

> 历史计划快照（2026-07-19，M9）：设备合同后来由 [M10](2026-07-19-m10-torchscript-cuda-inference.md) 和 [M11](2026-07-19-m11-onnx-runtime-cuda-inference.md) 扩展。当前合同见 [`docs/models/rtdetrv3`](../../../models/rtdetrv3/README.md)。

## 背景

M5/M8 已证明 R18/R34/R50 的 ONNX 和 traced TorchScript 能从归一化 tensor 输入运行到 `bbox/bbox_num`，但用户侧 Infer CLI 仍只能加载 PyTorch checkpoint。当前导出证据复用了 TestReader 做真实图诊断，却没有形成可安装包中的公开入口；用户仍需自行拼接图片解码、预处理、batch 分组、阈值、JSON 和可视化。

## 目标与非目标

### 目标

- 让 `rtdetrv3-infer` 在保持 checkpoint 用法兼容的同时，互斥接受 ONNX 或 TorchScript 导出产物。
- 三种后端复用同一 config/TestReader、图片批处理、`bbox_num` 分组、阈值、类别映射、JSON 和可视化路径。
- 导出后端默认并只声明 CPU/FP32；对 EMA、设备和输入尺寸等不成立的组合给出显式错误。
- 使用官方 R18 checkpoint、真实 COCO 图片和本轮临时导出产物验证 eager/ONNX/TorchScript 用户可见检测一致。

### 非目标

- 不增加 ONNX Runtime CUDA provider、TensorRT、C++、量化或 FP16/BF16 支持。
- 不声称单个导出产物支持动态高宽，也不自动修改导出图的固定输入尺寸。
- 不改变导出格式、checkpoint、权重或 `v0.1.0` Release assets。
- 不恢复 M4 标准 schedule 或多 seed 长训。

## 实施步骤

- [x] 扩展 Infer 参数合同，使用互斥模型源并验证后端专属选项。
- [x] 增加复用 session/module 的 ONNX/TorchScript batch runner，适配现有 Infer 输出合同。
- [x] 补参数、runner 和主流程回归，保持 checkpoint 既有路径不变。
- [x] 用 R18 真实图完成三后端对照，记录环境、命令、误差和限制。
- [x] 更新 ROADMAP、CLI/limitations 与独立报告，清理临时导出和推理产物。

## 依赖

- 仓库 UV `.venv` 与 `export` extra 中的 ONNX/ONNX Runtime。
- `v0.1.0` R18 checkpoint 和 COCO `val2017/000000000139.jpg`。
- M8 已建立的每图全候选一对一数值比较合同。

## 风险与回退

- 风险：Infer 的 batch dict 含导出图不需要的字段。缓解：runner 只提取 `image/im_shape/scale_factor`，输出重新包装为既有 `bbox/bbox_num` mapping。
- 风险：GPU 可用机器上的既有默认设备会误用于 CPU-only 导出后端。缓解：checkpoint 保持原默认选择，导出产物单独默认 CPU，并拒绝显式非 CPU 设备。
- 风险：配置 Resize 与导出固定高宽不一致。缓解：保留 `--imgsz` 显式合同和后端运行时 shape 错误，不宣称自动动态化。
- 回退：新路径只在选择 ONNX/TorchScript 参数时进入；checkpoint 分支和导出文件均不被修改。

## 验收

- [x] 旧 `--checkpoint` Infer 参数与主流程回归继续通过。
- [x] ONNX/TorchScript 模型源互斥，EMA/非 CPU 等无效组合在加载前失败。
- [x] runner 只创建一次 session/module，并正确处理最后一个不足 batch 的图片组。
- [x] R18 真实图片三后端在 `score >= 0.3` 下检测数、类别和框/score 满足 M8 数值合同，三者均生成可解码可视化和 JSON。
- [x] 非 Paddle 全仓测试、覆盖率门禁、Ruff 和 Mypy 通过；中间产物已清理。
- [x] 提交后的 GitHub Actions 托管矩阵通过。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | 复用 `rtdetrv3-infer`，不新增重复的部署 CLI | 已有入口完整实现预处理、阈值、JSON 和可视化；模型 runner 才是缺失边界 |
| 2026-07-19 | 本阶段导出后端只声明 CPU/FP32 | 当前 ONNX Runtime 只有 CPU provider；避免把 PyTorch CUDA 可用性误写为 ONNX GPU 证据 |
| 2026-07-19 | 新 TorchScript 归档内嵌 schema v1 `input_size`，旧归档仍可加载 | traced module 重载后不保留可查询 shape；最小元数据允许在执行前拒绝错误预处理尺寸 |

## 完成记录

已完成。官方 R18 在 COCO `000000000139.jpg`、CPU/FP32、640×640、阈值 `0.3` 下，checkpoint/ONNX/TorchScript 均输出 30 条检测；ONNX 相对 eager 的最大 score/框误差为 `1.49012e-6/9.15527e-5 px`，TorchScript 为 0，三张渲染图字节一致。640 产物与 `--imgsz 608` 的 ONNX/TorchScript 负例均在 backend 执行前明确失败。定向回归 `46 passed`；本地非 Paddle 全仓 `350 passed, 5 skipped, 34 deselected`，全包/直接维护范围覆盖率为 `51.41%/90.52%`，Ruff `174` 个文件与 Mypy `107` 个 source file 通过。提交 `545578a` 的 [GitHub Actions run 29689593612](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29689593612) 六个 job 全绿；托管 Python 3.12 为 `350 passed, 7 skipped, 17 deselected`、覆盖率 `51.42%/90.45%`，wheel smoke `59 passed`。临时模型、JSON、图片、日志、coverage 和 pytest 目录已清理。

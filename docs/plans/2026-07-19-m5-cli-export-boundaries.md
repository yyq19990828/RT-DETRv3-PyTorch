# M5——配置、CLI 与导出边界计划

- 状态：`in-progress`
- 创建日期：`2026-07-19`
- 最后更新：`2026-07-19`
- 负责人：`Codex / repository maintainers`

## 背景

M1–M3 已建立当前 PyTorch API 的训练、评估、恢复与权重转换链，M4 的同权重 COCO gate 也已通过；长期稳定性实验已经转为社区分片，不阻塞本地迁移。M5 需要把面向用户的配置、CLI 和导出边界从“代码存在”提升为“有当前合同和真实执行证据”。

2026-07-19 审计发现，Train、Eval 和 Convert 已有活跃测试，而 Infer 只有 `tests/legacy/` 中基于旧模型合同的历史测试。旧 Infer CLI 使用另一套配置加载器、手写 letterbox/ImageNet 预处理、外置 NMS，并假定模型接收 tensor、返回 `pred_logits/pred_boxes`；它在当前仓库首先失败于 `RTDETRv3` 未注册，即使补导入也仍与当前 batch dict 和 `bbox/bbox_num` 合同不兼容。

## 目标与非目标

### 目标

- 为 workspace 的 shared、inject、from_config、显式参数冲突和全局状态建立可重复合同。
- 为 Train、Eval、Infer、Convert 建立当前 CLI contract，显式记录 Paddle 参数差异。
- 以当前 TestReader 和模型内置后处理建立可信的 Infer 基线，并覆盖单图、目录、真实 batch、EMA 和 JSON 输出。
- 完成 ONNX/ONNXRuntime 与 TorchScript 导出、重新加载和输出回归。
- 验证已声明的动态尺寸、batch 1/4/8 和空阈值结果边界。

### 非目标

- 不恢复 `tests/legacy/` 使用的旧 Registry/builder 或 `pred_logits/pred_boxes` CLI 合同。
- 不为 RT-DETR 输出额外增加 NMS；当前模型后处理使用配置驱动的 top-k。
- 不在 M5 恢复 M4 的 72 epoch、多 seed 或 R34/R50 长训。
- 不在本阶段引入 TensorRT、C++ 或量化部署。

## 实施步骤

- [x] 审计四个 CLI 与 workspace/export 现状，确认 Infer 当前不可运行的第一处错误和后续语义冲突。
- [x] 重写 Infer CLI：复用统一 workspace、TestReader、batch dict、模型内置 `bbox/bbox_num` 后处理和 Eval checkpoint 加载规则。
- [x] 为 Infer 参数校验、旧下划线参数别名、预处理、batch、阈值、输出拆分和类别映射添加活跃单元测试。
- [x] 使用官方 R18 checkpoint 在 CPU/FP32 上验证真实 COCO 单图、JSON 和 batch 4 目录推理。
- [ ] 补齐 workspace 冲突矩阵、重复配置加载和进程内全局状态隔离测试，并修正文档中的过时描述。
- [ ] 补齐 Train/Eval/Infer/Convert 的统一 `--help`、错误路径和端到端 CLI contract；记录 Paddle-only 参数及替代方式。
- [ ] 建立导出适配层，完成 ONNX/ONNXRuntime 输出回归并记录算子和动态轴限制。
- [ ] 完成 TorchScript 保存/加载回归；验证 batch 1/4/8、支持的输入尺寸和空阈值结果。
- [x] 在 Infer 第一阶段后运行隐藏 GPU 的默认全量测试，并更新路线图和迁移文档证据。
- [ ] 在 M5 完成前构建 wheel，并运行最终默认全量测试。

## 风险与回退

- 风险：直接 trace/script 训练态模型会把 Python dict、内置后处理或动态 shape 固化。缓解：先锁定 eager Infer 基线，再为导出定义最小 tensor 输入/输出适配层。
- 风险：workspace 是进程级全局状态，连续加载配置可能残留前一份值。缓解：先用隔离测试暴露状态边界，再决定最小修复，不另建第二套注册表。
- 风险：自定义 TestReader 使用 `keep_ratio=True` 时，不同尺寸图片可能无法直接堆叠为 batch。缓解：动态/填充行为必须经 M5 边界测试后才能声明支持。
- 回退：Infer 改动集中在 CLI 和新增测试；若回归，可回退该入口而不改变训练、模型参数或 checkpoint 格式。

## 验收

- [x] `uv run --extra dev pytest tests/unit/cli/test_infer.py -q` 通过。
- [x] 官方 R18 checkpoint、COCO val2017 图片、CPU/FP32 单图命令成功，生成可解码可视化图片和 30 条阈值 `0.3` JSON 记录。
- [x] 同一环境 batch 4 目录命令一次前向处理 4 张图片并生成 4 个结果文件。
- [ ] 四个 CLI 的活跃 contract 测试全部通过。
- [ ] ONNXRuntime 与 eager 输出在记录的 dtype、输入和容差内通过。
- [ ] TorchScript 重新加载输出在记录的 dtype、输入和容差内通过。
- [x] Infer 第一阶段后的默认全量测试为 `200 passed, 8 skipped`，测试中间产物已清理。
- [x] Infer 第一阶段的 sdist/wheel 构建成功，wheel 包含当前 `ppdet_pytorch/cli/infer.py`；构建目录已清理。
- [ ] M5 最终默认全量测试和 wheel 构建通过。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | M4 长训转社区后，本地优先恢复 Infer 再做导出 | 导出回归需要一个与 Eval 共用预处理和后处理语义的 eager 基线 |
| 2026-07-19 | 删除 CLI 手写 letterbox/ImageNet 归一化和外置 NMS | 当前配置明确使用 TestReader Resize/Normalize，模型已完成 top-k 和原图坐标恢复；重复实现会制造语义分叉 |
| 2026-07-19 | 保留下划线参数作为别名，文档使用连字符参数 | 兼顾仓库 README 的历史命令和当前 Python CLI 命名习惯，不保留已失效的内部模型合同 |
| 2026-07-19 | Infer 复用 Eval 的受控 checkpoint 加载 | 官方转换权重允许两个可派生 buffer 缺失，但其他 missing/unexpected key 必须失败；训练 checkpoint 还可显式选择 EMA |

## 完成记录

进行中。第一阶段基于提交 `9d76bb6a17e1` 后的工作树完成：Python `3.12.11`、PyTorch `2.5.1+cu121`、OpenCV `4.5.5`、CPU/FP32；R18 checkpoint SHA-256 为 `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547`，验证图片 `000000000139.jpg` SHA-256 为 `ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d`。单图最高结果为 `chair / 0.9240278006`；隐藏 GPU 的默认全量回归为 `200 passed, 8 skipped, 6 warnings in 9.31s`，sdist/wheel 构建成功且产物随后清理。这些结果只证明当前 Infer CLI 与官方 checkpoint 可运行且未破坏活跃回归，不单独构成新的 Paddle/PyTorch 数值对齐结论。

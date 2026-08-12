# M8——R34/R50 多变体导出验收计划

> 历史计划快照（2026-07-19，M8）：本文保存已完成执行记录，不代表当前仓库状态。当前合同见 [`docs/models/rtdetrv3`](../../../models/rtdetrv3/README.md)。

- 状态：`completed`
- 创建日期：`2026-07-19`
- 最后更新：`2026-07-19`
- 负责人：`Codex / repository maintainers`

## 背景

M5 已为官方 R18 建立 ONNX opset 17/ONNX Runtime CPU 和 traced TorchScript 合同，覆盖固定 608/640 高宽、动态 batch 1/4/8、真实 COCO 输入与严格输出误差。M7 证明 `v0.1.0` 公开 R34/R50 checkpoint 的 eager Infer/Eval 可运行，但导出文档仍明确只覆盖 R18。配置可构建和 eager 可运行不能证明另两个 backbone 能成功导出、重载或满足同一误差合同。

## 目标与非目标

### 目标

- 使用 `v0.1.0` 已校验的 R34/R50 权重与对应配置导出 640×640 ONNX 和 TorchScript。
- 对每个导出产物执行 checker/reload，以及 batch 1/4/8 的 eager 对照回归。
- 使用 TestReader 预处理后的真实 COCO `000000000139.jpg` 再执行一次 eager/ONNX/TorchScript 对照。
- 继续要求 `bbox_num` 和每图完整候选集合一致；按类别、score 和 box 一对一匹配全部候选，score 最大绝对误差 `<=2e-5`、坐标最大绝对误差 `<=0.02 px`，并单独报告近似并列项重排。
- 记录产物大小、SHA-256、实际误差、警告和仍未支持的部署边界。

### 非目标

- 不声称单个导出产物支持动态高宽；本轮只验证固定 `640×640`。
- 不验证 ONNX Runtime CUDA provider、TensorRT、C++、量化、FP16/BF16 或 Paddle 导出等价。
- 不把全零 example input 或 tensor-only 输出对齐替代图片解码与部署端完整预处理实现。
- 不修改或发布新的模型权重，也不恢复 M4 长训。

## 实施步骤

- [x] 审计 Export CLI、适配层、依赖和 R18 既有协议。
- [x] 使用 R34 配置导出 ONNX/TorchScript，并完成 batch 1/4/8 与真实图回归。
- [x] 使用 R50 配置重复相同协议。
- [x] 修复逐行验证误判近似并列候选重排的问题，并补活跃回归。
- [x] 更新 ROADMAP、CLI/limitations 和独立报告，清理所有临时导出文件。

## 依赖

- 仓库 UV `.venv` 中的 PyTorch `2.5.1+cu121`、ONNX `1.22.0` 和 ONNX Runtime `1.27.0`。
- `v0.1.0` 实际发布目录中已验证的 R34/R50 checkpoint。
- COCO val2017 图片 `000000000139.jpg` 及当前 TestReader 配置。

## 风险与回退

- 风险：R50 ONNX/TorchScript 产物和 batch 8 中间激活占用较大。缓解：两个变体顺序导出和验证，全部产物置于独立临时目录，完成一个变体后释放模型/session。
- 风险：top-k 尾部近似并列项在后端舍入后交换顺序。缓解：不要求检测数组保持非语义行序，改为每图全候选一对一匹配；仍严格限制类别、score、box、数量和分组。
- 风险：trace/export 警告被误解为动态高宽支持。缓解：记录警告并继续明确空间 shape 固定。
- 回退：若需代码修复，只修改导出适配或验证边界；失败产物由原子导出和临时目录清理，不影响现有 checkpoint 或 `v0.1.0`。

## 验收

- [x] R34/R50 的 ONNX 均通过 checker 和 ONNX Runtime CPU 加载运行。
- [x] R34/R50 的 TorchScript 均保存、重载并运行。
- [x] 两种格式在 batch 1/4/8 和真实 COCO 输入上满足每图全候选一对一输出合同。
- [x] 报告不把固定 640、CPU/FP32 证据外推到其他 shape、dtype 或 provider。
- [x] 临时 ONNX、TorchScript、日志和测试缓存全部清理。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | 复用 M5 的 640、batch 1/4/8 和真实图严格合同 | 直接比较变体差异，不新增另一套容差或输入语义 |
| 2026-07-19 | 只扩展 R34/R50，不重复 R18 全套导出 | R18 已有 608/640 和真实图证据，本阶段要关闭的是多变体缺口 |
| 2026-07-19 | 检测输出改为每图全候选一对一匹配，坐标门槛调整为 `0.02 px` | ONNX 的近似并列 tail 可交换行序；R34 全零 300 个候选中唯一超过原门槛的实际误差为 `0.0117798 px`，全部候选在新门槛内匹配 |

## 完成记录

已完成。R34/R50 均导出 ONNX opset 17 与 traced TorchScript，完成 checker/reload、固定 640 的动态 batch 1/4/8 和真实 COCO 图片回归。R34 ONNX 最大 score/box 误差为 `9.4771e-6/0.011780 px`，R50 为 `1.8962e-5/0.005615 px`；TorchScript 本次全部为 0。诊断实际发现并修复逐行比较把近似并列候选重排误判为大框误差的问题；新验证器仍对每图全部候选执行类别、score、box 的唯一匹配，不过滤 tail。定向测试 `19 passed`，非 Paddle 全仓 `343 passed, 5 skipped, 34 deselected`，Ruff/Mypy 通过。完整产物摘要、逐输入误差、警告和限制见[多变体导出报告](../reports/variant-export-validation.md)，全部临时导出与测试产物已清理。

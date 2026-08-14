# RT-DETRv3 证据索引

> 本页把当前结论映射到可提交的正式报告与本轮机器收据名称。`.omo` 原始日志是本地执行材料，不是发布文档依赖。

## 当前计划收据

| 证据 | 状态 | 支持的结论 |
|---|---|---|
| Task 1，`task-1-rtdetrv4-merge.json` | APPROVE | R18 官方 checkpoint 身份、Paddle GPU 探针、CPU/FP32 数值基线、验证驱动与负例 |
| F2，`final-F2-quality.json` | APPROVE | Ruff、Mypy、覆盖率、unit/integration、上游 numerical 和图审计 |
| F4，`final-F4-scope-v3.json` | APPROVE | 子模块范围、完整非 Paddle 回归、R18 数值门和既有三后端基线身份 |
| F1，`final-F1-plan-compliance.md` | APPROVE | Task 1-23 与 F2-F4 的计划身份和状态完整性 |

Task 1 与 F2/F4 收据绑定 plan identity `60333d67db893e1b12be693d53a3873f7f028878e9f77e7e4aecb34c85613ac5`。F4 的 surface 状态继承 Task 1 baseline；逐项误差必须引用下列正式归档，不能从 F4 状态反推。

## 正式归档

| 报告 | 内容 |
|---|---|
| [精度验证](../../archive/rtdetrv3-v0.1.0/reports/accuracy-validation.md) | R18 同权重完整 COCO val2017、预测匹配和 AP 差 |
| [导出推理](../../archive/rtdetrv3-v0.1.0/reports/exported-inference-validation.md) | R18 eager/ONNX/TorchScript 单图用户链 |
| [TorchScript 设备](../../archive/rtdetrv3-v0.1.0/reports/torchscript-device-validation.md) | R18 CPU/CUDA 四图矩阵 |
| [ONNX Runtime 设备](../../archive/rtdetrv3-v0.1.0/reports/onnx-runtime-device-validation.md) | R18 ORT CPU/CUDA provider 与误差 |
| [多变体导出](../../archive/rtdetrv3-v0.1.0/reports/variant-export-validation.md) | R34/R50 CPU 导出和候选匹配 |
| [多变体设备矩阵](../../archive/rtdetrv3-v0.1.0/reports/variant-export-device-validation.md) | R34/R50 CPU/CUDA 结果及 ONNX CUDA 未通过门 |
| [发布验证](../../archive/rtdetrv3-v0.1.0/reports/release-validation.md) | `v0.1.0` 公开资产、checksum 与打包合同 |

## 可执行入口

- Manifest：[`configs/checkpoints/rtdetrv3_coco.yml`](../../../configs/checkpoints/rtdetrv3_coco.yml)
- 官方 R18 数值测试：[`tests/numerical/test_r18_official_checkpoint.py`](../../../tests/numerical/test_r18_official_checkpoint.py)
- 配置：[`configs/rtdetrv3`](../../../configs/rtdetrv3/)
- 转换方法：[权重转换](../../migrations/weight-conversion.md)
- 训练与验证方法：[训练与验证](../../migrations/training-and-validation.md)

## 证据边界

- 历史归档带日期，不自动代表任意未来依赖版本。
- 当前 F2/F4 证明集成后未观察到 v3 回归；不替代 `v0.1.0` 的原始逐数值报告。
- 未提供可选 checkpoint 环境变量导致的 skip 不是通过证据，也不是数值失败。

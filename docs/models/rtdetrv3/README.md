# RT-DETRv3

RT-DETRv3 是当前仓库首个完成迁移和发布验证的 DETR 系列模型。本目录只保存模型专属合同；共享训练框架和 Paddle-to-PyTorch 迁移方法见 [`docs/migrations`](../../migrations/README.md)。

## 当前合同

- [配置迁移指南](configuration-guide.md)：R18/R34/R50 配置字段、必要改写与未声明支持范围。
- [CLI 与导出边界](cli-and-export.md)：Train/Eval/Infer/Convert/Export/Models 的当前用户合同。
- [已知限制](limitations.md)：数值、环境、配置、API 与导出边界。
- [验证报告](validation-report.md)：官方 checkpoint、Paddle/PyTorch 保真、回归与最终范围审计。
- [指标记录](metrics.md)：COCO、权重转换及 eager/ONNX/TorchScript 的数值结果。
- [证据索引](evidence-index.md)：当前最终门与 `v0.1.0` 历史报告之间的映射。

## 版本证据

RT-DETRv3 `v0.1.0` 的已完成计划、技术报告、论文、图片和机器可读数据统一归档在 [`docs/archive/rtdetrv3-v0.1.0`](../../archive/rtdetrv3-v0.1.0/README.md)。未完成的 M4 长训、多 seed 与 R34/R50 长训仍以根目录 [`ROADMAP.md`](../../../ROADMAP.md) 和活动计划为准。

## 配置与实现

- 配置：[`configs/rtdetrv3`](../../../configs/rtdetrv3/)
- 权重清单：[`configs/checkpoints/rtdetrv3_coco.yml`](../../../configs/checkpoints/rtdetrv3_coco.yml)
- 架构实现：[`src/ppdet_pytorch/modeling/architectures/rtdetrv3.py`](../../../src/ppdet_pytorch/modeling/architectures/rtdetrv3.py)

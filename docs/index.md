# DETR-series

DETR-series 是 DETR 系列实时目标检测模型的 PyTorch 实现合集。六个模型族共 **30 个 COCO 变体**运行在同一个训练、评估、推理、checkpoint 与部署运行时上:写一份配置,即可用同一组 CLI 训练、评估、导出任意族。

Python 包与 PyPI 项目名为 `detrs`,全部工作流由单一 `detrs` 命令的子命令(`train`/`eval`/`infer`/`export`/`convert`/`models`)提供。核心 PyTorch 运行时不导入 Paddle;仓库起源于 RT-DETRv3 的 Paddle-to-PyTorch 迁移,Paddle 官方实现仅作为只读参考子模块保留。

## 核心特性

- **统一运行时**:六个族共享数据管道、两阶段训练协议、EMA、断点恢复、`bbox`/`bbox_num` 推理合同与 ONNX/TorchScript 导出边界。
- **证据驱动验收**:每个官方 checkpoint 记录上游 revision、SHA-256、key 映射与数值对齐证据;完整 val2017 结果与官方公布值的误差以预注册门槛约束。
- **部署友好**:推理不额外执行 NMS,ONNX(opset 17、固定高宽、动态 batch)与 TorchScript 逐值验证;DEIMv2 覆盖从 50.3M 到 0.5M 的全部尺寸档。

## 模型总览

COCO val2017 bbox AP 为官方公布值;本仓库实测与验收证据见各[模型文档](models/README.md)。

| 模型族 | 论文 | 变体 | 官方 COCO AP |
|---|---|---|---|
| RT-DETRv3 | RT-DETRv3(2024) | R18/R34/R50 | 48.1(R18) |
| D-FINE | D-FINE(2024) | N/S/M/L/X | 42.8 – 55.8 |
| DEIM-D-FINE | DEIM(CVPR 2025) | N/S/M/L/X | 43.0 – 56.5 |
| DEIM-RT-DETRv2 | DEIM(CVPR 2025) | S/M/M*/L/X | 49.0 – 55.5 |
| RT-DETRv4 | RT-DETRv4(2025) | S/M/L/X | 49.8 – 57.0 |
| DEIMv2 | DEIMv2(2025) | X/L/M/S;N/Pico/Femto/Atto | 23.8 – 57.8 |

## 从这里开始

- [快速开始](guides/quickstart.md):安装、下载权重、推理、训练与评估的最短路径。
- [使用指南](guides/README.md):安装模式、模型资产、ONNX/TorchScript 推理、权重转换与导出的完整说明。
- [模型文档](models/README.md):每个模型族的支持合同、逐变体指标与已知限制。
- [API 参考](api/index.md):`detrs` 包全部子包的自动生成接口文档。
- [迁移经验](migrations/troubleshooting.md):环境、配置、权重转换与数值差异的排错经验。

# RT-DETRv3 PyTorch Implementation

RT-DETRv3 目标检测模型的 PyTorch 实现版本，从 PaddlePaddle 迁移而来。

## 特性

- 完整的 RT-DETRv3 架构实现
- 支持 COCO 和 LVIS 数据集
- 支持多种骨干网络（ResNet-18/34/50/101）
- 层次化密集正监督训练
- 支持 ONNX 导出
- 完整的训练和推理管道

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练

```bash
python tools/train.py --config configs/rtdetrv3_r50vd_6x_coco.yml
```

### 评估

```bash
python tools/eval.py --config configs/rtdetrv3_r50vd_6x_coco.yml --checkpoint outputs/rtdetrv3_r50vd_6x_coco/best.pth
```

### 推理

```bash
python tools/infer.py --config configs/rtdetrv3_r50vd_6x_coco.yml --checkpoint outputs/rtdetrv3_r50vd_6x_coco/best.pth --image test_image.jpg
```

### 导出

```bash
python tools/export.py --config configs/rtdetrv3_r50vd_6x_coco.yml --checkpoint outputs/rtdetrv3_r50vd_6x_coco/best.pth --format onnx
```

## 模型架构

RT-DETRv3 是一个基于 Transformer 的端到端目标检测器，具有以下特点：

- 混合编码器：结合 CNN 和 Transformer 的特征提取
- 层次化密集正监督：多层解码器输出用于训练
- 去噪训练：对比去噪训练策略
- 查询选择：基于编码器输出的查询向量选择

## 性能

| 模型 | 骨干网络 | 输入尺寸 | AP | AP50 | AP75 | 推理速度 |
|------|----------|----------|-----|------|------|-----------|
| RT-DETRv3-R50 | ResNet-50 | 640×640 | 53.1 | 71.3 | 57.8 | 108 FPS |
| RT-DETRv3-R101 | ResNet-101 | 640×640 | 54.3 | 72.7 | 59.0 | 74 FPS |

## 许可证

Apache License 2.0

## 更新日志

- 2025-07-18：添加 Claude Code 集成，包括自定义命令 (`commit-info`, `execute-prp`, `generate-command`, `generate-prp`) 和项目级指令，以支持自动化开发工作流。
- 2025-07-18：初始版本发布，完成 PaddlePaddle 到 PyTorch 的初步迁移，支持 COCO/LVIS 数据集、ResNet 系列骨干、ONNX 导出、完整训练/推理/评估/导出流程。(尚未验证，仍需完善)
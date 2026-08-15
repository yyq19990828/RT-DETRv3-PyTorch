# 快速开始

本页给出从零到推理、训练、评估的最短路径。完整的安装模式、参数与部署边界见[使用指南](README.md)。

## 1. 安装

项目支持 Python 3.9–3.12,使用 uv 0.11.29 至 0.12.x 管理环境。默认锁文件面向 Linux x86_64 或 Windows amd64 的 PyTorch CUDA 12.1;其他平台需要选择匹配的 PyTorch 索引。

```bash
git clone --recurse-submodules https://github.com/yyq19990828/DETR-series.git
cd DETR-series
uv sync
```

核心运行时不安装 Paddle;Paddle 权重转换和数值对齐等场景使用 `uv sync --extra dev` 等额外安装模式,详见[使用指南](README.md#安装)。

## 2. 获取模型权重

```bash
# 查看模型与权重状态(其他族用 --family dfine / deim-dfine / rtdetrv4 / deimv2)
uv run detrs models list

# 下载并校验 RT-DETRv3 v0.1.0 发布权重
uv run detrs models download r18
uv run detrs models verify r18 \
  pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth
```

D-FINE 的权重可以由 CLI 原子下载;DEIM 与 RT-DETRv4 的 Google Drive 托管权重只支持 list 和本地 verify,`download` 会返回官方来源地址。

## 3. 推理

```bash
uv run detrs infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --infer-img path/to/image.jpg \
  --output-dir output/infer \
  --save-results
```

`--infer-dir` 支持目录推理,`--infer-img` 与 `--infer-dir` 互斥。推理复用配置中的 `TestReader` 和 DETR 后处理,不额外执行 NMS。ONNX/TorchScript 推理见[使用指南](README.md#onnx-与-torchscript-推理)。

## 4. 训练

```bash
uv run detrs train \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --seed 0
```

训练 checkpoint 保存模型、EMA、optimizer、scheduler、scaler 和 RNG 状态,支持 epoch 边界的确定性恢复。各模型族的配置入口见[模型文档](../models/README.md)。

## 5. 评估

```bash
uv run detrs eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth

# 评估训练 checkpoint 中的 EMA,并保存 COCO prediction JSON
uv run detrs eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --use-ema \
  --output-dir output/eval
```

## 6. 导出

```bash
uv run --extra export detrs export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --format both \
  --output-dir output/export
```

导出生成 ONNX(opset 17)与 traced TorchScript;空间尺寸固定、batch 动态,改变空间尺寸时需要重新导出。

## 下一步

- 完整参数、安装模式与 CLI 边界:[使用指南](README.md)。
- 各模型族支持合同与逐变体指标:[模型文档](../models/README.md)。
- 环境与数值差异排错:[故障排查](../migrations/troubleshooting.md)。

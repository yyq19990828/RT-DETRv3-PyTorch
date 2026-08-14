# 使用指南

本指南面向安装包和仓库用户，集中说明环境、模型资产、训练、评估、推理、转换与导出。模型支持状态、逐变体指标和限制以[模型文档](../models/README.md)为准。

## 安装

项目支持 Python 3.9–3.12 和 uv 0.11.29 至 0.12.x。默认锁文件使用 PyTorch CUDA 12.1 官方索引，面向 Linux x86_64 或 Windows amd64；CPU、macOS 和 ARM 环境需要改用平台匹配的 PyTorch 索引。

```bash
# 核心 PyTorch 训练与推理
uv sync

# 不依赖 Paddle 的测试
uv sync --extra test

# Paddle 权重转换和数值对齐
uv sync --extra dev

# ONNX CPU 或 CUDA provider
uv sync --extra export
uv sync --extra export-gpu

# Ruff/Mypy，或训练专用 DINOv3 teacher
uv sync --extra quality
uv sync --extra teacher
```

`dev` 与 `export`/`test`，以及 `export-gpu` 与 `export`/`test` 不能组合安装。`export` 使用 CPU `onnxruntime`，`export-gpu` 与 `dev` 使用同时包含 CUDA 和 CPU provider 的 `onnxruntime-gpu`。核心运行时不依赖 Paddle。

中国大陆 Linux x86_64 环境可以从阿里云或上海交大镜像预载锁定的 PyTorch wheel；脚本会使用 `uv.lock` 中的官方 SHA-256 校验，再执行 locked sync：

```bash
python3 scripts/sync_china.py --extra test
python3 scripts/sync_china.py --mirror sjtug --extra dev
```

如果缺少只读 Paddle 参考子模块：

```bash
git submodule update --init --recursive
```

## 模型与 checkpoint

Models CLI 默认使用 RT-DETRv3 manifest；`--family` 选择其他模型族，显式 `--manifest` 的优先级最高。

```bash
uv run detrs models list
uv run detrs models --family dfine list
uv run detrs models --family deim-dfine list --json
uv run detrs models --family rtdetrv4 verify \
  rtdetrv4-s path/to/RTv4-S-hgnet.pth
```

D-FINE 的固定 GitHub Release asset 可以由 CLI 原子下载。Google Drive 托管的 DEIM 与 RT-DETRv4 权重只支持 list 和本地 verify；download 会返回 manifest 中的官方来源地址。RT-DETRv3 `v0.1.0` 发布权重可以直接下载并校验：

```bash
uv run detrs models download r18
uv run detrs models verify r18 \
  pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth
```

## 训练与评估

```bash
uv run detrs train \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --seed 0

uv run detrs eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth

# 评估训练 checkpoint 中的 EMA，并保存 COCO prediction JSON
uv run detrs eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --use-ema \
  --output-dir output/eval
```

训练 checkpoint 使用 format-version-1，保存模型、EMA、optimizer、scheduler、scaler、epoch/global-step 和 RNG 状态。当前只声明 epoch-boundary 确定性恢复；各模型族特有的训练协议见对应[模型验证报告](../models/README.md)。

RT-DETRv4 的 DINOv3 teacher 只在训练构造。训练者需要自行准备固定 revision 的 DINOv3 checkout 和经 Meta 授权的权重；student eval、infer、export 和 checkpoint 不包含或访问 teacher 资产。

## Eager 推理

```bash
uv run detrs infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --infer-img path/to/image.jpg \
  --output-dir output/infer \
  --save-results
```

`--infer-dir` 支持非递归目录推理，`--batch-size` 控制实际 batch。只有训练 checkpoint 可以使用 `--use-ema`。推理复用配置中的 `TestReader` 和 DETR 后处理，不额外执行 NMS。

## ONNX 与 TorchScript 推理

```bash
# ONNX CPU
uv run --extra export detrs infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --onnx-model output/export/model.onnx \
  --infer-img path/to/image.jpg \
  --imgsz 640 \
  --device cpu \
  --output-dir output/infer-onnx

# ONNX CUDA
uv run --extra export-gpu detrs infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --onnx-model output/export/model.onnx \
  --infer-img path/to/image.jpg \
  --imgsz 640 \
  --device cuda:0 \
  --output-dir output/infer-onnx-cuda

# TorchScript
uv run detrs infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --torchscript-model output/export/model.torchscript.pt \
  --infer-img path/to/image.jpg \
  --imgsz 640 \
  --device cpu \
  --output-dir output/infer-torchscript
```

`--checkpoint`、`--onnx-model` 和 `--torchscript-model` 互斥。ONNX 默认使用 CPU；TorchScript 和 checkpoint 在 CUDA 可用时默认选择 CUDA，也可以显式指定 CPU。`--imgsz` 必须与导出时固定高宽一致。

## 权重转换与导出

Paddle 权重转换需要 `dev` extra：

```bash
uv run --extra dev detrs convert \
  --input path/to/model.pdparams \
  --output path/to/model.pth
```

导出需要 `export`、`export-gpu` 或 `dev` extra：

```bash
uv run --extra export detrs export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --format both \
  --output-dir output/export
```

导出入口使用 tensor-only 适配层，生成 ONNX opset 17 和 traced TorchScript；空间尺寸固定、batch 动态。改变空间尺寸时需要重新导出。各模型族的实际容差与训练残留检查见[模型文档](../models/README.md)。

## CLI 与配置边界

公开入口为单一 `detrs` 命令,子命令为 `train`、`eval`、`infer`、`convert`、`export` 和 `models`(亦可用 `python -m detrs`)。历史上的 `rtdetrv3-*` 命令与 `tools/*.py` 兼容包装器已随包名重命名为 `detrs` 移除。

未迁移的 Paddle CLI 参数会明确报错，不会静默忽略。RT-DETRv3 的配置覆盖与详细 CLI 合同见[配置迁移指南](../models/rtdetrv3/configuration-guide.md)和 [CLI 与导出边界](../models/rtdetrv3/cli-and-export.md)。

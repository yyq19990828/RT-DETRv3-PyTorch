# 推理

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

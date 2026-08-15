# 权重转换与导出

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

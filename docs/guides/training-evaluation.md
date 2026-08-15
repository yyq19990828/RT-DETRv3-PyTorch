# 训练与评估

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

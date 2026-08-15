# 模型与 checkpoint

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

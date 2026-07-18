# 数值验证

本目录保留 RT-DETRv3 PyTorch 实现的数值、确定性和 Paddle 对照用例。实际的 Paddle 对照需要 `dev` 附加依赖，部分用例还需要预训练权重或 COCO 数据集。

```bash
uv sync --extra dev
uv run pytest tests/numerical -v
```

只运行不依赖 Paddle 的数值用例：

```bash
uv run pytest tests/numerical -m "not paddle" -v
```

使用已在 [`configs/checkpoints/rtdetrv3_coco.yml`](../../configs/checkpoints/rtdetrv3_coco.yml) 校验过的官方 R18/R34/R50 checkpoint，运行可选的参数逐值、eval 分层、head/loss 输出梯度、确定性缩减训练 loss 和整体参数梯度方向对齐：

```bash
RTDETRV3_R18_PADDLE_CHECKPOINT=pretrained_models/paddle/rtdetrv3_r18vd_6x_coco.pdparams \
  uv run pytest -q -p no:cacheprovider tests/numerical/test_r18_official_checkpoint.py -k r18

RTDETRV3_R34_PADDLE_CHECKPOINT=pretrained_models/paddle/rtdetrv3_r34vd_6x_coco.pdparams \
  uv run pytest -q -p no:cacheprovider tests/numerical/test_r18_official_checkpoint.py -k r34

RTDETRV3_R50_PADDLE_CHECKPOINT=pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
  uv run pytest -q -p no:cacheprovider tests/numerical/test_r18_official_checkpoint.py -k r50
```

该用例不下载权重，转换输出写入 pytest 的 `tmp_path` 并在用例结束时显式删除。三个变体建议使用独立 pytest 进程，避免完整 Paddle/PyTorch 模型的内存池和 Paddle 全局 workspace 状态在同一进程累积。为避免 CPU 并行 reduction 的微小差异改变 transformer 边界 top-k 候选，用例会临时将 PyTorch CPU 线程数设为 1，结束后恢复。

缩减训练场景会关闭 label/box noise，并减少 query 数以隔离确定性训练语义。完整模型梯度按整体相对 L2、余弦和符号分歧率验收，不要求 AdamW 更新逐元素一致；因此仍不能外推为官方完整训练配置或训练收敛等价。R50 的容差和 2 个后处理 top-k 边界候选有单独记录，不能据此宣称逐候选完全一致。

当前数值测试仍处于迁移校准阶段。不在文档中固化“全部通过”或 mAP 一致性结论；以当前测试运行结果和可获得的官方权重/数据为准。

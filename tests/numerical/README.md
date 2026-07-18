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

当前数值测试仍处于迁移校准阶段。不在文档中固化“全部通过”或 mAP 一致性结论；以当前测试运行结果和可获得的官方权重/数据为准。

# RT-DETRv3 验证报告

> 验证快照：2026-08-14。当前用户合同见 [README](README.md)，逐项数值见[指标记录](metrics.md)。

## 结论

RT-DETRv3 R18/R34/R50 是已发布模型。三变体的官方 Paddle checkpoint 已完成目标感知转换、公开资产校验、eager 推理及 ONNX opset 17/TorchScript 导出验证；R18 另完成同权重 Paddle/PyTorch CPU/FP32 完整 COCO val2017 对齐、受控训练 loss 和梯度对齐。本轮新模型族集成的 F2/F4 最终门均为 `APPROVE`，确认共享运行时改动未使既有 v3 合同失效。

F4 的 eager、ONNX、TorchScript 状态是对 Task 1 和 `v0.1.0` 已批准基线的身份继承，不是本轮重新生成的逐 tensor 数值矩阵。具体数值仍以[归档报告](../../archive/rtdetrv3-v0.1.0/reports/README.md)为准。

## 验证环境

Task 1 使用 Python `3.12.13`、PyTorch `2.5.1+cu121`、Paddle `3.3.0`、NumPy `1.26.4`；GPU 探针为 RTX 4090，官方数值对齐本身在 CPU/FP32 下执行。F4 完整非 Paddle 回归为 `761 passed, 92 skipped, 34 deselected`，官方 R18 数值测试为 `1 passed`；R34/R50 的可选 Paddle 数值用例因该命令未提供资产变量而为 `2 skipped`，不覆盖或撤销其既有归档证据。

## 已验证范围

- R18/R34/R50 共 `2,041` 个检测 checkpoint tensor 完成名称、布局和逐 tensor 转换校验。
- R18 的 backbone、neck、transformer、head、后处理、全部受控 loss key 和 `384` 个参数梯度方向通过 Paddle/PyTorch 对齐。
- R18 同权重完整 val2017 的 Paddle/PyTorch CPU 主 AP 差为 `1.65599e-7`。
- 三变体公开 checkpoint 的 eager CPU 用户路径可运行；ONNX/TorchScript 固定 640、动态 batch 1/4/8 已验证，R18 另覆盖固定 608。
- F2 通过 Ruff、Mypy `123 source files`、覆盖率、`761` 项 unit/integration、`21` 项上游 numerical 和图审计。
- F4 通过完整非 Paddle回归、R18 官方数值门、子模块范围审计及受控 baseline mismatch 负例。

## 负面与限制

- R34/R50 ONNX CUDA 分别观察到 `0.00141865 / 0.0375671 px` 和 `0.000972390 / 0.0349426 px` 的最大 score/box 误差，未通过从 R18 外推的 `1e-3 / 0.03 px` 门；只能声明功能与候选匹配，不声明该严格门通过。
- 动态 batch 不表示动态高宽；输入空间尺寸变化必须重新导出。
- 未验证 FP16/BF16、TensorRT、C++/mobile、ORT I/O Binding 或任意依赖版本组合。
- 完整 72 epoch、多 seed、R34/R50 长训仍为 deferred；一轮训练和短 probe 只证明训练链可运行。
- 同权重数值对齐不证明 Paddle/PyTorch optimizer 每元素更新、随机增强序列或完整 schedule 收敛一致。

## 复现入口

```bash
uv run --extra dev pytest tests/numerical/test_r18_official_checkpoint.py
uv run --extra test pytest -m "not paddle"
uv run python tools/dev/compare_upstream_pytorch.py \
  --baseline <task-1-receipt.json> \
  --family rtdetrv3 \
  --surfaces eager,onnx,torchscript \
  --output <scope-receipt.json>
```

第一条命令需要按测试说明提供仓库外官方 Paddle checkpoint。已发布资产身份见 [`rtdetrv3_coco.yml`](../../../configs/checkpoints/rtdetrv3_coco.yml)。

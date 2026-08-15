# 部署导出

`detrs.deploy` 提供 ONNX 与 TorchScript 导出入口。导出使用 tensor-only 适配层,生成 ONNX opset 17 与 traced TorchScript;空间尺寸固定、batch 动态,改变空间尺寸时需要重新导出。运行导出的产物需要 `export`、`export-gpu` 或 `dev` extra 中的 onnxruntime。

::: detrs.deploy

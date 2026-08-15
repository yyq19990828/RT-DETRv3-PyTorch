# 使用指南

本指南面向安装包和仓库用户，集中说明环境、模型资产、训练、评估、推理、转换与导出。模型支持状态、逐变体指标和限制以[模型文档](../models/README.md)为准。

- [安装](install.md):Python/uv 环境要求、各安装模式、中国大陆镜像与子模块初始化。
- [模型与 checkpoint](model-assets.md):Models CLI 的 list/download/verify 与各模型族的权重获取边界。
- [训练与评估](training-evaluation.md):train/eval 工作流、EMA 评估与 format-version-1 checkpoint 合同。
- [推理](inference.md):eager、ONNX 与 TorchScript 三种推理入口及设备/provider 边界。
- [权重转换与导出](conversion-export.md):Paddle 权重转换与 ONNX/TorchScript 导出。
- [CLI 与配置边界](cli-config.md):`detrs` 单命令入口与未迁移参数的报错合同。

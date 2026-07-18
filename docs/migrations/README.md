# 迁移经验

本目录用于沉淀 PaddlePaddle 到 PyTorch 迁移中可复用的结论，包括框架语义对比、已知局限、数值对齐方法和排错经验。

## 文档索引

- [Paddle 与 PyTorch 模块对比](paddle-pytorch-comparison.md)：迁移早期的组件级对比快照。
- [历史一致性检查](consistency-check.md)：早期实现与 Paddle 结构对照记录。
- [迁移局限](limitations.md)：当前尚未覆盖或无法等价保证的边界。
- [排错经验](troubleshooting.md)：环境、子模块、配置、权重转换和数值差异的常见问题。
- [历史规格整合记录](spec-history.md)：原 `specs/001`–`005` 的去向、冲突和可信度说明。
- [权重转换经验](weight-conversion.md)：参数命名、张量布局、校验层级和已知边界。
- [注册与配置迁移](registry-and-configuration.md)：`workspace`、`__inject__`、`__shared__` 和 YAML 继承的语义。
- [RT-DETRv3 配置迁移指南](configuration-guide.md)：Paddle YAML 字段的直接支持、必要改写和未支持矩阵。
- [训练与数值验证](training-and-validation.md)：数据、优化器、调度器、DDP 及分层对齐方法。
- [CLI 与导出迁移](cli-and-export.md)：当前 Infer 数据流、Paddle 参数差异以及 ONNX/TorchScript 验收约束。

## 编写约定

- 记录 Python、Paddle、PyTorch、CUDA 和关键依赖版本。
- 区分“已通过测试”、“根据源码推断”和“待验证”，不把形状通过当作数值等价。
- 给出最小复现命令、关键报错和已验证的解决方法。
- 历史快照必须标注日期，并明确它不代表当前验收状态。

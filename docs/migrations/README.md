# 迁移经验

本目录只沉淀 PaddlePaddle 或原生 PyTorch 上游迁移中可跨模型复用的结论，包括框架语义对比、数值对齐方法和排错经验。各模型专属合同统一见[模型文档](../models/README.md)，RT-DETRv3 历史证据见[`v0.1.0` 归档](../archive/rtdetrv3-v0.1.0/README.md)。

## 文档索引

- [Paddle 与 PyTorch 模块对比](paddle-pytorch-comparison.md)：组件级语义差异与迁移注意事项。
- [排错经验](troubleshooting.md)：环境、子模块、配置、权重转换和数值差异的常见问题。
- [权重转换经验](weight-conversion.md)：参数命名、张量布局、校验层级、统一渲染可视化和已知边界。
- [注册与配置迁移](registry-and-configuration.md)：`workspace`、`__inject__`、`__shared__` 和 YAML 继承的语义。
- [训练与数值验证](training-and-validation.md)：数据、优化器、调度器、DDP、恢复及通用部署迁移方法。
- [上游 PyTorch 数值对齐](upstream-pytorch-parity.md)：原生 PyTorch 上游的 revision/资产预检、checkpoint container/state、具名张量比较、固定容差和证据驱动合同。

## 编写约定

- 记录 Python、Paddle、PyTorch、CUDA 和关键依赖版本。
- 区分“已通过测试”、“根据源码推断”和“待验证”，不把形状通过当作数值等价。
- 给出最小复现命令、关键报错和已验证的解决方法。
- 历史快照必须标注日期，并明确它不代表当前验收状态。

# API 参考

本节文档由 [mkdocstrings](https://mkdocstrings.github.io/) 从 `src/detrs` 源码自动生成,覆盖包的全部子包。`detrs` 的核心运行时不依赖 Paddle;`detrs.conversion` 中的 Paddle 权重转换属于 `dev` extra 场景,相关模块在导入时才惰性加载 Paddle。

| 子包 | 职责 |
|---|---|
| [`detrs.core`](core.md) | workspace 注册系统与 YAML 配置加载 |
| [`detrs.cli`](cli.md) | `detrs` 单命令入口与全部子命令 |
| [`detrs.data`](data.md) | 数据源、采样、增强与读取管道 |
| [`detrs.engine`](engine.md) | 训练器、评估循环、回调与分布式环境 |
| [`detrs.modeling`](modeling.md) | 模型结构:架构、骨干、检测头、损失、标签分配等 |
| [`detrs.optimizer`](optimizer.md) | AdamWDL、调度器与 EMA |
| [`detrs.metrics`](metrics.md) | COCO/YOLO 评估指标 |
| [`detrs.deploy`](deploy.md) | ONNX 与 TorchScript 导出 |
| [`detrs.conversion`](conversion.md) | Paddle checkpoint 到 PyTorch 的权重转换 |
| [`detrs.utils`](utils.md) | checkpoint、日志、分布式与可视化工具 |

配置文件的注册与注入语义(`__inject__`、`__shared__`、YAML 继承)见[注册与配置语义](../migrations/registry-and-configuration.md)。

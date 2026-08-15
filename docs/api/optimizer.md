# 优化器与 EMA

`detrs.optimizer` 提供带解耦 weight decay 与 layerwise lr decay 的 `AdamWDL`、`FlatCosine` 学习率调度、按配置构造优化器的 `OptimizerBuilder`,以及训练协议使用的 `ModelEMA`。

::: detrs.optimizer

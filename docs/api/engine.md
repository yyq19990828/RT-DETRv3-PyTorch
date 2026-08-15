# 训练引擎

`detrs.engine` 提供训练器、评估循环、回调机制、训练协议、朴素 SyncBN 实现与分布式环境封装。训练 checkpoint 采用 format-version-1,保存模型、EMA、optimizer、scheduler、scaler 和 RNG 状态,支持 epoch 边界的确定性恢复。

::: detrs.engine

# CLI 与配置边界

公开入口为单一 `detrs` 命令,子命令为 `train`、`eval`、`infer`、`convert`、`export` 和 `models`(亦可用 `python -m detrs`)。历史上的 `rtdetrv3-*` 命令与 `tools/*.py` 兼容包装器已随包名重命名为 `detrs` 移除。

未迁移的 Paddle CLI 参数会明确报错，不会静默忽略。全部子命令的完整参数说明见[CLI 参考](cli-reference.md)(由 `--help` 输出自动生成);RT-DETRv3 的配置覆盖与详细 CLI 合同见[配置迁移指南](../models/rtdetrv3/configuration-guide.md)和 [CLI 与导出边界](../models/rtdetrv3/cli-and-export.md)。

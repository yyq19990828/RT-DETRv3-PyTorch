# Legacy tests

这里保留迁移早期针对 `rtdetrv3_pytorch.models`、分类 Registry、旧版构建器和旧模型构造参数编写的测试。

当前可安装包使用 `ppdet_pytorch`、`ppdet_pytorch.core.workspace` 以及 Paddle 风格的统一注册系统；旧测试的前提与现有实现不兼容，因此默认 pytest 不收集本目录。若后续需要恢复其中的覆盖场景，应按当前公开 API 重写后移回 `tests/unit`、`tests/integration` 或 `tests/numerical`，不要重新引入旧包兼容层。

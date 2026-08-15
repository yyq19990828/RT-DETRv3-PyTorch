# 核心注册与配置

`detrs.core` 提供 workspace 注册系统与配置加载。所有模型组件通过 `register` 装饰器登记到全局 workspace,YAML 配置中的 `__inject__`/`__shared__` 语义与配置继承都在这里实现;`load_config` 是 CLI 加载配置文件的统一入口。

::: detrs.core.workspace

::: detrs.core.config

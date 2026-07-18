# 注册与配置迁移

当前实现使用 `ppdet_pytorch.core.workspace` 的统一注册表，保留 PaddleDetection 的 `@register`、`global_config`、`__inject__`、`__shared__`、`from_config()` 和 `_BASE_` YAML 继承概念。

## 已采用的方向

- 单一 `global_config` 同时存放配置值和注册类的 schema。
- `@register` 返回原类，不使用 metaclass，避免与 `torch.nn.Module` 的类机制冲突。
- 类的 `__inject__` 声明需要递归创建的依赖，`__shared__` 从全局配置填充共享值。
- `from_config()` 用于根据已有配置和上游 shape 生成构造参数。
- `_BASE_` 相对于当前 YAML 文件递归解析，子配置覆盖 base 配置。

## 当前 `create()` 的实际顺序

以当前代码为准，不沿用历史规格中的理想化描述：

1. 根据类名从 `global_config` 获取 schema 和默认构造参数。
2. 解析 `__shared__`，已在模块配置中给出的值优先，否则从全局值或 schema 默认值获取。
3. 调用 `from_config(config, **kwargs)` 并合并返回值。
4. 解析 `__inject__`，将字符串或带 `name` 的字典递归 `create()` 为实例。
5. 使用最终 `cls_kwargs` 调用构造函数。

一个重要边界是：当前 `create()` 只接受类或类名字符串，并不直接接受历史文档中的 `{'type': ...}` 字典。传入的显式 `kwargs` 会传给 `from_config()`，但不会自动作为最终构造参数覆盖。这两点需要在进一步追求 Paddle API 兼容前用测试锁定。

## 导入时注册

装饰器只在定义类的模块被导入时执行。历史上曾出现 DINOv3Loss 已加装饰器但未进入注册表，根因是 losses 模块没有被包初始化路径导入。

新增组件时必须同时验证：

- 定义模块能直接导入。
- 包的 `__init__.py` 或启动流程会触发该模块导入。
- `get_registered_modules()` 包含该组件。
- 从 YAML 合并后可通过 `create(name)` 构建。

## 全局配置状态

`load_config()` 会将结果合并进模块级 `global_config`，多次加载可以累积上一个测试或命令的值。因此：

- 单元测试应在前后备份/恢复或清空 `global_config`。
- 同一进程加载多个模型配置时不能假设完全隔离。
- 修改 schema 中的 list/dict 时要防止实例间共享可变对象。

## 从历史方案沉淀的规则

- 不要重建 BACKBONE/NECK/HEAD 等分类 Registry 作为新的并行系统。
- 不用 metaclass 做自动注册；保持装饰器对原类构造语义的侵入最小。
- 不把 Paddle 文档的参数解析顺序当作当前实现事实；用单元测试覆盖 shared、inject、from_config 和显式参数冲突。
- 配置错误应报出组件名、字段和可用注册项，而不是只报递归构造失败。
- 旧 Registry API 的用例保留在 `tests/legacy/`；恢复覆盖时应按当前 `workspace` 语义重写。

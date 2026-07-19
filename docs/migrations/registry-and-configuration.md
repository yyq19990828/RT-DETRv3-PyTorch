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

1. 输入可以是注册类、注册名字符串，或带 `name`/`type` 的配置映射。字符串若指向全局命名配置块，会先解析该块再递归构建。
2. 复制注册 schema 并合并配置块，避免嵌套构建修改后续命令共享的 schema。
3. 解析 `__shared__`；模块配置已给出的值优先，否则从全局值或 schema 默认值获取。
4. 只将 schema 之外的上下文参数（例如 `input_shape`）传给 `from_config()`，合并它返回的构造参数。
5. 解析 `__inject__`，将字符串或带 `name`/`type` 的字典递归 `create()` 为实例。
6. 合并显式构造参数，它们的优先级高于配置块和 `from_config()` 返回值；显式注入参数保留第 5 步解析后的实例，不会被原始字符串或字典覆盖，然后调用构造函数。

当前冲突优先级按字段类型区分：

- 普通构造字段：显式 kwarg > `from_config()` 返回值 > 组件配置 > schema 默认值。
- shared 字段：显式 kwarg > `from_config()` 返回值（若它返回同名字段）> 组件配置 > 全局 shared 值 > schema 默认值。
- inject 字段：显式注入目标 > 组件配置/非空 schema 目标 > `from_config()` 返回实例；字符串和带 `name`/`type` 的映射会递归构建。

**已验证（2026-07-19）**：命名配置块、上述冲突矩阵、显式注入映射解析、`from_config()` 上下文隔离，以及空的显式 `merge_config` 目标都有活跃单元测试。这只证明当前合同，不代表已复制 PaddleDetection 的全部冲突处理语义。

## 导入时注册

装饰器只在定义类的模块被导入时执行。历史上曾出现 DINOv3Loss 已加装饰器但未进入注册表，根因是 losses 模块没有被包初始化路径导入。

新增组件时必须同时验证：

- 定义模块能直接导入。
- 包的 `__init__.py` 或启动流程会触发该模块导入。
- `get_registered_modules()` 包含该组件。
- 从 YAML 合并后可通过 `create(name)` 构建。

## 全局配置状态

`load_config()` 仍把当前结果放进模块级 `global_config`，但每次调用会先成功解析完整 `_BASE_` 树，然后保留注册 schema、清除上一份 YAML 的运行时值，再合并新配置。因此：

- 单元测试应在前后备份/恢复或清空 `global_config`。
- 同一进程连续 `load_config()` 默认隔离；若需要增量 override，应在一次加载后显式调用 `merge_config()`。
- 解析新 YAML 失败时，当前活动 workspace 不会被清空。
- 修改 schema 中的 list/dict 时要防止实例间共享可变对象。
- 向显式空字典合并时，不得用容器的布尔值判断是否回退到 `global_config`；应只以 `None` 表示“未提供目标”。

**已验证（2026-07-19）**：同一进程 R18 → R50 → R18 构建恢复正确 depth 与参数量；定向测试同时证明上一份普通字段和命名配置块被移除、注册类仍可用、失败解析保留当前配置。

## Schema 类型检查

`SchemaDict` 在安装 Typeguard 时会校验带注解的普通构造参数；注入字段仍由注册表递归解析，不按构造参数注解直接检查。Typeguard 4 的 `check_type` 签名是 `check_type(value, expected_type)`，不能继续传入旧版的参数路径字符串，否则合法值也会因调用签名错误被记为类型不匹配。

**已验证（2026-07-19）**：活跃回归测试同时覆盖合法 `int` 值通过、错误 `str` 值被列入 mismatch 并由 `validate()` 拒绝。这个结果只证明当前 Schema/typeguard 边界，不等同于 YAML 已具有完整静态类型安全；未安装可选 Typeguard 时仍保持原有的跳过运行时类型检查行为。

## 从历史方案沉淀的规则

- 不要重建 BACKBONE/NECK/HEAD 等分类 Registry 作为新的并行系统。
- 不用 metaclass 做自动注册；保持装饰器对原类构造语义的侵入最小。
- 不把 Paddle 文档的参数解析顺序当作当前实现事实；用单元测试覆盖 shared、inject、from_config 和显式参数冲突。
- 配置错误应报出组件名、字段和可用注册项，而不是只报递归构造失败。
- 旧 Registry API 的用例保留在 `tests/legacy/`；恢复覆盖时应按当前 `workspace` 语义重写。

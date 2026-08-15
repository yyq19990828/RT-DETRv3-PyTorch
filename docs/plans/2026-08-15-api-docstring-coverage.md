# API docstring 全量补全计划

- 状态：`completed`
- 创建日期：`2026-08-15`
- 最后更新：`2026-08-15`
- 负责人：`yyq08228`

## 背景

文档站的 API 参考由 mkdocstrings 从 `src/detrs` 自动生成。评审发现注册组件(可写进 YAML 的公开配置面)docstring 覆盖不足:68 个注册类中仅 43 个有类 docstring、17 个有 Args;25 个完全缺失,包括 `TrainReader`/`EvalReader`/`TestReader`、`RTDETRV3`、`DETRPostProcess`、`PResNet` 等高频组件。

## 范围

- 包含:P1(25 个无 docstring 注册类补类说明与 Args)、P2(26 个"有说明缺 Args"中真正缺 Args 的 16 个类补 Args)、P3(`core/workspace.py` 的 `dump_value`/`get_registered_modules`/`make_partial`)。
- 不包含:内部实现函数(约 655 个无 docstring 函数属实现细节)、CLI argparse 处理器、vendored dinov3 代码。

## 依赖

- 无代码行为变化,全部为 docstring/注释级修改。

## 目标与非目标

### 目标

- 68/68 注册组件有类 docstring;参数说明按"基类集中文档化"原则覆盖。
- mkdocs API 页从裸签名变为可用的配置参考。

### 非目标

- 不补内部函数 docstring;不写中文 docstring(仓库惯例为英文 Google 风格)。

## 实施步骤

- [x] AST 扫描生成 P1/P2 清单(含 `@register` 签名与构造参数);验证:清单与源码一致。
- [x] P1:补 data 13 个、modeling 12 个注册类;验证:扫描 0 缺失。
- [x] P2:补 16 个类 Args(`TrainingProtocol` 为无构造参数的接口类、`ResNet` 的 `__init__` 已有 Args,均不重复补);验证:扫描确认。
- [x] P3:workspace 3 个函数;验证:mypy/ruff 通过。

## 风险与回退

- 风险:docstring 描述与实际行为不符误导用户。缓解:逐类阅读构造实现与校验逻辑后撰写,不臆测(GAM/`sm_use` 等语义均对照实现确认)。
- 回退:全部改动为 docstring 文本,revert 单个 commit 即可。

## 验收

- [x] `scripts/check_quality.py`(ruff format/lint + mypy)通过。
- [x] `mkdocs build --strict` 零警告;抽查 `DETRPostProcess`、`TwoStageDetectionProtocol` 的新 docstring 已在站点渲染。
- [x] `scripts/check_docs.py` 通过(104 个 markdown 文件)。
- [x] 覆盖率:注册组件类 docstring 43 → **68/68**,类级 Args 17 → **50**(其余 18 个为参数继承自基类的包装类,Args 集中在基类文档)。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-15 | 参数说明集中在基类,包装类注明"参数与基类共享" | Reader/架构包装类只改默认值,重复 Args 会产生两份维护点 |
| 2026-08-15 | DETRLoss/HungarianMatcher 的 `__init__` 旧 Args 并入类 docstring | mkdocstrings 会同时渲染两份,合并避免重复 |
| 2026-08-15 | `DETRLoss.__call__` 等行为细节不在本次范围 | 本次只覆盖配置面(构造参数),方法级 docstring 已有存量 |

## 完成记录

- 环境:Python 3.12、uv 0.11.29、Linux x86_64。
- 改动:24 个源文件、约 54 个对象(25 个 P1 类、16 个 P2 类、3 个 P3 函数、10 处合并/清理),全部为 docstring/注释级修改,无任何可执行代码变化。
- 验证命令与结果:`uv run --extra quality python scripts/check_quality.py` 通过;`uv run --extra docs mkdocs build --strict` 零警告;`uv run --no-sync python scripts/check_docs.py` 通过。
- 偏差:计划阶段将 `TrainingProtocol` 计入 P2,实施时确认其为无构造参数的接口类,已文档化的接口说明即为完整覆盖,不再补空 Args。
- 后续事项:内部函数 docstring 与方法级文档未排期;`docs/guides/README.md` 拆分多页仍未排期。

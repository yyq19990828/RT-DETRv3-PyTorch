# 使用指南拆分计划

- 状态：`completed`
- 创建日期：`2026-08-15`
- 最后更新：`2026-08-15`
- 负责人：`yyq08228`

## 背景

`docs/guides/README.md` 是 158 行的单页使用手册,在文档站导航中只占一个入口,读者无法直接跳到安装/推理/导出等主题。文档站计划阶段将其拆分列为未排期事项,本次补齐。

## 范围

- 包含:把 `guides/README.md` 的六个章节拆为独立页面,`README.md` 保留为分区总览;修正 quickstart 中两处带锚点的入链;更新 `mkdocs.yml` 导航。
- 不包含:内容重写(逐字搬运)、guides 以外文档的结构调整。

## 依赖

- 无。

## 目标与非目标

### 目标

- 使用指南在站点导航中按主题分页(安装/模型资产/训练评估/推理/转换导出/CLI 边界)。
- 仓库内与站内链接全部可解析,零死链。

### 非目标

- 不改写原文表述与命令;不改 quickstart/模型文档内容(除两处链接目标)。

## 实施步骤

- [x] 扫描 `guides/README.md` 全部入链;验证:仅 quickstart 两处带锚点,其余均指向落地页。
- [x] 拆分六页 + 重写落地页导航;验证:内容与原文件逐字一致(仅标题层级调整)。
- [x] quickstart 锚点链接改指新页面;`mkdocs.yml` 导航展开为分区;验证:门禁通过。

## 风险与回退

- 风险:外部(站点外)若有指向旧锚点 `guides/README.md#section` 的链接会失效。缓解:站点为新上线,尚无外部入链;GitHub 上阅读时 `README.md#锚点` 由落地页保留大部分标题无关紧要。
- 回退:六个新页面与 landing 改动可整体 revert。

## 验收

- [x] `scripts/check_docs.py` 通过(111 个 markdown)。
- [x] `mkdocs build --strict` 零警告;`site/guides/` 下 7 个页面(install/model-assets/training-evaluation/inference/conversion-export/cli-config/quickstart)全部生成。
- [x] quickstart 两处链接(`install.md`、`inference.md#onnx-与-torchscript-推理`)在站内解析正常。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-15 | Eager 与 ONNX/TorchScript 推理合并为一页 | 同属 `detrs infer` 入口,设备/provider 边界需要对照阅读 |
| 2026-08-15 | 保留 `guides/README.md` 作为总览落地页 | `docs/README.md` 索引与根 README 等 7 处入链无需改动;mkdocs 中作为分区首页 |

## 完成记录

- 环境:Python 3.12、uv 0.11.29、Linux x86_64。
- 改动:新增 6 个指南页面,`guides/README.md` 从 158 行手册改为总览导航;quickstart 修正 2 处链接;`mkdocs.yml` 导航从单入口展开为 7 项分区。
- 验证:`check_docs.py` 与 `mkdocs build --strict` 均通过。
- 后续事项:无;guides 域文档结构自本次起稳定。

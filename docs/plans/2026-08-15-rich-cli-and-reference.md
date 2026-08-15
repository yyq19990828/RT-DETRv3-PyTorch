# CLI rich 美化与 CLI 参考自动生成计划

- 状态：`completed`
- 创建日期：`2026-08-15`
- 最后更新：`2026-08-15`
- 负责人：`yyq08228`

## 背景

CLI 输出为朴素 print/logging 文本;`models list` 是手写定宽表,训练/评估过程无进度条。同时文档站缺少一份完整的 CLI 参数参考(参数只能跑 `--help` 查看)。本计划(1)用 rich 统一美化输出,(2)在构建管线中从真实 parser 自动生成 CLI 参考页。

## 范围

- 包含:rich 依赖引入与共享 console;CLI 层(models list 表格、infer 横幅与逐图行、export 摘要、train 启动面板)与 engine 层(LogPrinter 进度条、eval 进度与指标表)美化;`scripts/generate_cli_reference.py` 与 `docs/guides/cli-reference.md`(提交生成文件形态);docs workflow 防漂移。
- 不包含:`--json` 与 verify/download 的 JSON 机器接口、argparse help 文本(测试契约与文档源)、pycocotools 自带 AP 表、rich_argparse。

## 依赖

- rich>=13,<15(python 3.9–3.12 全区间可用),进入 base dependencies。

## 目标与非目标

### 目标

- TTY 下 CLI 与训练/评估输出有现代化呈现;管道/CI 下自动退化为纯文本(零 ANSI)。
- 文档站有与代码永远一致的 CLI 参考页,CI 防漂移。

### 非目标

- 不改变任何 stdout 机器接口与测试契约。

## 实施步骤

- [x] rich 依赖 + `utils/console.py` 单例;验证:uv lock/sync、单测。
- [x] CLI 层四个子命令 + engine 层 LogPrinter/eval 美化;验证:全量非 Paddle 测试 832 通过。
- [x] LogPrinter 双路径:TTY 走 rich Progress,非 TTY 走原 `logger.info` Paddle 风格单行;验证:`test_log_printer_uses_global_average_batch_time_for_eta` 原样通过。
- [x] 提交阶段一 `9faff31`。
- [x] `scripts/generate_cli_reference.py`(进程内调用 dispatch parser,`COLUMNS=100` 固定宽度,`--check` 幂等校验);验证:两次运行输出一致。
- [x] 生成并提交 `docs/guides/cli-reference.md`;nav、`cli-config.md` 交叉链接、docs workflow `--check` 防漂移与触发路径;验证:check_docs、`mkdocs build --strict` 通过。

## 风险与回退

- 风险:训练中其他 logger 行与 rich Progress Live 交错显示。缓解:epoch 结束与 eval 模式开始时停止进度条;逐步日志间隔内基本无其他输出。
- 风险:只改 CLI 源码不改文档时防漂移不触发。缓解:docs workflow 触发路径包含 `src/detrs/**` 与生成脚本。
- 回退:两个 commit 独立可 revert;rich 仅为展示层,JSON/日志契约未动。

## 验收

- [x] `check_quality.py`(ruff+mypy,135 源文件)、`check_docs.py`(113 markdown)、`mkdocs build --strict` 零警告。
- [x] 全量非 Paddle 测试 832 通过、96 跳过;新增 console/models-list 渲染单测。
- [x] 冒烟:`detrs models list` 管道输出为纯文本表格(grep ANSI 计数 0)。
- [x] `uv build` + `check_release.py` 通过(rich 入 wheel 依赖)。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-15 | LogPrinter 双路径而非全面替换 | 唯一日志内容测试断言拦截 `logger.info`;非 TTY(测试/CI)保持原契约,TTY 增强 |
| 2026-08-15 | 防漂移用脚本 `--check` 而非 `git diff` | 不依赖 checkout 状态,报错信息直接指向再生成命令 |
| 2026-08-15 | CLI 参考页采用提交生成文件形态 | 维护者选择;GitHub 上可读,CI `--check` 防漂移 |
| 2026-08-15 | `--help` 保持 argparse 纯文本 | 是测试契约(`usage: detrs`)与文档页生成源,不引入 rich_argparse |
| 2026-08-15 | 追加:引入 rich-argparse 美化 `--help`,自定义 formatter 保留小写标题 | 维护者确认期望帮助文本也美化;`DetrsHelpFormatter` 覆盖 `group_name_formatter` 保持 `usage:` 小写,测试契约与生成页格式不变 |

## 完成记录

- 环境:Python 3.12、uv 0.11.29、Linux x86_64、rich 14.3.4、torch 2.5.1+cu121。
- 提交:`9faff31`(阶段一,11 文件)与本提交(阶段二)。
- 验证:见验收;另本地补装 `nvidia-nccl-cu12` 修复了 venv 中文件缺失导致的 torch 导入失败(环境修复,不涉及仓库)。
- 偏差:无。
- 后续事项:rich 渲染的 TTY 视觉效果需维护者在真实终端确认(自动化环境无法验证颜色/动态刷新)。

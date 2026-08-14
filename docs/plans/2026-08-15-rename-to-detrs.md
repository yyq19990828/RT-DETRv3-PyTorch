# 包与 CLI 重命名:ppdet_pytorch → detrs

- 状态：`completed`
- 创建日期：`2026-08-15`
- 最后更新：`2026-08-15`
- 负责人：maintainer

## 背景

仓库起源于 RT-DETRv3 的 Paddle-to-PyTorch 迁移,Python 包名 `ppdet_pytorch` 与六个 `rtdetrv3-*` 公开命令均属历史兼容命名。仓库已更名为 DETR-series 并集成 D-FINE、DEIM、RT-DETRv4、DEIMv2 等模型家族,旧名称不再覆盖当前定位。经维护者决策:包与项目更名为 `detrs`(PyPI 未占用),CLI 收敛为单个 `detrs` 命令 + 子命令(`detrs train/eval/infer/export/convert/models`),不保留旧命令别名。

## 范围

- 包含:`src/ppdet_pytorch` → `src/detrs` 目录与全部 import 改名;项目名 `rtdetrv3-pytorch` → `detrs`;统一 CLI 分发器;`tools/` 四个兼容包装器删除;工件字符串(`rtdetrv3-export.json`、tempdir 前缀、User-Agent)更名;门禁脚本(check_release/check_coverage/check_quality)、CI 工作流、测试与当前面向用户的文档同步。
- 不包含:历史文档改写(`docs/archive/**`、`docs/plans/**`、带日期的验证证据记录);`rtdetrv3` 等模型家族命名(configs/rtdetrv3/、architectures/rtdetrv3.py、DEFAULT_FAMILY);third-party 子模块;本地目录名(仓库目录仍为 RT-DETRv3-PyTorch)。

## 依赖

- uv 管理的 `.venv`(dev extra 环境)。
- PyPI 名称 `detrs` 可用(2026-08-15 验证:JSON API 404)。

## 目标与非目标

### 目标

- 全仓(排除历史文档与 third-party)不再出现 `ppdet_pytorch` token。
- `detrs --help` 列出六个子命令;各子命令参数、后验证逻辑与原命令一致;`python -m detrs` 可用。
- CI 同款验证全绿:`pytest -m "not paddle"`、`scripts/check_quality.py`、`scripts/check_docs.py`、`uv build` + `scripts/check_release.py` + wheel 冒烟。

### 非目标

- 不重构六个子命令模块自身的参数解析(`-o` 语义差异保持原样)。
- 不迁移 tests/legacy/**(norecursedirs 已排除)。
- 不发布到 PyPI。

## 实施步骤

- [x] 创建本计划文档;验证:文档存在且索引已更新。
- [x] `git mv src/ppdet_pytorch src/detrs` + 全仓 token 替换(仅完整 token `ppdet_pytorch`;排除 third-party/、docs/archive/、docs/plans/、tests/legacy/、.venv/、uv.lock);删除 `tools/{train,eval,infer,convert_weights}.py`;验证:`git status` 改动面与预期清单一致(136 重命名、4 删除、3 新增)。
- [x] 新增 `src/detrs/cli/main.py` 分发器与 `src/detrs/__main__.py`,更新 `cli/__init__.py` 文档字符串;验证:`detrs --help`、`detrs <cmd> --help` 冒烟。
- [x] 更新 pyproject.toml(项目名、keywords、scripts、wheel、force-include、ruff/mypy 路径)与工件字符串;验证:`uv lock` 成功。
- [x] 门禁脚本、CI 工作流、测试文件同步;验证:`pytest tests/unit/cli tests/unit/scripts tests/unit/conversion/test_cli.py -q` 196 passed。
- [x] 当前面向用户的文档同步(README、docs/guides、docs/models/*/README、docs/migrations 现行文档、NOTICE);验证:`python scripts/check_docs.py` 通过。
- [x] 全量验证(镜像 CI):`uv sync --extra dev` → pytest → quality/docs → build+release+wheel 冒烟 → 残留 grep;验证:全部通过且口径内 0 残留(17 个 pre-existing 失败除外,见完成记录)。
- [x] 清理中间产物,填写完成记录;验证:`git status` 仅预期改动。

## 风险与回退

- 风险:旧 torchscript 导出目录内 `rtdetrv3-export.json` 元数据文件名变更,旧导出无法被 infer 加载;缓解:重新导出即可(干净切换的既定代价)。
- 风险:仓库外脚本调用 `rtdetrv3-*` 命令将失效;维护者已确认接受。
- 风险:批量替换误伤裸 `ppdet`(Paddle 原包);缓解:仅替换完整 token 并逐一核对动态字符串文件。
- 回退:单次提交承载全部改动,`git revert` 即可整体恢复;`git mv` 保留文件历史。

## 验收

- [x] `grep -rn "ppdet_pytorch"` 在排除 third-party/、docs/archive/、docs/plans/、tests/legacy/、docs/models/**/{validation-report,evidence-index,metrics}.md、.venv/ 后 0 命中(带日期证据记录按仓库规则不改写)。
- [x] `grep -rn "rtdetrv3-"` 同口径仅历史文档命中(唯一例外:docs/guides/README.md:156 的"旧命令已移除"说明句)。
- [x] `uv run --no-sync pytest -p no:cacheprovider -q -m "not paddle"`:794 passed、93 skipped、34 deselected、17 failed——17 个失败全部为 pre-existing(见完成记录),与重命名无关。
- [x] `uv run --no-sync python scripts/check_quality.py`(ruff + mypy 131 文件)与 `scripts/check_docs.py` 通过。
- [x] `uv build` 后 `scripts/check_release.py --wheel dist/*.whl --sdist dist/*.tar.gz` 通过(`detrs-0.1.0`),wheel 覆盖安装后在 checkout 外六个子命令 `--help` 与 packaged config 加载冒烟通过;CI 同款 wheel 冒烟测试 104 passed。
- [x] README 等当前文档不再声称包名/命令为旧名。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-15 | 包/项目名选 `detrs` | 与仓库定位 DETR-series 对齐;导入短;PyPI 未占用;不绑定单一模型家族 |
| 2026-08-15 | CLI 收敛为单 `detrs` 入口 + 子命令,分发器按需导入子模块 | UX 统一;`detrs models` 等轻量子命令不付 torch 导入成本 |
| 2026-08-15 | 不保留 `rtdetrv3-*` 别名,删除 tools/ 兼容包装器 | 维护者确认干净切换;包装器仅文档提及 |
| 2026-08-15 | `rtdetrv3-export.json` 改名 `detrs-export.json` | 彻底去历史品牌;旧导出重导出即可 |

## 完成记录

2026-08-15 完成。改动构成:136 个文件经 `git mv` 重命名(`src/ppdet_pytorch/` → `src/detrs/`)、4 个兼容包装器删除、3 个新文件(`src/detrs/cli/main.py`、`src/detrs/__main__.py`、本计划),其余为 import/字符串/文档同步;`uv.lock` 经 `uv lock` 重新生成(项目名 `detrs`)。

**实现要点与偏差**

- 分发器 `src/detrs/cli/main.py`:`detrs_command` 作为顶层 subparsers dest(避免与 `models` 二级 subparser 的 `command` dest 冲突);仅被请求的子命令挂真实参数解析器(`parents=` + `add_help=False`,prog 自动显示 `detrs <cmd>`),`detrs models` 等轻量子命令不付 torch 导入成本;其余参数原样转发给子模块 `main(argv)`,复用其后验证逻辑。
- `cli/models.py`:`--family/--manifest` 提取为 `_add_manifest_options`,在 `list/verify/download` 二级子命令上以 `default=SUPPRESS` 重复注册——`detrs models --family x list`(旧风格)与 `detrs models list --family x`(嵌套后更自然的位置)均可解析。此为计划外的小幅功能补充,动机是统一入口后选项位置约定改变。
- sed 缩短 import 后 16 个文件偏离 ruff format,经 `ruff format` 恢复;质量门禁全绿。
- 修复工作站 `.venv` 中 onnxruntime 损坏(orphan `onnxruntime` 目录覆盖 GPU 版模块,`InferenceSession` 缺失):`uv sync --extra dev --extra teacher --reinstall-package onnxruntime-gpu` 恢复,2 个 export 测试随之通过。环境原本缺 teacher extra,已一并恢复。

**Pre-existing 失败(与本次重命名无关,未处理)**

- `tests/unit/utils/test_validation_drivers.py` 中 17 个测试失败,根因全部为 `FAIL: [Errno 2] No such file or directory: '.omo/plans/rtdetrv4-merge.md'`——该文件与 `.omo/` 目录从未被 git 跟踪(`git log --all -- .omo` 为空),属工作站本地资产缺失;测试与 `tools/dev/validation_common.py` 的 `DEFAULT_PLAN` 硬编码引用它。任何干净检出上这些测试都会同样失败。后续需维护者决定恢复该文件或改造测试默认值。

**证据命令**(工作目录为仓库根,环境 `uv sync --extra dev --extra teacher`)

- `uv run --no-sync pytest -p no:cacheprovider -q -m "not paddle"` → 794 passed, 93 skipped, 34 deselected, 17 failed(全部 `.omo` 缺失类)
- `uv run --no-sync python scripts/check_quality.py` / `scripts/check_docs.py` → 通过
- `uv build && uv run --no-sync python scripts/check_release.py --wheel dist/*.whl --sdist dist/*.tar.gz` → "release checks passed"
- wheel 覆盖安装后在 checkout 外:`detrs --help`、六个 `detrs <cmd> --help`、packaged config 加载 → 全部通过
- CI 同款 wheel 冒烟:104 passed

**遗留**

- 旧 TorchScript 导出目录(`rtdetrv3-export.json` 元数据)需重新导出后方可被 `detrs infer` 加载。
- 仓库本地目录名仍为 `RT-DETRv3-PyTorch`(与远端 `DETR-series` 不一致),是否重命名由维护者决定,不在本计划范围。
- `.omo/plans/rtdetrv4-merge.md` 缺失问题待维护者决策(见上)。

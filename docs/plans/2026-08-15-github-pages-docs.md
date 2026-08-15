# GitHub Pages 文档站计划

- 状态：`completed`
- 创建日期：`2026-08-15`
- 最后更新：`2026-08-15`
- 负责人：`yyq08228`

## 背景

仓库已有 77 个中文 Markdown 文档,但没有文档站:README 与 docs/ 只能在 GitHub 上浏览,没有导航、搜索和自动生成的 API 参考。需要为 GitHub Pages 建立中文文档站,覆盖快速开始、详细使用手册和自动化 API 接口文档。

## 范围

- 包含:MkDocs Material 站点配置(`mkdocs.yml`)、`docs` extra、首页与快速开始页、`docs/api/` mkdocstrings 自动 API 参考、构建期链接改写 hook、GitHub Pages 部署 workflow、治理文件同步。
- 不包含:为 `src/detrs` 系统补写 docstring、`docs/guides/README.md` 拆分多页、中英双语站、Material Insiders 付费功能(中文 jieba 分词搜索)。

## 依赖

- GitHub Pages 已开通,Source 需手动选择 "GitHub Actions"(维护者一次性操作)。
- uv 锁文件能解析 mkdocs/mkdocs-material/mkdocstrings 在 Python 3.9–3.12 矩阵内的版本。

## 目标与非目标

### 目标

- push 到 main 自动部署中文文档站,PR 只做 `--strict` 构建校验。
- 导航只收录面向用户与维护者的文档;plans/archive 等过程文档仍参与构建(保证站内零死链),但不进导航。
- API 参考由 mkdocstrings 静态分析 `src/detrs` 生成,构建环境不需要 Paddle。

### 非目标

- 不修改既有文档正文(含归档),链接问题在构建期解决。
- 不处理 pyproject 版本号(0.1.0)与 CHANGELOG(1.0.0)不一致问题。

## 实施步骤

- [x] `pyproject.toml` 新增 `docs` extra(mkdocs、mkdocs-material、mkdocstrings[python]、ruff),更新 `Documentation` URL;验证:`uv lock` + `uv sync --extra docs` 成功。
- [x] 编写 `mkdocs.yml`(Material 主题、中文、显式导航、CJK 搜索 separator、mkdocstrings `paths: [src]`);验证:`mkdocs build --strict` 零警告。
- [x] 新写 `docs/index.md` 首页与 `docs/guides/quickstart.md` 快速开始;验证:页面渲染与站内链接检查。
- [x] 新写 `docs/api/` 页面(总览 + 10 个子包 + modeling 8 个子包);验证:每页 mkdocstrings 对象渲染非空。
- [x] 新增 `scripts/docs_hooks.py`:构建期把逃逸出 `docs/` 的相对链接改写为 GitHub 绝对 URL;验证:构建产物中 `blob/main/` 链接存在、零 404 警告。
- [x] 新增 `.github/workflows/docs.yml`(build + deploy 两个 job);验证:YAML 结构与 ci.yml 模式一致。
- [x] 治理同步:plans 索引、docs 索引、README 文档导航、CHANGELOG;验证:`scripts/check_docs.py` 通过。

## 风险与回退

- 风险:docs 文档大量相对链接指向仓库根(configs/src/tests/ROADMAP),共 84 处,直接构建会 404。缓解:hook 在构建期统一改写为 GitHub URL,源文档零改动。
- 风险:griffe 对无标注 docstring 产生大量提示,阻断 `--strict`。缓解:hook 中把 `mkdocs.plugins.griffe` 日志降级到 ERROR。
- 回退:删除 `mkdocs.yml`、`docs/api/`、`docs/index.md`、`docs/guides/quickstart.md`、`scripts/docs_hooks.py`、`.github/workflows/docs.yml`,恢复 pyproject/索引改动即可;不触及模型代码与既有文档正文。

## 验收

- [x] `uv run mkdocs build --strict` 零警告完成("Documentation built in 4.65 seconds")。
- [x] 14 个 API 页面均有渲染对象(锚点计数 19–710 不等;`modeling/necks` 为空占位包,不建页)。
- [x] 站内链接零 404 警告;逃逸链接已改写(models 各页含 `blob/main/` 绝对链接)。
- [x] `scripts/check_docs.py`、`scripts/check_quality.py` 通过(见完成记录)。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-15 | 工具链选 MkDocs Material + mkdocstrings | 现有素材全为中文 Markdown,零转换成本;mkdocstrings 静态分析,构建不需要安装 Paddle |
| 2026-08-15 | 全部构建、精选导航 | models 验证文档约 20 处链接指向 archive 证据;排除 archive 会产生死链,故只控制导航可见性 |
| 2026-08-15 | 逃逸链接构建期改写而非改源文档 | 84 处链接分布在 31 个文件(含归档);改写源文档违反"归档不得改写"且扩散 diff |
| 2026-08-15 | slugify 用小写 unicode slug | 与 GitHub 锚点风格一致,既有文档中的小写锚点链接在站内同样可用 |
| 2026-08-15 | `modeling/necks` 不建 API 页 | 该子包是空占位(`__all__ = []`),特征融合实际在 transformers |

## 完成记录

- 环境:Python 3.12、uv 0.11.29、Linux x86_64;mkdocs 1.6.x、mkdocs-material 9.x、mkdocstrings 1.0.6、mkdocstrings-python 2.0.5、griffe、pymdownx 11.0.1(以 uv.lock 为准)。
- 验证命令与结果:
  - `uv run mkdocs build --strict` → 通过,零 WARNING/INFO 以上的链接问题。
  - `uv run --no-sync python scripts/check_docs.py` → 通过(84 个 markdown 文件)。
  - `uv run --no-sync python scripts/check_quality.py` → 通过(docstring 修正 `optimizer.param_groups[0]["lr"]` 改为行内代码后无 lint/类型回归)。
- 偏差:计划阶段估计逃逸链接 3 处,实际扫描为 84 处,因此将"改 3 处链接"升级为构建期 hook 方案;`docs/api/` 中 core/cli/data/utils 四页因 `__init__.py` 无 re-export 而渲染为空,改为显式模块列表后解决。
- 维护者操作:仓库 Settings → Pages → Source 选择 "GitHub Actions"(一次性)。
- 后续事项:`src/detrs` docstring 补全、guides 大文档拆分、Material Insiders 中文分词搜索,均未排期。

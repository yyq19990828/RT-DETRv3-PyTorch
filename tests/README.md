# 测试指南

测试使用根目录 `pyproject.toml` 中的 pytest 配置，并从已同步的 `.venv` 运行。

```bash
uv run --extra dev pytest
```

常用的分类命令：

```bash
# 单元测试
uv run --extra dev pytest tests/unit

# 集成测试
uv run --extra dev pytest tests/integration

# 数值与 Paddle 对照测试
uv run --extra dev pytest tests/numerical

# 不运行需要 Paddle 的用例
uv run --extra dev pytest -m "not paddle"
```

目录约定：

- `unit/`：当前公开 API 的单元测试，其中 `conversion/` 覆盖 Paddle 到 PyTorch 权重转换。
- `integration/`：跨模块工作流与转换流程。
- `numerical/`：确定性、数值范围及 Paddle 对照验证。
- `legacy/`：迁移早期、依赖旧 API 的历史测试；默认不收集，待按当前 API 重写后再移回。
- `configs/`：测试专用配置。

Paddle 和测试工具均属于 `dev` 附加依赖，开发环境用以下命令创建或更新：

```bash
uv sync --extra dev
```

测试产生的缓存、临时目录和构建产物应在测试后清理，不提交到仓库。

# 测试指南

测试使用根目录 `pyproject.toml` 中的 pytest 配置，并从 uv 管理的 `.venv` 运行。完整测试、质量和文档维护流程见[开发者指南](../docs/development/README.md)。

## 常用命令

```bash
# 不依赖 Paddle 的测试环境与全量测试
uv sync --extra test
uv run --extra test pytest -m "not paddle"

# 单元、集成与覆盖率门禁
uv run --extra test pytest tests/unit
uv run --extra test pytest tests/integration
uv run --extra test python scripts/check_coverage.py

# Paddle 转换和数值对齐
uv sync --extra dev
uv run --extra dev pytest tests/numerical
uv run --extra dev pytest
```

需要官方 checkpoint、COCO、GPU 或 DINOv3 teacher 的用例可能需要额外资产和环境变量；缺失时应按测试合同明确跳过或失败。不要把 smoke、shape 或成功加载结果描述成数值、精度或训练收敛证据。

## 目录约定

- `unit/`：当前公开 API 的局部行为，其中 `conversion/` 覆盖 Paddle 到 PyTorch 权重转换。
- `integration/`：跨模块工作流、训练恢复、打包与运行时。
- `numerical/`：确定性、官方 checkpoint 及 Paddle/上游 PyTorch 数值对照。
- `legacy/`：迁移早期旧 API 历史测试，默认不收集；需要恢复时按当前 API 重写。
- `configs/`：测试专用配置。

测试完成后清理由本次工作产生的缓存、临时 checkpoint、导出和构建产物，不删除 `.venv`。

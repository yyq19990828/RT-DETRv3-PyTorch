# 开发者指南

本指南面向维护、测试、文档和发布工作。模型使用方法见[使用指南](../guides/README.md)，迁移语义与数值验证方法见[迁移经验](../migrations/README.md)。

## 环境选择

优先使用仓库的 uv 管理 `.venv`：

```bash
# 非 Paddle 测试
uv sync --extra test

# Paddle 转换、数值对齐和完整开发测试
uv sync --extra dev

# Ruff 与 Mypy
uv sync --extra quality
```

不要把 Paddle 或迁移专用依赖加入核心运行时。`third-party/RT-DETRv3-paddle` 是只读参考子模块。

## 测试

```bash
# 不依赖 Paddle 的全量测试
uv run --extra test pytest -m "not paddle"

# 单元与集成测试
uv run --extra test pytest tests/unit
uv run --extra test pytest tests/integration

# 直接维护范围覆盖率门禁
uv run --extra test python scripts/check_coverage.py

# 包含 Paddle 对齐测试
uv run --extra dev pytest
uv run --extra dev pytest tests/numerical
```

需要 checkpoint、COCO、GPU 或 DINOv3 teacher 的验证必须记录 Python、PyTorch、Paddle、CUDA/cuDNN、设备、配置、数据集、checkpoint checksum、seed、dtype 和容差。缺失外部资产时应明确跳过或失败，不以 shape/smoke 结果代替数值或精度证据。

测试目录约定：

- `tests/unit/`：当前 API 的局部行为。
- `tests/integration/`：跨模块工作流、训练恢复、打包和运行时。
- `tests/numerical/`：确定性、官方 checkpoint 及 Paddle/上游 PyTorch 数值对照。
- `tests/legacy/`：迁移早期旧 API 历史测试，默认不收集。

## 代码质量

```bash
# Ruff format、Ruff lint 和 Mypy
uv run --extra quality python scripts/check_quality.py

# 格式化并应用 Ruff 安全修复
uv run --extra quality python scripts/check_quality.py --fix
```

Ruff 排除只读子模块、历史测试和生成目录；Mypy 覆盖 `src/ppdet_pytorch` 及纳入门禁的脚本。不要顺手格式化或修改不属于当前任务的文件。

## 文档维护

```bash
uv run python scripts/check_docs.py
uv run pytest -q tests/unit/scripts/test_check_docs.py
```

文档职责：

- 根 `README.md` 只保留项目定位、快速开始和导航。
- `docs/guides/` 保存面向用户的任务流程。
- `docs/models/` 保存模型当前合同、指标、验证与证据索引。
- `docs/migrations/` 只保存可跨模型复用的迁移知识。
- `docs/plans/` 只保存活动或延期计划；完成后补齐实际证据并归档。
- `docs/archive/` 保存不可改写为当前状态的日期快照。
- `ROADMAP.md` 只展开仓库级未完成工作。

新增、移动或删除计划及迁移文档时同步更新对应 `README.md` 索引。文档使用仓库相对链接，不提交工作站绝对路径、临时日志或无结论排错记录。

## 构建与发布检查

wheel 包含受支持配置、LICENSE 和 NOTICE，但不携带模型权重、数据集、Paddle 子模块或 DINOv3 teacher 资产。

```bash
uv build

# 校验构建产物；按实际版本替换文件名
uv run python scripts/check_release.py \
  --wheel dist/*.whl \
  --sdist dist/*.tar.gz
```

RT-DETRv3 `v0.1.0` 的 11 个发布资产可以在不存在的目标目录中原子组装并严格校验：

```bash
release_workspace="$(mktemp -d)"
trap 'find "$release_workspace" -depth -delete' EXIT
uv run python scripts/check_release.py \
  --require-models \
  --wheel dist/rtdetrv3_pytorch-0.1.0-py3-none-any.whl \
  --sdist dist/rtdetrv3_pytorch-0.1.0.tar.gz \
  --stage-release-dir "$release_workspace/v0.1.0"
```

发布后把固定 tag 的全部资产下载到空目录并回读：

```bash
release_dir="$(mktemp -d)"
trap 'find "$release_dir" -depth -delete' EXIT
gh release download v0.1.0 --dir "$release_dir"
uv run python scripts/check_release.py --verify-release-dir "$release_dir"
```

历史 checksum 只证明当时产物；每次发布都必须重新构建、生成唯一 checksum 并验证公开下载。完成验证后清理测试缓存、临时 checkpoint、导出、`dist/` 和其他中间产物；保留 `.venv`，除非维护者明确要求删除。

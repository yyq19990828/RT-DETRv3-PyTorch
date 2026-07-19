# M6 覆盖率验证报告

- **状态**：当前已验证
- **验证日期**：2026-07-19
- **范围**：纯 `test` extra 中的非 Paddle 活跃测试对 `src/ppdet_pytorch/` 的语句覆盖率

## 方法

主基线使用 Python `3.12.11`、Pytest `8.4.2` 和 pytest-cov `7.0.0`，从锁文件安装独立的 `test` extra，不安装 Paddle 或 `dev` extra。开发环境可以在仓库 UV `.venv` 中执行同一命令：

```bash
uv run --extra test python scripts/check_coverage.py
```

该命令等价于运行非 Paddle 测试，并对整个 `ppdet_pytorch` 包生成覆盖率：

```bash
pytest -p no:cacheprovider -q -m "not paddle" \
  --cov=ppdet_pytorch --cov-report=term --cov-report=json:<temporary-path>
```

`scripts/check_coverage.py` 将 coverage data 和 JSON 报告写入临时目录，命令结束时自动清理。当前托管 `test` extra 的结果为 `217 passed, 7 skipped, 17 deselected, 6 warnings in 12.91s`；17 项因 `paddle` marker 被排除，7 项因当前环境缺少所需的 Paddle/CUDA 能力跳过。`tests/legacy/` 由 Pytest 配置明确排除，但 `src/ppdet_pytorch/` 内没有源文件被从全包统计中删除。

在提交 `19bcb60` 上，已安装 `dev` extra 的本机 `.venv` 另行观测到 `221 passed, 33 deselected`、全包 `43.11%`；其中 5 个在纯 `test` extra 中跳过的 loss 测试可以执行。为保证托管 CI 可重现，下表和门禁以不含 Paddle 的托管 `test` extra 为准。

GitHub Actions [run 29671674073](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29671674073) 在提交 `19bcb60` 上完成托管复验：Python 3.12 CPU job 为 `216 passed, 7 skipped, 17 deselected`，全包 `5,606/13,195`（`42.49%`），直接维护范围 `1,169/1,783`（`65.56%`），双门禁通过。托管环境比本地隔离环境多覆盖 1 条全包语句；该差异已观测但不影响当前回退下限。

Mypy 扩面后的 GitHub Actions [run 29672051076](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29672051076) 在提交 `6750f62` 上再次通过双门禁：全包 `5,610/13,200`（`42.50%`），直接维护范围 `1,173/1,788`（`65.60%`）。

新增 metrics 活跃测试后，隐藏 GPU 的本地 `.venv` 实测为全包 `5,800/13,200`（`43.94%`）和直接维护范围 `1,173/1,788`（`65.60%`）。GitHub Actions [run 29672789563](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29672789563) 在提交 `e22c2b4` 上完成托管复验：Python 3.12 CPU job 为 `221 passed, 7 skipped, 17 deselected`，全包 `5,801/13,200`（`43.95%`），直接维护范围 `1,173/1,788`（`65.60%`），双门禁通过。托管环境仍比本地多覆盖 1 条 `data` 语句；下表已更新为该次托管结果。

## 当前结果

| 模块 | 语句数 | 覆盖语句 | 覆盖率 |
|---|---:|---:|---:|
| package root | 4 | 4 | 100.0% |
| `cli` | 679 | 486 | 71.6% |
| `conversion` | 649 | 352 | 54.2% |
| `core` | 375 | 254 | 67.7% |
| `data` | 5,584 | 1,464 | 26.2% |
| `deploy` | 85 | 81 | 95.3% |
| `engine` | 705 | 389 | 55.2% |
| `metrics` | 576 | 265 | 46.0% |
| `modeling` | 3,386 | 2,012 | 59.4% |
| `optimizer` | 409 | 224 | 54.8% |
| `utils` | 748 | 270 | 36.1% |
| **全包** | **13,200** | **5,801** | **43.95%** |

直接维护范围指 M1–M5 首批质量门禁中的 `cli`、`conversion`、`core` 和 `deploy`，共 `1,788` 条语句，覆盖 `1,173` 条，覆盖率为 **65.60%**。这个子集用于屏蔽回退，不代表其他模块不维护，也不是对整个“已迁移核心”的最终定义。

## 门禁与限制

当前可执行门禁同时要求：

- 非 Paddle 全包语句覆盖率不低于 **43%**。
- `cli/conversion/core/deploy` 合计覆盖率不低于 **65%**。

两个阈值都低于当前实测值，是防止覆盖率回退的初始下限，不是完成目标。`ROADMAP.md` 中的 90% 目标仍未完成；不应通过排除 `data`、`metrics` 或其他低覆盖源文件来声称全包达标。

下一轮优先覆盖用户可见且当前低于直接维护范围平均值的路径：`conversion/converter.py`、`conversion/validation.py`、`cli/infer.py`、`cli/export.py`、`core/config/yaml_helpers.py` 和 `core/config/schema.py`。之后再以实际新增测试证据逐步提高门禁。

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

`scripts/check_coverage.py` 将 coverage data 和 JSON 报告写入临时目录，命令结束时自动清理。最近一次托管 `test` extra 的结果为 `241 passed, 7 skipped, 17 deselected, 6 warnings in 15.47s`；17 项因 `paddle` marker 被排除，7 项因当前环境缺少所需的 Paddle/CUDA 能力跳过。`tests/legacy/` 由 Pytest 配置明确排除，但 `src/ppdet_pytorch/` 内没有源文件被从全包统计中删除。

在提交 `19bcb60` 上，已安装 `dev` extra 的本机 `.venv` 另行观测到 `221 passed, 33 deselected`、全包 `43.11%`；其中 5 个在纯 `test` extra 中跳过的 loss 测试可以执行。为保证托管 CI 可重现，下表和门禁以不含 Paddle 的托管 `test` extra 为准。

GitHub Actions [run 29671674073](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29671674073) 在提交 `19bcb60` 上完成托管复验：Python 3.12 CPU job 为 `216 passed, 7 skipped, 17 deselected`，全包 `5,606/13,195`（`42.49%`），直接维护范围 `1,169/1,783`（`65.56%`），双门禁通过。托管环境比本地隔离环境多覆盖 1 条全包语句；该差异已观测但不影响当前回退下限。

Mypy 扩面后的 GitHub Actions [run 29672051076](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29672051076) 在提交 `6750f62` 上再次通过双门禁：全包 `5,610/13,200`（`42.50%`），直接维护范围 `1,173/1,788`（`65.60%`）。

新增 metrics 活跃测试后，隐藏 GPU 的本地 `.venv` 实测为全包 `5,800/13,200`（`43.94%`）和直接维护范围 `1,173/1,788`（`65.60%`）。GitHub Actions [run 29672789563](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29672789563) 在提交 `e22c2b4` 上完成托管复验：Python 3.12 CPU job 为 `221 passed, 7 skipped, 17 deselected`，全包 `5,801/13,200`（`43.95%`），直接维护范围 `1,173/1,788`（`65.60%`），双门禁通过。托管环境仍比本地多覆盖 1 条 `data` 语句；下表已更新为该次托管结果。

新增 utils 活跃测试后，显式设置 `CUDA_VISIBLE_DEVICES=''` 的本地 `.venv` CPU 实测为 `225 passed, 5 skipped, 34 deselected`，全包 `5,853/13,203`（`44.33%`），直接维护范围 `1,173/1,788`（`65.60%`）。未隐藏 GPU 时额外执行 5 个 CUDA 可用用例，得到 `44.96%`；该设备差异不用于 CPU 门槛。GitHub Actions [run 29673364179](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29673364179) 在提交 `7ee602b` 上完成托管复验：Python 3.12 CPU job 为 `225 passed, 7 skipped, 17 deselected`，全包 `5,854/13,203`（`44.34%`），直接维护范围 `1,173/1,788`（`65.60%`），双门禁通过。托管环境仍比本地多覆盖 1 条 `data` 语句；下表已更新为该次托管结果。

新增 core schema 类型检查回归后，显式设置 `CUDA_VISIBLE_DEVICES=''` 的本地 `.venv` CPU 实测为 `226 passed, 5 skipped, 34 deselected`，全包 `5,882/13,212`（`44.52%`），直接维护范围 `1,202/1,797`（`66.89%`）。新增 9 条 schema 声明/分支语句的同时覆盖语句净增 29 条，因此直接维护范围回退下限从 65% 提高到 66%。GitHub Actions [run 29673733080](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29673733080) 在提交 `230c6c3` 上完成托管复验：Python 3.12 CPU job 为 `226 passed, 7 skipped, 17 deselected`，全包 `5,883/13,212`（`44.53%`），直接维护范围 `1,202/1,797`（`66.89%`），新双门禁通过。托管环境仍比本地多覆盖 1 条 `data` 语句；下表已更新为托管结果。

新增 engine 的 4 个活跃回归后，显式设置 `CUDA_VISIBLE_DEVICES=''` 的本地 `.venv` CPU 实测为 `230 passed, 5 skipped, 34 deselected`，全包 `5,945/13,264`（`44.82%`），直接维护范围仍为 `1,202/1,797`（`66.89%`）。`engine` 新增 52 条实现语句，覆盖语句净增 63 条；全包尚未达到 45%，因此本批不提高 44% 下限。GitHub Actions [run 29674294425](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29674294425) 在提交 `7daad87` 上完成托管复验：Python 3.12 CPU job 为 `230 passed, 7 skipped, 17 deselected`，全包 `5,946/13,264`（`44.83%`），直接维护范围 `1,202/1,797`（`66.89%`），双门禁通过。托管环境仍比本地多覆盖 1 条 `data` 语句；下表已更新为托管结果。

新增 modeling 的 3 个实际边界回归后，显式设置 `CUDA_VISIBLE_DEVICES=''` 的本地 `.venv` CPU 实测为 `233 passed, 5 skipped, 34 deselected`，全包 `5,990/13,296`（`45.05%`），直接维护范围仍为 `1,202/1,797`（`66.89%`）。`modeling` 实现语句净增 32 条、覆盖语句净增 45 条，因此全包回退下限从 44% 提高到 45%。GitHub Actions [run 29674832957](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29674832957) 在提交 `c309c42` 上完成托管复验：Python 3.12 CPU job 为 `233 passed, 7 skipped, 17 deselected`，全包 `5,991/13,296`（`45.06%`），直接维护范围 `1,202/1,797`（`66.89%`），新双门禁通过。托管环境仍比本地多覆盖 1 条 `data` 语句；下表已更新为托管结果。

新增 data 的 8 个边界回归后，显式设置 `CUDA_VISIBLE_DEVICES=''` 的本地 `.venv` CPU 实测为 `241 passed, 5 skipped, 34 deselected`，全包 `6,284/13,345`（`47.09%`），直接维护范围为 `1,202/1,799`（`66.81%`）。实现语句净增 49 条，覆盖语句净增 294 条，因此全包回退下限从 45% 提高到 47%。GitHub Actions [run 29675617264](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29675617264) 在提交 `28ec38d` 上完成托管复验：Python 3.12 CPU job 为 `241 passed, 7 skipped, 17 deselected`，全包 `6,285/13,345`（`47.10%`），直接维护范围 `1,202/1,799`（`66.81%`），新双门禁通过。托管环境比本地多覆盖 1 条 `data` 语句；下表已更新为托管结果。

新增 Models CLI 和不依赖 Paddle 的转换验证核心测试后，显式设置 `CUDA_VISIBLE_DEVICES=''` 的本地 `.venv` CPU 实测为 `272 passed, 5 skipped, 34 deselected`，全包 `6,511/13,535`（`48.10%`），直接维护范围为 `1,429/1,989`（`71.85%`）。新增测试覆盖发布状态、size/SHA-256 校验、HTTPS 原子下载与既有文件保护，也实际发现并修复了转换验证器忽略 PyTorch 额外输出字段、非有限值可能被错误接受的问题。基于该证据，全包和直接维护范围回退下限分别从 47%/66% 提高到 48%/71%。GitHub Actions [run 29679546423](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29679546423) 在提交 `e1a306b` 上完成托管复验：Python 3.9–3.12 均为 `272 passed, 7 skipped, 17 deselected`；Python 3.12 全包 `6,512/13,535`（`48.11%`），直接维护范围 `1,429/1,989`（`71.85%`），新双门禁通过。托管环境比本地多覆盖 1 条 `data` 语句；下表采用托管结果。

新增转换器编排、批处理和 YAML `Callable` 的 16 个纯 CPU 回归后，本地隐藏 GPU 的非 Paddle 测试为 `288 passed, 5 skipped, 34 deselected`。全包覆盖率为 `6,689/13,538`（`49.41%`），直接维护范围为 `1,607/1,992`（`80.67%`）；`conversion/converter.py` 从 27.5% 提高到 98.0%，`core/config/yaml_helpers.py` 从 42.4% 提高到 91.7%。回归实际修复了不同 `Callable` 实例共享默认 `args/kwargs`，以及批量转换在输出不存在但同名 mapping 已存在时仍会无 `--force` 覆盖的问题。基于该证据，全包和直接维护范围门槛提高到 49%/80%。GitHub Actions [run 29680140237](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29680140237) 在提交 `3fbd4ec` 上完成托管复验：Python 3.9–3.12 均为 `288 passed, 7 skipped, 17 deselected`；Python 3.12 全包 `6,690/13,538`（`49.42%`），直接维护范围 `1,607/1,992`（`80.67%`），新门槛通过。托管环境比本地多覆盖 1 条 `data` 语句；下表采用托管结果。

## 当前结果

| 模块 | 语句数 | 覆盖语句 | 覆盖率 |
|---|---:|---:|---:|
| package root | 4 | 4 | 100.0% |
| `cli` | 847 | 630 | 74.4% |
| `conversion` | 659 | 569 | 86.3% |
| `core` | 401 | 327 | 81.5% |
| `data` | 5,630 | 1,758 | 31.2% |
| `deploy` | 85 | 81 | 95.3% |
| `engine` | 757 | 452 | 59.7% |
| `metrics` | 577 | 265 | 45.9% |
| `modeling` | 3,418 | 2,057 | 60.2% |
| `optimizer` | 409 | 224 | 54.8% |
| `utils` | 751 | 323 | 43.0% |
| **全包** | **13,538** | **6,690** | **49.42%** |

直接维护范围指 M1–M5 首批质量门禁中的 `cli`、`conversion`、`core` 和 `deploy`，共 `1,992` 条语句，覆盖 `1,607` 条，覆盖率为 **80.67%**。这个子集用于屏蔽回退，不代表其他模块不维护，也不是对整个“已迁移核心”的最终定义。

## 门禁与限制

当前可执行门禁同时要求：

- 非 Paddle 全包语句覆盖率不低于 **49%**。
- `cli/conversion/core/deploy` 合计覆盖率不低于 **80%**。

两个阈值都低于当前实测值，是防止覆盖率回退的初始下限，不是完成目标。`ROADMAP.md` 中的 90% 目标仍未完成；不应通过排除 `data`、`metrics` 或其他低覆盖源文件来声称全包达标。

下一轮优先覆盖用户可见且当前低于直接维护范围平均值的路径：`conversion/tensor_utils.py`、`cli/infer.py`、`cli/export.py` 和 `cli/convert.py`。之后再以实际新增测试证据逐步提高门禁。

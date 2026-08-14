# 模型文档

本目录按面向用户和验证驱动使用的模型族组织专属文档。跨模型复用的配置、训练、权重转换和排错经验仍保存在 [`docs/migrations`](../migrations/README.md)，这里不重复这些公共合同。

每个实际运行模型族都维护四个入口：`README.md` 描述当前用户合同，`validation-report.md` 记录验证方法、环境、结论和限制，`metrics.md` 保存逐变体 checkpoint、精度、数值对齐与部署指标，`evidence-index.md` 按验证能力域组织结论。机器日志中的临时路径、重复输出和无结论排错过程不会原样进入正式文档。

## 最终质量快照

2026-08-14 的跨模型最终验收使用 Python `3.12.13` 和 PyTorch `2.5.1+cu121`：Ruff 检查通过，Mypy 检查 `123` 个 source file 无问题；覆盖率运行结果为 `810 passed, 43 skipped, 34 deselected`，全包 `10,406/17,329 = 60.05%`，直接维护范围 `2,076/2,283 = 90.93%`；独立 unit/integration 为 `761 passed, 26 skipped, 31 deselected`，上游数值定向测试 `21 passed`，五族图审计通过。

上述计数分别属于覆盖率运行、独立 unit/integration 运行和上游数值运行，不能相互替代。最终结构化收据记录全部检查通过；当时保存的控制台日志只包含进度流的前段，没有形成独立完整 transcript，因此不作为长期证据保留。

安装包用户验收在独立 Python `3.11.15` CPU 环境中安装 wheel 的 `[teacher]` extra，使用 PyTorch `2.5.1+cpu`、torchvision `0.20.1+cpu`、NumPy `2.4.4`、Pillow `12.2.0`、OpenCV `5.0.0.93`、SciPy `1.17.1`、ONNX `1.22.0`、ONNX Runtime `1.28.0`、pycocotools `2.0.11` 和 pytest `9.1.1`，覆盖四个新模型族的最小变体及缺失 teacher、错误 checksum、错误大小、错误模型族和缺失 stage-1 companion 负例。五类负例均在 preflight 失败并记录 `state_mutated=false`。Teacher 依赖出现在安装日志中是该 extra 的预期行为，不表示 core wheel 强制依赖 teacher。

结构化打包收据和 checksum 文件记录的 wheel SHA-256 为 `e53c3fb28bff67e6c369c6c517c92a37d1b559ed7708a3fb9ef1e10ea510cbe0`；NOTICE 更新后的重建日志另记录 `7810ab5327ec0c66921f03d15aa5fe007948da3799e1432e9d03597c59ec0333`，但未生成第二份结构化 artifact receipt。临时 venv、wheel 和 `dist/` 已在验收后清理，因此两者都不是当前可下载发布资产，后续发布必须重新构建并生成唯一 checksum。

四图安装包验收使用 `000000000139.jpg`、`000000000285.jpg`、`000000000632.jpg`、`000000000724.jpg`，CPU/FP32、固定 640、推理 batch 4、阈值 `0.3`；上游 raw-output parity 为逐图 batch 1、单 torch thread。阈值后每图检测数如下：

| 模型 | 139 | 285 | 632 | 724 | 合计 |
|---|---:|---:|---:|---:|---:|
| D-FINE N | 34 | 1 | 78 | 4 | 117 |
| DEIM-D-FINE N | 29 | 1 | 28 | 2 | 60 |
| DEIM-RT-DETRv2 S | 26 | 1 | 29 | 4 | 60 |
| RT-DETRv4 S | 32 | 1 | 31 | 4 | 68 |

检测 JSON 可由验证驱动重建；其中图片路径属于临时目录。渲染 JPEG checksum 会随 Pillow、字体和编码器变化，不作为模型数值真值，因此原始 JSON 和渲染图不进入版本库。

## 已收录模型

- [RT-DETRv3](rtdetrv3/README.md)：已发布模型，包含配置支持、CLI/导出边界、已知限制和 `v0.1.0` 验证证据入口。
- [D-FINE](dfine/README.md)：N/S/M/L/X 的 checkpoint、数值、COCO、训练恢复和部署合同；集成与打包已验收，尚未发布权重。
- [DEIM](deim/README.md)：两个 DEIM 产品分支的索引与共同训练边界。
- [DEIM-D-FINE](deim-dfine/README.md)：使用 D-FINE decoder 的 DEIM N/S/M/L/X 合同；集成与打包已验收，尚未发布权重。
- [DEIM-RT-DETRv2](deim-rtdetrv2/README.md)：DEIM 所需的受限 RT-DETRv2 decoder 分支及 S/M/M*/L/X 合同；不作为独立 RT-DETRv2 产品族。
- [RT-DETRv4](rtdetrv4/README.md)：S/M/L/X 的 checkpoint、真实 DINOv3 reduced train、COCO 和 student-only 部署合同；模型级与打包验收已完成，尚未发布权重。

目录状态必须明确区分“已发布”、“已完成模型级验收但未发布”和“计划中”。新增模型时应创建一个同级目录；不要在每个迁移主题下重复建立模型子目录。

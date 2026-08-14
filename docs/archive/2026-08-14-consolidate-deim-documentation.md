# DEIM 模型文档目录合并计划

> 完成计划归档快照（2026-08-14）：本文保存 DEIM 文档目录合并的实际范围与验收证据，不代表后续仓库状态。

- 状态：`completed`
- 创建日期：`2026-08-14`
- 最后更新：`2026-08-14`
- 负责人：仓库维护者与当前执行代理

## 背景

当前 `docs/models/` 同时存在 `deim/` 汇总目录、`deim-dfine/` 和 `deim-rtdetrv2/` 两个产品分支目录。三层入口把运行时 family 与文档信息架构混为一谈，增加导航和状态维护成本。

## 范围

- 包含：将两个 DEIM profile 的合同、验证、指标和证据合并到 `docs/models/deim/`，更新全部内部链接、文档布局门禁和单测。
- 不包含：重命名 CLI family、配置目录、manifest、checkpoint alias 或模型实现；接入独立的 DEIMv2 上游。

## 依赖

- 固定上游仍为 `Intellindust-AI-Lab/DEIM@09d35d53d39ee3145a1e61e3a989b28b9468d1dd`。
- 官方项目名称为 DEIM；DEIMv2 是独立上游，因此本仓库不自行使用“DEIMv1”命名。

## 目标与非目标

### 目标

- `docs/models/deim/` 成为 DEIM 当前合同的唯一目录。
- 在同一组 README、验证报告、指标和证据索引中明确区分 D-FINE 与 RT-DETRv2 decoder profile。
- 保留两个运行时 family 的独立配置、checkpoint 与部署容差边界。

### 非目标

- 不把两个 profile 描述成可交换 checkpoint 或单一模型图。
- 不改变已有数值、AP、checksum 或验证结论。

## 实施步骤

- [x] 合并四类 DEIM 文档并删除两个分支目录；验证：原有合同、指标和命令均有对应位置。
- [x] 更新模型、迁移与计划文档中的内部链接；验证：仓库内不再引用旧目录。
- [x] 将文档布局门禁从运行时 family 映射改为文档目录集合；验证：只要求 `rtdetrv3`、`dfine`、`deim`、`rtdetrv4` 四个文档目录。
- [x] 运行文档、单测和质量检查；验证：相对链接、manifest、Ruff 和 Mypy 全部通过。

## 风险与回退

- 风险：合并时遗漏 profile-specific 容差或训练初始化资产。缓解：按 README、validation、metrics、evidence 四类逐项合并，并在删除前对照原文件。
- 风险：把文档合并误解成运行时 family 合并。缓解：在总览和门禁中明确文档目录与 CLI family 是两个维度。
- 回退：恢复两个目录并还原内部链接，不涉及模型代码、配置或资产。

## 验收

- [x] `rg` 检查不存在指向旧模型文档目录的 Markdown 链接。
- [x] `uv run python scripts/check_docs.py` 通过。
- [x] `uv run pytest -q -p no:cacheprovider tests/unit/scripts/test_check_docs.py` 通过。
- [x] `uv run --extra quality python scripts/check_quality.py` 通过。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-14 | 文档统一命名为 DEIM，不称 DEIMv1 | 固定上游正式名称为 DEIM，DEIMv2 是独立项目 |
| 2026-08-14 | 仅合并文档目录，保留两个 CLI family | checkpoint、配置、模型图和部署门槛仍然不同 |

## 完成记录

2026-08-14 完成：

- 删除 `docs/models/deim-dfine/` 与 `docs/models/deim-rtdetrv2/`，将 README、validation report、metrics 和 evidence index 合并为 `docs/models/deim/` 的四文件结构。
- 十个 detector 的 AP、大小、SHA-256、tensor 数、schedule、PResNet 初始化资产、两套验证命令与 family-specific ONNX 容差均已保留。
- 模型总览统一为一个 DEIM 文档域；根 README 中两个 CLI family 继续保留，以反映实际 manifest、配置和 checkpoint 边界。
- 文档明确使用官方名称 DEIM，不自行称 DEIMv1；独立 DEIMv2 上游不在当前支持范围。
- `scripts/check_docs.py` 现在验证四个模型文档域，并拒绝旧 DEIM 文档目录重新出现。

实际验收：

- `uv run python scripts/check_docs.py`：通过，覆盖 5 个运行时 family、23 个 artifact、19 个新增变体和 74 份 Markdown。
- `uv run pytest -q -p no:cacheprovider tests/unit/scripts/test_check_docs.py`：`12 passed`。
- `uv run --extra quality python scripts/check_quality.py`：265 个 Python 文件格式通过，Ruff lint 通过，Mypy 检查 123 个 source file 无问题。
- Markdown 链接检查未发现指向旧 DEIM 文档目录的引用，`git diff --check` 通过。

计划无范围偏差；CLI family、配置、manifest、checkpoint alias、模型实现与数值结论均未修改。

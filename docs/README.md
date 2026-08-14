# 文档

本目录集中存放 RT-DETRv3 PyTorch 迁移项目的文档。文档按用途分区，避免把历史快照误当作当前支持范围；判断“当前是否支持”请以模型合同和路线图为准。

## 分区

- [`migrations/`](migrations/README.md)：跨模型复用的 PaddlePaddle 到 PyTorch 迁移知识，包括框架语义对比、权重转换规则、注册与配置行为、训练与数值验证方法及排错经验。内容不绑定某个具体模型。
- [`models/`](models/README.md)：模型专属的当前合同，按模型族分目录。已收录 RT-DETRv3、D-FINE、DEIM 汇总、DEIM-D-FINE、DEIM-RT-DETRv2 与 RT-DETRv4；新模型族均明确区分模型级/打包验收与发布状态。
- [`plans/`](plans/README.md)：活动与延期计划。当前正在执行 [D-FINE、DEIM 与 RT-DETRv4 集成](plans/2026-08-12-dfine-deim-rtdetrv4-integration.md)，M4 COCO 精度与稳定性保持 deferred；已完成计划移入归档。新计划可从 [`plans/TEMPLATE.md`](plans/TEMPLATE.md) 复制。
- [`archive/`](archive/README.md)：已完成版本和迁移阶段的历史证据，包括计划、报告、论文、图片和机器可读数据。归档内容均为带日期的历史快照，不代表当前仓库状态。
- [`ROADMAP.md`](../ROADMAP.md)：未完成迁移大纲的唯一顶层文档，与归档快照分开维护。

## 使用约定

- 归档中的计划、报告和证据必须保留原日期、环境、提交、数值与限制，不得改写为当前状态。
- 迁移可复用结论沉淀到 `docs/migrations/`，模型专属合同沉淀到 `docs/models/<model>/`。
- 未完成工作以根目录 [`ROADMAP.md`](../ROADMAP.md) 为准。

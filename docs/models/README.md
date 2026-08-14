# 模型文档

本目录按面向用户和验证驱动使用的模型族组织专属文档。跨模型复用的配置、训练、权重转换和排错经验仍保存在 [`docs/migrations`](../migrations/README.md)，这里不重复这些公共合同。

## 已收录模型

- [RT-DETRv3](rtdetrv3/README.md)：已发布模型，包含配置支持、CLI/导出边界、已知限制和 `v0.1.0` 验证证据入口。
- [D-FINE](dfine/README.md)：N/S/M/L/X 的 checkpoint、数值、COCO、训练恢复和部署合同；集成与打包已验收，尚未发布权重。
- [DEIM](deim/README.md)：两个 DEIM 产品分支的索引与共同训练边界。
- [DEIM-D-FINE](deim-dfine/README.md)：使用 D-FINE decoder 的 DEIM N/S/M/L/X 合同；集成与打包已验收，尚未发布权重。
- [DEIM-RT-DETRv2](deim-rtdetrv2/README.md)：DEIM 所需的受限 RT-DETRv2 decoder 分支及 S/M/M*/L/X 合同；不作为独立 RT-DETRv2 产品族。
- [RT-DETRv4](rtdetrv4/README.md)：S/M/L/X 的 checkpoint、真实 DINOv3 reduced train、COCO 和 student-only 部署合同；模型级与打包验收已完成，尚未发布权重。

目录状态必须明确区分“已发布”、“已完成模型级验收但未发布”和“计划中”。新增模型时应创建一个同级目录；不要在每个迁移主题下重复建立模型子目录。

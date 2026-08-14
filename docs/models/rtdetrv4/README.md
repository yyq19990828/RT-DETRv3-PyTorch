# RT-DETRv4

本目录记录 RT-DETRv4 模型族的当前仓库合同。跨模型复用的上游 PyTorch 数值验证方法见[上游 PyTorch 数值对齐](../../migrations/upstream-pytorch-parity.md)。

- [验证报告](validation-report.md)：checkpoint、真实 teacher、DSI/GAM、训练恢复、推理和部署结论。
- [指标记录](metrics.md)：四变体 checkpoint、COCO、四图 parity、部署和 DINOv3 身份数值。
- [证据索引](evidence-index.md)：Task 18-23 与 F1-F4 的结论映射及复现入口。

## 当前状态

截至 2026-08-14，RT-DETRv4 S/M/L/X 已完成 Task 18-23 的模型、Models CLI、打包和文档验收，并通过 F1-F4 最终门；官方权重仍未由本项目发布。

| 变体 | Backbone | 参数量 | val2017 bbox AP |
|---|---|---:|---:|
| S | HGNetv2-B0 | 10,519,253 | 0.498371 |
| M | HGNetv2-B2 | 19,787,440 | 0.536396 |
| L | HGNetv2-B4 | 31,487,224 | 0.554134 |
| X | HGNetv2-B5 | 63,011,160 | 0.570014 |

四个 AP 相对上游公布的三位小数值最大绝对误差为 `0.000604`。这是同一官方 EMA checkpoint 的完整 COCO val2017 评估结果，不是本仓库完整 schedule 的训练收敛证据。

## 已验证合同

- 四个正式配置分别映射 B0/B2/B4/B5，并固定 teacher/projector、DSI/GAM、Dense O2O、FlatCosine 和两阶段 schedule。
- 官方 solver checkpoint 的评估 state 位于 `ema.module`，使用 PyTorch 原生 layout 和 identity key mapping 严格加载；S/M/L/X 分别覆盖 `796/1055/1255/1573` 个 tensor。
- CPU/FP32、固定 640 输入下，stem、backbone、encoder 和 raw logits/boxes 对固定上游 revision 通过 `rtol=1e-5, atol=1e-6`；四张真实 COCO 图也全部通过。
- S/M/L/X 均使用授权 DINOv3 ViT-B/16 权重完成真实 teacher reduced update；stage/GAM epoch-boundary resume、缺失 teacher preflight 和 stale GAM state 拒绝路径通过。Reduced run 不构成完整训练 schedule 收敛声明。
- 四个变体均通过官方 EMA eager、deploy、ONNX opset 17 与 TorchScript 验收；导出产物固定 640 空间尺寸并支持动态 batch 1/4，不包含 teacher、distillation 或 DSI projector residue。student-only 图声明由 Task 20 证据中四变体 `training_residue=false` 与图审计共同约束。
- student checkpoint 的 eval、infer 和 export 不构造 DINOv3，也不访问 teacher checkout 或授权权重。

## 配置与资产

| 变体 | 配置 | CLI alias |
|---|---|---|
| S | `configs/rtdetrv4/rtdetrv4_hgnetv2_s_coco.yml` | `rtdetrv4-s` |
| M | `configs/rtdetrv4/rtdetrv4_hgnetv2_m_coco.yml` | `rtdetrv4-m` |
| L | `configs/rtdetrv4/rtdetrv4_hgnetv2_l_coco.yml` | `rtdetrv4-l` |
| X | `configs/rtdetrv4/rtdetrv4_hgnetv2_x_coco.yml` | `rtdetrv4-x` |

- 配置：[`configs/rtdetrv4`](../../../configs/rtdetrv4/)
- 官方 checkpoint 清单：[`configs/checkpoints/rtdetrv4_coco.yml`](../../../configs/checkpoints/rtdetrv4_coco.yml)
- 架构：[`RTDETRV4`](../../../src/ppdet_pytorch/modeling/architectures/rtdetrv4.py)
- 教师边界：[`DINOv3TeacherModel`](../../../src/ppdet_pytorch/modeling/teachers/dinov3.py)
- 执行计划与逐任务证据摘要：[D-FINE、DEIM 与 RT-DETRv4 集成计划](../../plans/2026-08-12-dfine-deim-rtdetrv4-integration.md)

上游为 `RT-DETRs/RT-DETRv4@55fefaaed7efe2a5f72d0a18fd4e05965e35c292`（Apache-2.0）。官方 student checkpoint 继续由上游托管，不进入本项目 Release。

DINOv3 固定为 `facebookresearch/dinov3@346f38fee679c56a6888f91c51670fae61d364e0`，适用其[自定义 DINOv3 License](https://github.com/facebookresearch/dinov3/blob/346f38fee679c56a6888f91c51670fae61d364e0/LICENSE.md)。checkout 和经 [Meta 官方仓库](https://github.com/facebookresearch/dinov3)说明获取的门控权重仅作为仓库外训练资产，不进入仓库、wheel、sdist 或本项目 Release，也不由本项目再分发。使用者须自行取得授权并遵守许可证与 acknowledgment 要求。

# D-FINE

本目录记录 D-FINE 模型族的当前仓库合同。跨模型复用的上游 PyTorch 数值验证方法见[上游 PyTorch 数值对齐](../../migrations/upstream-pytorch-parity.md)。

- [验证报告](validation-report.md)：上游对齐、训练恢复、推理、部署与限制。
- [指标记录](metrics.md)：五变体 checkpoint、COCO、四图和导出数值。
- [证据索引](evidence-index.md)：组件、模型、用户接口和最终审计的结论映射。

## 当前状态

截至 2026-08-14，D-FINE N/S/M/L/X 已完成组件、模型、Models CLI、打包和文档验收，并通过全部最终审计；官方权重仍未由本项目发布。

| 变体 | Backbone | 参数量 | val2017 bbox AP |
|---|---|---:|---:|
| N | HGNetv2-B0 | 3,782,693 | 0.427997 |
| S | HGNetv2-B0 | 10,321,877 | 0.485145 |
| M | HGNetv2-B2 | 19,590,064 | 0.522783 |
| L | HGNetv2-B4 | 31,244,152 | 0.539703 |
| X | HGNetv2-B5 | 62,621,560 | 0.557650 |

五个 AP 相对上游公布的三位小数值误差均小于 `0.000350`。这是同一官方 checkpoint 的完整 COCO val2017 评估结果，不是本仓库完整 schedule 的训练收敛证据。

## 已验证合同

- 官方 `{"model": state_dict}` checkpoint 使用 PyTorch 原生 layout 和 identity key mapping 严格加载，N/S/M/L/X 分别覆盖 `674/794/1053/1173/1441` 个 tensor。
- CPU/FP32、固定 640 输入下，stem、backbone、encoder 和 raw logits/boxes 对固定上游 revision 通过 `rtol=1e-5, atol=1e-6`。
- Eval/Test 预处理使用 PIL bilinear；OpenCV cubic 曾使 N 的 AP 降至 `0.426412`，因此不能用相同模型输入 tensor 的 raw-output 对齐替代预处理验证。
- 五个变体均通过 reduced staged training、epoch-boundary resume、四图 eager 推理和两阶段 checkpoint/EMA 协议验证；reduced run 不代表完整训练 schedule。
- ONNX opset 17 与 TorchScript 已在 CPU/FP32、固定 640、动态 batch 1/4 下重载。ONNX 最大 score/box 误差为 `1.2443e-5 / 0.01879 px`，TorchScript 为零。
- 导出只声明固定空间尺寸，产物不得包含 criterion、denoising 或 auxiliary training residue；全零退化输入不能作为 TopK 部署验收 fixture。

## 配置与资产

| 变体 | 配置 | CLI alias |
|---|---|---|
| N | `configs/dfine/dfine_hgnetv2_n_coco.yml` | `dfine-n` |
| S | `configs/dfine/dfine_hgnetv2_s_coco.yml` | `dfine-s` |
| M | `configs/dfine/dfine_hgnetv2_m_coco.yml` | `dfine-m` |
| L | `configs/dfine/dfine_hgnetv2_l_coco.yml` | `dfine-l` |
| X | `configs/dfine/dfine_hgnetv2_x_coco.yml` | `dfine-x` |

- 配置：[`configs/dfine`](../../../configs/dfine/)
- 官方 checkpoint 清单：[`configs/checkpoints/dfine_coco.yml`](../../../configs/checkpoints/dfine_coco.yml)
- 架构：[`DFINE`](../../../src/ppdet_pytorch/modeling/architectures/dfine.py)
- Backbone：[`HGNetv2`](../../../src/ppdet_pytorch/modeling/backbones/hgnetv2.py)
- 执行计划与逐任务证据摘要：[D-FINE、DEIM 与 RT-DETRv4 集成计划](../../plans/2026-08-12-dfine-deim-rtdetrv4-integration.md)

上游为 `Peterande/D-FINE@267a6da6d04c8ad52e54120692896515b9e55981`（Apache-2.0）。官方 checkpoint 继续由上游托管，不进入本项目 Release；配置和 manifest 已进入 wheel/sdist。

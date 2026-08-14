# DEIM-D-FINE

本目录记录使用 D-FINE decoder 的 DEIM 模型族合同。它与 [D-FINE](../dfine/README.md) 共享已验证的 HGNetv2 和推理图，但训练使用 DEIM MAL、Dense O2O、FlatCosine 和两阶段 EMA 语义。

- [验证报告](validation-report.md)：MAL、训练恢复、上游 parity、推理和部署结论。
- [指标记录](metrics.md)：五变体 checkpoint、COCO 与部署数值。
- [证据索引](evidence-index.md)：训练语义、模型矩阵、用户接口和最终审计映射。

## 当前状态

截至 2026-08-14，DEIM-D-FINE N/S/M/L/X 已完成训练语义、模型矩阵、Models CLI、打包和文档验收，并通过全部最终审计；官方权重仍未由本项目发布。

| 变体 | Backbone | val2017 bbox AP | checkpoint tensors |
|---|---|---:|---:|
| N | HGNetv2-B0 | 0.430424 | 674 |
| S | HGNetv2-B0 | 0.489613 | 794 |
| M | HGNetv2-B2 | 0.526880 | 1053 |
| L | HGNetv2-B4 | 0.547392 | 1253 |
| X | HGNetv2-B5 | 0.564731 | 1571 |

五个 AP 相对上游公布值的最大绝对误差为 `0.000424`。完整 val2017 结果验证官方 checkpoint 的评估路径，不证明完整 schedule 训练收敛。

## 已验证合同

- 五个官方 checkpoint 均以 identity mapping 严格加载；固定 640 的 stem、backbone、encoder 和 raw output 对 `Intellindust-AI-Lab/DEIM@09d35d53` 通过 `rtol=1e-5, atol=1e-6`。
- `DEIM` 复用共享 `DFINE` eval graph，不新增推理分支；训练 criterion 覆盖 MAL、GO union、main/aux/pre/encoder/CDN/local、FGL 和 DDF。
- class-agnostic encoder 必须在 matcher 前构造零标签 targets。MAL fractional gamma 前将非有限 quality 置零并限制到 `[0,1]`，所有最终 loss 继续执行非有限 fail-fast，不复制上游静默 `nan_to_num`。
- 五个变体均通过 reduced optimizer/EMA、epoch-boundary resume、stage-1 companion 回载和四图 eager/parity 验证。
- ONNX opset 17 与 TorchScript 在 CPU/FP32、固定 640、动态 batch 1/4 下通过；ONNX 最大 score/box 误差为 `1.1861e-5 / 0.014901 px`，TorchScript 为零。

## 配置与资产

| 变体 | 配置 | CLI alias |
|---|---|---|
| N | `configs/deim/dfine/deim_hgnetv2_n_coco.yml` | `deim-dfine-n` |
| S | `configs/deim/dfine/deim_hgnetv2_s_coco.yml` | `deim-dfine-s` |
| M | `configs/deim/dfine/deim_hgnetv2_m_coco.yml` | `deim-dfine-m` |
| L | `configs/deim/dfine/deim_hgnetv2_l_coco.yml` | `deim-dfine-l` |
| X | `configs/deim/dfine/deim_hgnetv2_x_coco.yml` | `deim-dfine-x` |

- 配置：[`configs/deim/dfine`](../../../configs/deim/dfine/)
- 官方 checkpoint 清单：[`configs/checkpoints/deim_dfine_coco.yml`](../../../configs/checkpoints/deim_dfine_coco.yml)
- 架构：[`DEIM`](../../../src/ppdet_pytorch/modeling/architectures/deim.py)
- Criterion：[`DEIMCriterion`](../../../src/ppdet_pytorch/modeling/losses/deim_loss.py)
- 执行计划与逐任务证据摘要：[D-FINE、DEIM 与 RT-DETRv4 集成计划](../../plans/2026-08-12-dfine-deim-rtdetrv4-integration.md)

上游为 `Intellindust-AI-Lab/DEIM@09d35d53d39ee3145a1e61e3a989b28b9468d1dd`（Apache-2.0）。官方 checkpoint 继续由上游托管，不进入本项目 Release；配置和 manifest 已进入 wheel/sdist。

# DEIM

本目录是 DEIM 当前合同的唯一文档入口。固定上游的正式名称是 [DEIM](https://github.com/Intellindust-AI-Lab/DEIM)，不是 `DEIMv1`；[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) 是另一个独立上游，其合同见 [deimv2](../deimv2/README.md)，与本目录的 DEIM profile 不可混用。

仓库保留两个不能互换 checkpoint 的运行时 profile：

- DEIM-D-FINE：N/S/M/L/X，HGNetv2 与 D-FINE decoder。
- DEIM-RT-DETRv2：S/M/M*/L/X，PResNet 与受限 RT-DETRv2 decoder。

两者固定到 `Intellindust-AI-Lab/DEIM@09d35d53d39ee3145a1e61e3a989b28b9468d1dd`（Apache-2.0），共享 MAL、Dense O2O、FlatCosine 和两阶段 EMA 协议，但配置、checkpoint、训练初始化和部署容差不可混用。

- [验证报告](validation-report.md)：两套图的训练、恢复、数值、推理、部署与限制。
- [指标记录](metrics.md)：十个 detector、PResNet 初始化、COCO、schedule 与部署数值。
- [证据索引](evidence-index.md)：两个运行时 family 的 manifest、测试和可执行验证入口。

## 当前状态

截至 2026-08-14，两个 profile 共十个变体均已完成官方 checkpoint、上游 parity、完整 COCO val2017、reduced train/resume、Models CLI、ONNX/TorchScript、打包和文档验收；官方权重仍未由本项目发布。

| Profile | 变体 | Backbone | CLI family | Manifest |
|---|---|---|---|---|
| DEIM-D-FINE | N/S/M/L/X | HGNetv2-B0/B2/B4/B5 | `deim-dfine` | [`deim_dfine_coco.yml`](../../../configs/checkpoints/deim_dfine_coco.yml) |
| DEIM-RT-DETRv2 | S/M/M*/L/X | PResNet-18/34/50/101-vd | `deim-rtdetrv2` | [`deim_rtdetrv2_coco.yml`](../../../configs/checkpoints/deim_rtdetrv2_coco.yml) |

## DEIM-D-FINE

该 profile 与 [D-FINE](../dfine/README.md) 共享已验证的 HGNetv2 和 eval 图，但训练使用 DEIM MAL、Dense O2O、FlatCosine 和两阶段 EMA 语义。

| 变体 | Backbone | val2017 bbox AP | checkpoint tensors |
|---|---|---:|---:|
| N | HGNetv2-B0 | 0.430424 | 674 |
| S | HGNetv2-B0 | 0.489613 | 794 |
| M | HGNetv2-B2 | 0.526880 | 1053 |
| L | HGNetv2-B4 | 0.547392 | 1253 |
| X | HGNetv2-B5 | 0.564731 | 1571 |

五个 AP 相对上游公布值的最大绝对误差为 `0.000424`。完整 val2017 结果验证官方 checkpoint 的评估路径，不证明完整 schedule 训练收敛。

### DEIM-D-FINE 已验证合同

- 五个官方 checkpoint 均以 identity mapping 严格加载；固定 640 的 stem、backbone、encoder 和 raw output 通过 `rtol=1e-5, atol=1e-6`。
- `DEIM` 复用共享 `DFINE` eval graph，不新增推理分支；训练 criterion 覆盖 MAL、GO union、main/aux/pre/encoder/CDN/local、FGL 和 DDF。
- Class-agnostic encoder 在 matcher 前构造零标签 targets。MAL fractional gamma 前将非有限 quality 置零并限制到 `[0,1]`，最终 loss 继续执行非有限 fail-fast。
- 五个变体均通过 reduced optimizer/EMA、epoch-boundary resume、stage-1 companion 回载和四图 eager/parity。
- ONNX opset 17 与 TorchScript 在 CPU/FP32、固定 640、动态 batch 1/4 下通过；ONNX 最大 score/box 误差为 `1.1861e-5 / 0.014901 px`，TorchScript 为零。

### DEIM-D-FINE 配置与资产

| 变体 | 配置 | CLI alias |
|---|---|---|
| N | `configs/deim/dfine/deim_hgnetv2_n_coco.yml` | `deim-dfine-n` |
| S | `configs/deim/dfine/deim_hgnetv2_s_coco.yml` | `deim-dfine-s` |
| M | `configs/deim/dfine/deim_hgnetv2_m_coco.yml` | `deim-dfine-m` |
| L | `configs/deim/dfine/deim_hgnetv2_l_coco.yml` | `deim-dfine-l` |
| X | `configs/deim/dfine/deim_hgnetv2_x_coco.yml` | `deim-dfine-x` |

- 配置：[`configs/deim/dfine`](../../../configs/deim/dfine/)
- 架构：[`DEIM`](../../../src/detrs/modeling/architectures/deim.py)
- Criterion：[`DEIMCriterion`](../../../src/detrs/modeling/losses/deim_loss.py)

## DEIM-RT-DETRv2

该 profile 是 DEIM 所需的受限 RT-DETRv2 decoder 实现切片，不表示仓库支持独立 RT-DETRv2 产品族，也不能与 RT-DETRv3 配置或 checkpoint 混用。

| 变体 | Backbone | val2017 bbox AP | checkpoint tensors |
|---|---|---:|---:|
| S | PResNet-18-vd | 0.490525 | 540 |
| M | PResNet-34-vd | 0.509376 | 667 |
| M* | PResNet-50-vd-m | 0.531902 | 732 |
| L | PResNet-50-vd | 0.542924 | 801 |
| X | PResNet-101-vd | 0.554852 | 1107 |

五个 AP 相对上游公布值的最大绝对误差为 `0.000525`。这是官方 checkpoint 的完整 val2017 评估证据，不是完整训练 schedule 的收敛声明。

### DEIM-RT-DETRv2 已验证合同

- 五个 detector checkpoint 均使用 identity mapping 严格加载。固定 CPU/FP32/640 下，stem、backbone、encoder 和 raw logits/boxes 对上游最大绝对误差为零。
- 只支持表中五个 backbone/decoder 组合；M* 与 L 的 PResNet-50 配置不同，错误混用必须在推理前失败。
- X 的 encoder/FPN channel 为 384、encoder FFN 为 2048，decoder hidden dimension 仍为 256。
- 官方 checkpoint 包含固定 640 的 decoder anchors/valid mask，配置必须保留 `eval_spatial_size: [640, 640]`。
- 官方 ImageNet PResNet-vd state 只在 Trainer 构建 optimizer/EMA 前加载；eval/infer/export 不加载该资产。
- 裸 backbone state 除 PyTorch 专有 `num_batches_tracked` 外必须完整匹配。
- 后处理必须使用 focal sigmoid 后的 `queries x classes` 全局 TopK。错误 softmax/query TopK 实测使 S/M AP 降至 `0.4547 / 0.4805`。
- 五个变体均通过 reduced train/resume、两阶段 companion、四图 eager、ONNX opset 17 和 TorchScript。

### DEIM-RT-DETRv2 部署边界

TorchScript 在固定 640、CPU/FP32、batch 1/4 下逐值一致。ONNX 使用预注册的 family-specific 门槛：S/M/M*/L score 为 `2e-5`，X score 为 `4e-4`，五个变体 box 均为 `0.1 px`；实测最大 score/box 误差为 `3.8812e-4 / 0.078148 px`。该例外不得扩散到其他模型族。

### DEIM-RT-DETRv2 配置与资产

| 变体 | 配置 | CLI alias |
|---|---|---|
| S | `configs/deim/rtdetrv2/deim_r18vd_120e_coco.yml` | `deim-rtv2-s` |
| M | `configs/deim/rtdetrv2/deim_r34vd_120e_coco.yml` | `deim-rtv2-m` |
| M* | `configs/deim/rtdetrv2/deim_r50vd_m_60e_coco.yml` | `deim-rtv2-m-star` |
| L | `configs/deim/rtdetrv2/deim_r50vd_60e_coco.yml` | `deim-rtv2-l` |
| X | `configs/deim/rtdetrv2/deim_r101vd_60e_coco.yml` | `deim-rtv2-x` |

- 配置：[`configs/deim/rtdetrv2`](../../../configs/deim/rtdetrv2/)
- Backbone：[`PResNet`](../../../src/detrs/modeling/backbones/presnet.py)
- Decoder：[`RTDETRTransformerv2`](../../../src/detrs/modeling/transformers/rtdetr_transformerv2.py)

## 共同边界

- 两个 profile 的 detector、训练初始化、checkpoint 和部署容差不可交换；文档合并不改变两个 CLI family。
- 官方 checkpoint 继续由上游托管，不进入本项目 Release；配置和 manifest 已进入 wheel/sdist。
- Reduced train/resume 只证明有限更新和 epoch-boundary 恢复，不证明完整 schedule、多 seed、低精度、TensorRT 或性能。
- 执行计划与逐任务证据摘要见 [D-FINE、DEIM 与 RT-DETRv4 集成计划](../../archive/2026-08-12-dfine-deim-rtdetrv4-integration.md)。

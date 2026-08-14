# DEIM-RT-DETRv2

本目录记录 DEIM 的 RT-DETRv2 decoder 分支合同。该分支是 DEIM 产品族所需的受限实现切片，不表示仓库新增了独立 RT-DETRv2 产品族，也不能与 RT-DETRv3 配置或 checkpoint 混用。

## 当前状态

截至 2026-08-14，S/M/M*/L/X 已完成 Task 17 的模型级验收和 Task 21-22 的 Models CLI/打包验收；官方权重仍未由本项目发布。

| 变体 | Backbone | val2017 bbox AP | checkpoint tensors |
|---|---|---:|---:|
| S | PResNet-18-vd | 0.490525 | 540 |
| M | PResNet-34-vd | 0.509376 | 667 |
| M* | PResNet-50-vd-m | 0.531902 | 732 |
| L | PResNet-50-vd | 0.542924 | 801 |
| X | PResNet-101-vd | 0.554852 | 1107 |

五个 AP 相对上游公布值的最大绝对误差为 `0.000525`。这是官方 checkpoint 的完整 val2017 评估证据，不是完整训练 schedule 的收敛声明。

## 已验证合同

- 五个 detector checkpoint 均使用 identity mapping 严格加载。固定 CPU/FP32/640 下，stem、backbone、encoder 和 raw logits/boxes 对固定上游 revision 的最大绝对误差为零。
- 只支持表中五个 backbone/decoder 组合。M* 与 L 的 PResNet-50 配置不同，错误混用必须在推理前失败。
- X 的 encoder/FPN channel 为 384、encoder FFN 为 2048，但 decoder hidden dimension 仍为 256；不能从 encoder channel 推断 decoder hidden dimension。
- 官方 checkpoint 包含固定 640 的 decoder anchors/valid mask，配置必须显式保留 `eval_spatial_size: [640, 640]`。
- 官方训练起点是 manifest 固定的 ImageNet PResNet-vd state。Trainer 在 optimizer 和 exponential EMA 构建前加载；eval/infer/export 不加载或下载该资产。裸 backbone state 除 PyTorch 专有 `num_batches_tracked` 外必须完整匹配。
- 后处理必须显式使用 focal sigmoid 后的 `queries x classes` 全局 TopK。错误的 softmax/query TopK 虽不破坏 raw-output 对齐，但实测使 S/M AP 降至 `0.4547 / 0.4805`。
- 五个变体均通过 reduced train/resume、两阶段 companion、四图 eager、ONNX opset 17 和 TorchScript 验证。

## 部署边界

TorchScript 在固定 640、CPU/FP32、batch 1/4 下逐值一致。ONNX 存在预注册的 family-specific 门槛：S/M/M*/L score 为 `2e-5`，X score 为 `4e-4`，五个变体 box 均为 `0.1 px`；本轮最大 score/box 误差为 `3.8812e-4 / 0.078148 px`。该例外不得扩散到其他模型族。

## 配置与资产

| 变体 | 配置 | CLI alias |
|---|---|---|
| S | `configs/deim/rtdetrv2/deim_r18vd_120e_coco.yml` | `deim-rtv2-s` |
| M | `configs/deim/rtdetrv2/deim_r34vd_120e_coco.yml` | `deim-rtv2-m` |
| M* | `configs/deim/rtdetrv2/deim_r50vd_m_60e_coco.yml` | `deim-rtv2-m-star` |
| L | `configs/deim/rtdetrv2/deim_r50vd_60e_coco.yml` | `deim-rtv2-l` |
| X | `configs/deim/rtdetrv2/deim_r101vd_60e_coco.yml` | `deim-rtv2-x` |

- 配置：[`configs/deim/rtdetrv2`](../../../configs/deim/rtdetrv2/)
- detector 与 PResNet 初始化清单：[`configs/checkpoints/deim_rtdetrv2_coco.yml`](../../../configs/checkpoints/deim_rtdetrv2_coco.yml)
- Backbone：[`PResNet`](../../../src/ppdet_pytorch/modeling/backbones/presnet.py)
- Decoder：[`RTDETRTransformerv2`](../../../src/ppdet_pytorch/modeling/transformers/rtdetr_transformerv2.py)
- 执行计划与逐任务证据摘要：[D-FINE、DEIM 与 RT-DETRv4 集成计划](../../plans/2026-08-12-dfine-deim-rtdetrv4-integration.md)

上游为 `Intellindust-AI-Lab/DEIM@09d35d53d39ee3145a1e61e3a989b28b9468d1dd`（Apache-2.0）。官方资产继续由上游托管，不进入本项目 Release；配置和 manifest 已进入 wheel/sdist。

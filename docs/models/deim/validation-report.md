# DEIM 验证报告

> 验证快照：2026-08-14。本文合并 DEIM-D-FINE 与 DEIM-RT-DETRv2 两个运行时 profile；逐变体数据见[指标记录](metrics.md)。

## 结论

DEIM-D-FINE N/S/M/L/X 与 DEIM-RT-DETRv2 S/M/M*/L/X 均已通过官方 checkpoint strict load、固定输入与四图上游 parity、完整 COCO val2017、reduced train/resume、eager、ONNX opset 17 和 TorchScript。上游固定 `Intellindust-AI-Lab/DEIM@09d35d53d39ee3145a1e61e3a989b28b9468d1dd`。

文档统一为 DEIM，不把原项目改称 DEIMv1；独立 DEIMv2 上游不在本次验证范围。

## 共同训练与恢复合同

两个 profile 都使用 MAL `gamma=1.5`、Dense O2O、FlatCosine 和两阶段 EMA。Stage companion 的 basename/SHA、family、stage、配置和完整组件状态在修改 live state 前校验；十个变体恢复后的下一 update 与 uninterrupted 路径一致。

Mosaic affine 与固定上游一致地 clamp 边界但保留零面积框。Reduced run 不证明完整 schedule 收敛或 mid-epoch resume。

## DEIM-D-FINE

该 profile 共享 D-FINE eval 图；训练 criterion 覆盖 main/aux/pre/encoder/CDN 的 MAL、bbox、GIoU、local、FGL 与 DDF。Class-agnostic encoder 在 matcher 前使用零标签；MAL quality 在 fractional gamma 前将非有限值置零并限制到 `[0,1]`，最终 loss 继续执行非有限 fail-fast。

- 官方 checkpoint 为 `{"model": state_dict}`、identity mapping、PyTorch native layout。
- CPU/FP32、固定 640 下 stem、backbone、encoder 和 raw outputs 通过 `rtol=1e-5, atol=1e-6`。
- 五个完整 val2017 AP 与上游三位小数值最大误差 `0.000424`。
- X 变体使用已经完整生成的 prediction JSON 恢复后续 COCOeval；该恢复路径不表示模型推理失败。
- ONNX 固定 640、动态 batch 1/4，全族最大 score/box 误差 `1.18613e-5 / 0.0149012 px`；TorchScript 为零。
- 安装包用户验收在独立 Python 3.11 CPU wheel 中复验最小变体 N 的 verify/train-resume/eval/infer/export。

错误 MAL gamma、VFL 替代、重复 GO row、malformed CDN/pre-output、wrong-family/size/checksum checkpoint、缺失或篡改 stage companion 均在 mutation 前拒绝。

## DEIM-RT-DETRv2

该 profile 只实现 DEIM 所需的受限 RT-DETRv2 图：

- 五个 detector 为 `{"model": state_dict}`、identity mapping、PyTorch native layout，并包含固定 640 anchors/valid mask。
- M* 与 L 都使用 depth-50 初始化，但 encoder expansion、decoder 层数和 detector 图不同，checkpoint 不可交换。
- X encoder hidden 为 384、FFN 2048，decoder hidden 仍为 256。
- 四个 PResNet-vd pretrained 文件只在 Trainer 构建 optimizer/EMA 前加载；eval/infer/export 不需要该资产，核心运行时不导入 Paddle。

CPU/FP32、固定 640 下，stem、backbone、encoder 和 raw outputs 对上游最大绝对误差为零；五变体四张真实图同样为零。完整 val2017 AP 与官方三位小数值最大误差 `0.000525`。

Criterion 使用 MAL 与 box loss，不启用 D-FINE local/FGL/DDF。后处理必须 focal sigmoid 后展平 `queries x classes` 做全局 TopK 300；错误 softmax/query TopK 会使 S/M AP 降至 `0.4547 / 0.4805`，即使 raw-output parity 仍可能通过。

ONNX 固定 640、动态 batch 1/4。S/M/M*/L score 门为 `2e-5`，X 为 `4e-4`，全族 box 门为 `0.1 px`；该 family-specific 例外不得扩散到其他模型族。TorchScript 为零误差。

RT-DETRv3 checkpoint、D-FINE local loss、不支持的 profile、M*/L checkpoint 交换、wrong-family/size/checksum 和错误 stage companion 均在状态修改前拒绝。

## 共同限制

未验证完整 schedule、多 seed、动态高宽、低精度、TensorRT 或性能。官方 Google Drive checkpoint 不由本项目自动下载或发布；两个运行时 profile 的配置、资产和容差不能因文档合并而互换。

# DEIM-D-FINE 验证报告

> 验证快照：2026-08-14。逐变体数据见[指标记录](metrics.md)。

## 结论

DEIM-D-FINE N/S/M/L/X 已通过官方 checkpoint strict load、固定输入与四图上游 parity、完整 COCO val2017、reduced optimizer/EMA、两阶段 epoch-boundary resume、eager、ONNX opset 17 和 TorchScript。上游固定 `Intellindust-AI-Lab/DEIM@09d35d53d39ee3145a1e61e3a989b28b9468d1dd`。

## 训练合同

该分支共享 D-FINE eval 图，但训练使用 MAL `gamma=1.5`、GO union、Dense O2O、FlatCosine 和两阶段 EMA。Criterion 覆盖 main/aux/pre/encoder/CDN 的 MAL、bbox、GIoU、local、FGL 与 DDF。Class-agnostic encoder 在 matcher 前使用零标签；MAL quality 在 fractional gamma 前将非有限值置零并限制到 `[0,1]`，最终 loss 仍执行非有限 fail-fast。Mosaic affine 与固定上游一致地 clamp 边界但保留零面积框；后续 reader 不得擅自过滤这类框而改变上游语义。

Stage companion 的 basename/SHA、family、stage、配置和完整组件状态在修改 live state 前校验。五变体恢复后的下一 update 与 uninterrupted 路径一致。Reduced run 不证明完整 schedule 收敛或 mid-epoch resume。

## 数值、推理与部署

- 官方 checkpoint 为 `{"model": state_dict}`、identity mapping、PyTorch native layout。
- CPU/FP32、固定 640 下 stem、backbone、encoder 和 raw outputs 通过 `rtol=1e-5, atol=1e-6`。
- 五个完整 val2017 AP 与上游三位小数值最大误差 `0.000424`。
- X 变体推理已经完整生成 prediction JSON，外层进程在后续评估阶段超时；验收只对该完整 JSON 重新运行 COCOeval，没有重复推理。该恢复路径不表示模型推理失败。
- ONNX 固定 640、动态 batch 1/4，全族最大 score/box 误差 `1.18613e-5 / 0.0149012 px`；TorchScript 为零。
- 安装包用户验收在独立 Python 3.11 CPU wheel 中复验最小变体 N 的 verify/train-resume/eval/infer/export。

## 负例与限制

错误 MAL gamma、VFL 替代、重复 GO row、malformed CDN/pre-output、wrong-family/size/checksum checkpoint、缺失或篡改 stage companion 均在 mutation 前拒绝。未验证完整 schedule、多 seed、动态高宽、低精度、TensorRT 或性能。官方 Google Drive checkpoint 不由本项目发布或自动下载。

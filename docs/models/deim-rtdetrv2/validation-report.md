# DEIM-RT-DETRv2 验证报告

> 验证快照：2026-08-14。该实现只是 DEIM 所需的受限 RT-DETRv2 分支，不是独立 RT-DETRv2 产品族。

## 结论

S/M/M*/L/X 已通过官方 detector checkpoint strict load、固定输入与四图上游 parity、完整 COCO val2017、PResNet 训练初始化、reduced train/resume、eager、ONNX opset 17 和 TorchScript。上游固定 `Intellindust-AI-Lab/DEIM@09d35d53d39ee3145a1e61e3a989b28b9468d1dd`。

## 图与资产边界

- 五个 detector 为 `{"model": state_dict}`、identity mapping、PyTorch native layout，并包含固定 640 anchors/valid mask。
- M* 与 L 都使用 depth-50 初始化，但 encoder expansion、decoder 层数和 detector 图不同，checkpoint 不可交换。
- X encoder hidden 为 384、FFN 2048，decoder hidden 仍为 256。
- 四个 PResNet-vd pretrained 文件只在 Trainer 构建 optimizer/EMA 前加载；eval/infer/export 不需要该资产。核心运行时不导入 Paddle。

## 数值与训练

CPU/FP32、固定 640 下，stem、backbone、encoder 和 raw outputs 对上游最大绝对误差为零；五变体四张真实图同样为零。完整 val2017 AP 与官方三位小数值最大误差 `0.000525`。

Criterion 使用 MAL `gamma=1.5` 与 box loss，不启用 D-FINE local/FGL/DDF。后处理必须 focal sigmoid 后展平 `queries x classes` 做全局 TopK 300；观察到错误 softmax/query TopK 会使 S/M AP 降至 `0.4547 / 0.4805`，即使 raw-output parity 仍可能通过。

五变体 reduced resume 覆盖 model、optimizer、scheduler、EMA、global step、stage companion 和两阶段 restart；不证明完整 60/120 epoch 收敛或 mid-epoch resume。

## 部署与限制

ONNX 固定 640、动态 batch 1/4。S/M/M*/L score 门为 `2e-5`，X 为 `4e-4`，全族 box 门为 `0.1 px`；该 family-specific 例外不得扩散到其他模型族。TorchScript 为零误差。

RT-DETRv3 checkpoint、D-FINE local loss、不支持的 profile、M*/L checkpoint 交换、wrong-family/size/checksum 和错误 stage companion 均在状态修改前拒绝。未验证完整 schedule、多 seed、动态高宽、低精度、TensorRT 或性能。Google Drive 官方资产不由本项目自动下载或发布。

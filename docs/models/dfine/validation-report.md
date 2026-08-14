# D-FINE 验证报告

> 验证快照：2026-08-14。数值见[指标记录](metrics.md)，任务映射见[证据索引](evidence-index.md)。

## 结论

D-FINE N/S/M/L/X 已通过官方 checkpoint 严格加载、固定输入和四张真实图上游对齐、完整 COCO val2017、reduced 两阶段训练/epoch-boundary resume、eager 推理、ONNX opset 17 和 TorchScript。组件、模型、用户接口、打包、文档和最终审计均为 `APPROVE`。

上游固定 `Peterande/D-FINE@267a6da6d04c8ad52e54120692896515b9e55981`。所有数值对齐均针对该原生 PyTorch 上游，不是 Paddle parity。

## 验证环境与范围

- 主模型验收：Python `3.12.13`、PyTorch `2.5.1+cu121`、CPU/FP32、固定 640。
- 完整 val2017：5000 图、PIL bilinear、每变体 `1,500,000` 条 top-300 prediction。机器收据未能读取 pycocotools 的模块版本，记录为 `unknown`；同一锁文件当前解析为 `pycocotools 2.0.10`，这是锁文件重建值而非直接观测值。
- 安装包用户验收：Python `3.11.15`、PyTorch `2.5.1+cpu`，从 site-packages 复验最小变体 N。
- HGNetv2 B0/B2/B4/B5 的 stage-1 state、stem 和四个 stage activation 均对齐。
- D-FINE primitives、FDR/LQE、matcher、DN、criterion 的 loss key/数值/梯度与上游对齐。

Eval/Test 必须使用 PIL bilinear。观察到 OpenCV cubic 会使 N AP 从通过门的 `0.427997` 降至 `0.426412`；raw model input 对齐不能替代预处理验证。

## 训练与恢复

五变体均完成有限 optimizer/EMA update、stage-1 companion 发布与 SHA 校验、stage transition、EMA restart，以及恢复后下一 update 与不中断路径一致。失败前先验证 family/stage/config/companion 和完整组件状态，回滚覆盖 model、optimizer、scheduler、scaler、EMA、协议和 RNG。

这只证明 reduced update 和相同 world size 的 epoch-boundary resume，不证明完整 schedule 收敛、mid-epoch resume 或多 seed 稳定性。

## 推理与部署

五变体四图 raw boxes/logits 均通过 `rtol=1e-5, atol=1e-6`。ONNX 固定 640、动态 batch 1/4；score/box 门为 `2e-5 / 0.02 px`。TorchScript batch 1/4 逐值一致。图审计确认无 criterion、denoising 或 auxiliary training residue。

## 负例与限制

- 错误变体、key、shape、dtype、非有限 tensor 和错误 stage-1 companion 均在 mutation 前拒绝。
- D-FINE 使用 Dense O2O/Mosaic、动态高宽、training output 或错误固定尺寸导出均失败。
- 空目标只声明有限 loss/gradient，不声明上游零长度 DN key 集逐项一致。
- 未验证完整训练 schedule、CUDA/低精度部署、TensorRT、latency、吞吐或显存。
- 官方 checkpoint 由上游托管，不属于本项目 Release。

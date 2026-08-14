# RT-DETRv4 验证报告

> 验证快照：2026-08-14。逐变体数据见[指标记录](metrics.md)，任务映射见[证据索引](evidence-index.md)。

## 结论

RT-DETRv4 S/M/L/X 已通过官方 solver checkpoint 的 `ema.module` 严格加载、固定输入和四张真实图上游对齐、完整 COCO val2017、student-only eager/ONNX/TorchScript，以及真实 DINOv3 teacher 的 reduced update、DSI/GAM 和 epoch-boundary resume。Task 18-23 与 F1-F4 均为 `APPROVE`。

COCO AP 来自官方 EMA checkpoint。Reduced training 只证明有限更新、协议状态和确定性恢复，不证明本仓库从头运行完整 schedule 后收敛到这些 AP。

## 环境

- 任务级：Linux、Python `3.12.13`、PyTorch `2.5.1+cu121`。
- Checkpoint parity：CPU/FP32、固定 seed、单 torch thread。
- 完整 COCO：RTX 4090、CUDA/FP32、5000 张 val2017 图片。
- GAM DDP：CPU/FP32、双进程 Gloo。
- F3 wheel：Python `3.11.15`、PyTorch `2.5.1+cpu`、ONNX `1.22.0`、ONNX Runtime `1.28.0`。

## Student 验证

- 上游固定 `RT-DETRs/RT-DETRv4@55fefaaed7efe2a5f72d0a18fd4e05965e35c292`。
- checkpoint 使用 PyTorch native layout、identity key mapping，固定 640 输入的 stem、backbone、encoder、raw boxes/logits 对上游通过 `rtol=1e-5, atol=1e-6`。
- 四张真实 COCO 图中 S/M/X raw output 为零误差；L 最大 logits/box 误差为 `1.04904e-5 / 1.78814e-7`。
- 完整 val2017 每变体产生 `1,500,000` 条候选，实测 AP 与上游三位小数值最大误差为 `0.000604`。

## Teacher 与训练协议

Teacher 固定为 `facebookresearch/dinov3@346f38fee679c56a6888f91c51670fae61d364e0` 的 `dinov3_vitb16`：embed dim 768、patch size 16，输出 `x_norm_patchtokens`。Preflight 在 optimizer/EMA 前验证 Python `>=3.11`、干净 checkout、精确 revision、hub entry、`.pth` 文件身份、模型类型、patch geometry 和有限输出。

Teacher 始终 eval、frozen、`no_grad()`，feature 显式 detach。DSI 比较 projected AIFI F5 与 teacher feature；GAM 在 AMP unscale 后、clip 前统计并跨 rank 汇总梯度 L1。任一 rank 非有限或 AMP skip 时同步跳过。两阶段 checkpoint 持久化 stage、best、restart、EMA、companion SHA 和 GAM weight，但不保存 teacher。四变体 uninterrupted/resumed 的下一更新均通过。

DINOv3 使用自定义许可证和门控授权。checkout 与权重不进入仓库、wheel、sdist 或 Release。

## Student-only 部署

Eval 只返回 `bbox`/`bbox_num`。ONNX 为 opset 17、固定 640x640、动态 batch 1/4，四变体均记录 `training_residue=false`；图中不含 teacher、DSI projector、distillation 或 GAM residue。TorchScript 与 eager 逐值一致。

## 负例与限制

- 缺失、错误 revision、脏 checkout、错误权重大小/SHA、safetensors 替代、错误 embed dim/patch geometry 均在 teacher preflight 拒绝。
- Stale/非有限 GAM state、rank 权重分歧、错误 stage companion 和 wrong-family checkpoint 均在状态修改前拒绝。
- 未运行完整官方 schedule、多 seed、FP16/BF16、TensorRT、动态高宽或性能基准。
- 官方 student checkpoint 由上游托管；本项目不发布，也不对门控 URL执行自动下载。

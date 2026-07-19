# 迁移局限

本页记录当前 PaddlePaddle 到 PyTorch 迁移的已知边界。这些限制应在新计划、训练验收和数值对齐中显式处理。

## 验证范围

- 当前默认测试覆盖权重转换、数据操作、注意力、解码器和部分训练策略；通过并不等于已完成完整 COCO 训练、评估或 mAP 对齐。
- 迁移早期针对旧构建器、Registry 和模型参数的测试保留在 `tests/legacy/`，默认不收集。这些场景需要按当前 API 重写，不应通过强行恢复旧兼容层来绕过。
- Train/Eval/Infer/Convert/Export/Models 均有当前 CLI contract。公开 R18/R34/R50 checkpoint 已分别通过 Models CLI 固定 URL 下载、CPU/FP32 真实 COCO 单图 Infer 和统一四图 Eval；R18 另有 batch 4 和完整 val2017 证据。ONNX opset 17/ONNX Runtime CPU 与 traced TorchScript 已覆盖三个变体的固定 640、动态 batch 1/4/8 和真实图，R18 另有固定 608；Infer CLI 已补充 R18 TorchScript 和 ONNX Runtime 的 CUDA/CPU 四图证据。这些证据仍不证明 Paddle CLI 全参数、单产物动态高宽、R34/R50 导出后端 CUDA、TensorRT 或 C++ 部署已支持。

## 数值等价

- 形状一致、无 NaN/Inf 和确定性只是基础条件，不证明 Paddle 与 PyTorch 数值等价。
- 真正的对齐需要同一份权重、相同输入、相同预处理参数和分层激活对比，最终还需要在同一数据集上验证指标。
- 随机增强、插值、padding、NMS、浮点精度和不同后端算子都可能产生合理但可累积的差异。
- 官方 R18/R34/R50 已完成共 2,041 个参数逐值转换校验、eval 分层、确定性缩减训练 loss 和完整模型整体梯度方向对齐；不要求 AdamW 逐元素完全一致。R50 后处理仍有 2/300 个 top-k 离散边界候选差异。PyTorch 已实现 ResNet LR multiplier 参数组并完成一次真实 COCO epoch/val 可运行性验收，但与 Paddle 相同的标准 schedule、多 seed 收敛和 AP 对照仍未完成；基础 LR、warmup 长度和衰减配置也仍有意保留差异。
- 当前 batch conversion 只支持一个 config/架构；跨架构需要分别运行。低内存模式会释放源 tensor 并避免常驻完整目标模型，但 Paddle 文件与最终 PyTorch state dict 仍整体驻留，不是流式格式。
- Paddle 权重转换范围只包括模型参数，不迁移 Paddle optimizer 状态。PyTorch 训练侧已有 schema v1 保存 optimizer、scheduler、EMA、GradScaler、步数和 RNG，但它只用于恢复可信的 PyTorch 自有 checkpoint。

详细规则见[权重转换经验](weight-conversion.md)和[训练与验证经验](training-and-validation.md)。

## 框架与环境

- Paddle 参考实现是 `third-party/RT-DETRv3-paddle` 子模块，不会打包进 PyTorch wheel。开发工具还依赖子模块已初始化。
- wheel 包含当前 26 个 YAML 配置与 Apache-2.0/NOTICE，但不包含 checkpoint 或数据集；sdist 同样排除 Paddle 子模块和 `pretrained_models/`。用户仍需根据 manifest 单独获取权重并校验 SHA-256；`v0.1.0` 固定 tag 的 GitHub Release 已提供转换权重、mapping report 和覆盖所有资产的 `SHA256SUMS`。
- Paddle 和相关对齐工具位于 `dev` 附加依赖；核心 PyTorch 运行时不应直接导入 Paddle。
- CPU ONNX Runtime 位于 `export`/`test`，GPU ONNX Runtime 位于 `export-gpu`/`dev`；两类 distribution 由 UV conflicts 隔离。当前 CUDA 12 GPU extra 限制 `<1.27`，避免安装依赖 CUDA 13 的 ORT 1.27。核心训练/eager 运行时不要求安装或导入它们。
- Paddle 扩展算子不会随 `uv sync --extra dev` 自动编译，使用旋转框等特定路径时需要额外构建。
- 当前锁文件面向 Python 3.9–3.12，PyTorch 默认使用 CUDA 12.1 索引。CPU、macOS、ARM 或其他 CUDA 版本需要替换合适的 PyTorch 索引并重新锁定。

## 配置与 API

- PyTorch 保留了部分 PaddleDetection 风格的 YAML 和注册机制，但两个框架的对象创建、参数注入和 DataLoader 语义不能假设完全等价。
- 当前 `create()` 接受类、注册名、全局命名配置块或带 `name`/`type` 的配置映射；不带组件名的任意字典会失败。显式构造参数优先于配置块和 `from_config()`，显式注入目标会先递归解析。
- 组件依赖仍在导入时注册。`load_config()` 现已隔离连续 YAML 的运行时值并保留注册 schema，但 `global_config` 仍是当前进程共享的活动配置；测试中的临时 `merge_config()` 和手工修改仍必须在前后恢复。
- 当相对 `configs/...` 路径在当前工作目录不存在时，`load_config()` 会回退到 wheel 内配置；绝对路径、已存在的用户相对路径和不以 `configs/` 开头的自定义路径不会被改写。
- 数据集路径使用仓库相对默认值，实际训练仍需根据本机数据位置覆盖配置。

当前行为和设计约束见[注册与配置经验](registry-and-configuration.md)与[配置迁移指南](configuration-guide.md)。

## 导出

- 当前 ONNX/TorchScript 图只声明动态 batch；deformable attention 中的 Python 整数转换和按层循环会固化空间 shape。改变高宽必须先同步模型 `eval_size` 并重新导出，不能把 ONNX 的 batch 动态轴解释为任意高宽支持。
- 导出 tensor 合同从归一化后的 image tensor 开始，止于 raw `bbox/bbox_num`；图片解码、TestReader 和阈值过滤不在图中。本仓库 Infer CLI 已为 R18 验证复用 TestReader 的端到端路径，但任何外部部署端的预处理未经对比时，导出回归通过仍不证明其预测一致。
- Infer 的 ONNX 路径默认 CPU，并接受显式 `cuda[:id]`；GPU wheel 缺失或 session 完全落回 CPU 时失败。TorchScript 由 PyTorch runtime 执行，在 CUDA 可用时默认 CUDA并支持显式 CPU fallback。ONNX 从图读取固定空间 shape，新 TorchScript 归档内嵌输入尺寸元数据；旧无元数据 TorchScript 可兼容加载，但不能在运行前由 CLI 证明尺寸匹配。
- 当前回归使用官方 R18/R34/R50、CPU/FP32、ONNX opset 17 和 ONNX Runtime CPU provider；R34/R50 只验证固定 640。TorchScript 与 ONNX Runtime CUDA 都只覆盖 R18、固定 640、FP32 和当前 Python Infer CLI。其他配置、dtype、provider 和图优化级别需要独立证据。
- raw top-k 尾部会对近似并列 score 的微小后端误差敏感，R34/R50 ONNX 已稳定观测到每图最多两个低分候选重排行序。当前验收要求 `bbox_num`/分组严格相等，并在每张图内对全部候选按类别、score 和 box 一对一匹配；score/坐标门槛为 `2e-5/0.02 px`，同时报告重排行数。任何候选缺失、跨图配对或超容差仍失败，不能只比较最终可视化。
- CPU/CUDA 不应共用逐位一致假设。R18 四图中，TorchScript CUDA 相对 eager CUDA 的最大 score/box 误差为 `2.79218e-4/0.00872803 px`，ONNX CUDA 为 `6.06865e-4/0.0238647 px`；两条 CPU 对照分别为 `1.90735e-6/9.15527e-5 px` 与 `6.82473e-6/0.000183105 px`。ONNX CUDA 使用单独记录的 `1e-3/0.03 px` 门槛；跨设备比较只作为观测，不修改 M8 的 CPU 默认容差。

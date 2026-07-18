# 迁移局限

本页记录当前 PaddlePaddle 到 PyTorch 迁移的已知边界。这些限制应在新计划、训练验收和数值对齐中显式处理。

## 验证范围

- 当前默认测试覆盖权重转换、数据操作、注意力、解码器和部分训练策略；通过并不等于已完成完整 COCO 训练、评估或 mAP 对齐。
- 迁移早期针对旧构建器、Registry 和模型参数的测试保留在 `tests/legacy/`，默认不收集。这些场景需要按当前 API 重写，不应通过强行恢复旧兼容层来绕过。
- Train/Eval/Infer/Convert/Export 均有当前 CLI contract。Infer 已用官方 R18 checkpoint 完成 CPU/FP32 真实 COCO 单图、batch 4 和 608/640 尺寸验证；ONNX opset 17/ONNX Runtime CPU 与 traced TorchScript 已覆盖固定高宽、动态 batch 1/4/8。它仍不证明 Paddle CLI 全参数、单产物动态高宽、R34/R50 导出、GPU provider、TensorRT 或 C++ 部署已支持。

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
- Paddle 和相关对齐工具位于 `dev` 附加依赖；核心 PyTorch 运行时不应直接导入 Paddle。
- ONNX 和 ONNX Runtime 位于 `export` 附加依赖，并因导出回归测试同时包含在 `dev` 中；核心训练/eager 运行时不要求安装或导入它们。
- Paddle 扩展算子不会随 `uv sync --extra dev` 自动编译，使用旋转框等特定路径时需要额外构建。
- 当前锁文件面向 Python 3.9–3.12，PyTorch 默认使用 CUDA 12.1 索引。CPU、macOS、ARM 或其他 CUDA 版本需要替换合适的 PyTorch 索引并重新锁定。

## 配置与 API

- PyTorch 保留了部分 PaddleDetection 风格的 YAML 和注册机制，但两个框架的对象创建、参数注入和 DataLoader 语义不能假设完全等价。
- 当前 `create()` 接受类、注册名、全局命名配置块或带 `name`/`type` 的配置映射；不带组件名的任意字典会失败。显式构造参数优先于配置块和 `from_config()`，显式注入目标会先递归解析。
- 组件依赖仍在导入时注册。`load_config()` 现已隔离连续 YAML 的运行时值并保留注册 schema，但 `global_config` 仍是当前进程共享的活动配置；测试中的临时 `merge_config()` 和手工修改仍必须在前后恢复。
- 数据集路径使用仓库相对默认值，实际训练仍需根据本机数据位置覆盖配置。

当前行为和设计约束见[注册与配置经验](registry-and-configuration.md)与[配置迁移指南](configuration-guide.md)。

## 导出

- 当前 ONNX/TorchScript 图只声明动态 batch；deformable attention 中的 Python 整数转换和按层循环会固化空间 shape。改变高宽必须先同步模型 `eval_size` 并重新导出，不能把 ONNX 的 batch 动态轴解释为任意高宽支持。
- 导出 tensor 合同从归一化后的 image tensor 开始，止于 raw `bbox/bbox_num`；图片解码、TestReader 和阈值过滤不在图中。部署端预处理未经对比时，导出回归通过也不证明端到端预测一致。
- 当前回归只使用官方 R18、CPU/FP32、ONNX opset 17 和 ONNX Runtime CPU provider。其他模型、dtype、provider 和图优化级别需要独立证据。
- raw top-k 尾部可能对近似并列 score 的微小后端误差敏感。当前验收仍要求标签/行顺序和 `bbox_num` 完全相等，并分别限制 score 与坐标误差；若失败，应先定位第一个分歧候选，不能只比较最终可视化。

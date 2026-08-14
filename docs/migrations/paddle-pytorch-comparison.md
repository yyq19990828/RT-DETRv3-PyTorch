# PaddlePaddle 与 PyTorch 迁移语义对照

本文只保留当前可复用的框架语义差异和验证顺序。2025-10-20 的逐模块长篇分析已经移入[历史归档](../archive/rtdetrv3-v0.1.0/migration/paddle-pytorch-comparison.md)；其中的完成度和生产可用性描述只代表当时观察，不是当前仓库状态。

## 证据口径

- **已验证**：相同 checkpoint、预处理、输入、模式、dtype 和容差下已有可复现结果。
- **已观察**：在记录环境中出现过，但尚未形成跨环境或完整矩阵证据。
- **推断**：根据源码、API 或方程分析得到，仍需要数值实验确认。
- **计划中**：尚未执行，不能用于支持当前合同。

类名、参数名、tensor shape、成功加载和确定性输出都不能单独证明框架等价。最终预测不一致时，应比较第一个分歧中间激活。

## 当前语义矩阵

| 主题 | PaddlePaddle | PyTorch 迁移要求 | 证据入口 |
|---|---|---|---|
| 参数注册与构建 | workspace、`__inject__`、`__shared__`、YAML 继承 | 保留配置驱动依赖注入和重复加载一致性 | [注册与配置](registry-and-configuration.md) |
| Linear 权重 | 常见布局为 `[in, out]` | 对照目标 `state_dict` 决定是否转置，不能按名称盲转 | [权重转换](weight-conversion.md) |
| Conv/BN 状态 | Conv 通常无需转置；BN 有参数与运行状态 | 名称映射、layout 和加载集合分别验证 | [权重转换](weight-conversion.md) |
| 优化器与 LR | 参数组、正则、scheduler step 单位可能不同 | 核对方程、更新顺序、AMP skip、累积和恢复 suffix | [训练与验证](training-and-validation.md) |
| DataLoader 与增强 | collate、padding、插值和随机源由框架实现影响 | 固定预处理、seed、worker 与 epoch-aware 策略后再比较 | [训练与验证](training-and-validation.md) |
| 后处理 | TopK、坐标缩放和离散边界可能放大微差 | 同时比较 raw output、阈值前候选和最终检测 | [排错经验](troubleshooting.md) |
| 分布式与状态 | SyncBN、reduction、sampler 和 RNG 行为不同 | 记录 rank、同步边界、checkpoint 组件和恢复时点 | [训练与验证](training-and-validation.md) |

## 验证顺序

1. 固定上游 revision、配置、checkpoint 来源、大小和 SHA-256。
2. 分别验证名称映射、tensor layout、missing/unexpected key 和 dtype。
3. 在 eval 模式按 backbone → neck/encoder → transformer/decoder → head 比较具名激活。
4. 比较 raw logits/boxes，再比较后处理候选和最终检测。
5. 训练态依次比较 loss 分项、参数梯度、optimizer/scheduler/EMA 更新和恢复 suffix。
6. 完整 COCO、性能、低精度和多卡结果各自记录环境与容差，不从局部测试外推。

RT-DETRv3 的 Paddle 对齐结果见[模型验证报告](../models/rtdetrv3/validation-report.md)；D-FINE、DEIM 与 RT-DETRv4 来自原生 PyTorch 上游，使用[上游 PyTorch 数值对齐](upstream-pytorch-parity.md)协议，不宣称 Paddle parity。

## 当前边界

- RT-DETRv3 R18/R34/R50 已完成官方 checkpoint 转换、eval 分层、受控训练 loss 和梯度证据；R18 同权重完整 val2017 对齐已通过。
- RT-DETRv3 标准 schedule、多 seed 及 R34/R50 长训仍是延期工作，以 [`ROADMAP.md`](../../ROADMAP.md) 为准。
- 新模型族的 reduced train/resume 只证明有限更新和 epoch-boundary 恢复，不证明完整 schedule 收敛。
- Paddle 及迁移专用依赖继续限制在 `dev` extra，核心 PyTorch 运行时不得导入 Paddle。

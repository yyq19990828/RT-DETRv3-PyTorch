# 权重转换经验

本文整合历史权重转换规格与当前 `ppdet_pytorch.conversion` 实现。

## 当前可验证能力

- 加载 `.pdparams`，通过 NumPy 转为 PyTorch tensor。
- 按规则转换 BatchNorm、权重和 bias 名称，并允许 JSON 手工覆盖。
- 使用目标 `state_dict` 做形状检查，支持严格失败和宽松跳过。
- 保存模型权重、转换元数据、统计和参数映射报告。
- 小型 fixture 的单元和集成回归已纳入默认测试集。

## 参数名称映射

常见规则：

| Paddle | PyTorch | 说明 |
|---|---|---|
| `._mean` | `.running_mean` | BatchNorm 运行均值 |
| `._variance` | `.running_var` | BatchNorm 运行方差 |
| `._scale` | `.weight` | BatchNorm scale |
| `._offset` | `.bias` | BatchNorm offset |
| `.w_0` | `.weight` | 普通权重 |
| `.b_0` | `.bias` | 普通 bias |

规则匹配只说明名称候选合理，不证明该参数已正确对齐。必须同时检查目标 key、形状、dtype 和数值。自定义模块、不同层次命名或重构后的模型需要手工映射。

## 张量布局

- 卷积权重通常同为 `[out_channels, in_channels, H, W]`，不应因框架不同就无条件转置。
- Paddle Linear 权重通常为 `[in_features, out_features]`，PyTorch 为 `[out_features, in_features]`；当前转换器对 RT-DETRv3 已知 Linear 名称模式执行二维转置。
- 只看元素数相等不能盲目 reshape。应先确认该层的数学语义，再决定 transpose/permute。
- 默认保留源 dtype。FP16/BF16 权重应在转换后单独记录误差，不与 FP32 使用同一容差。

## 校验层级

1. **文件级**：后缀、可读性、大小、checksum、输出元数据。
2. **映射级**：源/目标 key 覆盖率、手工覆盖、未映射清单。
3. **参数级**：shape、dtype、transpose 规则和数值保留。
4. **模块级**：同输入下比较 backbone/neck/transformer/head 激活。
5. **模型级**：同一预处理后比较 boxes/logits 和 COCO 指标。

当前基线主要覆盖前三层的小型 fixture；官方 R18/R34/R50 的模块级与模型级验收仍在路线图中。

## 已遇到的框架边界

- 部分 Paddle 版本的 `paddle.load` 不接受 `pathlib.Path`，在边界使用 `paddle.load(str(path))`。
- `Path` 写入 JSON 元数据会触发序列化错误，应在构建元数据时转为 `str`。
- 转换优化器、调度器或 AMP scaler 状态风险很高；当前仅转换模型参数，转换后用 PyTorch 重新初始化训练状态。
- 分布式训练产物可能带 `module.` 等 wrapper 前缀，量化、剪枝和自定义算子元数据不能假设可跨框架保留。

## 尚未完成

- 批量 glob 发现、单文件失败后继续、批量统计和输出路径生成。
- 建议性 shape fix、真正的分块/低内存模式和内存/时间 profiling。
- 官方 R18/R34/R50 权重的覆盖率、转换时间、峰值内存和实际模型加载验收。
- 转换后的分层激活、预测与 COCO AP 对齐。

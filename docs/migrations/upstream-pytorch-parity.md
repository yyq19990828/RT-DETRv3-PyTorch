# 上游 PyTorch 数值对齐

本文记录原生 PyTorch 上游与本仓库实现之间可复用的验证合同。它不构成任何模型已完成数值对齐的声明；各模型的配置、实测结果和例外分别记录在 [`docs/models`](../models/README.md)。

## 固定来源

| 上游 | Revision | 许可证 |
|---|---|---|
| `Peterande/D-FINE` | `267a6da6d04c8ad52e54120692896515b9e55981` | Apache-2.0 |
| `Intellindust-AI-Lab/DEIM` | `09d35d53d39ee3145a1e61e3a989b28b9468d1dd` | Apache-2.0 |
| `RT-DETRs/RT-DETRv4` | `55fefaaed7efe2a5f72d0a18fd4e05965e35c292` | Apache-2.0 |

上游 checkout、checkpoint、COCO 和 DINOv3 资产必须位于仓库外。存在但 revision、大小或 SHA-256 不匹配是验证失败；资产不存在才是 blocked，不能用随机权重或 shape smoke 替代。

## 对齐顺序

1. 在构建模型前验证计划身份、上游 revision、manifest、资产大小和 SHA-256。
2. 对显式 key adapter 后的 state tensor 做逐值比较，分别验证名称映射和 layout。
3. 使用同一 checkpoint、预处理、输入、eval mode、dtype 和 seed，对注册 hook 的中间激活按稳定名称顺序比较。
4. 比较 raw logits/boxes，再比较后处理后的每图检测集合。
5. 训练路径另行比较 loss、输入梯度和模型梯度；成功加载或 shape 一致不算数值证据。

首个失败项必须记录 tensor 名、两侧 shape/dtype、最大绝对/相对误差和最大误差位置。所有 hook 在 `finally` 路径移除，采集结果在比较前 `detach`、复制到 CPU，避免后续 forward 修改证据。

## 原生 PyTorch checkpoint

- 先验证文件大小和 SHA-256，再反序列化；来源 URL、container、state 路径和 tensor 数分别记录，不能互相替代。
- `{"model": state_dict}`、裸 backbone state 和 solver checkpoint 的 `ema.module` 是不同合同。评估 state 必须由 manifest 明确指定，不能按键名猜测。
- identity mapping 仍需分别验证 key 集、shape、dtype、有限性和逐 tensor 值；“无需转置”不等于“无需校验”。
- 恢复训练前先验证全部 model/optimizer/scheduler/scaler/EMA/RNG 与 stage companion，再统一应用；任何失败不得留下部分 mutation。

## 固定容差

| 表面 | 合同 |
|---|---|
| state tensor | key adapter 后逐值一致 |
| intermediate activation、raw logits/boxes | `rtol=1e-5, atol=1e-6` |
| loss、gradient | `rtol=1e-4, atol=1e-6` |
| ONNX/TorchScript score | `atol=2e-5` |
| ONNX/TorchScript box | `atol=0.02 px` |
| `bbox_num`、label、一对一候选关系 | 严格一致 |

实测失败不得通过事后放宽容差改写为通过。应先定位首个分歧激活，再判断是实现、权重映射、预处理还是运行环境差异。

模型族专属容差不得提升为共享默认值。当前 DEIM-RT-DETRv2 的预注册 ONNX 例外及实测结果见其[模型合同](../models/deim-rtdetrv2/README.md#部署边界)；D-FINE、DEIM-D-FINE 和其他模型族继续使用表中的默认门槛。

## 驱动与证据

- `tools/dev/compare_upstream_pytorch.py`：具名 state、activation 和 output 比较。
- `tools/dev/validate_model_family.py`：manifest 绑定的 checkpoint、训练恢复、COCO、推理、导出和 teacher 矩阵。
- `tools/dev/audit_plan_evidence.py`：任务证据、当前计划身份和 revision 审计。
- `tools/dev/audit_model_family_graphs.py`：核心依赖、opset、训练节点残留、重复实现和容差合同审计。

驱动统一输出确定性 schema `{schema_version, plan_identity, family_results, negatives, status}`，状态和退出码分别为 `APPROVE=0`、`FAIL=1`、`BLOCKED=2`。计划身份直接对原始字节计算 SHA-256，只归一化列零任务 marker 的 checkbox 状态；正文、空格、编号和换行的任何其他变化都会使既有证据失效。

所有矩阵必须先完整 preflight，再构建模型或写产物。preflight 失败不得修改模型、optimizer、scheduler、scaler、EMA、RNG 或已有输出文件。

## 模型证据入口

- [D-FINE](../models/dfine/README.md)：官方 checkpoint、预处理、COCO、训练恢复和部署结果。
- [DEIM-D-FINE](../models/deim-dfine/README.md)：MAL/Dense O2O、两阶段训练和五变体验收结果。
- [DEIM-RT-DETRv2](../models/deim-rtdetrv2/README.md)：PResNet 初始化、后处理、family-specific ONNX 门槛和五变体验收结果。
- [RT-DETRv4](../models/rtdetrv4/README.md)：已验证的 student 与训练专用 DINOv3 教师边界。

跨模型可复用的结论是：预处理必须独立于模型图进行像素级验证；导出 fixture 应使用固定的非退化输入；checker/reload 成功不能替代 runtime parity 与训练节点残留审计；任何 family-specific 修复或容差必须留在对应模型合同中。

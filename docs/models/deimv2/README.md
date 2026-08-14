# DEIMv2

本目录是 DEIMv2 当前合同的唯一文档入口。固定上游为 [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)@`add5bcdb499bf7b8a366bfeac1a47d3dc278de27`（Apache-2.0；论文 arXiv 2509.20787）。DEIMv2 与本仓库既有 DEIM profile 是不同代的独立上游,checkpoint、配置与验收证据不可混用。

仓库维护两条不能互换 checkpoint 的运行时分支:

- DINOv3 分支:X/L/M/S,DINOv3 或蒸馏 ViT-Tiny 骨干 + STA 空间调优适配器 + sum-fusion 混合编码器。
- HGNetv2 分支:N/Pico/Femto/Atto,HGNetv2(含 Atto/Femto/Pico 剪枝)+ HybridEncoder 或 LiteEncoder 两尺度编码器。

- [验证报告](validation-report.md):数值对齐、完整 val2017、reduced train/resume、部署与限制。
- [指标记录](metrics.md):八个变体的 checkpoint、精度、部署数值与上游敏感性证据。
- [证据索引](evidence-index.md):manifest、测试与可复现入口。

## 当前状态

截至 2026-08-14,八个官方 COCO 变体完成官方 checkpoint identity strict-load、pinned 上游逐激活对齐、完整 val2017、reduced train/resume(两分支代表变体)、ONNX/TorchScript 导出、Models CLI 与文档验收;官方权重由上游 Google Drive 托管,本项目不重新发布。

| 变体 | Backbone/Encoder | 官方 AP | 本仓库实测 AP | 绝对误差 | 配置 | CLI alias |
|---|---|---:|---:|---:|---|---|
| X | DINOv3 ViT-S/16+ + STA | 0.578 | 0.578128 | 0.000128 | [`deimv2_dinov3_x_coco.yml`](../../../configs/deimv2/deimv2_dinov3_x_coco.yml) | `deimv2-x` |
| L | DINOv3 ViT-S/16 + STA | 0.560 | 0.559889 | 0.000111 | [`deimv2_dinov3_l_coco.yml`](../../../configs/deimv2/deimv2_dinov3_l_coco.yml) | `deimv2-l` |
| M | 蒸馏 ViT-Tiny+ + STA | 0.530 | 0.529714 | 0.000286 | [`deimv2_dinov3_m_coco.yml`](../../../configs/deimv2/deimv2_dinov3_m_coco.yml) | `deimv2-m` |
| S | 蒸馏 ViT-Tiny + STA | 0.509 | 0.508602 | 0.000398 | [`deimv2_dinov3_s_coco.yml`](../../../configs/deimv2/deimv2_dinov3_s_coco.yml) | `deimv2-s` |
| N | HGNetv2-B0(2 stage) | 0.430 | 0.429757 | 0.000243 | [`deimv2_hgnetv2_n_coco.yml`](../../../configs/deimv2/deimv2_hgnetv2_n_coco.yml) | `deimv2-n` |
| Pico | HGNetv2-Pico + LiteEncoder | 0.385 | 0.384677 | 0.000323 | [`deimv2_hgnetv2_pico_coco.yml`](../../../configs/deimv2/deimv2_hgnetv2_pico_coco.yml) | `deimv2-pico` |
| Femto | HGNetv2-Femto + LiteEncoder | 0.310 | 0.309933 | 0.000067 | [`deimv2_hgnetv2_femto_coco.yml`](../../../configs/deimv2/deimv2_hgnetv2_femto_coco.yml) | `deimv2-femto` |
| Atto | HGNetv2-Atto + LiteEncoder | 0.238 | 0.237765 | 0.000235 | [`deimv2_hgnetv2_atto_coco.yml`](../../../configs/deimv2/deimv2_hgnetv2_atto_coco.yml) | `deimv2-atto` |

八个 AP 相对官方公布值的最大绝对误差为 `0.000398`,全部满足预注册门槛 `≤ 0.001`。该结果验证官方 checkpoint 的评估路径,不证明完整 schedule 训练收敛。

## 已验证合同

- 八个官方 checkpoint 均为 `{"model": state_dict}` 容器,以 identity mapping 零未知 key 严格加载;checkpoint manifest 见 [`deimv2_coco.yml`](../../../configs/checkpoints/deimv2_coco.yml)。
- HGNetv2 分支四个变体(N/Pico/Femto/Atto)的 backbone/encoder 激活与 raw logits/boxes 在固定随机输入下与 pinned 上游逐位一致(max_abs = 0)。
- DINOv3 分支四个变体的 backbone 激活首分歧为 `≤ 1.9e-6`(容差 `rtol=1e-5, atol=1e-6` 内);解码器随机输入 raw 输出发散由 topk 近平局翻转放大,已用扰动敏感性(约 256 倍)与真实图像 val2017 收敛证据界定,见[指标记录](metrics.md)。
- 完整 val2017 于 CUDA/FP32、官方 checkpoint、逐变体固定输入尺寸(640/416/320)下执行;DINOv3 分支预处理为 ImageNet mean/std 归一化,HGNetv2 分支保持 0/1 归一化,与上游逐像素一致。
- reduced train/resume 以 val2017 派生 96 图 COCO 格式子集、两分支代表变体(dinov3_s、hgnetv2_atto)验证:loss 有限、两阶段协议、EMA、matcher epoch 切换(epoch 1 生效)与 epoch 边界恢复逐位一致(首步 loss 与 LR 完全一致)。该验证不构成精度收敛证据。
- ONNX(opset 17、固定空间尺寸、动态 batch)与 TorchScript 导出:七个变体 TorchScript 逐值一致(score/box max_abs = 0),ONNX 默认容差通过;`deimv2-x` 的 ONNX 随机输入验证存在 3/300 的 top-300 底部近平局交换(未匹配 score 均 < 0.003),真实图像验证为 300/300 或值级漂移(≤ 1.4e-4 score、box 一致),按 DEIM-RTv2 X 先例预注册为 family-specific 例外,不扩散到其他变体。

## DINOv3 许可边界

`DINOv3STAs` 依赖的 DINOv3 ViT 前向代码 vendored 于 `src/ppdet_pytorch/modeling/backbones/dinov3/`,按上游 DEIMv2 同样方式裁剪,保留 Meta DINOv3 License 头。该代码遵循 [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/346f38fee679c56a6888f91c51670fae61d364e0/LICENSE.md),不能按 Apache-2.0 处理;本项目不 vendor、不再分发任何 DINOv3 或作者私有骨干初始权重,训练初始化权重(`weights_path`)由使用者按上游说明自行获取。官方 COCO checkpoint 已包含微调后的骨干权重,评估、推理与导出不需要任何外部骨干资产。

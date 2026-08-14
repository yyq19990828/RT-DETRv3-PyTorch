# DEIMv2 证据索引

按能力域组织 2026-08-14 模型级验收结论。结论明细见[验证报告](validation-report.md)与[指标记录](metrics.md)。

## 组件与权重

| 能力 | 状态 | 入口 |
|---|---|---|
| vendored DINOv3 ViT 前向 + ViT-Tiny + STA 适配器构建 | APPROVE | `src/ppdet_pytorch/modeling/backbones/dinov3/`、`vit_tiny.py`、`deimv2_dinov3.py` |
| HGNetv2 Atto/Femto/Pico 剪枝变体与 B0 部分加载 | APPROVE | `src/ppdet_pytorch/modeling/backbones/hgnetv2.py` |
| DEIMTransformer(SwiGLU/RMSNorm/Gate/共享 query pos/eval_idx 裁剪) | APPROVE | `src/ppdet_pytorch/modeling/transformers/deimv2_decoder.py` |
| LiteEncoder 与 HybridEncoder deim 版本(sum fusion) | APPROVE | `deimv2_lite_encoder.py`、`dfine_hybrid_encoder.py` |
| 官方 checkpoint identity strict-load(8/8) | APPROVE | [`deimv2_coco.yml`](../../../configs/checkpoints/deimv2_coco.yml) |

## 数值

| 能力 | 状态 | 入口 |
|---|---|---|
| pinned 上游逐激活对齐(8/8,seed 42) | APPROVE | 上游 checkout `Intellindust-AI-Lab/DEIMv2@add5bcd`;随机输入对比脚本与结论见 metrics |
| 上游自评对照(s/femto) | APPROVE | 见 validation-report "数值对齐"节 |
| topk 近平界发散的敏感性与边界证据 | APPROVE | metrics "敏感性证据"节 |

## 训练与评估

| 能力 | 状态 | 入口 |
|---|---|---|
| 完整 val2017 ×8(≤ 0.001) | APPROVE | `rtdetrv3-eval -c configs/deimv2/<variant> --checkpoint <official>.pth --device cuda` |
| reduced train/resume(dinov3_s、hgnetv2_atto,逐位一致) | APPROVE | validation-report "训练与恢复"节 |
| Copy-Blend / matcher epoch 切换单测 | APPROVE | `tests/unit/modeling/test_deimv2_loss.py`、`tests/unit/data/test_dense_o2o.py` |

## 部署与用户接口

| 能力 | 状态 | 入口 |
|---|---|---|
| ONNX opset 17 + TorchScript ×8(TorchScript 逐值) | APPROVE(见 x 的 family-specific 例外) | `rtdetrv3-export -c configs/deimv2/<variant> --checkpoint <official>.pth --format both` |
| Models CLI list/verify(8 alias) | APPROVE | `rtdetrv3-models --family deimv2 list / verify deimv2-<variant>` |
| family 矩阵测试 | APPROVE | `tests/unit/core/test_deimv2_configs.py`、`tests/unit/conversion/test_deimv2_checkpoint_manifest.py`、`tests/unit/deploy/test_model_family_matrix.py`、`tests/unit/cli/test_model_families.py`、`tests/integration/test_packaged_model_families.py` |

## 边界

- 权重由上游 Google Drive 托管,本项目只 list/verify,不重新发布。
- vendored DINOv3 前向代码遵循 DINOv3 License,NOTICE 单列;不分发任何骨干初始权重。
- 完整 schedule、多 seed、低精度与吞吐基准未验证,不做声明。
- `deimv2-x` 的 ONNX 随机输入验证例外是 family-specific 合同,不可引用到其他模型族。

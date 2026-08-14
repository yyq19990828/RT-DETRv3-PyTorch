# DEIM-RT-DETRv2 证据索引

| 证据 | 状态 | 结论 |
|---|---|---|
| 依赖切片验证 | APPROVE | 受限 RT-DETRv2 encoder/decoder、五 profile 和上游组件数值 |
| 训练语义与协议验证 | APPROVE | DEIM MAL 与两阶段 checkpoint/EMA/resume |
| 模型矩阵验证 | APPROVE | 五 detector/pretrained、parity、COCO、训练恢复、推理和部署 |
| 用户接口与发布检查 | APPROVE | Models CLI、打包、许可和文档 |
| 最终审计 | APPROVE | 质量、安装包 S 用户链、v3 无回归和计划合规 |

- Manifest：[`configs/checkpoints/deim_rtdetrv2_coco.yml`](../../../configs/checkpoints/deim_rtdetrv2_coco.yml)
- 配置：[`configs/deim/rtdetrv2`](../../../configs/deim/rtdetrv2/)
- Checkpoint parity：[`tests/numerical/test_deim_rtdetrv2_official_checkpoints.py`](../../../tests/numerical/test_deim_rtdetrv2_official_checkpoints.py)
- Upstream components：[`tests/numerical/test_deim_rtdetrv2_components_upstream.py`](../../../tests/numerical/test_deim_rtdetrv2_components_upstream.py)
- Runtime：[`tests/integration/test_deim_rtdetrv2_runtime.py`](../../../tests/integration/test_deim_rtdetrv2_runtime.py)

```bash
DEIM_UPSTREAM_ROOT=<fixed-checkout> \
DEIM_RTDETRV2_PRETRAINED_ROOT=<pretrained-root> \
uv run python tools/dev/validate_model_family.py \
  --family deim-rtdetrv2 --variants all --phase all \
  --checkpoint-root <checkpoint-root> --coco-root <coco-root> \
  --evidence-dir <evidence-dir>
```

原始机器收据保存在本地执行证据目录；本页按能力域提供稳定的结论索引。

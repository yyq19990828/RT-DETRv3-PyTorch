# DEIM 证据索引

## 共同证据

| 证据 | 状态 | 结论 |
|---|---|---|
| 训练语义与协议验证 | APPROVE | MAL、Dense O2O、两阶段 checkpoint/EMA/resume、DDP 与事务回滚 |
| 用户接口与发布检查 | APPROVE | 两个 Models CLI family、打包、许可和文档 |
| 最终审计 | APPROVE | 质量、安装包用户链、RT-DETRv3 无回归和计划合规 |

## DEIM-D-FINE

| 证据 | 状态 | 结论 |
|---|---|---|
| Criterion 验证 | APPROVE | MAL、GO union、loss/gradient 和负例 |
| 模型矩阵验证 | APPROVE | 五变体 checkpoint、parity、COCO、训练恢复、推理和部署 |

- Manifest：[`configs/checkpoints/deim_dfine_coco.yml`](../../../configs/checkpoints/deim_dfine_coco.yml)
- 配置：[`configs/deim/dfine`](../../../configs/deim/dfine/)
- Checkpoint parity：[`tests/numerical/test_deim_dfine_official_checkpoints.py`](../../../tests/numerical/test_deim_dfine_official_checkpoints.py)
- Runtime：[`tests/integration/test_deim_dfine_runtime.py`](../../../tests/integration/test_deim_dfine_runtime.py)

```bash
DEIM_UPSTREAM_ROOT=<fixed-checkout> uv run python tools/dev/validate_model_family.py \
  --family deim-dfine --variants all --phase all \
  --checkpoint-root <checkpoint-root> --coco-root <coco-root> \
  --evidence-dir <evidence-dir>
```

## DEIM-RT-DETRv2

| 证据 | 状态 | 结论 |
|---|---|---|
| 依赖切片验证 | APPROVE | 受限 RT-DETRv2 encoder/decoder、五 profile 和上游组件数值 |
| 模型矩阵验证 | APPROVE | 五 detector/pretrained、parity、COCO、训练恢复、推理和部署 |

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

原始机器收据保存在本地执行证据目录；正式文档只保留可复现入口和稳定结论，不依赖临时导出或日志。

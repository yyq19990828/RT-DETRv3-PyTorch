# DEIM-D-FINE 证据索引

| 证据 | 状态 | 结论 |
|---|---|---|
| Task 14 | APPROVE | MAL、GO union、criterion loss/gradient 和负例 |
| Task 15 | APPROVE | 两阶段 checkpoint、EMA、resume、DDP 与事务回滚 |
| Task 16 | APPROVE | 五变体 checkpoint、parity、COCO、训练恢复、推理和部署 |
| Task 21-23 | APPROVE | Models CLI、打包、许可和文档 |
| F2/F3/F4/F1 | APPROVE | 最终质量、wheel N 用户链、v3 无回归和计划合规 |

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

原始 receipt 为 `task-14` 至 `task-16-rtdetrv4-merge.json` 以及最终 F1-F4 收据；正式报告不依赖生成日志或临时导出二进制。

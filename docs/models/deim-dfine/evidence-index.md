# DEIM-D-FINE 证据索引

| 证据 | 状态 | 结论 |
|---|---|---|
| Criterion 验证 | APPROVE | MAL、GO union、loss/gradient 和负例 |
| 训练协议验证 | APPROVE | 两阶段 checkpoint、EMA、resume、DDP 与事务回滚 |
| 模型矩阵验证 | APPROVE | 五变体 checkpoint、parity、COCO、训练恢复、推理和部署 |
| 用户接口与发布检查 | APPROVE | Models CLI、打包、许可和文档 |
| 最终审计 | APPROVE | 质量、安装包 N 用户链、v3 无回归和计划合规 |

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

原始机器收据保存在本地执行证据目录；正式报告不依赖生成日志或临时导出二进制。

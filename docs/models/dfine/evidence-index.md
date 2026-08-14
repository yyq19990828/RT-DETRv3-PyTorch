# D-FINE 证据索引

| 证据 | 状态 | 结论 |
|---|---|---|
| HGNetv2 验证 | APPROVE | B0/B2/B4/B5 stage-1 state 与逐 stage activation |
| 组件与配置验证 | APPROVE | Primitives、matcher/DN、数据策略、scheduler、checkpoint、architecture 和 criterion |
| 官方权重验证 | APPROVE | 五个 checkpoint、state、固定输入 activation/raw output |
| 训练与精度验证 | APPROVE | 五变体 train-resume、四图 eager/parity、完整 COCO |
| 部署验证 | APPROVE | 五变体 deploy eager、ONNX、TorchScript 和导出负例 |
| 用户接口与发布检查 | APPROVE | Models CLI、打包、许可和文档 |
| 最终审计 | APPROVE | 质量、安装包 N 用户链、v3 无回归和计划合规 |

原始机器收据位于本地执行证据目录；本页只保留按能力域组织的结论，不依赖临时日志和渲染图。

## 可执行入口

- Manifest：[`configs/checkpoints/dfine_coco.yml`](../../../configs/checkpoints/dfine_coco.yml)
- 配置：[`configs/dfine`](../../../configs/dfine/)
- Checkpoint parity：[`tests/numerical/test_dfine_official_checkpoints.py`](../../../tests/numerical/test_dfine_official_checkpoints.py)
- Runtime：[`tests/integration/test_dfine_runtime.py`](../../../tests/integration/test_dfine_runtime.py)
- Export：[`tests/unit/deploy/test_dfine_export.py`](../../../tests/unit/deploy/test_dfine_export.py)

```bash
uv run python tools/dev/validate_model_family.py \
  --family dfine --variants all --phase all \
  --checkpoint-root <checkpoint-root> --coco-root <coco-root> \
  --evidence-dir <evidence-dir>
```

上游数值测试另需把 `DFINE_UPSTREAM_ROOT` 指向固定 revision checkout。

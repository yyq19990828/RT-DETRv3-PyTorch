# RT-DETRv4 证据索引

> 原始机器日志已提炼为本目录报告并在验收后清理；正式文档不依赖临时产物。

| 证据 | 状态 | 结论 |
|---|---|---|
| Teacher 验证 | APPROVE | DINOv3 身份、授权权重、feature boundary、frozen/detached teacher |
| 训练协议验证 | APPROVE | DSI、GAM、AMP/accumulation/DDP/resume |
| 模型矩阵验证 | APPROVE | 四变体 checkpoint、parity、COCO、训练、推理和部署 |
| 用户接口与发布检查 | APPROVE | Models CLI、打包、许可、文档与 19 变体支持矩阵 |
| 最终质量审计 | APPROVE | 质量、覆盖率、unit/integration、上游 numerical 和图审计 |
| 安装包用户验收 | APPROVE | 独立 Python 3.11 CPU wheel 的 S 变体全用户链与 teacher 负例 |
| 最终兼容性审计 | APPROVE | RT-DETRv3 范围和兼容基线未回归 |
| 最终计划合规审计 | APPROVE | 全部实现与最终验证记录的身份和状态完整 |

验收后临时 venv、wheel 和 `dist/` 已清理；不把已删除 wheel 的 hash 当作当前发布资产。

## 可执行入口

- Manifest：[`configs/checkpoints/rtdetrv4_coco.yml`](../../../configs/checkpoints/rtdetrv4_coco.yml)
- 配置：[`configs/rtdetrv4`](../../../configs/rtdetrv4/)
- Checkpoint parity：[`tests/numerical/test_rtdetrv4_official_checkpoints.py`](../../../tests/numerical/test_rtdetrv4_official_checkpoints.py)
- DSI/GAM：[`tests/unit/modeling/test_rtdetrv4_loss.py`](../../../tests/unit/modeling/test_rtdetrv4_loss.py)、[`tests/unit/engine/test_rtdetrv4_gam.py`](../../../tests/unit/engine/test_rtdetrv4_gam.py)
- Resume/DDP：[`tests/integration/test_rtdetrv4_gam_resume.py`](../../../tests/integration/test_rtdetrv4_gam_resume.py)、[`tests/integration/test_rtdetrv4_gam_ddp.py`](../../../tests/integration/test_rtdetrv4_gam_ddp.py)

```bash
uv run python tools/dev/validate_model_family.py \
  --family rtdetrv4 --variants all --phase all \
  --checkpoint-root <checkpoint-root> --coco-root <coco-root> \
  --dinov3-repo <dinov3-checkout> \
  --dinov3-weights <dinov3-weights.pth> \
  --dinov3-sha256 73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c \
  --evidence-dir <evidence-dir>
```

同时设置 `RTDETRV4_UPSTREAM_ROOT` 为固定 revision checkout；DINOv3 资产须由使用者自行授权取得。

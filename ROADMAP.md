# RT-DETRv3 PyTorch Migration Roadmap

**Status**: Active

**Last updated**: 2026-08-14

**Current execution plan**: [D-FINE、DEIM 与 RT-DETRv4 集成](docs/plans/2026-08-12-dfine-deim-rtdetrv4-integration.md)（技术验收完成，等待维护者接受）

**Deferred plan**: [M4——COCO 精度与稳定性对齐](docs/plans/2026-07-18-m4-coco-accuracy-stability.md)

本路线图只展开仓库级未完成、延期或等待决策的迁移工作。已完成里程碑的执行细节、环境和数值结果保存在带日期的计划与报告中，不在这里重复维护。

## 目标

建立一个可安装、可训练、可评估、可恢复、可转换官方 Paddle 权重，并能在相同 checkpoint、预处理、输入、模式、dtype 和容差下提供数值、精度与部署证据的 PyTorch DETR 训练库。

## 当前未完成工作

### M4——RT-DETRv3 COCO 精度与稳定性（deferred）

同一官方 R18 checkpoint 的 Paddle/PyTorch CPU/FP32 完整 val2017 gate 已通过：bbox AP 分别为 `0.480477300367` 与 `0.480477134768`，绝对差 `1.65599e-7`。该结果验证同权重评估，不代表本仓库标准 schedule 的训练收敛。

- [ ] 完成 R18 seed 0 的 72 epoch 标准训练，记录环境、命令、配置、日志、checkpoint checksum 和 EMA COCO 指标。
- [ ] 对 seed 1、2 重复同一协议，报告均值、标准差和离群情况；发布级验收再扩展至 5 个 seed。
- [ ] R18 达到预注册门槛后，再依次验证 R34 和 R50，不在未定位差异时并行展开多变体长训。
- [ ] 在同设备、相同 batch 和同步边界下补齐 Paddle/PyTorch 训练吞吐、显存与端到端性能比较。

恢复该计划前需要维护者明确训练时间和算力预算。社区可用 [`scripts/run_stability_experiment.py`](scripts/run_stability_experiment.py) 按 `model + seed` 分片执行，但只有 commit、输入、配置和训练协议一致的结果才能合并。

### 新模型集成接受决策

D-FINE、DEIM-D-FINE、DEIM-RT-DETRv2 与 RT-DETRv4 共 19 个变体已经通过官方 checkpoint、pinned activation/raw-output、reduced train/resume、四图 eager、ONNX/TorchScript、完整 COCO val2017、打包与最终审计。

- [ ] 由维护者明确接受最终结果，再把[集成计划](docs/plans/2026-08-12-dfine-deim-rtdetrv4-integration.md)从 `in-progress` 改为 `completed` 并移入归档。

技术验收不自动表示官方权重由本项目发布，也不证明四个新模型族的完整 schedule、多 seed、低精度或性能收敛。

## 已完成里程碑索引

| Milestone | 结论 | 执行与证据 |
|---|---|---|
| M1 最小训练链 | config → data → model → loss → backward → optimizer 可重复 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-18-m1-minimal-training-chain.md) |
| M2 checkpoint 转换与数值对齐 | R18/R34/R50 完成目标感知转换、加载和分层对齐 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-18-m2-official-checkpoint-alignment.md) |
| M3 训练、评估与恢复 | 真实 COCO、AMP、EMA、DDP、累积与确定性恢复通过 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-18-m3-training-evaluation-recovery.md) |
| M5 CLI 与导出边界 | 六个入口、配置覆盖和 tensor-only 导出合同固定 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-19-m5-cli-export-boundaries.md) |
| M6 性能、质量与发布 | 质量、覆盖率、打包和 `v0.1.0` 发布验收完成 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-19-m6-performance-quality-release.md) |
| M7 多变体运行时 | R18/R34/R50 公开 eager CPU 链路通过 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-19-m7-variant-runtime-validation.md) |
| M8 多变体导出 | R34/R50 ONNX 与 TorchScript CPU 验收完成 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-19-m8-variant-export-validation.md) |
| M9 导出后推理 | 导出产物复用 Infer CLI 的端到端链路通过 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-19-m9-exported-inference.md) |
| M10 TorchScript 设备 | R18 CUDA 默认与 CPU fallback 合同通过 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-19-m10-torchscript-cuda-inference.md) |
| M11 ONNX Runtime 设备 | R18 CUDA/CPU provider 合同通过 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-19-m11-onnx-runtime-cuda-inference.md) |
| M12 导出设备矩阵 | R34/R50 CUDA/CPU 功能矩阵与容差边界完成 | [计划](docs/archive/rtdetrv3-v0.1.0/plans/2026-07-19-m12-variant-export-device-matrix.md) |

RT-DETRv3 `v0.1.0` 的统一证据入口见[版本归档](docs/archive/rtdetrv3-v0.1.0/README.md)，当前模型合同见 [`docs/models/rtdetrv3`](docs/models/rtdetrv3/README.md)。

## 依赖顺序

1. M4 先完成 R18 单 seed，再扩展多 seed，最后开展 R34/R50。
2. 任意精度结论先固定 checkpoint、预处理、输入、eval/train 模式、dtype、设备、seed 和容差。
3. 出现最终预测差异时，先比较第一个分歧中间激活，不用形状、成功加载或确定性输出代替数值证据。
4. 新模型集成计划只有在维护者接受后才归档；权重发布另行制定发布计划。

## 非当前阻塞项

- 四个新模型族的完整 schedule、多 seed、低精度、TensorRT/C++ 和性能基准。
- 动态空间尺寸导出；当前 ONNX/TorchScript 合同为固定高宽、动态 batch。
- 上游模型权重的重新托管或纳入本项目 Release。
- Objects365 完整预训练和 DINOv3 授权资产再分发。

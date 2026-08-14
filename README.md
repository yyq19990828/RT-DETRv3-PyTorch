# DETR-series

DETR-series 是 DETR 系列实时目标检测模型的 PyTorch 实现合集。六个模型族共 **30 个 COCO 变体**运行在同一个训练、评估、推理、checkpoint 与部署运行时上:写一份配置,即可用同一组 CLI 训练、评估、导出任意族。

仓库起源于 RT-DETRv3 的 Paddle-to-PyTorch 迁移,因此当前 Python 包和六个公开命令仍保留 `ppdet_pytorch` 与 `rtdetrv3-*` 名称以维持兼容性。Paddle 官方实现仅作为只读参考子模块保留;核心 PyTorch 运行时不导入 Paddle。

**核心特性**

- **统一运行时**:六个族共享数据管道、两阶段训练协议、EMA、断点恢复、`bbox`/`bbox_num` 推理合同与 ONNX/TorchScript 导出边界。
- **证据驱动验收**:每个官方 checkpoint 记录上游 revision、SHA-256、key 映射与数值对齐证据;完整 val2017 结果与官方公布值的误差以预注册门槛约束。
- **部署友好**:推理不额外执行 NMS,ONNX(opset 17、固定高宽、动态 batch)与 TorchScript 逐值验证;DEIMv2 覆盖从 50.3M 到 0.5M 的全部尺寸档,支持 GPU、边缘与移动端部署场景。

当前支持范围见[模型文档](docs/models/README.md);RT-DETRv3 标准 schedule、多 seed 与 R34/R50 长训仍处于延期状态,见[路线图](ROADMAP.md)。

## 模型总览

COCO val2017 bbox AP 为官方公布值;本仓库实测与验收证据见各模型文档。

| 模型族 | 论文 | 变体 | 官方 COCO AP | 模型文档 |
|---|---|---|---|---|
| RT-DETRv3 | RT-DETRv3(2024) | R18/R34/R50 | 48.1(R18) | [rtdetrv3](docs/models/rtdetrv3/README.md) |
| D-FINE | D-FINE(2024) | N/S/M/L/X | 42.8 – 55.8 | [dfine](docs/models/dfine/README.md) |
| DEIM-D-FINE | DEIM(CVPR 2025) | N/S/M/L/X | 43.0 – 56.5 | [deim](docs/models/deim/README.md) |
| DEIM-RT-DETRv2 | DEIM(CVPR 2025) | S/M/M*/L/X | 49.0 – 55.5 | [deim](docs/models/deim/README.md) |
| RT-DETRv4 | RT-DETRv4(2025) | S/M/L/X | 49.8 – 57.0 | [rtdetrv4](docs/models/rtdetrv4/README.md) |
| DEIMv2 | DEIMv2(2025) | X/L/M/S;N/Pico/Femto/Atto | 23.8 – 57.8 | [deimv2](docs/models/deimv2/README.md) |

各族的运行时 family 名、配置目录与 checkpoint manifest 如下;除 RT-DETRv3 外的 27 个 COCO 变体已完成官方 checkpoint、完整 val2017、reduced 训练恢复和部署验收,权重由上游托管,不属于仓库初始 `v0.1.0` Release。

| Runtime family | 变体 | 配置 | checkpoint manifest |
|---|---|---|---|
| `rtdetrv3` | R18/R34/R50 | [`configs/rtdetrv3`](configs/rtdetrv3/) | [`rtdetrv3_coco.yml`](configs/checkpoints/rtdetrv3_coco.yml) |
| `dfine` | N/S/M/L/X | [`configs/dfine`](configs/dfine/) | [`dfine_coco.yml`](configs/checkpoints/dfine_coco.yml) |
| `deim-dfine` | N/S/M/L/X | [`configs/deim/dfine`](configs/deim/dfine/) | [`deim_dfine_coco.yml`](configs/checkpoints/deim_dfine_coco.yml) |
| `deim-rtdetrv2` | S/M/M*/L/X | [`configs/deim/rtdetrv2`](configs/deim/rtdetrv2/) | [`deim_rtdetrv2_coco.yml`](configs/checkpoints/deim_rtdetrv2_coco.yml) |
| `rtdetrv4` | S/M/L/X | [`configs/rtdetrv4`](configs/rtdetrv4/) | [`rtdetrv4_coco.yml`](configs/checkpoints/rtdetrv4_coco.yml) |
| `deimv2` | X/L/M/S;N/Pico/Femto/Atto | [`configs/deimv2`](configs/deimv2/) | [`deimv2_coco.yml`](configs/checkpoints/deimv2_coco.yml) |

## 快速开始

项目支持 Python 3.9–3.12,使用 uv 0.11.29 至 0.12.x 管理环境。默认锁文件面向 Linux x86_64 或 Windows amd64 的 PyTorch CUDA 12.1;其他平台需要选择匹配的 PyTorch 索引。

```bash
git clone --recurse-submodules https://github.com/yyq19990828/DETR-series.git
cd DETR-series
uv sync

# 查看模型与权重状态
uv run rtdetrv3-models list

# 使用已准备的 checkpoint 推理
uv run rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --infer-img path/to/image.jpg \
  --output-dir output/infer \
  --save-results
```

完整安装模式、权重获取方式及 Train/Eval/Infer/Convert/Export 示例见[使用指南](docs/guides/README.md)。

## 常用工作流

```bash
# 训练
uv run rtdetrv3-train \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --seed 0

# 评估
uv run rtdetrv3-eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth

# 导出 ONNX 与 TorchScript
uv run --extra export rtdetrv3-export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --format both \
  --output-dir output/export
```

六个 `rtdetrv3-*` 命令继续作为统一入口。模型推理不额外执行 NMS;ONNX/TorchScript 使用动态 batch、固定导出高宽,改变空间尺寸时需要重新导出。详细参数和部署边界见[使用指南](docs/guides/README.md)与 [RT-DETRv3 CLI 合同](docs/models/rtdetrv3/cli-and-export.md)。

## 安装模式

| 目的 | 命令 | Paddle |
|---|---|---|
| 核心训练与推理 | `uv sync` | 不安装 |
| 非 Paddle 测试 | `uv sync --extra test` | 不安装 |
| 开发、Paddle 转换和数值对齐 | `uv sync --extra dev` | 安装 |
| ONNX CPU / CUDA | `uv sync --extra export` / `uv sync --extra export-gpu` | 不安装 |
| Ruff 与 Mypy | `uv sync --extra quality` | 不安装 |
| RT-DETRv4 DINOv3 teacher 训练 | `uv sync --extra teacher` | 不安装 |

Paddle 和迁移专用依赖只属于 `dev` extra。DINOv3 teacher 只在 RT-DETRv4 训练构造;student eval、infer 和 export 不需要 teacher checkout、授权权重或 `teacher` extra。各 extra 的平台与冲突边界见[使用指南](docs/guides/README.md)。

## 测试与质量

```bash
# 不依赖 Paddle 的测试
uv run --extra test pytest -m "not paddle"

# 包含 Paddle 对齐测试
uv run --extra dev pytest

# Ruff format、Ruff lint 和 Mypy
uv run --extra quality python scripts/check_quality.py
```

测试矩阵、覆盖率、文档门禁和发布检查见[开发者指南](docs/development/README.md)。

## 文档导航

- [使用指南](docs/guides/README.md):安装、模型资产、训练、评估、推理、转换与导出。
- [模型文档](docs/models/README.md):当前支持合同、逐变体指标、验证报告与证据索引。
- [迁移经验](docs/migrations/README.md):框架语义、配置、权重转换、训练验证和排错。
- [开发者指南](docs/development/README.md):测试、质量、发布和文档维护。
- [执行计划](docs/plans/README.md):活动、延期和已完成计划入口。
- [历史归档](docs/archive/README.md):带日期的计划、报告、论文和机器可读证据。

## 引用

以下 BibTeX 逐字取自各上游仓库官方 Citation 部分。若本仓库对你的工作有帮助,请同时引用对应模型论文与本仓库。

**RT-DETRv3**(上游 [clxia12/RT-DETRv3](https://github.com/clxia12/RT-DETRv3))

```bibtex
@article{wang2024rt,
  title={RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Supervision},
  author={Wang, Shuo and Xia, Chunlong and Lv, Feng and Shi, Yifeng},
  journal={arXiv preprint arXiv:2409.08475},
  year={2024}
}
```

**D-FINE**(上游 [Peterande/D-FINE](https://github.com/Peterande/D-FINE))

```bibtex
@misc{peng2024dfine,
      title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
      author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
      year={2024},
      eprint={2410.13842},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**DEIM**(上游 [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM),用于 DEIM-D-FINE 与 DEIM-RT-DETRv2 两个 profile)

```bibtex
@misc{huang2024deim,
      title={DEIM: DETR with Improved Matching for Fast Convergence},
      author={Shihua, Huang and Zhichao, Lu and Xiaodong, Cun and Yongjun, Yu and Xiao, Zhou and Xi, Shen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
}
```

**RT-DETRv2**(上游 [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR),DEIM-RT-DETRv2 profile 的 decoder 来源)

```bibtex
@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140},
}
```

**RT-DETRv4**(上游 [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4))

```bibtex
@article{liao2025rtdetrv4,
  title={RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models},
  author={Zijun Liao and Yian Zhao and Xin Shan and Yu Yan and Chang Liu and Lei Lu and Xiangyang Ji and Jie Chen},
  journal={arXiv preprint arXiv:2510.25257},
  year={2025}
}
```

**DEIMv2**(上游 [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2))

```bibtex
@article{huang2025deimv2,
  title={Real-Time Object Detection Meets DINOv3},
  author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal={arXiv},
  year={2025}
}
```

DEIMv2 骨干依赖的 [DINOv3](https://github.com/facebookresearch/dinov3) 前向代码按其 [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/346f38fee679c56a6888f91c51670fae61d364e0/LICENSE.md) vendored,许可与致谢边界见 [NOTICE](NOTICE)。

## 发布状态

仓库初始 [`v0.1.0`](https://github.com/yyq19990828/DETR-series/releases/tag/v0.1.0) 面向 RT-DETRv3,已发布 wheel、sdist、R18/R34/R50 检测权重、R18-vd backbone 权重、mapping reports 和 `SHA256SUMS`。固定 tag 的 11 个资产已完成公开下载与 checksum 回读;历史环境、命令和限制见[发布验证报告](docs/archive/rtdetrv3-v0.1.0/reports/release-validation.md)。

后续版本发布必须重新构建并生成唯一 checksum;当前工作树或历史构建记录不构成可发布资产。

## 仓库结构

```text
.
├── src/ppdet_pytorch/       # 可安装的 PyTorch 包
├── configs/                 # 模型与 checkpoint 配置
├── tests/                   # 单元、集成和数值测试
├── tools/dev/               # 开发期数值对齐与验证工具
├── docs/                    # 用户、模型、迁移、计划与归档文档
├── ROADMAP.md               # 未完成迁移大纲
└── third-party/
    └── RT-DETRv3-paddle/    # 只读 Paddle 参考子模块
```

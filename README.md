# RT-DETRv3 PyTorch

RT-DETRv3 的 PyTorch 迁移实现。仓库当前仍处于迁移与数值对齐阶段；Paddle 官方实现作为只读参考子模块保留，PyTorch 包使用独立的 `src-layout`。

## 环境安装

项目支持 Python 3.9–3.12，支持用 `uv` 0.11.29 至 0.12.x 创建和管理虚拟环境；CI 固定使用 0.11.29 验证锁文件兼容性。当前
`torch`/`torchvision` 从 PyTorch 官方 CUDA 12.1 索引安装，默认面向 Linux x86_64 或
Windows amd64；CPU、macOS 和 ARM 环境需要改用与平台匹配的 PyTorch 索引。

```bash
git clone --recurse-submodules https://github.com/yyq19990828/RT-DETRv3-PyTorch.git
cd RT-DETRv3-PyTorch

# 仅安装 PyTorch 训练/推理运行时
uv sync

# 开发、测试、Paddle 权重转换和数值对齐
uv sync --extra dev

# 仅增加 ONNX 导出和 CPU 回归依赖
uv sync --extra export

# 仅增加 ONNX 导出和 CUDA/CPU provider 依赖
uv sync --extra export-gpu

# 不安装 Paddle 的核心测试与导出回归依赖
uv sync --extra test

# 仅增加 Ruff/Mypy 质量工具
uv sync --extra quality

# 仅增加训练专用 DINOv3 teacher 依赖
uv sync --extra teacher
```

中国大陆网络可选择从阿里云（默认）或上海交大镜像预载锁定的
PyTorch CUDA 12.1 wheels，再执行正常的 locked sync：

```bash
python3 scripts/sync_china.py --extra test
# 或：python3 scripts/sync_china.py --mirror sjtug --extra dev
```

该脚本仅支持 Linux x86_64，并按当前 Python ABI 从镜像下载
`torch`/`torchvision`。每个 wheel 都必须通过 `uv.lock` 中官方
PyTorch SHA-256 后才会安装；默认 `uv sync`、锁文件和 CI 仍使用官方源。
阿里/上交的 PyTorch 页面是大型 flat wheel 列表，不建议直接设为全局
`index-url`，否则 uv 会长时间扫描或因不完整的通用包集合改变解析行为。

如果仓库已克隆但缺少 Paddle 参考实现：

```bash
git submodule update --init --recursive
```

Paddle 及其专用依赖不属于核心运行依赖，只在 `dev` 附加依赖中安装。
Linux x86_64 的 `dev` extra 固定使用 PaddlePaddle GPU 3.3.0/cu118；
该构建可用 `paddle.set_device("cpu")` 显式回退到 CPU。`export`/`test` 使用
CPU `onnxruntime`；`export-gpu`/`dev` 使用同时包含 CUDA 和 CPU provider 的
`onnxruntime-gpu`。两类 ORT distribution 不应共装，UV 会拒绝冲突的 extra 组合。
训练专用 DINOv3 teacher 的官方 Python 依赖位于 `teacher` extra；默认 student
训练、评估、推理和导出均不依赖该 extra。

## 模型族

当前可安装包保留六个 `rtdetrv3-*` 命令，并支持以下 Models family。19 个新增
COCO 变体已完成官方 checkpoint、完整 val2017、训练恢复与部署验收，但其权重
仍由上游托管，不属于本项目 `v0.1.0` Release。

| Models family | 变体 | 配置 | checkpoint manifest |
|---|---|---|---|
| `rtdetrv3` | R18/R34/R50 | [`configs/rtdetrv3`](configs/rtdetrv3/) | [`rtdetrv3_coco.yml`](configs/checkpoints/rtdetrv3_coco.yml) |
| `dfine` | N/S/M/L/X | [`configs/dfine`](configs/dfine/) | [`dfine_coco.yml`](configs/checkpoints/dfine_coco.yml) |
| `deim-dfine` | N/S/M/L/X | [`configs/deim/dfine`](configs/deim/dfine/) | [`deim_dfine_coco.yml`](configs/checkpoints/deim_dfine_coco.yml) |
| `deim-rtdetrv2` | S/M/M*/L/X | [`configs/deim/rtdetrv2`](configs/deim/rtdetrv2/) | [`deim_rtdetrv2_coco.yml`](configs/checkpoints/deim_rtdetrv2_coco.yml) |
| `rtdetrv4` | S/M/L/X | [`configs/rtdetrv4`](configs/rtdetrv4/) | [`rtdetrv4_coco.yml`](configs/checkpoints/rtdetrv4_coco.yml) |

```bash
uv run rtdetrv3-models --family dfine list
uv run rtdetrv3-models --family deim-dfine list --json
uv run rtdetrv3-models --family rtdetrv4 verify rtdetrv4-s /path/to/RTv4-S-hgnet.pth
```

`--manifest` 的优先级高于 `--family`。D-FINE 的固定 GitHub release asset 可由
CLI 原子下载；Google Drive 托管的 DEIM/RT-DETRv4 资产只支持 list 和本地
verify，download 会返回 manifest 中的官方来源 URL。RT-DETRv4 的 DINOv3
ViT-B/16 teacher 只在训练构造；student eval/infer/export 不需要 DINOv3 checkout、
授权权重或 `teacher` extra。模型级数值与限制见[模型文档](docs/models/README.md)。

## 常用命令

```bash
# 训练
uv run rtdetrv3-train \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --seed 0

# 评估
uv run rtdetrv3-eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth

# 评估训练 checkpoint 中的 EMA，并保留 COCO prediction JSON
uv run rtdetrv3-eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth --use-ema \
  --output-dir output/eval

# 推理
uv run rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --infer-img path/to/image.jpg \
  --output-dir output/infer \
  --save-results

# 使用导出的 ONNX CPU（需要 export、export-gpu 或 dev 附加依赖）
uv run --extra export rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --onnx-model output/export/rtdetrv3_r18vd_6x_coco.onnx \
  --infer-img path/to/image.jpg \
  --imgsz 640 \
  --device cpu \
  --output-dir output/infer-onnx

# 使用导出的 ONNX CUDA（需要 export-gpu 或 dev 附加依赖）
uv run --extra export-gpu rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --onnx-model output/export/rtdetrv3_r18vd_6x_coco.onnx \
  --infer-img path/to/image.jpg \
  --imgsz 640 \
  --device cuda:0 \
  --output-dir output/infer-onnx-cuda

# 使用导出的 TorchScript
uv run rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --torchscript-model output/export/rtdetrv3_r18vd_6x_coco.torchscript.pt \
  --infer-img path/to/image.jpg \
  --imgsz 640 \
  --output-dir output/infer-torchscript

# Paddle 权重转换（需要 dev 附加依赖）
uv run --extra dev rtdetrv3-convert \
  --input path/to/model.pdparams \
  --output path/to/model.pth

# 查看权重发布状态，或校验已有转换权重
uv run rtdetrv3-models list
uv run rtdetrv3-models verify r18 \
  pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth
uv run rtdetrv3-models verify r18-backbone \
  pretrained_models/pytorch/ResNet18_vd_pretrained.pth

# manifest 记录固定 HTTPS URL 后可原子下载并自动校验
uv run rtdetrv3-models download r18
uv run rtdetrv3-models download r18-backbone

# 导出 ONNX 和 TorchScript（需要 export 或 dev 附加依赖）
uv run --extra export rtdetrv3-export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --format both \
  --output-dir output/export
```

推理入口互斥接受 `--checkpoint`、`--onnx-model` 或 `--torchscript-model`，三者复用配置中的 `TestReader`、RT-DETR 后处理结果、阈值、JSON 和可视化，不额外执行 NMS。`--infer-dir` 支持非递归目录推理，`--batch-size` 控制实际 batch；只有训练 checkpoint 可加 `--use-ema`。ONNX 默认 CPU，也可在 GPU ORT 环境中显式选择 `--device cuda[:id]`；TorchScript 与 checkpoint 在 CUDA 可用时默认使用 CUDA，也可显式 `--device cpu`，无 CUDA 时自动回退 CPU。`--imgsz` 必须与导出时的固定高宽一致。当前参数与 Paddle Infer 的差异见 [CLI 与导出边界](docs/models/rtdetrv3/cli-and-export.md)。R18 设备合同见归档的 [TorchScript](docs/archive/rtdetrv3-v0.1.0/reports/torchscript-device-validation.md)和[ONNX Runtime](docs/archive/rtdetrv3-v0.1.0/reports/onnx-runtime-device-validation.md)报告；R34/R50 的 CUDA/CPU 功能矩阵及 ONNX CUDA 严格门槛偏差见[多变体设备报告](docs/archive/rtdetrv3-v0.1.0/reports/variant-export-device-validation.md)。

导出入口使用 tensor-only 适配层，默认生成动态 batch、固定导出高宽的 ONNX opset 17，以及相同固定高宽的 traced TorchScript；空间尺寸改变时需要按新尺寸重新导出。Train/Eval/Infer/Convert/Export 只声明文档中列出的当前合同；未迁移的 Paddle Train 参数会直接报错，不会静默忽略。完整参数和部署边界同样见 [CLI 与导出边界](docs/models/rtdetrv3/cli-and-export.md)。

`tools/train.py`、`tools/eval.py`、`tools/infer.py` 和 `tools/convert_weights.py` 保留为兼容入口。Paddle 对齐和诊断脚本位于 `tools/dev/`。

## 测试

```bash
# 不依赖 Paddle 的测试
uv run --extra test pytest -m "not paddle"

# 包含 Paddle 对齐测试
uv run --extra dev pytest

# 非 Paddle 全包覆盖率与直接维护范围门禁
uv run --extra test python scripts/check_coverage.py
```

当前覆盖率范围、排除规则和逐模块结果见归档的 [M6 覆盖率验证报告](docs/archive/rtdetrv3-v0.1.0/reports/coverage-validation.md)。

## 代码质量

全部活跃 Python 文件统一使用 Ruff 格式化和基础 lint，Mypy 单独负责类型检查。Ruff 已覆盖仓库根目录并排除只读子模块、历史测试和生成目录；Mypy 已覆盖完整 `src/ppdet_pytorch` 与纳入门禁的仓库脚本。

```bash
# 检查 Ruff format、Ruff lint 和 Mypy
uv run --extra quality python scripts/check_quality.py

# 格式化并应用 Ruff 安全修复，再运行 Mypy
uv run --extra quality python scripts/check_quality.py --fix
```

## v0.1.0 发布检查

wheel 包含 26 个受支持 YAML 配置与 Apache-2.0/NOTICE，但不携带模型权重、数据集或 Paddle 子模块。安装 wheel 后，从仓库外仍可使用 `configs/...` 路径访问包内配置。发布候选权重的来源、大小、SHA-256 和 mapping 数记录在 [`configs/checkpoints/rtdetrv3_coco.yml`](configs/checkpoints/rtdetrv3_coco.yml)。

[`v0.1.0`](https://github.com/yyq19990828/RT-DETRv3-PyTorch/releases/tag/v0.1.0) 已通过 GitHub Releases 对外发布，包含 wheel、sdist、四个 PyTorch 权重、四份 mapping report 和 `SHA256SUMS`，共 11 个 asset。固定 tag 的全部资产已通过匿名公开下载、严格 checksum 回读和系统 `sha256sum` 复核；具体环境、命令和限制见归档的 [release 验证报告](docs/archive/rtdetrv3-v0.1.0/reports/release-validation.md)。R18/R34/R50 的 Paddle 原权重与 PyTorch 转换权重 COCO 统一渲染见归档的[预测可视化报告](docs/archive/rtdetrv3-v0.1.0/reports/prediction-visualization.md)。

`v0.1.0` 的 manifest 已将 R18/R34/R50 检测权重和 `r18-backbone` 训练初始化权重标记为 `published`，下载地址固定到 `v0.1.0` tag 和对应 asset 文件名，不使用 `latest`。`verify` 可校验本地权重的大小和 SHA-256；三个公开检测 asset 均已通过 Models CLI 下载校验，并使用各自配置完成 CPU 单图 Infer 和统一四图 COCO Eval 链路冒烟，详见归档的[多变体运行时报告](docs/archive/rtdetrv3-v0.1.0/reports/variant-runtime-validation.md)。

```bash
# 构建 wheel 与 sdist
uv build

# 在不存在的目标目录中原子组装并严格验证 11 个 Release assets
release_workspace="$(mktemp -d)"
trap 'find "$release_workspace" -depth -delete' EXIT
uv run python scripts/check_release.py \
  --require-models \
  --wheel dist/rtdetrv3_pytorch-0.1.0-py3-none-any.whl \
  --sdist dist/rtdetrv3_pytorch-0.1.0.tar.gz \
  --stage-release-dir "$release_workspace/v0.1.0"
```

`--stage-release-dir` 会先校验本地 Paddle 源权重、转换权重、mapping report、wheel 和 sdist，再在同一父目录的隐藏临时目录中复制资产、生成 `SHA256SUMS` 并执行严格回读。全部通过后才原子更名为目标目录；目标已存在或任意校验失败都不会留下半成品。checksum 覆盖四个 `.pth`、四份 `.mapping.json`、wheel 和 sdist；再加上 `SHA256SUMS`，GitHub Release 共有 11 个 asset。Paddle 源权重仍由上游托管，不作为本项目的发布资产。

发布后先把固定 tag 的全部 asset 下载到一个空目录，再执行严格回读。校验器要求目录恰好包含 11 个普通文件，拒绝缺失/额外资产、路径型 checksum 名称、重复条目和摘要不匹配，并将四个权重与四份 mapping report 再对照 manifest：

```bash
release_dir="$(mktemp -d)"
trap 'find "$release_dir" -depth -delete' EXIT
gh release download v0.1.0 --dir "$release_dir"
uv run python scripts/check_release.py --verify-release-dir "$release_dir"
```

## 仓库结构

```text
.
├── src/ppdet_pytorch/       # 可安装的 PyTorch 包
│   ├── cli/                 # 训练、评估、推理、转换和导出入口
│   ├── conversion/          # 权重转换逻辑
│   ├── core/
│   ├── data/
│   ├── deploy/              # ONNX/TorchScript tensor 适配与回归
│   ├── engine/
│   ├── metrics/
│   ├── modeling/
│   ├── optimizer/
│   └── utils/
├── configs/                 # PyTorch 配置
├── tests/                   # 单元、集成和数值对齐测试
├── tools/dev/               # 仅开发期使用的 Paddle 对齐工具
├── docs/
│   ├── migrations/        # 跨模型复用的迁移经验
│   ├── models/            # 按模型族组织的当前合同与支持状态
│   ├── plans/             # 活动计划与计划模板
│   └── archive/           # 已完成版本的计划、报告与证据
├── ROADMAP.md              # 未完成迁移大纲
└── third-party/
    └── RT-DETRv3-paddle/    # 官方 Paddle Git 子模块
```

当前 Paddle 子模块固定在官方仓库提交 `349e7d99a5065e7b684118912e6a74178d4f4625`，与本仓库此前内置的 Paddle 源码快照内容一致。

共享迁移经验见 [`docs/migrations`](docs/migrations/README.md)，各模型族合同与当前支持状态见
[`docs/models`](docs/models/README.md)，活动计划见
[`docs/plans`](docs/plans/README.md)，已完成证据见
[`docs/archive`](docs/archive/README.md)；未完成工作以 [`ROADMAP.md`](ROADMAP.md) 为准。

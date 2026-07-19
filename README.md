# RT-DETRv3 PyTorch

RT-DETRv3 的 PyTorch 迁移实现。仓库当前仍处于迁移与数值对齐阶段；Paddle 官方实现作为只读参考子模块保留，PyTorch 包使用独立的 `src-layout`。

## 环境安装

项目支持 Python 3.9–3.12，推荐用 `uv` 0.11.29.x 创建和管理虚拟环境；该版本范围与锁文件和 CI 一致。当前
`torch`/`torchvision` 从 CUDA 12.1 索引安装，默认面向 Linux x86_64 或
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

# 不安装 Paddle 的核心测试与导出回归依赖
uv sync --extra test

# 仅增加 Ruff/Mypy 质量工具
uv sync --extra quality
```

如果仓库已克隆但缺少 Paddle 参考实现：

```bash
git submodule update --init --recursive
```

Paddle 及其专用依赖不属于核心运行依赖，只在 `dev` 附加依赖中安装。
Linux x86_64 的 `dev` extra 固定使用 PaddlePaddle GPU 3.3.0/cu118；
该构建可用 `paddle.set_device("cpu")` 显式回退到 CPU。

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

# Paddle 权重转换（需要 dev 附加依赖）
uv run --extra dev rtdetrv3-convert \
  --input path/to/model.pdparams \
  --output path/to/model.pth

# 查看权重发布状态，或校验已有转换权重
uv run rtdetrv3-models list
uv run rtdetrv3-models verify r18 \
  pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth

# manifest 记录固定 HTTPS URL 后可原子下载并自动校验
uv run rtdetrv3-models download r18

# 导出 ONNX 和 TorchScript（需要 export 或 dev 附加依赖）
uv run --extra export rtdetrv3-export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --format both \
  --output-dir output/export
```

推理入口复用配置中的 `TestReader` 和模型内置 RT-DETR 后处理，不额外执行 NMS。`--infer-dir` 支持非递归目录推理，`--batch-size` 控制实际 batch；训练 checkpoint 可加 `--use-ema`。当前参数与 Paddle Infer 的差异见 [CLI 与导出迁移经验](docs/migrations/cli-and-export.md)。

导出入口使用 tensor-only 适配层，默认生成动态 batch、固定导出高宽的 ONNX opset 17，以及相同固定高宽的 traced TorchScript；空间尺寸改变时需要按新尺寸重新导出。Train/Eval/Infer/Convert/Export 只声明文档中列出的当前合同；未迁移的 Paddle Train 参数会直接报错，不会静默忽略。完整参数和部署边界同样见 [CLI 与导出迁移经验](docs/migrations/cli-and-export.md)。

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

当前覆盖率范围、排除规则和逐模块结果见 [M6 覆盖率验证报告](docs/reports/coverage-validation.md)。

## 代码质量

全部活跃 Python 文件统一使用 Ruff 格式化和基础 lint，Mypy 单独负责类型检查。Ruff 已覆盖仓库根目录并排除只读子模块、历史测试和生成目录；Mypy 已覆盖完整 `src/ppdet_pytorch` 与纳入门禁的仓库脚本。

```bash
# 检查 Ruff format、Ruff lint 和 Mypy
uv run --extra quality python scripts/check_quality.py

# 格式化并应用 Ruff 安全修复，再运行 Mypy
uv run --extra quality python scripts/check_quality.py --fix
```

## 发布候选检查

wheel 包含 26 个受支持 YAML 配置与 Apache-2.0/NOTICE，但不携带模型权重、数据集或 Paddle 子模块。安装 wheel 后，从仓库外仍可使用 `configs/...` 路径访问包内配置。发布候选权重的来源、大小、SHA-256 和 mapping 数记录在 [`configs/checkpoints/rtdetrv3_coco.yml`](configs/checkpoints/rtdetrv3_coco.yml)。

当前发布候选已通过本地和托管 CI 验证，但权重还没有对外发布。计划以 GitHub Releases 作版本绑定的主下载源，可选同步 Hugging Face Model Hub；具体限制、checksum 规则和剩余发布步骤见[release 验证报告](docs/reports/release-validation.md)。R18/R34/R50 的 Paddle 原权重与 PyTorch 转换权重 COCO 统一渲染见[预测可视化报告](docs/reports/prediction-visualization.md)。

`rtdetrv3-models list` 会明确显示当前 R18/R34/R50 为 `unpublished`；未发布时 `download` 会显式失败，不猜测 URL。`verify` 可立即用于校验本地权重的大小和 SHA-256。发布后 manifest 必须同时改为 `published` 并写入固定 tag 的 HTTPS URL，下载命令才会启用。

```bash
# 构建 wheel 与 sdist
uv build

# 检查许可元数据、包内配置和发布归档内容
uv run python scripts/check_release.py \
  --wheel dist/rtdetrv3_pytorch-0.1.0-py3-none-any.whl \
  --sdist dist/rtdetrv3_pytorch-0.1.0.tar.gz

# 发布权重前还必须实际校验本地源权重、转换权重和 mapping report
uv run python scripts/check_release.py --require-models
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
│   ├── plans/             # 实施计划与阶段任务
│   ├── migrations/        # 框架对比、迁移局限与排错经验
│   ├── reports/           # 历史技术报告
│   └── papers/            # 论文与参考资料
├── ROADMAP.md              # 未完成迁移大纲
└── third-party/
    └── RT-DETRv3-paddle/    # 官方 Paddle Git 子模块
```

当前 Paddle 子模块固定在官方仓库提交 `349e7d99a5065e7b684118912e6a74178d4f4625`，与本仓库此前内置的 Paddle 源码快照内容一致。

文档约定见 [`docs/plans`](docs/plans/README.md) 和
[`docs/migrations`](docs/migrations/README.md)；未完成工作以
[`ROADMAP.md`](ROADMAP.md) 为准。

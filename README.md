# RT-DETRv3 PyTorch

RT-DETRv3 的 PyTorch 迁移实现。仓库当前仍处于迁移与数值对齐阶段；Paddle 官方实现作为只读参考子模块保留，PyTorch 包使用独立的 `src-layout`。

## 环境安装

项目支持 Python 3.9–3.12，推荐用 `uv` 创建和管理虚拟环境。当前
`torch`/`torchvision` 从 CUDA 12.1 索引安装，默认面向 Linux x86_64 或
Windows amd64；CPU、macOS 和 ARM 环境需要改用与平台匹配的 PyTorch 索引。

```bash
git clone --recurse-submodules https://github.com/yyq19990828/RT-DETRv3-PyTorch.git
cd RT-DETRv3-PyTorch

# 仅安装 PyTorch 训练/推理运行时
uv sync

# 开发、测试、Paddle 权重转换和数值对齐
uv sync --extra dev
```

如果仓库已克隆但缺少 Paddle 参考实现：

```bash
git submodule update --init --recursive
```

Paddle 及其专用依赖不属于核心运行依赖，只在 `dev` 附加依赖中安装。

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
```

推理入口复用配置中的 `TestReader` 和模型内置 RT-DETR 后处理，不额外执行 NMS。`--infer-dir` 支持非递归目录推理，`--batch-size` 控制实际 batch；训练 checkpoint 可加 `--use-ema`。当前参数与 Paddle Infer 的差异见 [CLI 与导出迁移经验](docs/migrations/cli-and-export.md)。

`tools/train.py`、`tools/eval.py`、`tools/infer.py` 和 `tools/convert_weights.py` 保留为兼容入口。Paddle 对齐和诊断脚本位于 `tools/dev/`。

## 测试

```bash
# 不依赖 Paddle 的测试
uv run --extra dev pytest -m "not paddle"

# 包含 Paddle 对齐测试
uv run --extra dev pytest
```

## 仓库结构

```text
.
├── src/ppdet_pytorch/       # 可安装的 PyTorch 包
│   ├── cli/                 # 训练、评估、推理和转换入口
│   ├── conversion/          # 权重转换逻辑
│   ├── core/
│   ├── data/
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

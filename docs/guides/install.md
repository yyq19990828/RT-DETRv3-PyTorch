# 安装

项目支持 Python 3.9–3.12 和 uv 0.11.29 至 0.12.x。默认锁文件使用 PyTorch CUDA 12.1 官方索引，面向 Linux x86_64 或 Windows amd64；CPU、macOS 和 ARM 环境需要改用平台匹配的 PyTorch 索引。

```bash
# 核心 PyTorch 训练与推理
uv sync

# 不依赖 Paddle 的测试
uv sync --extra test

# Paddle 权重转换和数值对齐
uv sync --extra dev

# ONNX CPU 或 CUDA provider
uv sync --extra export
uv sync --extra export-gpu

# Ruff/Mypy，或训练专用 DINOv3 teacher
uv sync --extra quality
uv sync --extra teacher
```

`dev` 与 `export`/`test`，以及 `export-gpu` 与 `export`/`test` 不能组合安装。`export` 使用 CPU `onnxruntime`，`export-gpu` 与 `dev` 使用同时包含 CUDA 和 CPU provider 的 `onnxruntime-gpu`。核心运行时不依赖 Paddle。

中国大陆 Linux x86_64 环境可以从阿里云或上海交大镜像预载锁定的 PyTorch wheel；脚本会使用 `uv.lock` 中的官方 SHA-256 校验，再执行 locked sync：

```bash
python3 scripts/sync_china.py --extra test
python3 scripts/sync_china.py --mirror sjtug --extra dev
```

如果缺少只读 Paddle 参考子模块：

```bash
git submodule update --init --recursive
```

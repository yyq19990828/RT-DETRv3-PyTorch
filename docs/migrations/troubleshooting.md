# 排错经验

以下命令默认在仓库根目录执行。排错时先保留完整报错和最小复现输入，再分别检查环境、配置、权重和数值语义。

## 子模块为空或找不到 Paddle 源码

```bash
git submodule update --init --recursive
git submodule status --recursive
```

正常情况下 `third-party/RT-DETRv3-paddle` 应指向根 README 记录的固定提交。不要在子模块内直接保留未提交修改。

## 缺少 Paddle、VisualDL、imgaug 或 gdown

这些包属于开发附加依赖：

```bash
uv sync --extra dev
```

如果只执行 `uv sync`，开发依赖可能不在环境中。确认锁文件与声明一致：

```bash
uv lock --check
```

## Paddle 加载 Path 对象报类型错误

部分 Paddle 版本的 `paddle.load` 不接受 `pathlib.Path`。在框架边界显式转换：

```python
state_dict = paddle.load(str(checkpoint_path))
```

同理，把路径写入 JSON 元数据前也应转为 `str`，避免 `PosixPath is not JSON serializable`。

## 配置文件或数据集路径错误

- 使用 `configs/rtdetrv3/*.yml` 中现存的入口，不要沿用旧的 `configs/pytorch/` 或 `configs/rtdetrv3_r50vd.yml` 路径。
- 从仓库根目录执行命令，确保 YAML 的相对 include 可正确解析。
- `data/coco` 只是默认相对路径；数据在其他位置时需要显式覆盖。

可先做最小配置加载检查：

```bash
uv run --no-sync python -c "from ppdet_pytorch.core.workspace import load_config; print(load_config('configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml').architecture)"
```

## 测试报旧 Registry、builder 或 `targets=` API 错误

先确认测试是否来自 `tests/legacy/`。该目录保留迁移早期的历史用例，不代表当前公开 API，也不参与默认 pytest 收集。需要恢复覆盖时，应根据当前实现重写用例。

```bash
uv run --extra dev pytest
```

## 配置中的组件提示未注册

注册发生在 Python 模块导入时。先确认声明该类的模块已经被入口导入，再检查注册名是否与 YAML 完全一致。不要为解决导入顺序问题重新引入旧的分类 Registry 或元类系统；当前约定见[注册与配置经验](registry-and-configuration.md)。

如果同一进程连续加载多份配置，还要留意 `global_config` 的累积状态。测试中应显式隔离或恢复全局配置，避免前一个用例掩盖缺失注册或配置项。

## 转换权重时出现形状不匹配

- 优先传入目标 PyTorch `state_dict`，让转换器同时校验名称和目标形状。
- 卷积权重通常不需要转置；Paddle `Linear` 常为 `[in_features, out_features]`，PyTorch 则为 `[out_features, in_features]`。
- 严格模式用于正式验收；宽松模式只适合定位少量缺失或额外参数，不能作为转换成功证据。
- 不要把优化器或其他训练状态当作模型参数一起映射。

映射表和验证层级见[权重转换经验](weight-conversion.md)。

## Paddle 与 PyTorch 输出差异较大

1. 两侧都切换到 `eval` 模式，关闭随机增强和 dropout。
2. 固定 Python、NumPy、Paddle 和 PyTorch 随机种子。
3. 确认输入的 NCHW/NHWC、RGB/BGR、dtype、归一化和 padding 完全一致。
4. 先在 CPU/float32 上对齐，再引入 CUDA、AMP 或 float16。
5. 从 backbone 开始逐层比较激活，找到第一个超出容差的节点，不要只对比最终预测。
6. 核对 BatchNorm running statistics、卷积权重排布、线性层转置和参数名映射。

## 数值差异只出现在 GPU 或 AMP

不同 CUDA/cuDNN 算法、TF32、AMP 和并行归约顺序都可以放大差异。先建立 CPU/float32 基线，再分别开启 CUDA、TF32 和 AMP，并为每种精度设定独立容差。

如果差异只在续训后出现，还需核对 scheduler 的步进单位和调用顺序，以及 optimizer、EMA、GradScaler、全局步数和随机数状态是否完整恢复。详见[训练与验证经验](training-and-validation.md)。

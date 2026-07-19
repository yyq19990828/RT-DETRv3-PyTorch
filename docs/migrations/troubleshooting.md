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

## UV 为 Paddle nightly 选中错误架构

**已验证（2026-07-19）**：只写 `paddlepaddle>=3.0` 时，UV 0.11.29 曾把 nightly 索引中版本排序更优先、但只有 Linux aarch64 wheel 的 `3.4.0.post20260717` 写进锁文件。`uv lock --check` 只证明声明与锁一致，不证明当前平台一定有可安装 wheel；Linux x86_64 上随后执行 `uv sync --extra dev --locked` 会失败。

当前仓库把 Linux x86_64 声明为 UV 必需解析环境，并固定迁移证据实际使用、同时提供 Python 3.9–3.12 x86_64 wheel 的 `paddlepaddle==3.3.0.dev20251015`。排查类似问题时应同时执行：

```bash
uv lock --check
uv sync --extra dev --locked --dry-run
```

如果要升级 Paddle，先确认目标 Python 和平台的 wheel 实际存在，再有意更新版本、锁文件和数值验证环境；不要只放宽 nightly 版本下限后把“解析成功”当作“可安装”。核心 PyTorch 和托管 CPU 测试仍应使用不含 Paddle 的默认/`test` extra。

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

## 数据加载和随机变换边界

**已验证（2026-07-19）**：类型审计可以揭示 Paddle 继承数据代码中真实的运行时边界，不应仅用宽泛的 `Any` 或全局 ignore 消除报错。当前回归确认了以下规则：

- `ImageFolder` 纯图片推理不需要 annotation，不应在 `do_eval=False` 时构造 `COCO(None)`；需要映射回原 COCO image id 的评估路径则必须显式提供 annotation。
- `get_categories("keypoint")` 的 class-id 映射按 Paddle 合同可为 `None`。仅支持 detection 的 Infer/COCO metric 消费者应在边界处明确拒绝或窄化，不要直接遍历可空映射。
- VOC XML 的 `size`、`name` 和 `bndbox` 字段属于必需输入；缺失时应报出包含字段路径和 XML 文件的 `ValueError`。所有 bbox 均无效的图片应按 empty record 处理，不能因 XML 中存在无效 `object` 而误分类。
- 可重复迭代的 SSOD loader 应在 `__iter__` 内创建并循环两个子 iterator；批量固定尺寸 resize 没有随机选择索引，弱/强增广共享的 selection 应为 `None`，不能返回未初始化局部变量。
- NumPy 的类型声明不接受 callable list 直接传入 `np.random.permutation`。需保留 NumPy RNG 语义时，可打乱整数索引后再回取 callable；不要为类型通过改用不同的随机数源。

定向回归位于 `tests/unit/data/test_dataset_boundaries.py`。这些用例证明上述边界在当前 CPU/锁定依赖环境中可执行，不等同于所有随机数序列、多进程 DataLoader 时序或 Paddle/PyTorch 数值输出已完全对齐。

## 输出目录中的 config.yaml 只有 `{}`

先检查配置对象是否是 `dict`/`Mapping` 子类。某些配置容器同时具有空 `__dict__`；如果序列化代码先判断 `hasattr(obj, '__dict__')`，会丢掉实际 mapping 项并输出空配置。转换顺序应先处理 `Mapping`，再处理普通对象属性，同时把 tuple、`Path`、`torch.device` 和 NumPy scalar 转成 YAML 可表示类型。

不要覆盖已完成训练的空原始快照后再声称它是运行时证据。应保留缺陷事实、记录完整 CLI overrides，并把事后重建文件明确命名为 reconstructed/effective config。修复后用新的真实训练直接检查 YAML 字节数和 checkpoint `config` 项数。

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

## eval 对齐但训练 loss 分歧

先检查只在训练态生成的 attention mask、denoising query 和多分组 query。RT-DETRv3 的 Paddle/PyTorch decoder 都把布尔 mask 的 `True` 解释为“允许注意”，因此多分组复合 mask 必须先初始化为全 `False`，再只把每个组的对角块填为 `True`。如果从全 `True` 开始，组间 query 会错误地相互注意；eval 不生成这张训练 mask，所以完整 eval 对齐也发现不了该问题。

2026-07-18 的 R18 对齐中，正是这个初始化错误导致 30 个训练 loss 分项出现分歧。修正为 block-diagonal mask 后，同一官方 checkpoint 和确定性缩减训练配置下，30 项 loss 全部通过 `rtol=1e-4, atol=1e-5`。对应单元测试直接断言主分组与 O2M 分组的交叉块全为 `False`，避免只靠最终 loss 间接覆盖。

2026-07-19 的类型审计还验证了空标注批次边界：denoising helper 在整批没有 GT 时会返回 `None`，不能继续按 Tensor 列表拼接。当前 PyTorch transformer 会退化为只使用 matching/O2M queries，仍保持分组间 attention block 为 `False`，并返回 `dn_meta=None`。这只证明空 GT 前向不再因缺失 denoising tensor 崩溃，不等同于完整随机 denoising 训练已对齐。

如果差异只出现在 `freeze_norm=True` 的 backbone，还要分别检查“参数是否求导”和“前向使用哪组统计”。Paddle ResNet 的冻结 BN 会使用全局 running statistics；PyTorch 仅设置 `requires_grad=False` 后，外层 `model.train()` 仍会把 BN 切到 batch statistics。R50 曾因此出现 eval 完全对齐、训练 backbone 大幅分歧。当前 ResNet 冻结 BN 会在训练态保持 eval/global-statistics 模式，并有直接回归。

## 前向和 loss 对齐但完整梯度分歧

先在边界 tensor 上使用完全相同的上游梯度，不要直接从最终参数梯度猜根因。R18 排查中，`_get_decoder_input`、Conv2D 和 BatchNorm 分别隔离时都通过；真正的第一个分歧是 BatchNorm 收到 NCHW → NLC 重排反传的非连续 grad-output。其逻辑数值与 contiguous 梯度相同，但 PyTorch input-gradient 明显不同。

当前 `ContiguousGradBatchNorm2d` 用 backward-pre-hook 将 grad-output 规范为 contiguous，不改变参数名、state_dict 或前向结果。单元测试用同一输入和结构化非连续梯度对照标准 contiguous BatchNorm；官方三变体可选用例进一步检查整体梯度相对 L2、余弦和符号分歧率，其中 R18/R34/R50 分别比较 384/462/445 个有梯度参数。

如果梯度方向已经接近但 AdamW 更新仍差异很大，再检查参数级 LR multiplier。Paddle `ParamAttr(learning_rate=...)` 不会自动迁移到 PyTorch `Parameter`；必须由 optimizer param groups 显式表达。不要把这种 10 倍 LR 差异误判为 AdamW 浮点误差。

## 只有少量后处理 label 或 box 不一致

先比较后处理前的 logits/boxes，再看 top-k 边界候选的分数 margin。top-k 和类别索引是离散操作；极小浮点差异可能替换一个边界候选，使按行比较的 label 与 box 同时跳变。此时应明确统计边界候选数量，继续验收全部 score、`bbox_num` 和 label 稳定行的坐标，而不是无限放宽所有坐标的容差。

迁移 Softmax 后处理时不要照搬 `scores.max(-1)` 的返回合同：Paddle 返回最大值 Tensor，而 PyTorch `Tensor.max(dim)` 返回 `(values, indices)`。PyTorch 应直接解包这两个 Tensor；当 query 数小于 `num_top_queries` 时还要为 mask 分支构造完整 query index，并让 `bbox_num` 记录实际 query 数。2026-07-19 的回归已覆盖这一路径；成功 import 或 focal-loss 默认路径不能证明 Softmax 分支正确。

R50 官方权重回归中，decoder box 仅 1/1200 个值在 `rtol=1e-4` 下超差，最大绝对误差 `3.95e-5`；使用记录过的 `rtol=3e-4` 后，中间输出通过。后处理 300 个 label 中仍有 2 个边界差异，但全部 score、298 个稳定候选坐标和 `bbox_num` 通过。该证据属于“观察到的离散边界”，不能写成逐候选完全一致。

## 数值差异只出现在 GPU 或 AMP

不同 CUDA/cuDNN 算法、TF32、AMP 和并行归约顺序都可以放大差异。先建立 CPU/float32 基线，再分别开启 CUDA、TF32 和 AMP，并为每种精度设定独立容差。

如果差异只在续训后出现，还需核对 scheduler 的步进单位和调用顺序，以及 optimizer、EMA、GradScaler、全局步数和随机数状态是否完整恢复。详见[训练与验证经验](training-and-validation.md)。

## CUDA 训练报 CPU/GPU tensor 混用

首先检查不带 `device=` 的 `torch.arange/full/zeros/tensor`，尤其是 anchor 生成、assigner batch index、空 GT 分支和 SciPy matcher 返回的索引。这些路径在 CPU 单测中不会报错，但会在真实 CUDA 前向的第一个索引或 bbox decode 处失败。

修复原则是让创建点跟随语义来源：anchor/stride 跟随 feature，batch index 跟随 label，匹配索引跟随被索引 tensor。修复后应同时覆盖非空 GT 和空 GT CUDA 用例。

## AMP 报 BCE 不安全或首步梯度 Inf

- `binary_cross_entropy` 在 CUDA autocast 中会被 PyTorch 主动拒绝。如果模型接口已输出 sigmoid probability，将 BCE 局部置于 `autocast(enabled=False)` 并以 FP32 计算；不要在未改上游输出的情况下直接换成 `binary_cross_entropy_with_logits`。
- GradScaler 的初始 scale 过高时，前几个 batch 可能出现 Inf gradient。在 `unscale_` 之后记录 gradient norm，并根据 scale 是否下降判断 optimizer 是否被跳过。跳过时不要推进 scheduler/global-step/EMA；非 AMP 或 scaler 未跳步时出现非有限梯度，仍应立即失败。

## DDP 正常结束时警告未销毁 NCCL 进程组

由 `torchrun` 启动的 CLI 应在 `finally` 中调用 `torch.distributed.destroy_process_group()`。这不替代 rank-0 checkpoint 和 barrier 语义，但可以避免正常退出时由进程析构隐式回收 NCCL 资源。

如果 rank 1 仍重复输出应用日志，检查 logger 是否在 `init_process_group()` 前导入。此时 `dist.is_initialized()` 为假，但 `torchrun` 已设置全局 `RANK`；logger 应使用该环境变量决定控制台 rank。非日志 rank 若没有任何 handler，Python 的 `logging.lastResort` 仍会把 WARNING 裸写到 stderr，因此还需要 `NullHandler`。PyTorch/NCCL 自身带 `[rankN]` 的原生 warning 不受应用 logger 控制，应与重复的应用日志区分。

## 评估时官方转换权重缺少辅助头 anchor buffer

`aux_o2m_head.anchor_points` 和 `aux_o2m_head.stride_tensor` 由 `eval_size` 与 stride 在构建时派生，官方 Paddle 权重转换结果可以不包含它们。Eval CLI 可以放行这两个已知 missing key，但必须对任何其他 missing/unexpected key 失败。不要为了跳过派生 buffer 而把整个评估权重加载改成无审计的宽松模式。

评估入口还应使用当前 EvalReader 的 batch dict 和模型已后处理的 `bbox/bbox_num` 输出。如果入口仍假定 Dataset 返回 `(image, target)` 或模型返回 `pred_logits/pred_boxes`，说明它还停留在旧 API，即使单独的 postprocess 函数存在也不代表 CLI 可用。

Infer 同样必须复用 TestReader 和 `bbox/bbox_num`。旧入口曾同时使用手写 letterbox、ImageNet mean/std、`pred_logits/pred_boxes` 和外置 NMS；只补上 modeling 注册导入会越过第一处报错，但不会修复后续数据与输出合同。当前排查顺序应是：确认配置注册成功，再打印 batch 的 `image/im_shape/scale_factor`，最后检查模型输出是否已经完成后处理；不要在 CLI 叠加第二套解码。

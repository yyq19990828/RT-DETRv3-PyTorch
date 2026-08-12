# M2——官方 checkpoint 转换与分层数值对齐计划

> 历史计划快照（2026-07-18，M2）：本文保存已完成执行记录，不代表当前仓库状态。当前合同见 [`docs/models/rtdetrv3`](../../../models/rtdetrv3/README.md)。

- 状态：`completed`
- 创建日期：`2026-07-18`
- 最后更新：`2026-07-18`
- 负责人：`maintainer`
- 对应路线图：[`ROADMAP.md` Milestone 2](../../../../ROADMAP.md)
- 前置提交：`f95f4e0`（M1 最小训练链）

## 背景

M1 已证明 PyTorch R18 可通过当前 API 构建并完成最小训练链。M2 开始使用官方 Paddle checkpoint 锁定名称映射、张量布局和第一个分层数值分歧。

### 规划时基线

- **已验证**：M1 提交为 `f95f4e0`，全量活跃测试为 `127 passed, 1 skipped`。
- **已验证**：官方 R18 Google Drive checkpoint 大小为 `91,945,530` 字节，SHA-256 为 `f32dbd008bd7e5311c877d522f6d8c9e349795978c889f53823588b5e5d74a5f`。
- **已验证**：R18 源 checkpoint 包含 571 个 key；当前规则将 571 个 key 全部映射到 PyTorch 目标且转换后 shape 全部匹配。
- **已验证**：571 个转换 tensor 已分别按目标 Linear 布局转置或直接保留，并与源值逐个精确比较通过。
- **观察到**：PyTorch R18 `state_dict` 比 Paddle 多 77 个 key，其中 75 个是 BatchNorm `num_batches_tracked`，2 个是 auxiliary head 可根据 eval size 生成的 `anchor_points`/`stride_tensor`。
- **观察到**：官方 README 中 R18 示例使用的 BCEBos URL 在 2026-07-18 返回 HTTP 404；官方 model-zoo 中的 Google Drive 链接仍可下载。
- **已修复**：conversion CLI 现在默认通过 `--config` 构建目标模型，只有显式 `--no-validate` 才跳过目标感知校验。

上述 key 和 shape 结果只是映射基线，不能证明 Linear 方阵转置、BatchNorm 运行状态或模型输出已数值对齐。

## 目标与非目标

### 目标

- 为 R18/R34/R50 建立官方来源、固定配置、文件大小和 SHA-256 清单。
- 使 conversion CLI 通过 `--config` 构建目标 PyTorch 模型，默认执行 key/shape 校验，只有显式 `--no-validate` 才跳过。
- 对每个变体输出可审核的映射、shape mismatch、未映射源 key 和未填充目标 key 报告。
- 将转换权重加载到目标模型，为派生 buffer 与真正缺失参数建立分类规则。
- 在 CPU/float32 上使用同一 checkpoint、eval mode 和确定性输入，按 backbone → neck → transformer → head 定位第一个超容差激活。
- 在 R18 闭环稳定后再复用流程到 R34/R50，不同时分散排错。

### 非目标

- 不在 M2 声称 COCO AP 或训练等价；真实数据评估和精度门槛属于 M3/M4。
- 不转换 optimizer、scheduler、AMP scaler 或 RNG 状态。
- 不把 shape 相同、成功 `load_state_dict` 或最终预测接近单独作为数值等价证据。
- 不将官方 checkpoint 或转换后二进制文件提交到 Git。

## 依赖与执行约束

- 使用 `uv sync --extra dev` 管理的 `.venv`；Paddle 仅用于读取官方 checkpoint 和运行参考模型。
- `third-party/RT-DETRv3-paddle` 保持只读，对齐脚本从根目录显式引用子模块。
- 权重放在已忽略的 `pretrained_models/`；测试输出使用 `tmp_path`，验证后清理。
- 每个数值报告记录 Git、Python、Paddle、PyTorch、CUDA/cuDNN、device、seed、dtype、输入、预处理和容差。

## 实施步骤

### 阶段 1：checkpoint 来源与目标感知转换

- [x] 建立 `configs/checkpoints/rtdetrv3_coco.yml`，固定官方仓库 revision、三个变体的 config 和下载入口。
- [x] 下载 R18 官方 checkpoint，记录字节数与 SHA-256，并验证 Paddle 可读取。
- [x] 将 checkpoint checksum 从未注明的 MD5 升级为显式 SHA-256，补充元数据和单元测试。
- [x] 为 conversion CLI 增加 `--config`，默认构建目标 `state_dict`；修正当前形同虚设的 `--no-validate`。
- [x] 增加 manifest 和 CLI contract 单元测试，不在常规测试中下载官方权重。

### 阶段 2：R18 映射、加载与参数级校验

- [x] 通过目标感知 CLI 转换 R18，导出 mapping report 并记录耗时、源/目标 checksum 和覆盖率。
- [x] 审核 571 个源 key 的目标存在性与 shape，将 77 个目标缺失分为 BatchNorm 计数器、可派生 buffer 或真正缺失。
- [x] 根据目标 module 类型而不只是名称模式审核二维权重转置，特别处理 shape 为方阵时无法从 shape 发现的错误。
- [x] 将转换结果以明确允许的 missing/unexpected key 集合加载到 PyTorch R18。
- [x] 对非转置参数验证逐值保留；对 Linear 转置分别验证布局和数值。

### 阶段 3：R18 分层激活与训练单步对齐

- [x] 构建 Paddle/PyTorch 显式隔离的可选模型加载用例，使用独立命名空间和可恢复的 PyTorch workspace。
- [x] 用固定 NumPy 输入、eval mode、CPU/float32 捕获 backbone 每个返回层，定位第一个超容差激活。
- [x] 在 backbone 通过后依次比较 neck、transformer、head 和后处理，不跳过中间差异只看最终 boxes/scores。
- [x] 使用同一确定性 GT 隔离 head/loss，比较 12 个 loss 分项、总 loss 及其对四个 transformer 输出的梯度。
- [x] 使用缩减 query、关闭 noise 的确定性训练配置比较完整模型前向的 30 个 loss 分项，并修复多分组 attention mask 语义。
- [x] 修复非连续 BatchNorm grad-output 边界，并以整体相对 L2、余弦和符号分歧率验收完整模型参数梯度方向。
- [x] 记录 AdamW 单步差异及 ResNet 参数级 LR multiplier 缺口；按项目决策不要求更新逐元素一致，移交 M3 参数组验收。

### 阶段 4：扩展 R34/R50 与汇总报告

- [x] 下载 R34/R50 官方 checkpoint，补全 manifest 大小和 SHA-256。
- [x] 复用 R18 的映射、加载和分层对齐流程，只为变体特有差异增加规则。
- [x] 输出三变体转换汇总，包含覆盖率、missing/unexpected key、shape/layout、耗时、峰值内存和数值结果。
- [x] 更新本计划、进度快照、`ROADMAP.md` 和 `docs/migrations/weight-conversion.md`。

### 阶段 5：批量转换与受控内存

- [x] CLI 支持从单文件、目录或 glob 稳定发现 `.pdparams`，为同一 config/架构批量生成同名 `.pth`。
- [x] 单文件失败不阻断后续输入；批量结果记录成功、失败、错误、session、耗时和参数计数，并可导出 JSON。
- [x] checkpoint 先写同目录临时文件，成功后原子替换；写入失败时保留已有目标并清理临时文件。
- [x] `--memory-efficient` 只保留目标 shape 规格，并按参数批次释放源 Paddle tensor；输出 metadata 记录模式和批次大小。
- [x] 使用小型 fixture 验证批量 CLI/失败隔离，并用官方 R18 严格转换记录低内存路径的实际峰值 RSS。

## 风险与回退

- 风险：官方外链可变或需要 Google Drive cookie。缓解：固定官方仓库 revision、Drive file ID、字节数和 SHA-256，不只记录可点击 URL。
- 风险：方阵 Linear 权重转置错误不会被 shape 校验发现。缓解：以目标 module 类型和参数级数值测试确认布局。
- 风险：Paddle/PyTorch 同进程导入与全局注册状态干扰对齐。缓解：优先用双进程交换 NumPy/NPZ 中间结果。
- 风险：CPU 并行 reduction 的微小差异可能改变 transformer 边界 top-k 候选顺序。缓解：当前可选回归固定 PyTorch CPU 单线程；后续增加 top-k 前激活与 margin 证据。
- 风险：为追求最终输出接近而同时修改多层。缓解：每次只修复第一个超容差层，并保留层级回归。
- 风险：一个 config 被误用于不同架构的 batch。缓解：当前 batch 明确限定同 config/架构，跨架构官方模型仍分别运行目标感知转换。
- 风险：将参数分批误解为 checkpoint 流式读取。缓解：文档明确 Paddle 文件仍会整体加载，最终 PyTorch state dict 仍会整体驻留；该模式只降低双份 tensor 和目标模型的中间驻留。
- 回退：将 manifest/CLI 合同、R18 映射、分层对齐和变体扩展保持为独立提交；数值修复若破坏 M1，回退该层实现而保留失败证据。

## 验收

- [x] manifest 中所有 config 路径存在，verified 项的 SHA-256 和字节数与本地文件一致。
- [x] conversion CLI 默认要求 `--config`，`--no-validate` 的跳过行为有 contract 测试。
- [x] R18/R34/R50 均有官方文件 checksum、mapping report 和目标模型加载结果。
- [x] R18 完成 backbone、neck、transformer、head、loss 和完整参数整体梯度方向的分层报告；优化器差异已记录但不作为 M2 阻塞项。
- [x] 全量活跃测试（包含 M1 短训练）通过，官方 R18/R34/R50 可选回归分别通过，测试后无持久中间产物。
- [x] 批量转换在一个损坏、一个有效输入下继续执行并正确汇总；原子写入失败不会破坏已有输出。
- [x] 官方 R18 的低内存严格转换完成 571/571，输出 metadata 和重新加载结果正确。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-18 | 先闭环 R18，再扩展 R34/R50 | 防止同一映射/数值问题在三个变体重复排查 |
| 2026-07-18 | checksum 使用 SHA-256 | 比未标注的 MD5 更适合可复现完整性记录 |
| 2026-07-18 | 目标感知校验为 CLI 默认行为 | 无目标 `state_dict` 时无法识别错误 key 和 shape |
| 2026-07-18 | 官方权重不进 Git | 二进制体积大，应由来源、file ID 和 checksum 复现 |
| 2026-07-18 | 以目标 `torch.nn.Linear` 而非名称决定转置 | 12 个方阵 Linear 的错误无法由 shape 检查发现 |
| 2026-07-18 | R18 eval 回归固定 PyTorch CPU 单线程 | 避免并行 reduction 差异干扰 top-k 边界候选的可重复比较 |
| 2026-07-18 | 多分组训练 mask 从全 `False` 开始，仅开放对角组块 | 两侧 decoder 都把 `True` 解释为允许注意；eval 不会覆盖该训练态路径 |
| 2026-07-18 | 将 loss 输出梯度与完整模型参数梯度分开验收 | 前者使用逐 tensor 容差，后者使用整体相对 L2、余弦和符号分歧率，避免把近零元素当成主要信号 |
| 2026-07-18 | BatchNorm 反向入口统一 contiguous grad-output | NCHW → NLC 重排产生的非连续梯度会让 PyTorch input-gradient 偏离相同数值的 contiguous 参考；修复不改变前向或 state_dict |
| 2026-07-18 | 不要求 AdamW 更新逐元素完全一致 | 优化器方程和参数组应服务于收敛/AP；参数级 LR multiplier 缺口转入 M3 显式验收 |
| 2026-07-18 | R50 冻结 BatchNorm 在训练态保持全局统计 | Paddle `freeze_norm=True` 同时冻结参数并启用全局统计；仅设置 PyTorch `requires_grad=False` 仍会使用 batch statistics |
| 2026-07-18 | R50 后处理按稳定候选验收坐标 | 300 个候选中 2 个 label 处于 top-k 离散边界；全部 score 与 298 个稳定候选坐标通过，边界差异必须显式记录而不是放宽全局坐标容差 |
| 2026-07-18 | batch 限定一个 config/架构 | 不用文件名猜测模型结构；跨架构转换需要各自目标 shape 与 Linear 布局 |
| 2026-07-18 | 低内存模式释放源 tensor，但不宣称流式 checkpoint | Paddle load 与最终 PyTorch state dict 仍需完整驻留，避免夸大当前内存能力 |

## 当前完成证据

### R18 目标感知转换

- 命令：`.venv/bin/python -m ppdet_pytorch.cli.convert --input pretrained_models/paddle/rtdetrv3_r18vd_6x_coco.pdparams --output pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth --config configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --save-mapping pretrained_models/reports/rtdetrv3_r18vd_6x_coco.mapping.json --force`
- 已验证：571/571 转换，0 skipped，0 未映射源 key，77 未填充目标 key。
- 已验证：源 SHA-256 `f32dbd008bd7e5311c877d522f6d8c9e349795978c889f53823588b5e5d74a5f`；本次输出 SHA-256 `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547`，输出文件 `92,075,629` 字节。输出包含时间戳和 session ID，因此输出 hash 是本次运行证据，不是稳定发布 checksum。
- 观察到：本次转换 wall time `4.75 s`，`/usr/bin/time` 报告的最大 resident set size 为 `1,056,232 KiB`；该值包含 Python、Paddle、PyTorch 和目标模型驻留内存，不是纯权重转换增量。
- 已验证：加载后 unexpected key 为 0，missing key 仅为 `aux_o2m_head.anchor_points` 与 `aux_o2m_head.stride_tensor`。
- 已验证：571 个转换 tensor 全部与 Paddle 源值逐个精确相等；实际目标 Linear 先转置，其余 tensor 直接比较。

### R18 eval 分层对齐

- 环境：Python 3.12.11、NumPy 1.26.4、Paddle 3.3.0、PyTorch 2.5.1+cu121、CPU/float32/eval、PyTorch CPU 单线程，CUDA 未参与计算。
- 输入：NumPy `default_rng(2026)` 生成 `[1, 3, 640, 640]` 张量，`im_shape=[[640, 640]]`，`scale_factor=[[1, 1]]`。
- 容差：中间激活、head box/logit 和 score 使用 `rtol=1e-4, atol=1e-5`；后处理坐标先除以 640 再使用同一容差。
- 已验证：backbone 最大绝对误差 `8.17e-6`，neck `5.25e-6`；transformer decoder box/logit 分别 `1.54e-5`/`3.62e-5`，encoder box/logit 分别 `3.10e-6`/`6.68e-6`，全部通过。
- 已验证：head 通过；300 个后处理 label 和 `bbox_num` 完全一致，score 通过，像素坐标最大绝对误差 `0.00358 px`，归一化坐标通过。
- 回归：分别设置三个 checkpoint 环境变量并使用 `-k r18`、`-k r34`、`-k r50` 独立运行，实际结果均为 `1 passed, 2 deselected, 2 warnings`；警告来自只读 Paddle 子模块中的两个既有 `SyntaxWarning`。独立进程可避免三个完整双框架模型的内存池与 Paddle 全局状态累积。
- 全量活跃测试：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider`，最终结果 `144 passed, 3 skipped`；三个官方 checkpoint 用例未提供环境变量时按设计跳过。

### R18 确定性训练态对齐

- 环境与通用容差沿用 eval 用例：CPU/float32、PyTorch 单线程、seed `2026`、`rtol=1e-4, atol=1e-5`。
- 已验证：将 eval transformer 的四个输出作为独立叶子 tensor 输入训练 head，12 个 loss 分项、总 loss，以及四个输入梯度全部通过。
- 已验证：使用 `96×96` 合成输入、两条固定 GT、两组各 20 个 query、4 个 denoising query，并把 label/box noise 设为 0；完整训练前向的 30 个 loss 分项全部通过。总 loss 为 Paddle `68.8954163`、PyTorch `68.8954239`，绝对差 `7.63e-6`。
- 已修复：PyTorch 多分组 attention mask 曾错误地从全 `True` 初始化，允许组间 query 互相注意；改为全 `False` 后只开放组内对角块，并新增直接 mask 单元测试。
- 已修复：BN 本身、Conv2D 和 `_get_decoder_input` 隔离时均通过；首个真实分歧来自 BN 收到 NCHW → NLC 重排产生的非连续 grad-output。`ContiguousGradBatchNorm2d` 规范反向输入布局后，核心单测与官方 R18 回归通过。
- 已验证：384 个可训练参数的整体梯度相对 L2 误差 `0.00434`、余弦 `0.999991`、有效元素符号分歧率 `0.1304%`。可选回归门槛分别为 `<0.01`、`>0.9999`、`<0.5%`。
- 观察到：相同全局 LR 的 AdamW 探针里，Paddle ResNet stage 的 `ParamAttr` 仍应用 `lr_mult_list=0.1`，当前 PyTorch 参数/optimizer group 未承接，因此部分 backbone 更新约相差 10 倍。该缺口转入 M3，不计为 M2 数值算子失败。
- 限制：上述结果不证明官方完整 query/noise 配置、真实预处理、优化器参数组、训练收敛或 COCO AP 对齐。

### R34/R50 扩展结果

| 变体 | 源文件 / SHA-256 | 转换与加载 | 本次转换 wall / 最大 RSS | 数值结果 |
|---|---|---|---|---|
| R34 | `137,016,081` 字节 / `29b09c64d6c372cde46d94caee1b57a23cee0aae24bd7bd3e2937cf57e581a68` | 681/681；91 个 BN counter；missing 仅 2 个 auxiliary buffer；unexpected 0 | `5.13 s` / `1,188,888 KiB` | `rtol=1e-4, atol=1e-5` 下分层 eval、后处理、loss 及 462 个参数的整体梯度方向通过 |
| R50 | `182,331,170` 字节 / `e8b1d5db3208ce0f9edba5a914f23c918141b608ab4cd409db9d9204f7ed4b08` | 789/789；103 个 BN counter；missing 仅 2 个 auxiliary buffer；unexpected 0 | `5.50 s` / `1,322,312 KiB` | `rtol=3e-4, atol=1e-5` 下分层 eval、loss 及 445 个可训练参数的整体梯度方向通过 |

- **已验证**：R34 与 R50 的所有转换 tensor 均按目标 Linear 布局转置或原样保留，并与 Paddle 源数组逐个精确相等。
- **已修复**：R50 的 `freeze_norm=True` 曾只冻结 PyTorch BN 参数，`model.train()` 后仍错误使用 batch statistics；现在冻结 BN 保持 eval/global-statistics 语义，并有单元回归。
- **观察到**：R50 decoder box 在 `rtol=1e-4` 下有 1/1200 个边界值超差，最大绝对误差 `3.95e-5`，因此该变体使用有证据的 `rtol=3e-4`。后处理 300 个 label 中有 2 个 top-k 边界差异；全部 score、298 个稳定候选坐标和 `bbox_num` 通过，未把两个离散候选强行按行比较。
- **已验证**：三个变体的梯度统一按 `relative_l2 < 0.01`、`cosine > 0.9999`、有效元素符号分歧率 `<0.5%` 验收；这证明受控场景下整体梯度方向接近，不证明优化器更新或训练收敛完全一致。

### 批量与低内存转换证据

- **已验证**：小型 batch fixture 发现 1 个 `.pdparams`，生成 1 个 `.pth`、独立 mapping 和 JSON summary；summary 为 `total=1, succeeded=1, failed=0`，转换 9 个 tensor。
- **已验证**：集成用例输入顺序为损坏 checkpoint → 有效 checkpoint；结果为 `failed=1, succeeded=1`，失败项无残留输出，有效项仍可加载且 mapping 存在。
- **已验证**：模拟 `torch.save` 写入部分临时文件后抛错，已有目标文件内容保持不变，临时文件清零。
- **已验证**：官方 R18 使用 `--strict --memory-efficient --parameter-batch-size 64` 完成 571/571，重新加载得到 571 个 tensor，metadata 为 `memory_efficient_mode=true`、`parameter_batch_size=64`。
- **观察到**：上述低内存运行 wall time `5.71 s`、最大 RSS `925,780 KiB`；此前普通目标感知运行为 `1,056,232 KiB`，观测下降约 `12.4%`。这是目标 shape map 与分批释放的组合结果，不归因于单一优化，也不外推到 R34/R50。
- **已验证**：默认全量测试更新为 `144 passed, 3 skipped`；转换定向测试为 `26 passed`。所有 pytest 与转换临时产物均在验证后清理。

## 完成记录

M2 已完成。三个官方变体的来源、转换、加载、分层 eval、受控 loss 和整体梯度方向均有回归；批量失败隔离、JSON 汇总、原子输出和受控内存路径也已验收。优化器参数组、真实训练/评估与恢复转入 [`M3——训练、评估与恢复验收计划`](2026-07-18-m3-training-evaluation-recovery.md)。

# M1——R18 最小训练链迁移计划

- 状态：`completed`
- 创建日期：`2026-07-18`
- 最后更新：`2026-07-18`
- 负责人：`maintainer`
- 对应路线图：[`ROADMAP.md` Milestone 1](../../ROADMAP.md)

**完成状态**：M1 实现、本机验收和证据整理已完成；提交号在本计划的后续进度快照中记录。

## 背景

当前仓库已能加载 `configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml`，活跃测试也覆盖了部分注意力、解码器、训练策略和权重转换功能。但是还没有一条按当前 `workspace` API 验证过的 config → dataset → batch → model → loss → backward → optimizer 链路；`tests/legacy/` 的历史通过状态不能作为当前证据。

### 规划时基线

- **已验证**：在仓库 `.venv` 中加载 R18 COCO 配置后，`create(cfg.architecture)` 能构建 `RTDETRV3`，参数量为 `22,942,893`。这不证明训练前向或数值等价。
- **代码观察**：`Trainer` 的数据集构建会将 `TrainDataset` 当作注册类名，但当前配置将它定义为含 `name: COCODataSet` 的配置块。
- **代码观察**：`Trainer.__init__()` 调用 `_build_model()` 和 `_build_optimizer()` 时与当前方法签名不一致；optimizer 和 scheduler 的绑定、调用顺序需要用单步测试锁定。
- **代码观察**：配置和注册类使用 `RTDETRV3`，部分 Eval/Infer 路径使用 `RTDETRv3`。本计划只处理阻塞最小训练链的同类命名问题，完整 CLI contract 留给 M5。

## 目标与非目标

### 目标

- 用未修改的 R18 主配置构建完整模型，并验证 backbone、neck、transformer、DETR head、auxiliary head 和 post-process 的实际类型。
- 由测试生成的最小 COCO fixture 构建 `COCODataSet` 和 `TrainReader`，固定一个 batch 的字段、shape、dtype、bbox 坐标系与 padding 合同。
- 在 CPU/float32 上完成一次训练前向，产生有限的总 loss 和各 loss 分项。
- 完成 backward 与一次 optimizer step，验证关键子模块有有限梯度且至少一个可训练参数实际更新。
- 用同一 fixture 完成 5–10 iteration 短训练，全程无 NaN/Inf，且不留下 checkpoint、日志、cache 等中间产物。
- 将与 M1 直接相关的历史测试场景按当前 API 重写到活跃测试集，不恢复旧 Registry/builder 兼容层。

### 非目标

- 不在本计划中下载完整 COCO 数据集、跑完整 epoch 或声称 COCO AP 达标。
- 不做 Paddle/PyTorch 激活、loss、梯度或优化器数值对齐；这些属于 M2。
- 不验证 AMP、EMA、checkpoint 恢复、DDP 或真实 COCO 评估；这些属于 M3。
- 不完成 Eval/Infer/Convert 的全量 CLI 兼容和导出边界；这些属于 M5。
- 不将 Paddle 引入核心运行时或默认测试集。

## 依赖与执行约束

- 使用 `uv sync --extra dev` 维护的 `.venv`；此阶段的默认验收不依赖 Paddle 进程或官方 checkpoint。
- 最小 COCO fixture 在 pytest `tmp_path` 中生成，包含至少 2 张图像、多个有效 bbox 和可验证的 category 映射；不向仓库提交临时二进制图像。
- 默认数据测试设置 `worker_num=0`、固定 seed 和确定性 resize，先排除 worker 和随机增强差异。
- 保留主配置构建测试；为控制 CPU 内存和时间，前向及短训练允许通过测试专用 override 缩小 batch 和输入尺寸，但必须记录 override，不得把它写成原始 640 配置训练证据。
- 测试前后隔离并恢复 `global_config`，避免测试顺序污染注册和配置状态。

## 实施步骤

### 阶段 1：锁定当前配置构建合同

- [x] 为 `workspace` 增加活跃单元测试，覆盖配置加载隔离、`name` 引用、显式 kwargs 与 `from_config()` 所需的最小优先级。
- [x] 明确 `TrainDataset` 这类命名配置块的构建入口，使用单一当前 API 解析 `name: COCODataSet`；不引入第二套 Registry。
- [x] 修正 `Trainer` 构建方法的参数合同，并将配置中的 architecture 名作为唯一模型构建来源。
- [x] 验证未缩减的 R18 主配置构建完整模型，断言注入子模块类型和输出 shape 元数据。

**阶段验证**：配置构建测试可独立、重复和乱序执行，且不依赖 `tests/legacy/`。

### 阶段 2：建立最小 COCO batch 合同

- [x] 在 `tests/conftest.py` 中建立 `tmp_path` COCO fixture factory，使用可预期像素、图像尺寸、bbox 和 category id。
- [x] 验证 `COCODataSet.parse_dataset()` 产生的单样本字段、xyxy 原始 bbox 和 class id。
- [x] 验证 Decode、BatchRandomResize、NormalizeImage、NormalizeBox、BboxXYXY2XYWH、Permute 和 PadGT 的确定性链路。
- [x] 通过 `TrainReader` 构建 batch，断言模型消费字段的 shape/dtype/device。
- [x] 增加非法 bbox 过滤、不等长 GT 和 padding mask 回归；空 GT 明确留待后续边界测试。

**阶段验证**：一个 batch 的合同可由测试失败信息直接定位到数据集、单样本 transform 或 batch transform。

### 阶段 3：训练前向与 loss

- [x] 用 R18 架构和确定性 batch 执行 CPU/float32 训练前向。
- [x] 断言输出为包含 `loss` 的字典，总 loss 为标量且 30 个 loss 总项/分项全部有限。
- [x] 覆盖 DETR one-to-one/one-to-many、denoising 和 auxiliary O2M head 路径。
- [x] 为 M1 直接需要的 backbone、head、loss、post-process、配置与模型集成场景重写当前 API 回归测试；其他历史边界已分类延后。

**阶段验证**：固定 seed 后重复前向均为有限值；相同输出不得被描述为 Paddle 数值对齐证据。

### 阶段 4：backward、optimizer 与 scheduler 单步

- [x] 使 `OptimizerBuilder` 先根据配置构建 optimizer，再将其传入 `LearningRate`，并用 Trainer 构建测试锁定。
- [x] 断言 backbone、neck、transformer 和 auxiliary head 的参与参数梯度存在且有限；`DINOv3Head` 本身是无参数 loss wrapper。
- [x] 断言 backbone 参数在 step 后变化，并断言 auxiliary DFL 的冻结 projection 权重保持不变。
- [x] 验证 gradient clipping 返回有限全局范数，以及 optimizer step 后 scheduler 改变 LR。

**阶段验证**：单步测试同时证明 loss 可反传、梯度有限、参数更新和 LR 调度时序，不只断言“没有抛异常”。

### 阶段 5：5–10 iteration 短训练与历史测试收敛

- [x] 提供一条通过当前 Trainer epoch 实现执行的 5 iteration 烟雾测试，设置固定 seed、CPU/float32、`worker_num=0` 和 pytest 临时输出目录。
- [x] 每步断言 loss、梯度范数和 LR 有限，并记录首步/末步值；未将短序列 loss 下降作为要求。
- [x] 将短训练标记为 `slow`/`integration`，并保留快速单步回归。
- [x] 在 `tests/legacy/README.md` 建立已重写/延后场景清单；本轮没有误删未完整替代的历史文件。
- [x] 更新本计划、进度快照、`ROADMAP.md` 与可复用迁移经验。

**阶段验证**：开发机可从干净 checkout 重复最小训练链，测试结束后没有遗留临时 checkpoint、日志、`__pycache__`、`.pytest_cache` 或生成数据。

## 候选变更范围

以下是当前预计直接涉及的路径；实施时以最小差异为准，不顺手重构无关模块。

- `src/ppdet_pytorch/core/workspace.py`
- `src/ppdet_pytorch/data/reader.py`
- `src/ppdet_pytorch/engine/trainer.py`
- `src/ppdet_pytorch/optimizer/optimizer.py`
- 首个失败所在的数据 transform、模型或 loss 文件
- `tests/unit/core/`、`tests/unit/data/`、`tests/unit/modeling/`、`tests/unit/optimizer/`
- `tests/integration/test_rtdetrv3_training_chain.py`

## 风险与回退

- 风险：扩大 `create()` 的输入语义可能改变 shared/inject/from_config 优先级。缓解：先写合同测试，只实现 M1 所需的命名配置块和显式参数路径，完整兼容矩阵留给 M5。
- 风险：R18 全模型在 CPU 上训练较慢或占用较多内存。缓解：将“原始配置构建”与“明确记录 override 的最小前向”分开验收，并将 5–10 step 用例标记为 slow。
- 风险：训练 transform 包含随机增强，直接固定框架 seed 不保证跨框架随机序列一致。缓解：M1 只使用确定性参数验证 PyTorch 数据合同，Paddle 对齐留给 M2。
- 风险：auxiliary head 或空 GT 路径可能扩大修复范围。缓解：保留完整 auxiliary head 作为 M1 必验路径；空 GT 若不阻塞有标注最小链，记录局限后排入后续边界测试。
- 回退：每个阶段保持独立可测的小提交。如新的配置构建语义破坏现有用例，回退该阶段的实现，保留失败合同测试和决策记录，不恢复旧 Registry。

## 验收

- [x] `.venv/bin/pytest tests/unit/core tests/unit/data tests/unit/modeling` 通过：`66 passed`。
- [x] `.venv/bin/pytest tests/integration/test_rtdetrv3_training_chain.py -m "integration and not slow"` 通过：`3 passed, 1 deselected`。
- [x] `.venv/bin/pytest tests/integration/test_rtdetrv3_training_chain.py -m "integration and slow"` 通过：`1 passed, 3 deselected`。
- [x] `.venv/bin/pytest -m "not paddle and not slow"` 通过：`99 passed, 29 deselected`。
- [x] 验收记录已包含环境、device、seed、dtype、命令、override 和实际结果；提交号由后续进度快照引用，避免在本提交内循环引用自身。
- [x] 测试禁用 pytest cache 与 Python bytecode，fixture/输出使用 pytest 临时目录；本轮无持久 checkpoint、日志或生成数据。

## 完成证据模板

| 证据 | 要求 | 实际结果 |
|---|---|---|
| 环境 | Git/Python/PyTorch/CUDA/cuDNN/device | 父提交 `9771881`；M1 完成提交见后续进度快照；Python 3.12.11；PyTorch 2.5.1+cu121；CUDA runtime 12.1；cuDNN 91300；CUDA 可用但本轮使用 CPU |
| 输入 | fixture 版本、seed、dtype、配置 override | 2 张合成图像、3 个有效 bbox、2 类；seed 2026；float32；96×96、batch 2、20 queries、2 decoder layers、8 denoising queries、CPU |
| 模型 | 子模块类型、参数量、构建命令 | 未修改 R18 主配置构建 `RTDETRV3/ResNet/HybridEncoder/RTDETRTransformerv3/DINOv3Head/PPYOLOEHead/DETRPostProcess`；`22,942,893` 参数 |
| batch | 字段、shape、dtype、bbox/padding 断言 | `image=(2,3,64,64)` float32；GT 分别 `(1,4)/(2,4)`；class 转 int64；origin padding mask 和非法 bbox 过滤已验证 |
| 前向 | loss key、有限性、执行命令 | 30 个总项/分项，包含 class/bbox/GIoU/DFL/L1、aux、DN 和 O2M；全部为有限标量 |
| 反向 | 代表性梯度、参数更新、LR 时序 | backbone/neck/transformer/aux head 梯度有限；backbone 参数更新；冻结 DFL projection 不变；裁剪范数与 LR 更新有限 |
| 短训练 | step 数、首末 loss/LR、总耗时 | 5 step；loss `45.69997 → 46.30236`；grad norm `142.58978 → 92.68174`；LR `1.999e-7 → 5.995e-7`；单独 slow 命令 2.67 s |
| 回归 | pytest 命令、passed/skipped/failed、产物清理 | 全量 `127 passed, 1 skipped` / 7.94 s；`git diff --check` 通过；无持久测试产物 |

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-18 | 将 M1 作为下一个执行计划 | M2–M5 都依赖当前 API 的可训练端到端链路 |
| 2026-07-18 | 原始 R18 配置构建与缩小输入的前向/烟雾测试分开验收 | 保留真实配置证据，同时控制 CPU 测试成本 |
| 2026-07-18 | M1 默认测试不依赖 Paddle | 维持 core-only 运行时边界，数值对齐在 M2 单独验收 |

## 完成记录

2026-07-18 已完成 M1 源码、测试和文档实施，并通过全量活跃测试。训练使用明确缩减的测试 override，不是原始 640 训练、Paddle 数值等价或 COCO 精度证据。空 GT、真实数据、AMP/EMA/DDP、checkpoint 恢复和 Eval/Infer 继续由 M2–M5 跟踪。

可复用结论已推广至 [`training-and-validation.md`](../migrations/training-and-validation.md) 和 [`registry-and-configuration.md`](../migrations/registry-and-configuration.md)；历史用例去向记录在 [`tests/legacy/README.md`](../../tests/legacy/README.md)。

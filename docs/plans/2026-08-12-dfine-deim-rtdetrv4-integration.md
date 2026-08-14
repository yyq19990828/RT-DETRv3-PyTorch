# D-FINE、DEIM 与 RT-DETRv4 集成计划

- 状态：`in-progress`
- 创建日期：`2026-08-12`
- 最后更新：`2026-08-14`
- 负责人：仓库维护者与后续执行代理
- 执行规范：本计划的任务、验证矩阵与完成记录；模型级结果见 [`docs/models`](../models/README.md)
- 审查记录：2026-08-12，Momus 与独立 Oracle 对同一执行计划完成双重批准；审查轮次 `rtdetrv4-merge-20260812-r6`

## 背景

当前仓库已经具备 RT-DETRv3 的配置、训练、恢复、评估、推理、权重、导出、打包与发布合同，但尚未实现 HGNetv2、D-FINE、DEIM 或 RT-DETRv4。此次工作把三个官方 PyTorch 上游的 COCO 模型族接入同一运行时，并以当前 RT-DETRv3 数值与用户接口作为不可回归基线。

权威上游固定为：

- D-FINE：`Peterande/D-FINE@267a6da6d04c8ad52e54120692896515b9e55981`，Apache-2.0。
- DEIM：`Intellindust-AI-Lab/DEIM@09d35d53d39ee3145a1e61e3a989b28b9468d1dd`，Apache-2.0。
- RT-DETRv4：`RT-DETRs/RT-DETRv4@55fefaaed7efe2a5f72d0a18fd4e05965e35c292`，Apache-2.0。

官方范围共 19 个 COCO 变体：D-FINE N/S/M/L/X、DEIM-D-FINE N/S/M/L/X、DEIM-RT-DETRv2 S/M/M*/L/X，以及 RT-DETRv4 S/M/L/X。

## 目标与非目标

### 目标

- 让 19 个官方配置均可从安装后的包构建，并支持其适用的训练、确定性恢复、评估、推理、模型目录和部署路径。
- 只实现一次 HGNetv2 与经源码和数值证明等价的 D-FINE 基元，同时保留 D-FINE criterion、DEIM MAL/Dense O2O/两阶段 EMA、RT-DETRv4 DSI/GAM/教师编排的独立语义。
- 直接加载官方 PyTorch `{"model": state_dict}` 推理权重，并继续使用仓库 format-version-1 完整训练 checkpoint。
- 让 RT-DETRv4 的 DINOv3 教师只存在于训练阶段；student checkpoint、评估、推理、ONNX 和 TorchScript 不构造教师且不依赖教师资产。
- 保持现有 `rtdetrv3-*` 命令、配置、manifest、`bbox`/`bbox_num` 输出、ONNX opset 17、固定空间尺寸与动态 batch 合同。
- 为所有上游代码与资产记录 revision、许可证、URL、大小、SHA-256、映射和数值证据。

### 非目标

- 不修改 `third-party/RT-DETRv3-paddle`，不在核心运行时导入 Paddle，也不宣称这些原生 PyTorch 上游与 Paddle 数值对齐。
- 不把 RT-DETRv3 当作 RT-DETRv2，不新增独立 RT-DETRv2 产品族。
- 不重命名包或 console scripts，不新建并行 Registry、Trainer、checkpoint 或导出框架。
- 不执行 Objects365 完整预训练，不移植 TensorRT/C++ 工具，不复现全部上游硬件性能。
- 不把上游权重重新发布到本项目 Release；manifest 只描述并校验上游托管资产。
- 不做无关 RT-DETRv3 重构，不因实测结果放宽预注册数值门槛。

## 决策与边界

1. 所有模型继续通过现有 workspace `@register`、`BaseArch` 和 Trainer 构建；新架构名为 `DFINE`、`DEIM` 与 `RTDETRV4`。
2. D-FINE 家族使用独立 encoder/decoder 模块，不改变当前 RT-DETRv3 `HybridEncoder` 或 `RTDETRTransformerv3`。
3. 只有源码 diff 与固定 checkpoint/输入激活对齐均通过时才复用 matcher、denoising、后处理或其他数学组件；名称和 shape 相同不算证据。
4. DEIM-RT-DETRv2 只引入 r18vd、r34vd、r50vd_m、r50vd、r101vd 所需的 RT-DETRv2 decoder 切片。
5. D-FINE、DEIM 与 RT-DETRv4 的两阶段训练通过一个通用可选 `TwoStageDetectionProtocol` 接缝接入 Trainer；未配置协议的 RT-DETRv3 路径保持原行为。
6. checkpoint 版本继续为 1，通过可选 `training_state` 保存 family/protocol identity、stage、最佳指标、EMA decay、`best_stg1.pth` basename/SHA-256 与 GAM 当前权重；不保存教师对象。
7. 仅承诺 epoch-boundary 确定性恢复；checkpoint 采用先反序列化/验证全部组件与 companion SHA、再统一应用的事务式恢复，任何错误都必须发生在模型、优化器、scheduler、scaler、EMA 修改前。任意 mid-epoch resume 不在本计划范围。
8. DINOv3 采用本地 `facebookresearch/dinov3` checkout 和 Meta 单独授权的 ViT-B/16 LVD-1689M 权重；训练前验证 repo revision、hub 入口、权重大小/SHA 与 patch geometry，不自动下载或再分发。
9. Models CLI 增加显式 family 选择，无参数仍默认 RT-DETRv3，`--manifest` 优先级最高；上游不能直接下载的链接只支持 list/verify，并在 download 时明确给出官方地址。

## 前置资产

- COCO 2017 train2017、val2017 与 annotations，通过配置 override 指向本机位置，不提交到仓库。
- 19 个上游官方 student checkpoint 与 HGNetv2 预训练权重，按 manifest 记录 URL、大小和 SHA-256。
- RT-DETRv4 真实训练验收所需的外部 `facebookresearch/dinov3@346f38fee679c56a6888f91c51670fae61d364e0` checkout 与经 Meta 表单授权获得的 ViT-B/16 LVD-1689M `.pth`；DINOv3 无 tag/release，使用自定义 DINOv3 License，不能按 Apache-2.0 上游处理。
- 支持 PyTorch 2.5.1 的 CPU 环境；student-only 包路径保持 Python 3.9-3.12，DINOv3 教师训练按其源码合同使用 Python 3.11+。完整精度矩阵和真实教师训练建议使用 CUDA，但 CPU 门仍必须通过。

若官方资产无法获得，依赖该资产的步骤标记为 `[blocked]` 并附 URL、错误和校验记录，不得以 fake teacher、shape 测试或小样本 smoke 冒充完整验收。

执行时统一设置：

```bash
export COCO_ROOT=/path/to/coco
export UPSTREAM_CHECKPOINT_ROOT=/path/to/official-checkpoints
export DINOV3_REPO=/path/to/dinov3
export DINOV3_WEIGHTS=/path/to/dinov3_vitb16_pretrain_lvd1689m.pth
export DINOV3_WEIGHTS_SHA256=<authorized-file-sha256>
```

计划中的模型矩阵统一通过 `tools/dev/validate_model_family.py` 执行 checkpoint、训练恢复、COCO、推理与导出验证。该驱动必须在构建模型前验证上述变量、目录布局与 manifest checksum；缺失资产时非零退出并命名缺失项。稳定结论、复现入口和限制已整理到各模型族的验证报告、指标记录与证据索引。

## 实施步骤

### Wave 1：基线与共享基础

- [x] 1. 冻结 RT-DETRv3 与上游对齐基线
  - 在任何模型族任务使用前新增并测试四个驱动：`tools/dev/compare_upstream_pytorch.py`（state/activation/output diff）、`tools/dev/validate_model_family.py`（manifest-bound checkpoint/train-resume/COCO/infer/export/teacher 矩阵）、`tools/dev/audit_plan_evidence.py`（计划/SHA/证据映射）和 `tools/dev/audit_model_family_graphs.py`（重复实现、依赖、opset、训练节点残留）。固定其参数、JSON schema、证据布局、preflight-before-mutation 与 `APPROVE|BLOCKED|FAIL`/退出码合同。
  - 新增固定输入、hook、失败诊断与驱动 CLI/负例测试，记录三上游 SHA、环境、当前非 Paddle 测试、质量、覆盖率与 R18 官方数值结果。
  - 错误 SHA 或超过容差的单个张量必须以张量名和最大误差失败；临时 checkout 与 checkpoint 位于仓库外。

- [x] 2. 实现共享 HGNetv2 B0/B2/B4/B5
  - 新增 `modeling/backbones/hgnetv2.py`，保留官方 freeze、freeze_norm、stage 输出和 `out_shape`。
  - 四个变体逐 stage 对齐官方 PyTorch 权重；错误变体、key 或 layout 在推理前失败。

- [x] 3. 实现共享 D-FINE decoder、分布数学与家族 encoder
  - 新增 FDR、LQE、GO-LSD、bbox distance/weighting、deploy conversion 和 D-FINE HybridEncoder。
  - encoder 默认只返回 FPN/PAN 特征；仅 RT-DETRv4 训练且配置投影维度时额外返回 projected AIFI F5。

- [x] 4. 固定 matcher、denoising、target bridge 与后处理语义
  - 先对现有 `HungarianMatcher` 和 `DETRPostProcess` 做源码/数值验证；等价则复用，不等价则使用唯一的新注册名。
  - 覆盖空目标、DN group、top-300、坐标还原和 `bbox`/`bbox_num`，不接受仅 shape 验证。

- [x] 5. 实现 Dense O2O 的 epoch-aware Mosaic/MixUp/multiscale
  - D-FINE 仅使用普通 transform/multiscale stop policy，不启用 Mosaic/MixUp：N stop 148、S/M stop 120、L/X stop 72。
  - Dense O2O 只用于 DEIM/RT-DETRv4，并按配置参数化验证：DEIM-D-FINE N `[4,78,148]`、S `[4,64,120]`、M `[4,49,90]`、L/X `[4,29,50]`；DEIM-RTv2 S/M `[4,64,117]`、M*/L/X `[4,34,58]`；RTv4-S `[4,64,120]`、M `[4,49,90]`、L/X `[4,29,50]`，以及各自 MixUp 区间。
  - 现有 v3 配置不含策略时，同 seed batch 必须保持不变。

- [x] 6. 实现可恢复的 FlatCosineLRScheduler
  - 对齐 quadratic warmup、flat、cosine 和 no-aug 阶段，包括 N 配置 warmup 后恒定 LR 的官方行为。
  - AMP skip 与累积 microbatch 不得前移 scheduler；完整 trace 与恢复 suffix 必须一致。

- [x] 7. 增加可选训练协议与 checkpoint-v1 `training_state`
  - 在 Trainer 增加 preflight、epoch 前后、validation、save/load、`after_backward` 和 `after_successful_optimizer_step` hooks；明确 AMP unscale、累积最终 microbatch、clip、overflow skip 与成功 update 的顺序。
  - 旧 v3 checkpoint、原子保存、RNG 与组件恢复不得回归；错误 checkpoint 通过事务式 preflight 保证所有组件 fingerprint 不变。

- [x] 8. 实现 DEIM 所需 RT-DETRv2 decoder 切片
  - 只支持五个已批准 backbone/decoder 组合，复用经验证的 ResNet-vd 和 attention 基元。
  - v3 checkpoint/config 或未支持 backbone 必须作为 RT-DETRv2 不兼容失败，不能部分加载。

### Wave 2：D-FINE 与共享训练协议

- [x] 9. 实现 `DFINE(BaseArch)` 与 `DFINECriterion`
  - 组装 backbone、D-FINE encoder/decoder、VFL/L1/GIoU/FGL/DDF/GO union 与后处理。
  - 固定 batch 的所有 loss、梯度、raw predictions 和最终检测必须对齐上游。
  - 2026-08-13：正常目标的全部 loss key/value 与全部 decoder 参数梯度对齐 pinned 上游，raw eval 和最终 `bbox`/`bbox_num` 对齐；全空目标完成 finite loss/gradient 验证。本地既有 DN helper 在全空目标时省略上游零长度 DN 键，因此不把该特殊键集合声明为逐项 parity。

- [x] 10. 添加 D-FINE N/S/M/L/X 官方 COCO 配置
  - 精确映射 B0/B0/B2/B4/B5、feature levels、hidden dims、decoder layers、schedule、optimizer、EMA、augment stop 和 top-300。
  - 五个 YAML 连续重复加载、构建、train/eval forward 均通过，无 Registry 状态泄漏。
  - 2026-08-13：五变体参数量为 `3,782,693 / 10,321,877 / 19,590,064 / 31,244,152 / 62,621,560`；官方 global batch、多尺度重复概率、stop epoch、500-step warmup、EMA warmup 1000 和 runtime 字段已映射到仓库合同。两阶段 `best_stg1` reload 与 stop-epoch EMA restart 由后续训练协议验证覆盖。

- [x] 11. 验证五个 D-FINE 官方 checkpoint
  - 记录不可变 URL、大小、SHA-256、container keys、映射和 layout；要求零未知 key。
  - 使用同 checkpoint、预处理、输入、eval mode、dtype 比较首个分歧激活和 raw output。
  - 2026-08-13：五个官方资产均完成 size/SHA-256/container 校验；N/S/M/L/X 分别以 identity mapping 严格覆盖 `674/794/1053/1173/1441` 个 tensor，零 missing/unexpected/shape/dtype 差异。固定 640 输入的 stem、backbone、encoder 与 raw `pred_logits`/`pred_boxes` 对 pinned runner 全部通过 `rtol=1e-5, atol=1e-6`，本次实测最大绝对误差均为 0；损坏资产与错变体在预测前失败。

- [x] 12. 验证 D-FINE 训练、恢复、推理与 COCO 精度
  - 对齐官方 D-FINE 两阶段 checkpoint/EMA：stage-1 best、stop-epoch companion SHA/reload、EMA restart/decay 与 stage-2 best/no-improvement restart；epoch-boundary 中断/恢复在下一 optimizer update 的 LR、loss、EMA、RNG、参数一致。
  - N/S/M/L/X 均完成真实图片推理、四图上游对照和 val2017；bbox AP 与官方四舍五入值相差不超过 0.001（0-1 标度）。
  - 2026-08-13：修正 D-FINE Eval/Test 从 OpenCV cubic 到上游 PIL bilinear 的预处理差异，并以真实图片逐像素对齐验证。N/S/M/L/X val2017 AP 为 `0.427997 / 0.485145 / 0.522783 / 0.539703 / 0.557650`，相对官方值误差均小于 `0.000350`；五变体 reduced staged training、epoch-boundary resume、四图 raw-output parity、eager JSON/render 与指定负例全部通过。

- [x] 13. 导出并重载全部 D-FINE 变体
  - 使用现有三输入/两输出 adapter，ONNX opset 17、固定空间尺寸、动态 batch 与 TorchScript。
  - 五个变体 batch 1/4 均通过 checker/reload 和逐图候选匹配，产物不含训练辅助输出。
  - 2026-08-13：修复 cached anchors 的 batch=1 trace 固化、deploy conversion 非幂等、CLI 未调用 deploy、D-FINE `eval_spatial_size` 未同步及 parity 失败后仍发布产物的问题。N/S/M/L/X 的 ONNX opset 17 与 TorchScript 均在 CPU/FP32、固定 640、batch 1/4 下通过；ONNX 最大 score/box 误差分别为 `1.2443e-5 / 0.01879 px`，TorchScript 为零，双格式均确认无 criterion、denoising 或 auxiliary residue。计划指定的 wrong-size、dynamic-height 和 training-output 负例通过，临时导出已清理。

- [x] 14. 实现 `DEIM(BaseArch)` 与 DEIM MAL/GO criterion
  - 配置选择 D-FINE 或 RT-DETRv2 decoder；实现 MAL `gamma=1.5`、union-set box/local、FGL/DDF（仅 D-FINE）及 CDN/aux/encoder loss。
  - 同权重 eval 必须与对应非 DEIM graph 一致，DEIM 不新增推理分支。
  - 2026-08-13：`DEIM` 直接继承共享 `DFINE(BaseArch)` adapter，不复制模型图或增加 eval 分支。pinned DEIM 对照覆盖 MAL `mal_alpha=None/0.5`、RT-DETRv2 main/aux/encoder/CDN、D-FINE main/aux/pre/encoder/CDN/FGL/DDF、union on/off、IoU weighting 的全部 loss key、值和 prediction gradient；实际两类 decoder 也完成 batch-2（含空 GT）一步反传。额外修复 class-agnostic encoder 在匹配前归零 labels、MAL fractional gamma 的负/非有限 quality、CDN metadata 与 `dn_pre_outputs` 验证；正式十模型配置/checkpoint/COCO/export 由后续模型矩阵验证覆盖。

- [x] 15. 实现确定性的两阶段 validation/EMA 协议
  - 通用协议服务 D-FINE、DEIM 与 RT-DETRv4；在 stop epoch 校验并加载 `best_stg1.pth`、重启/调整 EMA decay，生成 `best_stg2.pth`，保存全部协议状态。
  - 对三个家族参数化使用合成指标序列与真实 reduced run，验证 stage 1/2 的提升、未提升、DDP rank-0、打断和恢复。
  - 2026-08-13：实现 action 驱动的 `TwoStageDetectionProtocol`，协议不直接持有 live 训练对象；Trainer 负责 rank-0 原子发布、SHA 广播、事务式完整组件回载与实际 EMA decay 更新。状态持久化 family/stage、global/local metric、restart count、decay、companion 与 GAM weight；epoch-boundary 恢复、双进程 Gloo 和全部负例通过。

### Wave 3：DEIM 与 RT-DETRv4

- [x] 16. 完成 DEIM-D-FINE N/S/M/L/X
  - 五个配置对齐 BN 解冻、SiLU、Dense O2O、FlatCosine、MAL 与 staged EMA。
  - 每变体完成 checkpoint state/activation/output、train/resume、eager、ONNX/TorchScript 与完整 COCO 门。
  - 2026-08-13：N/S/M/L/X 官方 checkpoint 以 identity mapping 严格加载 `674/794/1053/1253/1571` 个 tensor；固定 640 的 stem/backbone/encoder/raw output 对 pinned DEIM 通过 `rtol=1e-5, atol=1e-6`。五变体 reduced optimizer/EMA、epoch-boundary resume、stage-1 回载、四图 eager/parity 均通过。val2017 AP 为 `0.430424 / 0.489613 / 0.526880 / 0.547392 / 0.564731`，相对官方值最大误差 `0.000424`。ONNX opset 17 和 TorchScript 在 batch 1/4 通过，ONNX 最大 score/box 误差 `1.1861e-5 / 0.014901 px`，TorchScript 为零。

- [x] 17. 完成 DEIM-RT-DETRv2 S/M/M*/L/X
  - 固定 S=r18vd、M=r34vd、M*=r50vd_m、L=r50vd、X=r101vd，criterion 只使用 MAL/boxes。
  - 每变体执行同样的 state/activation/output、train/resume、部署与 COCO 门；r50/r50_m 混用必须失败。
  - 2026-08-14：五个官方 checkpoint identity strict-load `540/667/732/801/1107` 个 tensor；固定 640 与四张真实 COCO 图的 raw logits/boxes 对 pinned DEIM 最大绝对误差均为零。S/M/M*/L/X val2017 AP 为 `0.490525 / 0.509376 / 0.531902 / 0.542924 / 0.554852`，最大官方四舍五入误差 `0.000525`。五变体 reduced train/resume、官方 PResNet 初始化、eager/infer、ONNX opset 17 和 TorchScript 全部通过；ONNX 最大 score/box 误差 `3.8812e-4 / 0.078148 px`，TorchScript 为零。

- [x] 18. 增加训练专用 DINOv3 teacher preflight 与 feature boundary
  - 新增 `modeling/teachers/dinov3.py`，固定 `facebookresearch/dinov3@346f38fee679c56a6888f91c51670fae61d364e0`，要求 Python 3.11+、`dinov3_vitb16`、`embed_dim=768`、patch 16 与 `x_norm_patchtokens`；训练前验证 checkout SHA、hub entry、权重实际文件名/size/SHA、model type 和 patch geometry，teacher 固定 eval/no-grad。
  - 权重须由 Meta 门控下载页授权取得；`73cec8be` 只是上游文件名 slug，不当作 SHA-256，Hugging Face safetensors 不得静默替代官方 `.pth`。checkout/权重受自定义 DINOv3 License 约束，不进入 wheel、sdist、仓库或本项目 Release。
  - 删除所有 teacher 资产后，student checkpoint 的 eval/infer/export 仍必须构建和运行。
  - 2026-08-14：训练专用 adapter、Trainer 构建/forward 接缝、验证驱动和 fake local hub 正反测试已完成；Meta 授权 ViT-B/16 `.pth` 与固定 checkout 的真实 `teacher-preflight` 已 APPROVE。DINOv3 依赖隔离在 `teacher` extra，student/core 环境不引入这些依赖。

- [x] 19. 实现 `RTDETRV4(BaseArch)`、DSI 与 GAM
  - DSI 对齐 projected AIFI F5 与 DINOv3 patch features 的 normalized cosine loss。
  - GAM 在 AMP unscale 后、clip 前且仅最终累积 microbatch 观察梯度；跨 rank all-reduce encoder/total L1，AMP skip 不计入，epoch 末由 rank 0 更新并广播 distillation weight，恢复时校验全 rank 一致；不得走旧 `slim_type=Distill`。
  - 2026-08-14：`RTDETRV4` 复用 D-FINE student 图，仅在训练路径解包 projected AIFI F5 并要求 detached teacher feature；`RTDETRV4Criterion` 复用 DEIM 的 MAL/boxes/local，只增加一次主 DSI。GAM 通过两阶段协议观察 unscaled/unclipped 成功 update，按 global raw L1 sums 计算占比，epoch 末广播并写回 criterion；transition/restart/resume 保留且校验当前权重，单 rank AMP 非有限梯度同步为全 rank skip。

- [x] 20. 完成 RT-DETRv4 S/M/L/X
  - 四个配置精确映射 B0/B2/B4/B5、teacher/projector、DSI/GAM、Dense O2O、FlatCosine 和 staged schedule。
  - 每变体完成官方 student checkpoint 对齐、真实 teacher reduced train、stage/GAM 恢复、student-only eager/export、四图对照和 val2017。

### Wave 4：模型目录、打包与文档

- [x] 21. 泛化 manifest 与 Models CLI
  - 增加 `--family {rtdetrv3,dfine,deim-dfine,deim-rtdetrv2,rtdetrv4}`，默认仍为 v3，`--manifest` 优先。
  - schema v2 记录 project/upstream hosting、artifact format 与 source metadata，同时继续读取已发布 schema v1；alias 全局无冲突。

- [x] 22. 扩展 package、CLI、deployment、quality 与 CI 矩阵
  - wheel/sdist 包含全部新 YAML/manifest；六个原命令不改名；CPU CI 使用 core-only extra 做配置、训练、推理、导出 smoke。
  - `uv build`、包外 wheel smoke、Ruff/Mypy、coverage、全部变体部署矩阵通过，不降低阈值或扩大排除。记录已校验 wheel 及 SHA-256 供最终用户验收，其余构建/cache/export 产物立即清理。

- [x] 23. 完成许可证、模型合同、迁移知识与实际验收记录
  - 更新 `NOTICE`、README、ROADMAP、计划/模型/迁移索引，并新增 D-FINE、DEIM、RT-DETRv4 模型合同。
  - 本计划完成时记录真实环境、命令、模型矩阵、数值、偏差、限制、blocked 项与产物清理，不能把 smoke 写成精度证据。

## 风险与回退

- 风险：三个上游虽同源，但同名 decoder、matcher、encoder 或 postprocessor 可能存在语义差异。
  - 缓解：每次共享前执行源码 diff、同 checkpoint/输入的首分歧激活检查；不通过则保留唯一命名的族内实现。
- 风险：DEIM 官方 stage 2 会回载 stage 1，若状态与 companion 不完整会破坏确定性恢复。
  - 缓解：在 format-version-1 中保存协议状态与 companion SHA，并在任何状态修改前验证。
- 风险：RT-DETRv4 上游教师通过外部本地 torch hub 和 Meta 授权权重加载，要求 Python 3.11+，CI 无法携带真实权重，且其自定义许可证不同于本项目 Apache-2.0。
  - 缓解：固定 DINOv3 SHA；CI 用 fake hub 测边界并保持 student Python 3.9-3.12；最终真实训练验收必须使用合法本地资产，否则任务保持 blocked；不 vendor/打包/再分发 DINOv3 代码或权重，并在 NOTICE/模型文档说明许可与致谢边界。
- 风险：19 个完整 val2017 和导出矩阵耗时/磁盘较大。
  - 缓解：按 family/variant 分片、复用只读 checkpoint cache、将证据写到 attempt 目录，并在验收后删除临时导出与 cache；不得减少矩阵或放宽门槛。
- 风险：共享运行时改动导致 v3 数值或 CLI 回归。
  - 缓解：无协议配置走原路径；每波运行 v3 定向回归，最终运行完整非 Paddle、R18 官方数值、wheel 与 eager/export 基线。
- 回退：所有新增路径必须是可删除的 additive 模块/config；若共享基元无法证明等价，回退到唯一命名的 family-local 模块，不回滚或覆盖用户已有修改。

## 验收

### 数值合同

- state tensor 在显式 key adapter 后逐值一致。
- intermediate activations 与 raw logits/boxes：`rtol=1e-5, atol=1e-6`。
- losses/gradients：`rtol=1e-4, atol=1e-6`。
- ONNX/TorchScript：`bbox_num`、labels 和一对一候选匹配严格一致；默认 score `atol=2e-5`、box `atol=0.02 px`。DEIM-RT-DETRv2 使用预注册的 family-specific ONNX 门：S/M/M*/L score 仍为 `2e-5`，X score 为 `4e-4`，五变体 box 为 `0.1 px`；TorchScript 仍要求逐值一致，其他 family 不放宽。
- 每个官方 checkpoint 的完整 val2017 bbox AP 与官方四舍五入值相差不超过 0.001（0-1 标度）。

### 最终验证波

- [x] 最终计划合规审计：在其他最终审计批准后，使用 `tools/dev/audit_plan_evidence.py` 审计完整执行计划与机器收据；normalized plan identity 忽略 checkbox 状态但对其他字节敏感，并通过 wrong-plan-identity fixture 验证失败路径。原始收据在结论提炼后清理。
- [x] 最终质量与数值审计：顺序运行 quality、coverage、非 Paddle unit/integration、全部新增上游数值测试和 graph auditor，使用 `pipefail + tee` 保存完整日志；再用 training-node/opset16/Paddle-import/tolerance fixtures 验证 auditor 的非零失败。
- [x] 最终真实用户验收：核对保留 wheel 的 SHA，使用 `python3.11 -m venv` 在仓库外创建 venv，明确安装 wheel、pytest、onnx、onnxruntime；从源码 checkout 用该 venv 解释器运行非打包 `validate_model_family.py`，并强制 `ppdet_pytorch` 只能解析到 venv site-packages。四族最小变体及全部负例通过后，flush 证据并清理 venv、wheel 和 `dist/`。
- [x] 最终范围与 v3 审计：检查 `git status` 与 Paddle submodule diff，运行完整非 Paddle、R18 官方数值，并对照已批准的 v3 eager/ONNX/TorchScript 基线；使用 submodule-diff 和 baseline-mismatch fixtures 验证失败路径。

先并行运行质量、用户和兼容性审计，三者批准后再串行运行计划合规审计。全部最终审计必须无条件批准；任何 blocked 资产或失败门都使计划保持未完成。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-12 | 一个计划覆盖 D-FINE、DEIM 两分支与 RT-DETRv4 | 共享基础只实现一次，同时让各族独立验收 |
| 2026-08-12 | 固定三个上游 commit SHA | 上游没有覆盖本次状态的稳定语义版本 |
| 2026-08-12 | 保留 `rtdetrv3-*` 命令与包名 | 避免与模型集成无关的破坏性重命名 |
| 2026-08-12 | 保留 checkpoint format v1 与 ONNX opset 17 | 延续现有训练恢复和部署合同 |
| 2026-08-12 | 权重保持上游托管 | 再发布涉及额外许可证、托管与 Release 决策 |
| 2026-08-12 | RT-DETRv4 teacher 仅训练时构造 | 官方 student inference/export 不需要教师依赖 |
| 2026-08-12 | DINOv3 固定 `346f38fee679c56a6888f91c51670fae61d364e0`，作为外部 Python 3.11+ 训练资产 | 与固定 RT-DETRv4 提交同期、包含所需安全加载/API；避免移动 main 和混淆自定义许可证 |

## 完成记录

2026-08-12 已启动 Wave 1。任务 1 的四个验证驱动、确定性 schema、计划身份、preflight、图审计与合成负例已完成首版实现；定向驱动测试 `18 passed`，完整非 Paddle 回归 `383 passed, 2 skipped, 17 deselected`，覆盖率为全包 `52.09%`、直接维护范围 `90.50%`。本机使用 `uv 0.12.1`、Python `3.12.13`、PyTorch `2.5.1+cu121`，`uv lock --check` 通过且锁文件未改变。

任务 1 已于 2026-08-13 完成。`paddlepaddle-gpu==3.3.0` 的 CPython 3.12/cu118 wheel 在仓库外完成 `1,300,069,989` bytes 与 SHA-256 `c2a1f5e05c74776a7780e1c0b6a3692019f769a18f61b152837c2321bc86f6ad` 校验后安装，未绕过 TLS 校验；RTX 4090 上确认 Paddle CUDA `11.8`、cuDNN `8.9.7`、单卡可见且 GPU 矩阵运算通过。官方 R18 checkpoint 再次通过 `91,945,530` bytes 与 SHA-256 `f32dbd008bd7e5311c877d522f6d8c9e349795978c889f53823588b5e5d74a5f` 校验，CPU/FP32 官方 checkpoint 对齐测试 `1 passed`，覆盖 571 个转换权重、前向激活/输出及 384 个梯度合同。仓库全量 quality 当时被 `scripts/render_prediction_comparison.py` 的既有 Ruff format 差异阻塞，最终质量审计已通过。结果已整理到 [RT-DETRv3 验证报告](../models/rtdetrv3/validation-report.md)；临时 checkpoint、wheel 与测试缓存已清理，`.venv` 按仓库规则保留。

任务 2 已于 2026-08-13 完成。新增独立注册的 HGNetv2 B0/B2/B4/B5，保留 D-FINE 的非对称 stem padding、LAB/ESE/SE aggregation、stage 返回、freeze/frozen-BN 与 `ShapeSpec` 合同；显式 `load_pretrained()` 在任何状态变更前拒绝错误变体、key、shape、dtype 与非有限 tensor，不进行隐式下载。固定 `Peterande/D-FINE@267a6da6d04c8ad52e54120692896515b9e55981`，四个官方 stage-1 checkpoint 的 size/SHA-256 均校验通过，并以同一 state、CPU/FP32、seed 0、eval mode 对比 stem、四个 stage 与最终返回特征，数值门 `4 passed`；单元与 R18 定向回归 `28 passed`，完整非 Paddle 回归 `401 passed, 4 skipped, 34 deselected`（四个 skip 仅因该命令未传仓库外资产变量），覆盖率 `52.56%/90.50%`，graph auditor `APPROVE`。资产身份已整理到 [D-FINE 指标记录](../models/dfine/metrics.md)。

任务 3-8 已于 2026-08-13 完成。D-FINE 顶层 transformer、FDR/LQE/分布数学与 family encoder 对固定 D-FINE state、真实 decoder layer、train/eval/deploy 递归输出完成直接数值比较；matcher 与 DN 使用唯一 D-FINE 支持实现，现有 postprocess 经 top-300 直接上游比较后复用。Dense O2O 覆盖全部 epoch 组、普通 D-FINE stop、Mosaic affine 像素/bbox 上游对齐与双 worker 重放。FlatCosine 的 21 点 trace、N 常量特例和 cosine 阶段 checkpoint 恢复通过；训练协议补齐 validation/save/load、成功 update observation、stage/companion 校验与事务回滚。RT-DETRv2 仅暴露五个批准 decoder profile，五组合 state、首层 activation 与输出对固定 DEIM 对齐。带齐仓库外上游与 HGNetv2 资产的非 Paddle 全量回归 `519 passed, 34 deselected`，覆盖率 `56.07%/90.59%`，全 `src/tests` Ruff 通过；稳定结论已整理到模型和迁移文档。

任务 9-11 已于 2026-08-13 完成。D-FINE architecture/criterion、五变体配置与全部官方 checkpoint 均完成固定上游对齐。Manifest 锁定五个 GitHub asset ID、大小与 SHA-256；官方 `{"model": state_dict}` 通过公共评估加载路径，且加载前后均验证 identity key、shape、dtype 和逐 tensor 值。N/S/M/L/X 在 CPU/FP32、固定 640 输入下的 stem/backbone/encoder/raw output 全部 APPROVE，数值测试 `7 passed`，manifest/config/backbone 定向回归 `31 passed`。结果见 [D-FINE 验证报告](../models/dfine/validation-report.md)和[指标记录](../models/dfine/metrics.md)。

任务 15 已于 2026-08-13 完成。通用两阶段协议覆盖 D-FINE、DEIM 与 RT-DETRv4 的 stage-1 best、stop transition、stage-2 global/local improvement 和 no-improvement restart；所有 checkpoint action 由 Trainer 执行，协议保持不可直接修改 live component 的边界。companion 在任何 mutation 前校验 basename/SHA/family/stage/config，回载 model/optimizer/scheduler/scaler/EMA/RNG 但不回退主循环 epoch cursor；EMA restart/decrement 实际参与后续更新，validation snapshot 无副作用。指定 happy/negative QA 为 `12 passed`/`4 passed`，非 Paddle 全量回归 `545 passed, 36 skipped, 34 deselected`；稳定语义已整理到各模型族验证报告。

任务 16 已于 2026-08-13 完成。DEIM-D-FINE N/S/M/L/X 的正式配置、官方 checkpoint manifest、pinned activation/raw-output、reduced train/resume、两阶段 companion、eager 四图、ONNX/TorchScript 和完整 COCO val2017 全部通过。五个 AP 与官方四舍五入值的最大绝对误差为 `0.000424`；部署最大 ONNX score/box 误差为 `1.1861e-5 / 0.014901 px`，TorchScript 为零。结果见 [DEIM 文档的 D-FINE profile](../models/deim/README.md#deim-d-fine)；结论不外推为完整训练 schedule 收敛。

任务 17 已于 2026-08-14 完成。DEIM-RT-DETRv2 S/M/M*/L/X 的正式配置、五个官方 detector checkpoint、四个官方 PResNet-vd 初始化资产、pinned 固定输入/真实图 raw-output、reduced train/resume、stage companion、eager 四图、ONNX/TorchScript 和完整 COCO val2017 均通过。五个 AP 的最大官方四舍五入误差为 `0.000525`；错误 softmax/query TopK 路径曾使 S/M AP 降至 `0.4547 / 0.4805`，当前显式 focal sigmoid/global TopK 已通过上游逐元素测试与完整 AP 门。结果见 [DEIM 文档的 RT-DETRv2 profile](../models/deim/README.md#deim-rt-detrv2)；reduced run 不构成完整 schedule 收敛声明。

任务 18 已于 2026-08-14 完成。`DINOv3TeacherModel` 在 optimizer/EMA 前校验 Python、干净的 `facebookresearch/dinov3@346f38fee679c56a6888f91c51670fae61d364e0` checkout、hub entry、`.pth` 文件名/size/SHA-256、`dinov3_vitb16`、768 channel 与 16x16 patch geometry；教师保持 eval/frozen，并将归一化后 2x 下采样得到的 detached `x_norm_patchtokens` 空间特征仅注入训练 batch。Trainer 不把教师注册到 student model，eval/test 忽略 teacher 配置；DINOv3 官方 requirements 隔离在 `teacher` extra。Meta 授权文件 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` 实测为 `342,860,279` bytes、SHA-256 `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`，真实本地 hub 严格加载后，64x64 固定输入输出 `(1, 768, 2, 2)` 有限 detached feature，S/M/L/X 四项均 APPROVE。fake hub 定向测试 `13 passed`，训练协议/checkpoint/CLI/驱动回归 `54 passed`，非 Paddle 全量 `661 passed, 79 skipped, 34 deselected`，graph auditor `APPROVE`，锁文件、本任务 Ruff 与 Python 3.12 Mypy 通过。结果见 [RT-DETRv4 验证报告](../models/rtdetrv4/validation-report.md)和[指标记录](../models/rtdetrv4/metrics.md)；Torch Hub 重复 cache 已清理，授权原文件仅保存在仓库外。

任务 19 已于 2026-08-14 完成。新增 `RTDETRV4(BaseArch)` adapter 和继承 `DEIMCriterion` 的 `RTDETRV4Criterion`，DSI 对 projected AIFI F5 与 detached DINOv3 patch feature 执行上游一致的 flatten、L2 normalize、cosine mean，并在空间不同时使用 `bilinear/align_corners=False`；eval 不读取或输出 teacher/distill key。GAM 只在最终 accumulation microbatch 的 AMP unscale 后、clip 前观察 AIFI transformer 与全模型 gradient L1 raw sums，成功 step 后才跨 rank SUM 并计入 epoch；任一 rank 非有限梯度会在 step 前同步 skip。epoch 末按官方 `rho/delta/default/current` 公式由 rank 0 更新和广播，criterion 权重在 init/resume/transition/restart 后同步，分歧、缺失及非有限状态均拒绝。合同测试 `14 passed`，engine/model 定向回归 `94 passed`，非 Paddle 全量 `693 passed, 79 skipped, 34 deselected`，双进程 Gloo、graph auditor、Ruff、Python 3.12 Mypy、锁文件与 diff 检查均通过。训练语义与限制已整理到 [RT-DETRv4 验证报告](../models/rtdetrv4/validation-report.md)。

任务 20 已于 2026-08-14 完成。RT-DETRv4 S/M/L/X 官方 solver checkpoint 分别验证 `796/1055/1255/1573` 个 `ema.module` tensor，固定输入 raw output 为零误差，四张真实 COCO 图通过上游对照；完整 val2017 AP 为 `0.498371 / 0.536396 / 0.554134 / 0.570014`，最大官方四舍五入误差 `0.000604`。四变体真实 DINOv3 reduced update、DSI/GAM resume、student-only eager、ONNX opset 17 和 TorchScript 均通过，导出证据均为 `training_residue=false`。统一九阶段驱动、graph audit 与 evidence audit 均为 `APPROVE`；该 reduced run 不构成完整 schedule 收敛声明。

任务 21-22 已于 2026-08-14 完成。Models CLI 保持 schema-v1 RT-DETRv3 默认行为，并增加 `dfine`、`deim-dfine`、`deim-rtdetrv2`、`rtdetrv4` schema-v2 family；23 个 artifact alias 全局唯一，`--manifest` 优先。上游 Google Drive 资产只 list/verify，download 返回准确官方 URL 且不留 partial；D-FINE 固定 GitHub asset 支持 HTTPS 原子下载。真实 wheel 在仓库外安装后加载五族 manifest 和 19 个配置，CLI/deploy/package 回归 `146 passed`；Ruff 全仓通过，Mypy `123 source files` 通过，覆盖率为全包 `59.57%`、直接维护范围 `90.30%`。结构化收据记录 SHA-256 `e53c3fb28bff67e6c369c6c517c92a37d1b559ed7708a3fb9ef1e10ea510cbe0`；NOTICE 更新后的重建日志另记录 `7810ab5327ec0c66921f03d15aa5fe007948da3799e1432e9d03597c59ec0333`，但没有对应的第二份结构化 artifact receipt。产物已清理，后续发布必须重新构建并生成唯一 checksum。

任务 23 已于 2026-08-14 完成。`NOTICE` 分别记录 D-FINE、DEIM、RT-DETRv4 的 Apache-2.0 上游 revision，并将 DINOv3 自定义许可证、Meta 门控授权、acknowledgment 和不打包/不再分发边界独立列出；根 README、路线图、五族模型合同和迁移索引同步到全部实现和验证的当前状态。文档检查器将 manifest 的 23 个 alias、19 个新配置、相对链接、归属、工作站路径和四变体 student-only 图合同绑定，并覆盖 stale variant、缺失归属、绝对路径和 teacher graph contradiction 四类负例。

本计划的执行环境为 Python `3.12.13`、PyTorch `2.5.1+cu121`，student/core 验收覆盖 CPU/FP32，真实 DINOv3 与 Paddle 探针使用 RTX 4090；各数值阶段的上游、checkpoint、COCO、seed、dtype 与容差已整理到正式报告。19 个新变体均完成官方 checkpoint 的完整 val2017 和固定 640 部署矩阵；reduced train/resume 只证明有限更新和确定性恢复，不证明完整 schedule 收敛。DEIM-RT-DETRv2 保留预注册的 family-specific ONNX 门槛，DINOv3 只属于 RT-DETRv4 训练，R34/R50 长训与多 seed 仍按路线图 deferred。全部最终审计均为 `APPROVE`；临时 venv、wheel、`dist/`、导出、重复 checkpoint cache、sdist 和测试 cache 已清理。技术验收已完成，计划保持 `in-progress` 直到维护者明确接受最终结果。

最终机器验收记录绑定基线 revision `41961f796dca06aee47bde01bd41a8ed807635ad` 与 normalized plan identity `60333d67db893e1b12be693d53a3873f7f028878e9f77e7e4aecb34c85613ac5`。这两个值描述验收时的代码和计划身份，不是当前 HEAD；原始机器收据、重复渲染和代理续跑状态在结论进入正式文档后清理。

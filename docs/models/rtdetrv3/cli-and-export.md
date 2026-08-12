# CLI 与导出迁移经验

本文记录 M5 及后续阶段中面向用户入口与部署边界的当前合同。状态结论以活跃测试和实际命令为准，不能由 CLI 能导入或 `--help` 能显示推断端到端可用。

## 六个公开入口的共同边界

**已验证（2026-07-19）**：Train/Eval/Infer/Convert/Export/Models 的 parser 都有活跃 `--help` contract，安装后的六个 console script 均可用；参数解析错误使用 argparse code 2。Train/Eval/Convert 的核心参数和 main wiring 另有定向测试，既有 M2–M4 真实转换、训练和评估证据继续作为端到端依据。Infer 与 Export 的真实 R18 证据见本页对应章节；Models 已完成四个转换产物的本地校验、未发布失败路径和统一下载合同。`v0.1.0` 固定 URL 已进入 manifest，三个检测 asset 均完成公开下载、checksum、Infer 和 Eval 端到端回读。

| 入口 | 当前必需输入 | 已声明支持 | 显式边界 |
|---|---|---|---|
| Train | `-c/--config` | `-o`、resume、seed、AMP、DDP；`--enable_ce` 只保留历史确定性兼容 | `--eval`、slim、TensorBoard、W&B、profiler、proposal/save-prediction 选项会在解析阶段失败；半监督 teacher/student 权重未迁移并会显式失败；训练后评估需单独运行 Eval |
| Eval | config + checkpoint | annotation/image override、batch/worker、持久输出、EMA、device、轻量 override | batch 必须 `>=1`、worker 必须 `>=0`；只声明 COCO 当前数据 API |
| Infer | config + checkpoint/ONNX/TorchScript 三选一 + 单图/目录之一 | 三后端共用 batch、阈值、可视化、JSON、类别映射和 TestReader；checkpoint/TorchScript 支持 PyTorch CUDA/CPU；ONNX 默认 CPU并支持显式 CUDA provider | 三个公开变体的 CUDA 证据只覆盖 FP32/固定 640/Python CLI；R34/R50 ONNX CUDA 不满足 R18 的严格数值门槛；不提供外置 NMS、切片、多尺度或 Paddle `infer_list` 合同 |
| Convert | Paddle checkpoint + PyTorch 输出 | 严格/宽松、目标 config 校验、批量失败隔离、mapping/summary、受控内存 | Paddle 是 dev extra；默认要求目标 config，只有显式 `--no-validate` 才跳过 shape 审核 |
| Export | config + checkpoint | R18/R34/R50 的 ONNX opset 17、ONNX Runtime CPU 回归、traced TorchScript、动态 batch、固定高宽重新导出、EMA | Export 自带验证仍固定 CPU；不声明单个产物动态高宽、TensorRT、C++ 或 Paddle 导出参数兼容 |
| Models | manifest + `list/verify/download` | R18/R34/R50 检测权重与 `r18-backbone` 训练初始化权重；本地 size/SHA-256 校验；HTTPS 临时文件下载、校验后原子替换 | `v0.1.0` 只接受固定 tag 的 GitHub Release URL；三个检测权重均经 CLI 回读，backbone 由 11-asset 整体回读覆盖 |

Train/Eval/Infer 推荐连字符参数；为既有仓库命令保留的下划线别名只覆盖文档列出的参数，不等于完整复刻 Paddle ArgsParser。各入口的 override 语法仍不同，复杂结构应放进派生 YAML。

Models 入口使用包内 `configs/checkpoints/rtdetrv3_coco.yml`，开发仓库中则回退到根目录同一 manifest。每个 `converted_artifact.alias` 是 CLI 的唯一用户别名；重复或非法别名会被 Models CLI 和发布检查共同拒绝。`distribution_status=unpublished` 必须没有 URL；`published` 必须有 HTTPS URL。下载先写目标同目录的 `.part` 临时文件，完成 size/SHA-256 校验后才原子替换目标；不匹配的既有文件默认保留，只有 `--force` 允许替换。

**公开回读（2026-07-19）**：`v0.1.0` 的 11 个 Release asset 已从无认证固定 URL 下载，并同时通过严格目录校验和系统 checksum。Models CLI 另行下载 R18/R34/R50，size/SHA-256 均与 manifest 一致；三个文件都完成 CPU 单图 Infer 和同一四图 COCO Eval 冒烟。四图指标不代表完整 val2017 精度，实际输入和输出规模见[多变体运行时报告](../../archive/rtdetrv3-v0.1.0/reports/variant-runtime-validation.md)。

## Infer 的当前数据流

```text
YAML + overrides
  -> core.workspace.load_config/create
  -> TestReader.sample_transforms
  -> batch dict(image/im_shape/scale_factor/im_id)
  -> checkpoint: RTDETRV3 eager model + built-in DETRPostProcess
     or ONNX/TorchScript: tensor-only exported model
  -> bbox[N, 6] + bbox_num[B]
  -> threshold / visualization / optional JSON
```

**已验证（2026-07-19）**：官方 R18 checkpoint 在 CPU/FP32 上完成真实 COCO 单图和 batch 4 目录推理。单图阈值 `0.3` 生成 30 条 JSON 记录；batch 4 生成 4 张可视化图片。环境、checksum 和命令边界见[M5 计划](../../archive/rtdetrv3-v0.1.0/plans/2026-07-19-m5-cli-export-boundaries.md)。这证明当前 PyTorch CLI 可运行，不新增跨框架数值等价声明。

同日使用 `v0.1.0` 公开 R34/R50 checkpoint 和各自配置补齐相同 CPU/FP32 单图合同，阈值 `0.3` 分别生成 `31/28` 条 JSON 记录和可解码图片；两者还完成统一四图 Eval 链路。该项是 eager 证据，后续 M8 已另行补齐三个变体的导出 tensor 回归，详见[多变体运行时报告](../../archive/rtdetrv3-v0.1.0/reports/variant-runtime-validation.md)和[多变体导出报告](../../archive/rtdetrv3-v0.1.0/reports/variant-export-validation.md)。

M9 进一步把导出 tensor 合同接回同一 Infer 用户链路。官方 R18、COCO `000000000139.jpg`、640×640、CPU/FP32、阈值 `0.3` 下，checkpoint/ONNX/TorchScript 均产生 30 条检测；ONNX 相对 eager 的 score/框最大绝对误差为 `1.49e-6/9.16e-5 px`，TorchScript 为 0，三份可视化文件字节一致。完整命令和限制见[导出产物推理报告](../../archive/rtdetrv3-v0.1.0/reports/exported-inference-validation.md)。

M10 当时将 provider 与 device 合同拆开：ONNX Runtime 只走 CPU provider；TorchScript 使用 PyTorch runtime，在 CUDA 可用时默认 CUDA并接受显式 CPU。R18 四图 batch 4 的 TorchScript CUDA 相对 eager CUDA 最大 score/box 误差为 `2.79218e-4/0.00872803 px`，TorchScript CPU 相对 eager CPU 为 `1.90735e-6/9.15527e-5 px`；两组同设备渲染均逐字节一致。跨 CPU/CUDA 有两条近似候选换序，因此同设备和跨设备证据必须分开记录。详见[TorchScript 设备验证报告](../../archive/rtdetrv3-v0.1.0/reports/torchscript-device-validation.md)。

M11 在不改变默认 CPU 的前提下补充显式 ONNX CUDA provider。R18 四图 batch 4 的 ONNX CUDA 相对 eager CUDA 最大 score/box 误差为 `6.06865e-4/0.0238647 px`，ONNX CPU 相对 eager CPU 为 `6.82473e-6/0.000183105 px`，两组均无候选重排。CUDA 使用独立的 `1e-3/0.03 px` 实测门槛；CPU 保持 M8 的 `2e-5/0.02 px`。完整 provider、TF32 A/B、依赖和可视化证据见[ONNX Runtime 设备验证报告](../../archive/rtdetrv3-v0.1.0/reports/onnx-runtime-device-validation.md)。

M12 把用户侧设备矩阵扩展到 R34/R50。两变体的 TorchScript CUDA/CPU 均与同设备 eager 逐值一致，ONNX CPU 最大误差分别为 `2.38419e-6/0.000183105 px` 和 `3.24845e-6/0.000213623 px`。ONNX CUDA 的功能、检测数和同图同类别一对一匹配成立，但 R34/R50 分别观测到 `0.00141865/0.0375671 px` 和 `0.000972390/0.0349426 px`，未通过 R18 的 `1e-3/0.03 px` 严格门槛。该失败是当前支持边界，不以 `2e-3/0.05 px` 观测包络改写全局合同。详见[多变体设备验证报告](../../archive/rtdetrv3-v0.1.0/reports/variant-export-device-validation.md)。

### 为什么不能保留旧推理链

- 当前模型接收包含 `image`、`im_shape` 和 `scale_factor` 的 batch dict，不接收裸 image tensor。
- `configs/rtdetrv3/_base_/rtdetr_reader.yml` 的 TestReader 使用配置驱动的 Decode、Resize、NormalizeImage 和 Permute。另写 letterbox 与 ImageNet mean/std 会改变输入。
- `RTDETRV3` 已调用 `DETRPostProcess`，返回原图坐标的 `bbox/bbox_num`。CLI 不应再次读取 `pred_logits/pred_boxes`、解码 box 或执行 NMS。
- RT-DETR 当前后处理是配置驱动的 top-k，不暴露外置 `--nms-threshold`。切片推理的合并 NMS 是另一条尚未迁移的合同，不能与普通推理混为一谈。

## 当前 Infer CLI 合同

推荐使用连字符参数；`--infer_img`、`--infer_dir`、`--output_dir`、`--save_results` 和 `--batch_size` 暂作为等价别名保留。

```bash
uv run rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint <R18_CHECKPOINT> \
  --infer-img <COCO_ROOT>/val2017/000000000139.jpg \
  --output-dir output/infer \
  --save-results \
  --device cpu
```

将 `--checkpoint` 分别替换为 `--onnx-model <MODEL.onnx>` 或 `--torchscript-model <MODEL.pt>` 即可复用同一图片和输出参数；ONNX CPU 路径需安装 `export`，CUDA 路径需安装 `export-gpu` 或 `dev`。三个模型源互斥且必须指定一个。

- `--infer-img` 与 `--infer-dir` 互斥且必须指定一个；目录扫描不递归，并按路径排序。
- `--threshold` 同时控制可视化和 `detections.json` 的最小 score；范围是 `[0, 1]`。
- `--batch-size` 现在控制真实 batch，而不是只改变日志或被忽略。
- 模型输出的 `bbox_num` 必须是一维整数、计数非负、总和等于 `bbox` 行数，并且 group 数等于当前输入 batch；任何不一致都会失败，避免目录推理因 `zip` 截断而静默漏图。
- 默认完全使用 TestReader；`--imgsz N` 把其中的第一个 Resize 和模型 `eval_size` 缓存同时覆盖为方形 `[N, N]`，不会另建预处理链。官方 R18 已实际验证 608 和 640；其他尺寸仍需逐一验证。
- `--use-ema` 只接受含 EMA 状态的训练 checkpoint；否则显式失败。
- checkpoint 和 TorchScript 未指定 `--device` 时优先使用可用 CUDA，无 CUDA时回退 CPU；二者都可显式选择 PyTorch device。ONNX 未指定设备时保持 CPU，显式 `cuda[:id]` 需要 GPU ORT；缺少 provider 或 session 完全回退 CPU 时明确失败。导出模型不能使用 `--use-ema`。
- ONNX runner 从图输入读取固定空间 shape；当前 Export 生成的 TorchScript 在归档内嵌 schema v1 `input_size` 元数据。预处理尺寸不一致会在 backend 执行前失败。旧的无元数据 TorchScript 仍可加载，但无法由 CLI 预先证明固定尺寸，建议重新导出。
- `--anno-file` 可为自定义数据集提供类别 ID/名称。未提供且配置的 annotation 不存在时，COCO 配置使用内置 COCO 类别映射。
- JSON bbox 采用 `[x, y, width, height]`，并同时记录输入路径、连续 image ID、数据集 category ID、名称和 score。

## 与 Paddle Infer 的已知差异

当前只声明普通单图/目录检测路径。Paddle 入口中的 `infer_list`、`do_eval`、`slice_infer`、切片合并策略、`visualize=False`、VisualDL 输出和多尺度测试尚未形成 PyTorch 当前合同；使用这些能力前应先增加计划、实现和回归，不能默默忽略同名参数。

## Export CLI 与部署 tensor 合同

ONNX checker 与 CPU ONNX Runtime 属于 `export` 附加依赖；`test` 使用同一 CPU distribution。`export-gpu` 和 `dev` 改用同时包含 CUDA/CPU provider 的 GPU distribution。CPU/GPU ORT extras 由 UV 声明为互斥，核心训练和 eager 推理不导入它们。

```bash
uv sync --extra export
# 或在 CUDA 12/cuDNN 9 环境安装 GPU provider
uv sync --extra export-gpu
uv run --extra export rtdetrv3-export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/rtdetrv3_r18vd_6x_coco.pth \
  --format both \
  --input-size 640 640 \
  --output-dir output/export
```

适配层输入为 FP32 `image[B,3,H,W]`、`im_shape[B,2]` 和 `scale_factor[B,2]`，输出为 `bbox[N,6]` 与 `bbox_num[B]`。它只隔离模型现有 batch dict，不包含图片解码、TestReader、阈值过滤或可视化。使用者必须复用 Infer 的预处理，或者建立等价且单独验证的预处理。

未显式传 `--input-size` 时，配置中的 `TestReader.inputs_def.image_shape` 必须是整数且为正高宽的 `[3, H, W]`；字符串、浮点数、非三通道、零或负高宽在构建模型前失败，不留到 example input 或导出后端产生间接错误。

| 格式 | 已验证行为 | 明确限制 |
|---|---|---|
| ONNX | opset 17；checker 通过；Export 使用 `CPUExecutionProvider` 回归；默认 batch 轴动态，batch 1/4/8 运行；R18/R34/R50 Infer 已验证显式 CUDA/CPU | 高宽固定为导出时数值；`--fixed-batch` 可关闭动态 batch；R34/R50 CUDA 只声明功能矩阵，不满足 R18 严格门槛；不覆盖外部客户端或性能 |
| TorchScript | `torch.jit.trace` 保存、CPU/CUDA 重新加载与回归；同一固定高宽下 batch 1/4/8 运行；R18/R34/R50 Infer 四图 batch 4 已验证 CUDA/CPU；归档内嵌 schema v1 固定输入尺寸元数据供 Infer 预检 | 不是 `script`；高宽被 trace 固化；CUDA 证据只覆盖 FP32 和当前 Python/PyTorch runtime，未声明 C++/TensorRT 合同 |

**已验证（2026-07-19）**：Python `3.12.11`、PyTorch `2.5.1+cu121`、ONNX `1.22.0`、ONNX Runtime `1.27.0`、CPU/FP32 下，官方 R18/R34/R50 的 640 产物完成 batch 1/4/8 和真实 COCO 图片 `000000000139.jpg` 回归；R18 的 608 ONNX/TorchScript 也通过导出时回归。验收要求 `bbox_num` 和每图候选数严格一致，每图全部候选按类别、score 和 box 一对一匹配；score 最大绝对误差 `<=2e-5`，坐标最大绝对误差 `<=0.02 px`。近似并列的低分 top-k 可以交换非语义行序，但不能丢失、跨图匹配或超过数值门槛。

R34 ONNX 在全零 batch 1/4/8 上的最大 score/box 误差为 `2.37e-6/0.01178 px`，真实图为 `9.48e-6/0.00214 px`；真实图有两个低分候选重排。R50 全零输入最高为 `1.90e-5/0.00562 px`，每张图有两个低分候选重排；真实图为 `5.91e-6/0.00461 px` 且不重排。两变体 TorchScript 本次全部逐值为 0。产物摘要和诊断见[多变体导出报告](../../archive/rtdetrv3-v0.1.0/reports/variant-export-validation.md)。

### 动态边界和排错限制

- “动态 batch”不等于“动态高宽”。deformable attention 会把 `value_spatial_shapes` 转成 Python 整数并按层循环，trace/export 警告表明高宽进入常量控制流。当前做法是在构建模型前同步 `cfg.eval_size`，按 608 或 640 等目标尺寸分别导出。
- ONNX exporter 对 advanced indexing 会给出负 index 警告；当前 RT-DETR 路径生成的是非负索引并已通过实际回归，但不能据此外推到自定义负索引路径。
- ORT GPU wheel 必须与 PyTorch 的 CUDA/cuDNN 主版本一致。当前 PyTorch 为 CUDA 12.1/cuDNN 9，因此 GPU extra 限制 `<1.27`；ORT 1.27.0 已实际因依赖 `libcudart.so.13` 无法导入。CPU/GPU distributions 也不能在同一环境共装。
- R34 真实图和 R50 全零输入稳定复现了低分 top-k 近似并列项重排。旧验证器按行比较时会把不同候选框相减，产生几十到上百像素的假误差。当前验证器在同一 image 内按类别、score 和 box 对全部候选做一对一最大匹配，禁止跨图或漏项，并额外报告重排行数；这不是忽略低分 tail。
- 空预测属于 Infer 的阈值过滤层：`threshold=1.0` 已验证可返回 shape 为 `[0,4]` 的 boxes。模型导出层仍返回配置规定的 raw top-k 行，不能声称导出图本身产生空候选。
- Infer 可直接运行 ONNX/TorchScript 并复用当前 TestReader，但这只验证了本仓库 Python CLI；外部 C++/TensorRT/mobile 客户端仍需独立实现并对齐图片解码、预处理、类别映射和阈值。
- CPU trace 的普通 Tensor 属性或显式 `device=src.device` factory 会把 CPU 写入 traced graph。固定位置编码、anchors 和 valid mask 应注册为 `persistent=False` buffer；临时 Tensor 应从活跃输入使用 `new_tensor`/`zeros_like`/`ones_like` 等派生。只验证 CPU reload 不能发现这类错误，必须至少做一次 CPU trace → CUDA load → 前向。
- `--no-verify` 只适合缺少运行后端的受控场景；发布产物应保留默认回归。`--force` 才允许覆盖既有文件。

这些证据覆盖官方 R18/R34/R50 的 CPU/FP32、R18 的 608/640 与 R34/R50 的 640，以及三个变体 TorchScript 与 ONNX Runtime 640 的 CUDA/CPU Infer；不应外推为任意配置、动态高宽、低精度、TensorRT 或跨框架导出等价。R34/R50 ONNX CUDA 的 provider 漂移必须与功能支持分开陈述。

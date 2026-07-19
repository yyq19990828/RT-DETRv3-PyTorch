# CLI 与导出迁移经验

本文记录 M5 中面向用户入口与部署边界的当前合同。状态结论以活跃测试和实际命令为准，不能由 CLI 能导入或 `--help` 能显示推断端到端可用。

## 五个公开入口的共同边界

**已验证（2026-07-19）**：Train/Eval/Infer/Convert/Export 的 parser 都有活跃 `--help` contract，安装后的五个 console script 均可用；参数解析错误使用 argparse code 2。Train/Eval/Convert 的核心参数和 main wiring 另有定向测试，既有 M2–M4 真实转换、训练和评估证据继续作为端到端依据。Infer 与 Export 的真实 R18 证据见本页对应章节。

| 入口 | 当前必需输入 | 已声明支持 | 显式边界 |
|---|---|---|---|
| Train | `-c/--config` | `-o`、resume、seed、AMP、DDP；`--enable_ce` 只保留历史确定性兼容 | `--eval`、slim、TensorBoard、W&B、profiler、proposal/save-prediction 选项会在解析阶段失败；半监督 teacher/student 权重未迁移并会显式失败；训练后评估需单独运行 Eval |
| Eval | config + checkpoint | annotation/image override、batch/worker、持久输出、EMA、device、轻量 override | batch 必须 `>=1`、worker 必须 `>=0`；只声明 COCO 当前数据 API |
| Infer | config + checkpoint + 单图/目录之一 | batch、阈值、可视化、JSON、EMA、device、TestReader Resize override | 不提供外置 NMS、切片、多尺度或 Paddle `infer_list` 合同 |
| Convert | Paddle checkpoint + PyTorch 输出 | 严格/宽松、目标 config 校验、批量失败隔离、mapping/summary、受控内存 | Paddle 是 dev extra；默认要求目标 config，只有显式 `--no-validate` 才跳过 shape 审核 |
| Export | config + checkpoint | ONNX opset 17、ONNX Runtime CPU 回归、traced TorchScript、动态 batch、固定高宽重新导出、EMA | 不声明单个产物动态高宽、GPU provider、TensorRT、C++ 或 Paddle 导出参数兼容 |

Train/Eval/Infer 推荐连字符参数；为既有仓库命令保留的下划线别名只覆盖文档列出的参数，不等于完整复刻 Paddle ArgsParser。各入口的 override 语法仍不同，复杂结构应放进派生 YAML。

## Infer 的当前数据流

```text
YAML + overrides
  -> core.workspace.load_config/create
  -> TestReader.sample_transforms
  -> batch dict(image/im_shape/scale_factor/im_id)
  -> RTDETRV3 eager model
  -> built-in DETRPostProcess
  -> bbox[N, 6] + bbox_num[B]
  -> threshold / visualization / optional JSON
```

**已验证（2026-07-19）**：官方 R18 checkpoint 在 CPU/FP32 上完成真实 COCO 单图和 batch 4 目录推理。单图阈值 `0.3` 生成 30 条 JSON 记录；batch 4 生成 4 张可视化图片。环境、checksum 和命令边界见[M5 计划](../plans/2026-07-19-m5-cli-export-boundaries.md)。这证明当前 PyTorch CLI 可运行，不新增跨框架数值等价声明。

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

- `--infer-img` 与 `--infer-dir` 互斥且必须指定一个；目录扫描不递归，并按路径排序。
- `--threshold` 同时控制可视化和 `detections.json` 的最小 score；范围是 `[0, 1]`。
- `--batch-size` 现在控制真实 batch，而不是只改变日志或被忽略。
- 默认完全使用 TestReader；`--imgsz N` 把其中的第一个 Resize 和模型 `eval_size` 缓存同时覆盖为方形 `[N, N]`，不会另建预处理链。官方 R18 已实际验证 608 和 640；其他尺寸仍需逐一验证。
- `--use-ema` 只接受含 EMA 状态的训练 checkpoint；否则显式失败。
- `--anno-file` 可为自定义数据集提供类别 ID/名称。未提供且配置的 annotation 不存在时，COCO 配置使用内置 COCO 类别映射。
- JSON bbox 采用 `[x, y, width, height]`，并同时记录输入路径、连续 image ID、数据集 category ID、名称和 score。

## 与 Paddle Infer 的已知差异

当前只声明普通单图/目录检测路径。Paddle 入口中的 `infer_list`、`do_eval`、`slice_infer`、切片合并策略、`visualize=False`、VisualDL 输出和多尺度测试尚未形成 PyTorch 当前合同；使用这些能力前应先增加计划、实现和回归，不能默默忽略同名参数。

## Export CLI 与部署 tensor 合同

ONNX checker 与 ONNX Runtime 属于 `export` 附加依赖；`dev` 已包含同一组依赖以运行导出测试。核心训练和 eager 推理不导入它们。

```bash
uv sync --extra export
uv run --extra export rtdetrv3-export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/rtdetrv3_r18vd_6x_coco.pth \
  --format both \
  --input-size 640 640 \
  --output-dir output/export
```

适配层输入为 FP32 `image[B,3,H,W]`、`im_shape[B,2]` 和 `scale_factor[B,2]`，输出为 `bbox[N,6]` 与 `bbox_num[B]`。它只隔离模型现有 batch dict，不包含图片解码、TestReader、阈值过滤或可视化。使用者必须复用 Infer 的预处理，或者建立等价且单独验证的预处理。

| 格式 | 已验证行为 | 明确限制 |
|---|---|---|
| ONNX | opset 17；checker 通过；ONNX Runtime `CPUExecutionProvider` 回归；默认 batch 轴动态，batch 1/4/8 运行 | 高宽固定为导出时数值；`--fixed-batch` 可关闭动态 batch；未验证其他 execution provider |
| TorchScript | `torch.jit.trace` 保存、重新加载和 CPU 回归；同一固定高宽下 batch 1/4/8 运行 | 不是 `script`；高宽被 trace 固化；未声明 C++/GPU 部署合同 |

**已验证（2026-07-19）**：Python `3.12.11`、PyTorch `2.5.1+cu121`、ONNX `1.22.0`、ONNX Runtime `1.27.0`、CPU/FP32、官方 R18 checkpoint 与真实 COCO 图片 `000000000139.jpg` 下，640 产物完成 batch 1/4/8 回归；608 的 ONNX 和 TorchScript 也分别通过导出时回归。验收要求 `bbox_num`、标签和行顺序完全一致，score 最大绝对误差 `<=2e-5`，坐标最大绝对误差 `<=0.01 px`。真实图观测到的 ONNX score/坐标最大绝对误差不超过 `8.76e-6/0.00123 px`，TorchScript 不超过 `5.79e-6/0.000916 px`；全零 640 导出样例的 ONNX 坐标误差最高观测为 `0.00550 px`，因此不能用真实图的更小误差替代公开门槛。

### 动态边界和排错限制

- “动态 batch”不等于“动态高宽”。deformable attention 会把 `value_spatial_shapes` 转成 Python 整数并按层循环，trace/export 警告表明高宽进入常量控制流。当前做法是在构建模型前同步 `cfg.eval_size`，按 608 或 640 等目标尺寸分别导出。
- ONNX exporter 对 advanced indexing 会给出负 index 警告；当前 RT-DETR 路径生成的是非负索引并已通过实际回归，但不能据此外推到自定义负索引路径。
- 一次跨进程 batch 8 诊断曾触发 raw top-k 标签/顺序严格门槛，分别复跑 ONNX 与 TorchScript 时 2,400 行又全部一致，未形成稳定复现。严格检查因此继续保留；若再次出现，应先检查 top-k 尾部近似并列项和第一个分歧 score，而不是直接放宽标签合同。
- 空预测属于 Infer 的阈值过滤层：`threshold=1.0` 已验证可返回 shape 为 `[0,4]` 的 boxes。模型导出层仍返回配置规定的 raw top-k 行，不能声称导出图本身产生空候选。
- `--no-verify` 只适合缺少运行后端的受控场景；发布产物应保留默认回归。`--force` 才允许覆盖既有文件。

这些证据只覆盖官方 R18、CPU/FP32、记录的两个固定尺寸和当前 opset，不应外推为 R34/R50、任意配置、GPU provider 或跨框架导出等价。

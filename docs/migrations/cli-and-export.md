# CLI 与导出迁移经验

本文记录 M5 中面向用户入口与部署边界的当前合同。状态结论以活跃测试和实际命令为准，不能由 CLI 能导入或 `--help` 能显示推断端到端可用。

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
- 默认完全使用 TestReader；`--imgsz N` 只把其中的第一个 Resize 覆盖为方形 `[N, N]`，不会另建预处理链。
- `--use-ema` 只接受含 EMA 状态的训练 checkpoint；否则显式失败。
- `--anno-file` 可为自定义数据集提供类别 ID/名称。未提供且配置的 annotation 不存在时，COCO 配置使用内置 COCO 类别映射。
- JSON bbox 采用 `[x, y, width, height]`，并同时记录输入路径、连续 image ID、数据集 category ID、名称和 score。

## 与 Paddle Infer 的已知差异

当前只声明普通单图/目录检测路径。Paddle 入口中的 `infer_list`、`do_eval`、`slice_infer`、切片合并策略、`visualize=False`、VisualDL 输出和多尺度测试尚未形成 PyTorch 当前合同；使用这些能力前应先增加计划、实现和回归，不能默默忽略同名参数。

## 导出前的约束

- eager CLI 是导出结果的基准，导出适配层应复用同一 TestReader 输入或记录等价的 tensor 预处理。
- ONNX/TorchScript 应使用最小 tensor 输入/输出适配层隔离 Python dict 和可视化逻辑，不应把整个 CLI trace 进去。
- 动态轴必须分别验证 batch、高宽和输出数量；仅导出成功不证明动态 shape 可用。
- 对比 eager/ONNXRuntime/TorchScript 时记录 checkpoint、输入 checksum、eval mode、dtype、device/opset 和绝对/相对容差，并从第一个分歧输出开始定位。
- 空预测指阈值过滤后的零条结果；模型内置 top-k 仍可能固定输出候选行，不能混淆这两个层级。

ONNX、ONNXRuntime、TorchScript 和动态尺寸的实际支持矩阵仍是计划项，完成前不得把本页的 eager 结论外推为部署支持。

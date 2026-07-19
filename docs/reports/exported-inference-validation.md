# 导出产物端到端推理验证报告

- 状态：`verified`
- 验证日期：`2026-07-19`
- 代码基线：`545578a`
- 模型：`v0.1.0` 官方 R18 转换 checkpoint

> 历史快照（2026-07-19，M9）：本文记录 M9 验收时的 CPU 导出后端；后续 TorchScript 与 ONNX 设备合同分别见 [M10](torchscript-device-validation.md) 和 [M11](onnx-runtime-device-validation.md) 验证报告。

## 结论

**已验证**：`rtdetrv3-infer` 现在互斥接受 `--checkpoint`、`--onnx-model` 或 `--torchscript-model`。三种模型源使用同一 config/TestReader、batch、`bbox_num` 分组、阈值、类别映射、JSON 和 OpenCV 可视化代码；ONNX session 和 TorchScript module 各只加载一次后复用于全部 batch。

**真实图片结果**：COCO `000000000139.jpg`、640×640、CPU/FP32、阈值 `0.3` 下，三后端均输出 30 条检测。ONNX 相对 eager 的 score/框最大绝对误差为 `1.49012e-6/9.15527e-5 px`，TorchScript 为 0；标签、顺序和数量一致。三张 JPEG 可视化均为 `640×426×3`，SHA-256 同为 `b7c90a17e4e6b237960218ad4770c2a02679421ceb316fd9838dafed40930e8e`。

**不作声明**：本报告只覆盖 M9 时的 R18、Python Infer CLI、CPU/FP32 和固定 640。它不证明 ONNX Runtime CUDA、TorchScript GPU、动态高宽、TensorRT、C++、量化或外部客户端预处理等价；后续设备支持应以 M10/M11 报告为准。R34/R50 已有 tensor 级导出证据，但尚未重复本轮用户侧三后端命令。

## 环境与输入

| 项目 | 实际值 |
|---|---|
| Python | `3.12.11` |
| PyTorch | `2.5.1+cu121` |
| ONNX / ONNX Runtime | `1.22.0 / 1.27.0` |
| ONNX Runtime provider | `CPUExecutionProvider`；本环境没有 CUDA provider |
| NumPy / OpenCV | `1.26.4 / 4.5.5` |
| 设备 / dtype | CPU / FP32；`CUDA_VISIBLE_DEVICES=''` |
| checkpoint SHA-256 | `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547` |
| 图片 SHA-256 | `ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d` |
| annotation SHA-256 | `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |
| 输入 / 阈值 / batch | `640×640 / 0.3 / 1` |

## 协议

先用当前 Export CLI 在临时目录生成 ONNX opset 17 和 TorchScript：

```bash
CUDA_VISIBLE_DEVICES='' uv run --extra export rtdetrv3-export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth \
  --format both \
  --output-dir <TMP>/export \
  --input-size 640 640 \
  --batch-size 1
```

随后执行三次相同 Infer 命令，只替换互斥模型源和输出目录：

```bash
CUDA_VISIBLE_DEVICES='' uv run rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  <MODEL_SOURCE> \
  --infer-img <COCO_ROOT>/val2017/000000000139.jpg \
  --anno-file <COCO_ROOT>/annotations/instances_val2017.json \
  --output-dir <TMP>/<BACKEND> \
  --save-results \
  --threshold 0.3 \
  --batch-size 1 \
  --imgsz 640 \
  --device cpu
```

`<MODEL_SOURCE>` 分别为 `--checkpoint <CHECKPOINT>`、`--onnx-model <MODEL.onnx>` 和 `--torchscript-model <MODEL.pt>`。JSON 的 COCO `[x,y,w,h]` 恢复为 `[x1,y1,x2,y2]` 后，使用 M8 的每图一对一类别/score/box 合同比较。

## 结果

| 后端 | 阈值后检测 | score 最大绝对误差 | box 最大绝对误差 | 顺序 | JSON / 可视化 |
|---|---:|---:|---:|---|---|
| checkpoint eager | `30` | reference | reference | reference | 成功 / 可解码 |
| ONNX Runtime CPU | `30` | `1.49012e-6` | `9.15527e-5 px` | 完全一致 | 成功 / 可解码 |
| TorchScript CPU | `30` | `0` | `0 px` | 完全一致 | 成功 / 可解码 |

本次临时 ONNX 为 `81,570,986` 字节、SHA-256 `4b2bd9043ceb6be7f35d6314677c398df8edc3bc8ad0a96f1ac8e10ef4102c90`；嵌入元数据后的 TorchScript 为 `93,499,457` 字节、SHA-256 `b8356dc7fdc0cc611f24990d5df20d70ee38f7f73b0cb7c55bc7bd7c92d8521b`。这些不是 Release assets，也不承诺跨环境按字节可重现。

## 参数与固定尺寸边界

- 三个模型源由 argparse 互斥组约束，缺失或重复会以 code 2 失败。
- checkpoint 保持既有设备默认值并可选 EMA；M9 验收时 ONNX/TorchScript 默认且只接受 CPU，任何 `--use-ema` 或显式非 CPU 组合在模型加载前失败。
- ONNX runner 从 `image` 输入读取固定高宽。新 TorchScript 导出在归档内嵌 `rtdetrv3-export.json`，当前 schema v1 为 `{"input_size":[640,640],"schema_version":1}`；旧无元数据归档仍可加载。
- 将两个 640 产物配合 `--imgsz 608` 时，均在 backend 执行前明确报告 `expects fixed spatial size 640x640, got 608x608`，且没有创建输出目录。

## 测试、清理与剩余项

参数、runner、session 复用、固定尺寸和既有 checkpoint 主流程的定向回归为 `46 passed`。显式隐藏 GPU 的本地非 Paddle 全仓为 `350 passed, 5 skipped, 34 deselected`；全包/直接维护范围覆盖率为 `7,059/13,730 (51.41%)` 和 `1,977/2,184 (90.52%)`，通过 `50.5%/90%` 门槛；Ruff `174` 个文件、Mypy `107` 个 source file 通过。

提交 `545578a` 的 [GitHub Actions run 29689593612](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29689593612) 六个 job 全部通过。托管 Python 3.12 为 `350 passed, 7 skipped, 17 deselected`，全包/直接维护范围为 `7,063/13,735 (51.42%)` 和 `1,980/2,189 (90.45%)`；Python 3.9–3.11、Ruff/Mypy、构建/发布校验、包外配置和 `59 passed` wheel smoke 同时通过。临时 ONNX、TorchScript、JSON、图片、负例日志、coverage 和 pytest 目录已清理，UV `.venv` 保留。

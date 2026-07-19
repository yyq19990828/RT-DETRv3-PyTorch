# R34/R50 多变体导出验证报告

- 状态：`R34/R50 ONNX and TorchScript verified`
- 验证日期：`2026-07-19`
- 代码基线：`de5a805` 后的 M8 工作树；实现与本报告在同一提交固化
- 输入权重：`v0.1.0` R34/R50 已发布 checkpoint

> 历史快照（2026-07-19，M8）：本文记录 CPU 导出 tensor 合同；后续 CUDA/CPU 用户侧矩阵及 ONNX CUDA 数值偏差见 [M12 验证报告](variant-export-device-validation.md)。

## 结论

**已验证**：R34 和 R50 均能从对应 640×640 COCO 配置导出 ONNX opset 17 与 traced TorchScript。两个 ONNX 均通过 checker，确认 batch 轴动态、空间轴固定为 640，并在 ONNX Runtime `CPUExecutionProvider` 下完成 batch 1/4/8 与真实 COCO 图片回归；两个 TorchScript 均完成保存、重载和相同输入矩阵回归。

**验证合同**：`bbox` shape 和 `bbox_num` 必须一致，所有值必须有限；每张图的全部候选必须按类别、score 和 box 在一对一匹配中全部配对，不允许跨 image 匹配或忽略低分 tail。score 最大绝对误差不超过 `2e-5`，坐标最大绝对误差不超过 `0.02 px`。近似并列候选的行序不是跨后端语义保证，验证器单独报告 `order_equal/reordered_detections`。

**不作声明**：这些结果只覆盖 M8 时的 CPU/FP32、固定 640×640、当前 opset 和当前依赖。M8 本身不证明单产物动态高宽、ONNX Runtime CUDA provider、TensorRT、C++、FP16/BF16、量化或 Paddle 导出等价；后续 CUDA/CPU 功能证据以 M12 报告为准。

## 环境与输入

| 项目 | 实际值 |
|---|---|
| Python | `3.12.11` |
| PyTorch | `2.5.1+cu121` |
| ONNX | `1.22.0` |
| ONNX Runtime | `1.27.0` / `CPUExecutionProvider` |
| 设备 / dtype | CPU / FP32；`CUDA_VISIBLE_DEVICES=''` |
| 导出尺寸 / opset | `640×640` / `17` |
| 动态范围 | batch 动态；高宽固定 |
| 合成输入 batch | `1, 4, 8`；每图 300 个 raw top-k 候选 |
| 真实图片 | COCO `val2017/000000000139.jpg`，SHA-256 `ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d` |
| 线程 | `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` |

checkpoint 机器可读真值来自 `configs/checkpoints/rtdetrv3_coco.yml`：

- R34：`137,170,947` 字节，SHA-256 `e69207749b37e493596086579f435d5f08e9f058b66322452456053b78a4f272`
- R50：`182,510,207` 字节，SHA-256 `5e3e34ac3d3d14f57ebf6100b146b5702f8dface24fbe57cbc993f59381b67f7`

## 命令与扩展回归

两个变体先运行真实 CLI；`<CONFIG>`、`<CHECKPOINT>` 和输出前缀分别替换为 R34/R50：

```bash
CUDA_VISIBLE_DEVICES='' uv run rtdetrv3-export \
  -c <CONFIG> \
  --checkpoint <CHECKPOINT> \
  --format both \
  --output-dir <TMP>/<VARIANT> \
  --input-size 640 640 \
  --batch-size 1 \
  --opset-version 17
```

CLI 默认用确定性全零 batch 1 建立 eager reference，并对两个后端调用同一 `validate_detection_outputs()`。随后在独立进程中：

1. 对 ONNX 运行 checker，并检查 image 输入的 batch 维是符号维、高宽是数值 `640/640`。
2. 对 TorchScript 重新 `torch.jit.load()`。
3. 分别创建 batch 1/4/8 的 640×640 全零输入，与同一 eager adapter 比较。
4. 复用 Infer 的 TestReader `Compose/BatchCompose` 处理真实 COCO 图片，再比较 eager/ONNX/TorchScript。

## 产物快照

| 变体 | 格式 | 大小 | 本次 SHA-256 |
|---|---|---:|---|
| R34 | ONNX | `126,656,358` | `e675eb5ebd6810cd5a09ae0c1b0f87e5cef006e58417adcd22d713a535d9eb5e` |
| R34 | TorchScript | `138,740,083` | `50d7fc7e5d86bbc17e399e8282e1546d5c1455ce19268d4ff6da134f2aae169c` |
| R50 | ONNX | `172,838,704` | `7191997921fba1640aefbbf41b67bec8816d44564892aa22867f92f2aa527984` |
| R50 | TorchScript | `184,230,529` | `a5cac3784ce82f74809ffc88c191598d658a2841f1fdfe3cd22f5f8a1ff4d1cc` |

这些是本机临时验收产物，不是 `v0.1.0` Release asset，也不承诺跨 PyTorch/归档版本按字节可重现；验证后已删除。

## 数值结果

### ONNX Runtime

| 变体 / 输入 | 候选数 | score 最大绝对误差 | box 最大绝对误差 | 重排数 |
|---|---:|---:|---:|---:|
| R34 / zero batch 1 | `300` | `2.3693e-6` | `0.011780 px` | `0` |
| R34 / zero batch 4 | `1,200` | `2.3693e-6` | `0.011780 px` | `0` |
| R34 / zero batch 8 | `2,400` | `2.3693e-6` | `0.011780 px` | `0` |
| R34 / real image | `300` | `9.4771e-6` | `0.002136 px` | `2` |
| R50 / zero batch 1 | `300` | `1.8068e-5` | `0.005615 px` | `2` |
| R50 / zero batch 4 | `1,200` | `1.8962e-5` | `0.005493 px` | `8` |
| R50 / zero batch 8 | `2,400` | `1.8962e-5` | `0.005493 px` | `16` |
| R50 / real image | `300` | `5.9083e-6` | `0.004608 px` | `0` |

重排数随相同全零图片的 batch 数线性增加：R50 每张图有两个近似并列的低分候选交换顺序，并不表示候选丢失。R34 真实图同样只有两个低分候选重排；每张图内全部 300 个候选仍在明确容差内一对一匹配。三个已发布模型在真实图 score `>=0.3` 的用户可见候选均完整匹配，详细 eager 数量见[多变体运行时报告](variant-runtime-validation.md)。

### TorchScript

R34/R50 的 zero batch 1/4/8 和真实图片在本次环境中均为 score/box 最大绝对误差 `0`，没有候选重排。该观测不应外推成所有硬件或未来 PyTorch 版本都逐位一致。

## 验证器缺陷与修复

初始 `validate_detection_outputs()` 按行要求标签相等，再按行计算框误差。R34 真实图的两个 tail 候选重排时，按行比较把不同候选的坐标相减，产生 `133.05 px` 的假性大误差；R50 全零输入同样产生 `32.84 px`。关闭 ONNX Runtime 图优化没有稳定消除重排，因此问题不是单一优化开关。

进一步对全部 300 个候选做每图一对一诊断：R34 真实图和 R50 全零输入在原 `0.01 px` 内全部匹配；R34 全零输入为 299/300，最后一个实际误差为 `0.0117798 px`，在 `0.02 px` 内达到 300/300。因此验证器改为：

- 先严格检查 shape、`bbox_num`、分组行数和有限值。
- 只在同一 image 内建立类别相同、score `<=2e-5`、box `<=0.02 px` 的候选边。
- 使用一对一最大匹配要求全部候选被唯一配对；跨 image、候选缺失、类别变化或超容差仍失败。
- 对配对后的候选计算真实最大误差，并额外报告重排行数。

该修复没有过滤低分候选，也没有把 300 个 raw 输出缩成阈值后的子集；它只移除了“近似并列检测必须跨后端保持同一行序”这一不成立的假设。新增回归同时覆盖合法重排、禁止跨 image 匹配和超坐标容差失败。

## 警告、门禁与清理

每个变体导出时观测到 9 条既有 TracerWarning 和 1 条 advanced-indexing UserWarning，来源与 R18 一致：空间 shape 转 Python 值、按层迭代以及 ONNX opset 17 对 advanced indexing 的组合导出。当前路径的索引非负，产物已通过实际回归；这些警告继续支持“高宽固定”的限制，不能静默解释为动态高宽。

修复后导出定向测试为 `19 passed`；显式隐藏 GPU 的非 Paddle 全仓回归为 `343 passed, 5 skipped, 34 deselected`；Ruff `174` 个文件、Mypy `107` 个 source file 通过。R34/R50 ONNX、TorchScript、诊断日志和临时测试目录均已清理，UV `.venv` 保留。

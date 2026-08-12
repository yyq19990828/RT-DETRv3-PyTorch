# R34/R50 导出后端设备矩阵验证报告

> 历史报告快照（2026-07-19，M12）：本文保存已完成验证记录，不代表当前仓库状态。当前合同见 [`docs/models/rtdetrv3`](../../../models/rtdetrv3/README.md)。

- 状态：`verified with documented ONNX CUDA tolerance deviation`
- 验证日期：`2026-07-19`
- 代码基线：`0067cfa`
- 输入权重：`v0.1.0` R34/R50 转换 checkpoint

## 结论

**运行路径已验证**：R34 和 R50 均从各自 checkpoint 重建固定 640、动态 batch 的 ONNX 与 TorchScript，并完成 eager、ONNX、TorchScript × CUDA、CPU 六条 Infer 路径。每条路径都处理同一四张 COCO 图片和 batch 4，R34 每图检测数为 `[31, 1, 28, 4]`，R50 为 `[28, 1, 31, 3]`；12 份 JSON 分别保留 `64/63` 条阈值后检测，48 张渲染图全部可解码且尺寸与输入一致。

**严格数值结论必须拆开**：TorchScript 在 R34/R50 的 CUDA 和 CPU 上均与同设备 eager 逐值一致；ONNX CPU 也通过 M8 的 `2e-5/0.02 px` 门槛。ONNX CUDA 的全部阈值后候选仍能同图、同类别一对一匹配，但 R34/R50 没有通过 M11 为 R18 记录的 `1e-3/0.03 px` 门槛：

| 变体 | 后端 / 设备 | 检测 | score 最大绝对误差 | box 最大绝对误差 | 重排 | R18 CUDA 门槛 |
|---|---|---:|---:|---:|---:|---|
| R34 | ONNX / CUDA | `64` | `0.00141865` | `0.0375671 px` | `2` | 未通过 score/box |
| R34 | TorchScript / CUDA | `64` | `0` | `0 px` | `0` | 通过 |
| R34 | ONNX / CPU | `64` | `2.38419e-6` | `0.000183105 px` | `0` | CPU 门槛通过 |
| R34 | TorchScript / CPU | `64` | `0` | `0 px` | `0` | CPU 门槛通过 |
| R50 | ONNX / CUDA | `63` | `0.000972390` | `0.0349426 px` | `0` | 未通过 box |
| R50 | TorchScript / CUDA | `63` | `0` | `0 px` | `0` | 通过 |
| R50 | ONNX / CPU | `63` | `3.24845e-6` | `0.000213623 px` | `0` | CPU 门槛通过 |
| R50 | TorchScript / CPU | `63` | `0` | `0 px` | `0` | CPU 门槛通过 |

因此本报告声明 R34/R50 两种导出格式的 CUDA/CPU **功能矩阵可运行**，并声明 TorchScript 与 ONNX CPU 的既有数值合同；不声明 R34/R50 ONNX CUDA 满足 R18 的严格数值门槛。`2e-3/0.05 px` 只是包住本次默认 provider 观测的诊断范围，不是新的全局发布容差。

## 环境与输入

| 项目 | 实际值 |
|---|---|
| Python | `3.12.11` |
| PyTorch | `2.5.1+cu121` |
| ONNX / ONNX Runtime GPU | `1.22.0 / 1.23.2` |
| CUDA / cuDNN | `12.1 / 91300` |
| GPU / driver | `2 × NVIDIA GeForce RTX 3090, 24 GiB / 595.71.05` |
| ONNX CUDA provider | `CUDAExecutionProvider,CPUExecutionProvider`；`device_id=0`、`use_tf32=1` |
| ONNX CPU provider | `CPUExecutionProvider` |
| 输入 / 阈值 / batch | `640×640 / FP32 / 0.3 / 4` |
| annotation SHA-256 | `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |

权重沿用 manifest 真值：

| 变体 | checkpoint 大小 | SHA-256 |
|---|---:|---|
| R34 | `137,170,947` | `e69207749b37e493596086579f435d5f08e9f058b66322452456053b78a4f272` |
| R50 | `182,510,207` | `5e3e34ac3d3d14f57ebf6100b146b5702f8dface24fbe57cbc993f59381b67f7` |

四图 image ID 为 `139/285/632/724`，SHA-256 依次为：

- `ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d`
- `f3a2974ce3686332609124c70e3e6a2e3aca43fccf1cd1bd7c5c03820977f57d`
- `a4cd7f45ac1ce27eaafb254b23af7c0b18a064be08870ceaaf03b2147f2ce550`
- `5c0e559c75d3969c8e3e297b61f61063f78045c9d4802b526ba616361f3823fd`

## 导出与产物

两个变体分别执行同一命令；`<VARIANT>` 替换为 `r34` 或 `r50`：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync rtdetrv3-export \
  -c configs/rtdetrv3/rtdetrv3_<VARIANT>vd_6x_coco.yml \
  --checkpoint pretrained_models/pytorch/rtdetrv3_<VARIANT>vd_6x_coco.pth \
  --format both --output-dir <TMP>/export/<VARIANT> \
  --input-size 640 640 --batch-size 4 --opset-version 17
```

两个 ONNX 均通过 checker，输入 shape 为 `image=['batch',3,640,640]`、`im_shape/scale_factor=['batch',2]`。两个 TorchScript 归档的 schema v1 元数据均为 `input_size=[640,640]`，映射到 CUDA 后参数和 buffer 设备均回读为 `cuda:0`。

| 变体 | 格式 | 大小 | 本次 SHA-256 | CPU raw batch 4 回归 |
|---|---|---:|---|---|
| R34 | ONNX | `126,654,755` | `29b625be67219a29429b1803585369847974b600482bd037e30cf9c3b9e13d7d` | `1.33365e-6/0.00457764 px`，无重排 |
| R34 | TorchScript | `138,740,728` | `9e266a82efdecb3be810667f9443516cd0331d6c6ff7a0f4358c1d2660f616c1` | 逐值为 0 |
| R50 | ONNX | `172,837,101` | `946e59f6b7f1ce31845dc2e0d134ca5443cd4d6580f5240733cd4017d2a583c9` | `1.65477e-5/0.00582886 px`，重排 `8/1200` |
| R50 | TorchScript | `184,230,982` | `c11ab35dc856390aa26ded9db497a150b68802e49c832eacd456f6b0c875dd70` | 逐值为 0 |

这些是临时验证产物，不是 Release assets，也不承诺未来导出按字节复现。

## Infer 与比较协议

保持 config、四图目录、annotation、batch、阈值和尺寸不变，只替换模型源与设备：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_<VARIANT>vd_6x_coco.yml \
  <CHECKPOINT_OR_EXPORTED_MODEL> \
  --infer-dir <COCO_FOUR_IMAGES> \
  --anno-file <COCO_ROOT>/annotations/instances_val2017.json \
  --batch-size 4 --threshold 0.3 --imgsz 640 --save-results \
  --device cuda:0 --output-dir <TMP>/<VARIANT>-<BACKEND>-cuda
```

CPU 对照只把 `CUDA_VISIBLE_DEVICES` 置空并把 `--device` 改为 `cpu`。JSON 的 COCO `xywh` 先恢复为 xyxy，再按 image、同类别、score 和 box 做全部阈值后候选一对一匹配；CPU 与 CUDA 各自使用同设备 eager 参考。匹配不丢弃低分项，也不允许跨图配对。

## Provider A/B 诊断

预注册的 R18 ONNX CUDA 门槛失败后，诊断只改变 provider 选项，不改 checkpoint、输入或 eager 参考：

- 默认 `use_tf32=1`：R34 为 `0.00141865/0.0375671 px`、2 条重排；R50 为 `0.000972390/0.0349426 px`、无重排。
- `use_tf32=0`：R34 的 score 缩小到 `0.000851989`，但 box 扩大到 `0.0579224 px` 且重排增至 4；R50 有一个阈值后候选即使在 `0.01/0.2 px` 诊断范围内也无法与默认 eager 一对一匹配。关闭 TF32 不是更可靠的默认值。
- `cudnn_conv_algo_search=HEURISTIC` 与默认 `EXHAUSTIVE` 数值相同。`DEFAULT` 让大量卷积进入 fallback，R34 仍超出 box 门槛，R50 再次出现不可匹配候选，因此不应为追求数值接近而切换算法。

TorchScript CUDA 对两变体逐值为 0，ONNX CPU 只有约 `3e-6/0.00022 px`，所以这些观测把额外漂移隔离到当前 ORT CUDA provider/算法组合，而不是 checkpoint 映射、TestReader、JSON 分组或通用导出图合同。这里是基于对照的归因，不是逐节点 activation 等价证明。

## 可视化

| 变体 | 候选相对同设备 eager | 逐字节一致图片 | 最大像素通道差 | 变化通道值 |
|---|---|---:|---:|---:|
| R34 | ONNX CUDA | `2/4` | `243` | `3,626/3,432,900` |
| R50 | ONNX CUDA | `3/4` | `189` | `8,217/3,432,900` |
| R34/R50 | TorchScript CUDA | `8/8` | `0` | `0` |
| R34/R50 | ONNX CPU | `8/8` | `0` | `0` |
| R34/R50 | TorchScript CPU | `8/8` | `0` | `0` |

ONNX CUDA 的非零像素差来自亚像素框变化经过整数取整、绘制与 JPEG 编码后的局部差异；最大通道差不能解释为整图视觉误差。R34/R50 分别只有约 `0.106%/0.239%` 的解码后通道值变化，图片 shape、检测数、类别和一对一语义候选保持一致。因此本报告不把“可解码”伪装成像素等价，也不把局部像素差写成检测语义变化。

## 本地门禁与剩余边界

- 两变体导出、checker/reload、12 条 Infer 路径、8 组同设备 JSON 比较和 48 张图片解码均已完成。
- 隐藏 GPU 的非 Paddle 全仓为 `358 passed, 7 skipped, 34 deselected`；全包覆盖率 `7,078/13,748 (51.48%)`，直接维护范围 `1,991/2,200 (90.50%)`，通过 `50.5%/90%` 门槛。
- Ruff format/lint：`174` 个文件通过；Mypy：`107` 个 source file 通过；wheel/sdist 构建和发布内容检查通过。
- 临时导出约 594 MiB、四图副本、12 组 Infer 图片/JSON、日志、临时 distribution 与测试缓存均已清理；UV `.venv` 保留 GPU `dev` 环境。
- 证据提交 `fc3a6f8` 的 [GitHub Actions run 29693029694](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29693029694) 六个 job 全部通过。Python 3.9–3.12 均为 `358 passed, 9 skipped, 17 deselected`；Python 3.12 全包/直接维护范围覆盖率为 `7,079/13,748 (51.49%)` 和 `1,991/2,200 (90.50%)`。托管 Ruff `174` 个文件、Mypy `107` 个 source file、wheel/sdist 发布检查、六个安装后 CLI、包外配置加载和 `65 passed` wheel smoke 同时通过。

本报告只声明 R34/R50、当前 Python Infer CLI、固定 640、FP32、ORT 1.23.2 和本机 CUDA 12.1/cuDNN 9。它不证明动态高宽、低精度、I/O Binding、TensorRT、外部客户端预处理、性能收益或其他 GPU/driver/provider 组合。特别是 R34/R50 ONNX CUDA 不应使用 R18 的严格门槛作已通过声明。

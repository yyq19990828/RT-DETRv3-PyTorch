# 公开模型多变体运行时验证报告

- 状态：`R18/R34/R50 public eager runtime verified`
- 验证日期：`2026-07-19`
- 代码基线：`72fd9ef9844689e49d35949b9cc37ae5a4ae12f5`
- 发布版本：[`v0.1.0`](https://github.com/yyq19990828/RT-DETRv3-PyTorch/releases/tag/v0.1.0)

> 历史快照（2026-07-19，M7）：本文记录公开 checkpoint 的 eager CPU 验收；后续导出与设备矩阵见 [M8](variant-export-validation.md) 和 [M12](variant-export-device-validation.md) 报告。

## 结论

**已验证**：`v0.1.0` 的 R18、R34、R50 三个检测权重都能通过 `rtdetrv3-models download` 从 manifest 固定 URL 下载并通过 size/SHA-256 校验。每个权重使用对应配置在 CPU/FP32 下严格加载，完成同一真实 COCO 图片的 Infer，以及同一四图 COCO 子集的 Eval。三个 Infer 都生成可解码可视化和非空 JSON，三个 Eval 都处理四张图并写出 1,200 条 raw top-k 候选。

**不作声明**：四图子集只验证配置、checkpoint、预处理、前向、后处理和 COCO metric 链路，样本量不足以形成正式 AP。M7 本身不证明 R34/R50 完整 val2017 AP、标准训练收敛、GPU eager、ONNX/TorchScript、TensorRT 或 LVIS 支持；后续导出设备证据应以 M8/M12 报告为准，跨框架数值证据仍以[预测可视化报告](prediction-visualization.md)和既有分层对齐报告为准。

## 环境与输入

| 项目 | 实际值 |
|---|---|
| Python | `3.12.11` |
| PyTorch | `2.5.1+cu121` |
| pycocotools | `2.0.10` |
| Pillow | `12.0.0` |
| NumPy | `1.26.4` |
| 设备 / dtype | CPU / FP32；`CUDA_VISIBLE_DEVICES=''` |
| Eval batch / workers | `2 / 0`，共两个 batch |
| Infer threshold / batch | `0.3 / 1` |
| COCO annotation | `instances_val2017.json`，SHA-256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |
| 四图 image ID | `139, 285, 632, 724`；共 43 条 annotation |

四张图片的 SHA-256 依次为：

- `000000000139.jpg`: `ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d`
- `000000000285.jpg`: `f3a2974ce3686332609124c70e3e6a2e3aca43fccf1cd1bd7c5c03820977f57d`
- `000000000632.jpg`: `a4cd7f45ac1ce27eaafb254b23af7c0b18a064be08870ceaaf03b2147f2ce550`
- `000000000724.jpg`: `5c0e559c75d3969c8e3e297b61f61063f78045c9d4802b526ba616361f3823fd`

## 协议

R34/R50 在空临时目录中分别执行以下协议；`<ALIAS>` 为 `r34` 或 `r50`，`<CONFIG>` 为对应 COCO YAML。R18 在 M6 发布回读中按相同协议完成。

```bash
CUDA_VISIBLE_DEVICES='' uv run rtdetrv3-models download <ALIAS> \
  --output <TMP>/<ALIAS>-public.pth

CUDA_VISIBLE_DEVICES='' uv run rtdetrv3-infer \
  -c <CONFIG> \
  --checkpoint <TMP>/<ALIAS>-public.pth \
  --infer-img <COCO_ROOT>/val2017/000000000139.jpg \
  --anno-file <TMP>/instances_val2017-subset.json \
  --output-dir <TMP>/<ALIAS>-infer \
  --save-results \
  --threshold 0.3 \
  --batch-size 1 \
  --device cpu

CUDA_VISIBLE_DEVICES='' uv run rtdetrv3-eval \
  -c <CONFIG> \
  --checkpoint <TMP>/<ALIAS>-public.pth \
  --anno-file <TMP>/instances_val2017-subset.json \
  --image-dir <TMP>/val2017 \
  --batch-size 2 \
  --num-workers 0 \
  --output-dir <TMP>/<ALIAS>-eval \
  --device cpu
```

子集 JSON 从上述完整 annotation 中保留 `info/licenses/categories`、四条 image 记录和 image ID 对应的 43 条 annotation；图片是原文件副本，没有重新编码。

## 实际结果

| 变体 | 公开权重大小 | SHA-256 | 单图检测数 | 可视化 | Eval 图数 / 候选数 |
|---|---:|---|---:|---|---:|
| R18 | `92,075,629` | `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547` | `30` | 可解码 | `4 / 1,200` |
| R34 | `137,170,947` | `e69207749b37e493596086579f435d5f08e9f058b66322452456053b78a4f272` | `31` | 可解码 | `4 / 1,200` |
| R50 | `182,510,207` | `5e3e34ac3d3d14f57ebf6100b146b5702f8dface24fbe57cbc993f59381b67f7` | `28` | 可解码 | `4 / 1,200` |

三个 checkpoint 都没有未知 missing/unexpected key。`aux_o2m_head.anchor_points` 和 `aux_o2m_head.stride_tensor` 是按既有加载合同重新生成的派生 buffer，不属于 checkpoint 不兼容。

**观测**：R34/R50 的单图检测数分别与既有同图跨框架可视化证据中的 `31/28` 一致。这是协议一致性的旁证，不替代逐预测数值报告。

## 清理与剩余边界

公开下载的 R34/R50 checkpoint、四图子集、Infer 图片/JSON、Eval `bbox.json` 和临时日志均在验收后删除；仓库保留 UV `.venv`，没有保留新的模型副本或测试缓存。

后续若扩展部署矩阵，应对 R34/R50 分别建立 ONNX/TorchScript 的固定高宽、动态 batch 和运行时误差证据；在此之前，导出支持范围仍只声明 R18。R18-vd backbone 是训练初始化权重，不适用检测 Infer/Eval 协议，其公开文件只由 Release 整体 checksum 回读覆盖。

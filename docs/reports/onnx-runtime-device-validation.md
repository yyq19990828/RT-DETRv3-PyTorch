# ONNX Runtime CUDA/CPU 推理验证报告

- 状态：`verified`
- 验证日期：`2026-07-19`
- 实现提交：`dc97927`
- 模型：`v0.1.0` 官方 R18 转换 checkpoint

## 结论

**已验证**：`rtdetrv3-infer --onnx-model` 保持默认 `--device cpu`，并可通过显式 `--device cuda[:id]` 使用 `CUDAExecutionProvider`。CUDA session 按优先级注册 CUDA、CPU 两个 provider；后者只为 CUDA 不支持的节点回退。若 GPU wheel 未安装，或 session 创建后完全落回 CPU，CLI 会明确失败而不把 CPU 结果伪装成 CUDA 结果。

官方 R18、固定 640×640、四张真实 COCO 图片、batch 4、FP32、阈值 `0.3` 下，eager CUDA、ONNX CUDA、eager CPU 和 ONNX CPU 的每图检测数均为 `[30, 1, 25, 2]`。同设备比较结果如下：

| 参考 | 候选 | score 最大绝对误差 | box 最大绝对误差 | 重排 | 验收门槛 |
|---|---|---:|---:|---:|---|
| eager CUDA | ONNX CUDA | `6.06865e-4` | `0.0238647 px` | `0/58` | `1e-3 / 0.03 px` |
| eager CPU | ONNX CPU | `6.82473e-6` | `0.000183105 px` | `0/58` | `2e-5 / 0.02 px` |

CUDA 门槛是本阶段基于默认 TF32 模式单独记录的合同，不覆盖 M8 的 CPU 导出门槛。ONNX CUDA 与 ONNX CPU 跨设备比较为 `0.00165835/0.0289612 px`，有两条近似候选交换行序；58 条候选仍在同图、同类别条件下完成一对一匹配。

## 环境与输入

| 项目 | 实际值 |
|---|---|
| Python | `3.12.11` |
| PyTorch | `2.5.1+cu121` |
| ONNX / ONNX Runtime GPU | `1.22.0 / 1.23.2` |
| CUDA / cuDNN | `12.1 / 91300` |
| GPU / driver | `2 × NVIDIA GeForce RTX 3090, 24 GiB / 595.71.05` |
| provider | CUDA：`CUDAExecutionProvider,CPUExecutionProvider`；CPU：`CPUExecutionProvider` |
| NumPy / OpenCV | `1.26.4 / 4.5.5` |
| checkpoint | `92,075,629` bytes；SHA-256 `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547` |
| ONNX | `81,569,383` bytes；SHA-256 `136942f484d5ab8d0d953c7e4c6bdcd89ef4f31c47b79f9a2401327e269b6b13` |
| 输入 / 阈值 / batch | `640×640 / FP32 / 0.3 / 4` |

输入图片与 M10 相同：

| 图片 | SHA-256 |
|---|---|
| `000000000139.jpg` | `ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d` |
| `000000000285.jpg` | `f3a2974ce3686332609124c70e3e6a2e3aca43fccf1cd1bd7c5c03820977f57d` |
| `000000000632.jpg` | `a4cd7f45ac1ce27eaafb254b23af7c0b18a064be08870ceaaf03b2147f2ce550` |
| `000000000724.jpg` | `5c0e559c75d3969c8e3e297b61f61063f78045c9d4802b526ba616361f3823fd` |

annotation SHA-256 为 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。

## Provider 与依赖合同

- `export` 和 `test` extras 安装 CPU `onnxruntime`，因此默认安装、CPU 用户和托管 CI 不承担 GPU wheel。
- `export-gpu` 和 `dev` extras 安装 `onnxruntime-gpu`；GPU wheel 同时包含 `CPUExecutionProvider`，开发环境可用同一 `.venv` 显式回退 CPU。
- CPU/GPU distributions 提供同名 Python 模块和共享库。UV `conflicts` 会拒绝 `dev + export/test`、`export-gpu + export/test`，避免安装顺序决定实际模块。
- 当前 CUDA 12 环境把 GPU ORT 限制为 `<1.27`。实装未设上界的 `1.27.0` 立即因缺少 `libcudart.so.13` 导入失败；[ORT 1.26 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0) 明确说明 1.27 将移除 CUDA 12 支持。[CUDA EP 官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)同时要求 ORT 与 PyTorch 的 CUDA/cuDNN 主版本匹配，并建议在创建 session 前导入 PyTorch。
- GPU session 显式使用 `device_id` 和 `use_tf32=1`。真实 `cuda:1` 单图前向生成 30 条检测，`get_provider_options()` 回读为 `device_id='1'`、`use_tf32='1'`，证明 device id 不只停留在单元测试。

ONNX Runtime 接受 NumPy feed，因此 Infer 继续在 CPU 完成 TestReader 预处理，再由 ORT 复制到 CUDA。日志观测到主图和两个子图分别增加 `5/1/2` 个 Memcpy node；这可能影响性能，但不影响当前功能与数值结论。本阶段没有实现 I/O Binding，也不作吞吐或显存声明。

## 协议

先从同一 checkpoint 导出固定 640、动态 batch 的 ONNX；Export 自带的 CPU batch 4 raw 输出回归为 1,200 条候选，score/box 最大误差 `5.05522e-6/0.00604248 px`：

```bash
uv run --no-sync rtdetrv3-export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth \
  --format onnx --output-dir <TMP>/export \
  --input-size 640 640 --batch-size 4
```

保持 config、四图目录、annotation、batch、阈值与尺寸相同，只替换模型源和设备：

```bash
# eager CUDA reference
CUDA_VISIBLE_DEVICES=0 uv run --no-sync rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth \
  --infer-dir <COCO_FOUR_IMAGES> --anno-file <COCO_ROOT>/annotations/instances_val2017.json \
  --batch-size 4 --threshold 0.3 --imgsz 640 --save-results \
  --device cuda:0 --output-dir <TMP>/eager-cuda

# ONNX CUDA candidate；CPU 对照把两个 device 都改为 cpu
CUDA_VISIBLE_DEVICES=0 uv run --no-sync rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --onnx-model <TMP>/export/rtdetrv3_r18vd_6x_coco.onnx \
  --infer-dir <COCO_FOUR_IMAGES> --anno-file <COCO_ROOT>/annotations/instances_val2017.json \
  --batch-size 4 --threshold 0.3 --imgsz 640 --save-results \
  --device cuda:0 --output-dir <TMP>/onnx-cuda
```

四份 JSON 按 `bbox_num` 的图像分组恢复为 xyxy，再使用 M8 的同图、同类别、全部阈值后候选一对一匹配。CPU 和 CUDA 各自以同设备 eager 为主参考；跨设备只作为独立观测。

## TF32 A/B 与可视化

ORT CUDA 默认 `use_tf32=1` 时相对默认 eager CUDA 为 `6.06865e-4/0.0238647 px`，58 条顺序不变。临时设置 `use_tf32=0` 后，ONNX CUDA 相对 ONNX CPU 缩小到 `2.59280e-6/0.000122070 px`，但相对用户实际默认的 eager CUDA 反而扩大到 `0.00738016/1.41972 px`，并出现两条重排。因此实现明确保留 TF32，而不是为了接近 CPU 数值改变默认 CUDA 计算模式。

四条路径均生成四张可解码 JPEG。`000000000285/632/724.jpg` 在四条路径下逐字节一致；首图的 eager CPU/ONNX CPU 一致，eager CUDA 与 ONNX CUDA 因亚像素框坐标差异生成不同字节。图片形状和 58 条阈值后语义检测保持一致，因此这里如实记录视觉输出差异，不用“可解码”代替像素等价声明。

## 本地门禁与剩余边界

- Infer/Export 定向回归：`52 passed`；新增 ONNX provider 定向回归另为 `7 passed`。
- GPU `dev`/`export-gpu` 与 CPU `test` extras 均完成实际安装；四组互斥组合均由 UV 拒绝。GPU 环境回读 `CUDAExecutionProvider,CPUExecutionProvider`，CPU wheel 对显式 CUDA 请求在创建输出目录前失败。
- 隐藏 GPU 的非 Paddle 全仓：`358 passed, 9 skipped, 17 deselected`。全包覆盖率 `7,078/13,748 (51.48%)`，直接维护范围 `1,991/2,200 (90.50%)`，通过 `50.5%/90%` 门槛。
- Ruff format/lint：`174` 个文件通过；Mypy：`107` 个 source file 通过；wheel/sdist 构建和发布内容检查通过。
- 临时 ONNX、四图副本、五组 Infer 输出与 pytest/cache 产物均已清理；UV `.venv` 保留 GPU `dev` 环境。

实现与本地证据提交 `dc97927`/`983821f` 的 [GitHub Actions run 29692163999](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29692163999) 六个 job 全部通过。Python 3.9–3.12 均为 `358 passed, 9 skipped, 17 deselected`；Python 3.12 全包/直接维护范围覆盖率为 `7,079/13,748 (51.49%)` 和 `1,991/2,200 (90.50%)`。托管 Ruff `174` 个文件、Mypy `107` 个 source file、wheel/sdist 发布检查、六个安装后 CLI、包外配置加载和 `65 passed` wheel smoke 同时通过。

本报告只声明 R18、当前 Python Infer CLI、固定 640、FP32、ORT 1.23.2 和本机 CUDA 12.1/cuDNN 9。它不证明 R34/R50 ONNX CUDA、动态高宽、AMP/FP16、I/O Binding、TensorRT、C++/mobile 或性能收益。CPU/GPU 的独立门槛也不能外推为 Paddle/PyTorch 逐位等价。

# TorchScript CUDA/CPU 推理验证报告

- 状态：`verified`
- 验证日期：`2026-07-19`
- 实现提交：`85b956d`
- 模型：`v0.1.0` 官方 R18 转换 checkpoint

> 历史快照（2026-07-19，M10）：本文的 ONNX CPU-only 结论记录 M10 验收时状态；当前 ONNX CUDA/CPU 合同见 [M11 验证报告](onnx-runtime-device-validation.md)。

## 结论

**已验证**：`rtdetrv3-infer --torchscript-model` 与 checkpoint 一样，在 CUDA 可用时默认选择 CUDA，也接受显式 `--device cpu`；无 CUDA 时自动回退 CPU。在 M10 验收时，ONNX 固定使用 ONNX Runtime `CPUExecutionProvider`，显式 `--device cuda` 在模型文件加载前以 argparse code 2 失败。

官方 R18、固定 640×640、四张真实 COCO 图片、batch 4、FP32、阈值 `0.3` 下，eager CUDA、TorchScript CUDA、eager CPU 和 TorchScript CPU 的每图检测数均为 `[30, 1, 25, 2]`。同设备比较结果如下：

| 参考 | 候选 | score 最大绝对误差 | box 最大绝对误差 | 重排 | 渲染 |
|---|---|---:|---:|---:|---|
| eager CUDA | TorchScript CUDA | `2.79218e-4` | `0.00872803 px` | `0/58` | 四张 JPEG 均逐字节一致 |
| eager CPU | TorchScript CPU | `1.90735e-6` | `9.15527e-5 px` | `0/58` | 四张 JPEG 均逐字节一致 |

跨设备的 TorchScript CPU 相对 eager CUDA 最大 score/box 误差为 `0.001755/0.0285645 px`，有两条近似候选交换行序；58 条候选仍在同图、同类别条件下完成一对一匹配。这是本次观测，不用于放宽 M8 的同设备默认门槛，也不作 CPU/CUDA 逐位一致声明。

## 设备可移植性修复

第一次真实 CUDA 运行依次暴露了三类 CPU trace 固化问题：

1. `HybridEncoder` 的固定位置编码是普通 Tensor 属性，且 eval 分支显式调用 `.to(src.device)`；CPU trace 把目标设备写成 CPU 常量。
2. `RTDETRTransformerv3` 的固定 anchors/valid mask 同样是普通属性，并在 trace 中产生 CPU 常量。
3. transformer 和后处理用带显式 `device=` 的 tensor factory 构造空间 shape、零值和 batch index；trace 后这些构造仍绑定 CPU。

当前实现把固定位置编码、anchors 和 valid mask 注册为 `persistent=False` buffer，使 `torch.jit.load(..., map_location=device)` 能随模块迁移，同时保持既有 checkpoint `state_dict` 键集合不变。运行时常量改为从活跃 Tensor 派生的 `new_tensor`、`zeros_like`、`ones_like` 和 `new_full`，避免在 CPU trace 中写死设备。两条 CUDA 回归直接覆盖 HybridEncoder 的 CPU trace/CUDA reload 前向和后处理的同类路径。

## 环境与输入

| 项目 | 实际值 |
|---|---|
| Python | `3.12.11` |
| PyTorch | `2.5.1+cu121` |
| CUDA / cuDNN | `12.1 / 91300` |
| GPU | `2 × NVIDIA GeForce RTX 3090, 24 GiB` |
| NVIDIA driver | `595.71.05` |
| checkpoint SHA-256 | `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547` |
| TorchScript | `93,499,278` bytes；SHA-256 `603dac6adaec16d97b79c127bbc6419f8a05cd8342470d000dd703865113ecf6` |
| 输入 / 阈值 / batch | `640×640 / FP32 / 0.3 / 4` |

输入图片来自 COCO val2017；按确定顺序扫描：

| 图片 | SHA-256 |
|---|---|
| `000000000139.jpg` | `ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d` |
| `000000000285.jpg` | `f3a2974ce3686332609124c70e3e6a2e3aca43fccf1cd1bd7c5c03820977f57d` |
| `000000000632.jpg` | `a4cd7f45ac1ce27eaafb254b23af7c0b18a064be08870ceaaf03b2147f2ce550` |
| `000000000724.jpg` | `5c0e559c75d3969c8e3e297b61f61063f78045c9d4802b526ba616361f3823fd` |

本次没有把本机 annotation 绝对路径写入命令，COCO 配置使用内置 category 映射；这不影响四条路径之间的同输入比较。

## 协议

先从当前实现和同一 checkpoint 在 CPU 上导出固定 640 TorchScript。Export 自带的 batch 4 raw 输出回归为 1,200 条候选，score/box 误差均为 0：

```bash
uv run rtdetrv3-export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth \
  --format torchscript \
  --output-dir <TMP>/export \
  --batch-size 4
```

然后保持 config、图片、batch、阈值和输出逻辑不变，只替换模型源或设备：

```bash
# eager CUDA 参考
uv run rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth \
  --infer-dir <COCO_FOUR_IMAGES> --batch-size 4 --threshold 0.3 \
  --save-results --device cuda --output-dir <TMP>/eager-cuda

# CUDA 可用时 TorchScript 默认选择 CUDA
uv run rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --torchscript-model <TMP>/export/rtdetrv3_r18vd_6x_coco.torchscript.pt \
  --infer-dir <COCO_FOUR_IMAGES> --batch-size 4 --threshold 0.3 \
  --save-results --output-dir <TMP>/torchscript-cuda

# 显式 CPU fallback
uv run rtdetrv3-infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --torchscript-model <TMP>/export/rtdetrv3_r18vd_6x_coco.torchscript.pt \
  --infer-dir <COCO_FOUR_IMAGES> --batch-size 4 --threshold 0.3 \
  --save-results --device cpu --output-dir <TMP>/torchscript-cpu
```

阈值后 JSON 按 `bbox_num` 分成四组，COCO `[x,y,w,h]` 恢复成 xyxy 后，使用 M8 的同图、同类别一对一匹配。CPU 和 CUDA 分别以同设备 eager 为主参考；跨设备结果单列为观测。

## 本地门禁与剩余边界

- CUDA 定向与 CLI/Export 回归：`56 passed`。
- 隐藏 GPU 的非 Paddle 全仓：`353 passed, 7 skipped, 34 deselected`。
- 覆盖率：全包 `7,068/13,738 (51.45%)`；直接维护范围 `1,981/2,190 (90.46%)`，通过 `50.5%/90%` 门槛。
- Ruff format/lint：`174` 个文件通过；Mypy：`107` 个 source file 通过。
- 四个后端均生成 58 条 JSON 记录和四张可解码图片；临时模型、图片、JSON 和 pytest 目录在验收结束后清理，UV `.venv` 保留。

实现和本地证据提交 `85b956d`/`f8b7439` 的 [GitHub Actions run 29690660612](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29690660612) 六个 job 全部通过。Python 3.9–3.12 均为 `353 passed, 9 skipped, 17 deselected`；Python 3.12 全包/直接维护范围覆盖率为 `7,069/13,738 (51.46%)` 和 `1,981/2,190 (90.46%)`。托管 Ruff `174` 个文件、Mypy `107` 个 source file、wheel/sdist 发布检查、六个安装后 CLI、包外配置加载和 `60 passed` wheel smoke 同时通过。

本报告只声明 M10 时的 R18、Python Infer CLI、固定 640、FP32 和本机 PyTorch CUDA/CPU。M10 当时的 ONNX Runtime CPU-only 边界已由 M11 扩展；R34/R50 TorchScript CUDA 后续由 [M12](variant-export-device-validation.md) 验证。本报告仍不覆盖动态高宽、AMP/FP16、TensorRT、C++/mobile 或外部客户端预处理，只做功能与数值合同，不给出吞吐或显存排名。

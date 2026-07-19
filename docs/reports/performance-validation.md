# R18 Paddle/PyTorch 性能验证

- 状态：已验证 model-only 基准；end-to-end 数据管线尚未验证
- 日期：2026-07-19
- 证据提交：`39e12b33587d554115f10a8cd138c5d55bbc5613`（四次采样均记录 `git_dirty=false`）
- 执行器：[`scripts/run_framework_benchmark.py`](../../scripts/run_framework_benchmark.py)

## 证据边界

**已验证**：执行器在隔离子进程中加载官方 R18 配置和对应 checkpoint，使用相同的 batch、输入尺寸、FP32、seed、CPU 线程数、warmup 和同步边界。Paddle GPU 构建已分别完成 CUDA 和 CPU 的实际 R18 前向，不是只检查可导入性。

**已观测**：下表是单台机器上的短采样结果，只用于定位当前性能量级。比值不是正确性门禁，也不是长时稳定性证据。

**已推断**：CUDA 推理显存接近，但训练 step 中 PyTorch 的峰值 allocated 显存比 Paddle 高约 16%，因此首个粗粒度差异位于训练专属的 activation、gradient、loss 或 optimizer state，而不是纯推理模型常驻量。未运行 memory snapshot/operator profile，不将该范围推断夸大为已定位到具体算子。

## 环境与协议

- 主机：Linux 5.15，Python 3.12.11，Intel Xeon Gold 6238R，112 逻辑 CPU，251.5 GiB RAM。
- GPU：2 × NVIDIA GeForce RTX 3090 24 GiB，driver 595.71.05；单进程使用可见的第 0 张卡。
- Paddle：3.3.0 GPU wheel，CUDA 11.8，cuDNN 8.9.7。
- PyTorch：2.5.1+cu121，CUDA 12.1，cuDNN 9.13.0。
- 模型：R18，batch 1，`640×640`，FP32，seed 2026，model-only 合成预处理输入。CPU 固定为 1 线程。
- 推理：CPU 为 3 warmup + 10 采样，CUDA 为 10 warmup + 50 采样。
- 训练 step：1 warmup + 3 采样；每次包含清梯度、前向、loss 汇总、反向和 `AdamW.step()`，lr/weight decay 为 0。
- 计时前后都同步设备；不包含 DataLoader、预处理、scheduler、EMA、AMP、DDP 和完整 Trainer。

Paddle checkpoint SHA-256 为 `f32dbd008bd7e5311c877d522f6d8c9e349795978c889f53823588b5e5d74a5f`，PyTorch checkpoint SHA-256 为 `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547`。

## 观测结果

| 设备 / workload | Paddle mean / p95 | PyTorch mean / p95 | Paddle throughput | PyTorch throughput | PyTorch/Paddle throughput |
|---|---:|---:|---:|---:|---:|
| CPU inference | 1259.476 / 1284.043 ms | 611.156 / 629.711 ms | 0.794 image/s | 1.636 image/s | 2.061× |
| CPU train-step | 11231.893 / 11369.981 ms | 7350.001 / 7521.216 ms | 0.089 image/s | 0.136 image/s | 1.528× |
| CUDA inference | 24.131 / 25.806 ms | 12.674 / 13.146 ms | 41.441 image/s | 78.899 image/s | 1.904× |
| CUDA train-step | 264.226 / 266.241 ms | 221.075 / 225.962 ms | 3.785 image/s | 4.523 image/s | 1.195× |

| 设备 / workload | Paddle 峰值内存 | PyTorch 峰值内存 | 口径 |
|---|---:|---:|---|
| CPU inference | 1258.2 MiB | 776.3 MiB | 进程生命期 peak RSS |
| CPU train-step | 3675.7 MiB | 3362.7 MiB | 进程生命期 peak RSS |
| CUDA inference | 161.1 / 288.4 MiB | 166.1 / 232.0 MiB | device peak allocated / reserved |
| CUDA train-step | 1928.6 / 2291.0 MiB | 2237.8 / 2484.0 MiB | device peak allocated / reserved |

PyTorch 在四个 workload 中的吞吐都超过 Paddle，达到 ROADMAP 的 95% 观测目标。CUDA 训练 allocated 峰值为 Paddle 的约 116%，未达“不超过 110%”的原始目标；与维护者不追求训练优化完全对齐的决策一致，本阶段只记录差距，不为该约 300 MiB 差异引入专项优化。

## 复现命令

以下命令会将新结果写入生成目录，不覆盖本次证据：

```bash
.venv/bin/python scripts/run_framework_benchmark.py --framework both --workload inference --device cpu --batch-size 1 --input-size 640 --warmup 3 --samples 10 --threads 1 --seed 2026 --output output/benchmarks/r18-cpu-inference.json
.venv/bin/python scripts/run_framework_benchmark.py --framework both --workload train-step --device cpu --batch-size 1 --input-size 640 --warmup 1 --samples 3 --threads 1 --seed 2026 --output output/benchmarks/r18-cpu-train-step.json
.venv/bin/python scripts/run_framework_benchmark.py --framework both --workload inference --device cuda --batch-size 1 --input-size 640 --warmup 10 --samples 50 --threads 1 --seed 2026 --output output/benchmarks/r18-cuda-inference.json
.venv/bin/python scripts/run_framework_benchmark.py --framework both --workload train-step --device cuda --batch-size 1 --input-size 640 --warmup 1 --samples 3 --threads 1 --seed 2026 --output output/benchmarks/r18-cuda-train-step.json
```

## 原始证据与局限

- [CPU inference JSON](data/r18-cpu-inference.json)
- [CPU train-step JSON](data/r18-cpu-train-step.json)
- [CUDA inference JSON](data/r18-cuda-inference.json)
- [CUDA train-step JSON](data/r18-cuda-train-step.json)

运行时 Paddle 报告了编译与加载 cuDNN 的兼容性提示，尽管版本 API 报告为 8.9.7，四份 workload 和已有 Paddle 测试均成功。这是已观测警告，不应用短采样反推长训稳定。

合成训练 loss 不作为此报告的对齐证据；它受框架随机路径、训练状态和实现细节影响。数值正确性应继续依赖固定 checkpoint/输入/中间 activation 的独立对齐测试。本次不含真实 COCO DataLoader、预处理或多卡，因此 DataLoader 占比、end-to-end 吞吐和关键算子 profile 仍是后续可选工作；没有实际用例需求时不扩展 R34/R50 性能采样。

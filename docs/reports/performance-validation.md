# R18 Paddle/PyTorch 性能验证

- 状态：已验证 model-only 基准与 R18 CUDA/COCO 端到端推理数据管线
- 日期：2026-07-19
- model-only 证据提交：`39e12b33587d554115f10a8cd138c5d55bbc5613`
- end-to-end 证据提交：`d823edf57b8bc7c758ed296702c6115fc9d2c4ec`
- 托管源码验证：[GitHub Actions run 29685452042](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29685452042)
- 执行器：[`scripts/run_framework_benchmark.py`](../../scripts/run_framework_benchmark.py)

## 证据边界

**已验证**：执行器在隔离子进程中加载官方 R18 配置和对应 checkpoint，使用相同的 batch、输入尺寸、FP32、seed、CPU 线程数、warmup 和同步边界。Paddle GPU 构建已分别完成 CUDA 和 CPU 的实际 R18 前向，不是只检查可导入性。五份正式 JSON 均记录 `git_dirty=false`。

**已验证**：端到端采样使用同一 COCO val2017 annotation（SHA-256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`）和各自原生 EvalReader；10 次 warmup 后的 50 个 measured image ID 逐批完全一致。比较器会在 annotation、数据集大小或 image ID 不一致时拒绝生成比值。

**已观测**：下表是单台机器上的短采样结果，只用于定位当前性能量级。比值不是正确性门禁，也不是长时稳定性证据。

**已推断**：CUDA 推理显存接近，但训练 step 中 PyTorch 的峰值 allocated 显存比 Paddle 高约 16%，因此首个粗粒度差异位于训练专属的 activation、gradient、loss 或 optimizer state，而不是纯推理模型常驻量。端到端结果显示 PyTorch 的可见 input-pipeline stall 已占 29.68%，是除模型前向之外最明确的推理侧次级瓶颈；它不解释训练显存差异。

## 环境与协议

- 主机：Linux 5.15，Python 3.12.11，Intel Xeon Gold 6238R，112 逻辑 CPU，251.5 GiB RAM。
- GPU：2 × NVIDIA GeForce RTX 3090 24 GiB，driver 595.71.05；每个隔离 worker 只暴露一张卡，并在进程内使用 `cuda:0`。
- Paddle：3.3.0 GPU wheel，CUDA 11.8，cuDNN 8.9.7。
- PyTorch：2.5.1+cu121，CUDA 12.1，cuDNN 9.13.0。
- model-only：R18，batch 1，`640×640`，FP32，seed 2026，合成预处理输入；CPU 固定为 1 线程。CPU 推理为 3 warmup + 10 采样，CUDA 推理为 10 warmup + 50 采样。
- 训练 step：1 warmup + 3 采样；每次包含清梯度、前向、loss 汇总、反向和 `AdamW.step()`，lr/weight decay 为 0。
- end-to-end：COCO val2017，R18，CUDA，batch 1，`640×640`，FP32，4 个 EvalReader worker，10 warmup + 50 采样。input-pipeline 定义为 `next(DataLoader)`、预处理、tensor 转换和传输，直到目标设备同步且 batch 可供模型使用；end-to-end 为该时长与同步模型前向之和。
- 算子 profile 在正式计时结束后对额外一次模型前向执行，保留 Top 10，因此 profiler 开销不进入吞吐。Paddle 使用 inclusive host trace 与独立 kernel 列表，PyTorch 按 self CUDA time 排序；两套 taxonomy 不作逐项时长比值。
- model-only 不包含 DataLoader、预处理、scheduler、EMA、AMP、DDP 和完整 Trainer；end-to-end 只额外包含推理数据管线，不包含 COCO metric accumulate。

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

### COCO 端到端推理

| 框架 | end-to-end mean / p95 | 吞吐 | input-pipeline mean / p95 | 可见管线占比 | model mean / p95 | GPU allocated / reserved | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paddle | 32.663 / 36.050 ms | 30.616 image/s | 0.160 / 0.237 ms | 0.49% | 32.503 / 35.833 ms | 161.1 / 288.4 MiB | 2018.0 MiB |
| PyTorch | 20.683 / 26.282 ms | 48.349 image/s | 6.139 / 11.291 ms | 29.68% | 14.544 / 15.423 ms | 166.1 / 232.0 MiB | 1231.1 MiB |

PyTorch 端到端吞吐为 Paddle 的 `1.579×`，模型前向平均时延为 Paddle 的 `44.75%`。与此同时，PyTorch 模型更快后，4-worker 管线未能像 Paddle 一样把预处理完全隐藏在前向期间，因此仍有平均 `6.139 ms` 可见等待。这里的占比是消费者实际等待，不等于 DataLoader 的总 CPU 工作占比；存储缓存、worker 数和模型速度都会改变该值。

单次额外 profile 的首要热点在两边都是卷积：Paddle 的前五个框架算子为 `conv2d`、`batch_norm`、`matmul`、`add`、`relu`；PyTorch 按 self CUDA time 的前五个 ATen 算子为 `aten::cudnn_convolution`、`aten::addmm`、`aten::cudnn_batch_norm`、`aten::clamp_min`、`aten::bmm`。Paddle 的 inclusive host duration 与 PyTorch 的 self CUDA time 口径不同，原始数值只用于各自框架内部排序，不用于声称算子级快慢对齐。

## 复现命令

以下命令会将新结果写入生成目录，不覆盖本次证据：

```bash
.venv/bin/python scripts/run_framework_benchmark.py --framework both --workload inference --device cpu --batch-size 1 --input-size 640 --warmup 3 --samples 10 --threads 1 --seed 2026 --output output/benchmarks/r18-cpu-inference.json
.venv/bin/python scripts/run_framework_benchmark.py --framework both --workload train-step --device cpu --batch-size 1 --input-size 640 --warmup 1 --samples 3 --threads 1 --seed 2026 --output output/benchmarks/r18-cpu-train-step.json
.venv/bin/python scripts/run_framework_benchmark.py --framework both --workload inference --device cuda --batch-size 1 --input-size 640 --warmup 10 --samples 50 --threads 1 --seed 2026 --output output/benchmarks/r18-cuda-inference.json
.venv/bin/python scripts/run_framework_benchmark.py --framework both --workload train-step --device cuda --batch-size 1 --input-size 640 --warmup 1 --samples 3 --threads 1 --seed 2026 --output output/benchmarks/r18-cuda-train-step.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_framework_benchmark.py --framework both --workload e2e-inference --device cuda --dataset-root /path/to/coco2017 --batch-size 1 --input-size 640 --warmup 10 --samples 50 --threads 1 --num-workers 4 --profile-top-k 10 --seed 2026 --output output/benchmarks/r18-cuda-e2e-inference.json
```

## 原始证据与局限

- [CPU inference JSON](data/r18-cpu-inference.json)
- [CPU train-step JSON](data/r18-cpu-train-step.json)
- [CUDA inference JSON](data/r18-cuda-inference.json)
- [CUDA train-step JSON](data/r18-cuda-train-step.json)
- [CUDA COCO end-to-end inference/profile JSON](data/r18-cuda-e2e-inference.json)

运行时 Paddle 报告了编译与加载 cuDNN 的兼容性提示，尽管版本 API 报告为 8.9.7，四份 workload 和已有 Paddle 测试均成功。这是已观测警告，不应用短采样反推长训稳定。

合成训练 loss 不作为此报告的对齐证据；它受框架随机路径、训练状态和实现细节影响。数值正确性应继续依赖固定 checkpoint/输入/中间 activation 的独立对齐测试。端到端证据只覆盖 R18、单卡、batch 1、50 个缓存状态未知的 val2017 batch，不覆盖训练 DataLoader、多卡、metric accumulate 或长时运行；单次算子 profile 也不是稳定统计。没有实际用例需求时不扩展 R34/R50 性能采样，也不为管线占比或训练显存数字追求完全对齐。

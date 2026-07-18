# COCO 准确率与稳定性验证报告

- 状态：`in-progress`
- 创建日期：`2026-07-18`
- 执行计划：[`M4——COCO 精度与稳定性对齐计划`](../plans/2026-07-18-m4-coco-accuracy-stability.md)

## 证据口径

- **已验证**：由当前代码、固定权重/数据和实际命令直接得到，并保留足以复现的环境与 checksum。
- **观察到**：本机运行或日志中的事实，但不自动外推为框架等价或多次运行结论。
- **发布参考**：官方仓库模型表或说明中的结果，本机尚未直接复现时不能标为已验证。
- **推断**：基于实测进行的成本或趋势估算，必须与直接结果分开。
- **计划**：尚未执行，不作为完成证据。

COCO 指标在命令日志中使用 `[0, 1]` 标度，在模型表中通常写成百分数。本报告的 `0.5 AP` 门槛表示 0.5 个百分点，即日志标度的绝对差 `≤ 0.005`。

## 固定环境与数据

| 项目 | 值 | 证据状态 |
|---|---|---|
| Python | `3.12.11` | 已验证 |
| PyTorch | `2.5.1+cu121` | 已验证 |
| CUDA/cuDNN/driver | `12.1` / `9.1.3` / `595.71.05` | 已验证 |
| GPU | 2×NVIDIA GeForce RTX 3090 24 GiB | 已验证 |
| Paddle | `3.3.0` CPU build | 已验证 |
| train2017 | 118287 图、860001 标注；annotation SHA-256 `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d` | 已验证 |
| val2017 | 5000 图、36781 标注；annotation SHA-256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` | 已验证 |

## R18 同权重 val2017

固定来源 checkpoint：

- Paddle：`pretrained_models/paddle/rtdetrv3_r18vd_6x_coco.pdparams`，SHA-256 `f32dbd008bd7e5311c877d522f6d8c9e349795978c889f53823588b5e5d74a5f`。
- PyTorch 转换结果：`pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth`，SHA-256 `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547`。

两侧目标条件为 val2017、640×640、keep-ratio false、bilinear interpolation、FP32、eval mode、batch size 16。只有直接运行的结果才能进入“本机直接评估”行。

| 来源 | AP | AP50 | AP75 | APs | APm | APl | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| 官方模型库 R18 | 0.481 | 0.662 | — | — | — | — | 发布参考 |
| Paddle checkpoint 本机 CPU 评估 | 0.480477 | 0.656152 | 0.519500 | 0.307266 | 0.514807 | 0.639255 | 已验证（固定 JSON 独立复算） |
| 转换 PyTorch checkpoint 本机 CUDA 评估 | 0.480502 | 0.656089 | 0.519446 | 0.307272 | 0.514910 | 0.639605 | 已验证（固定 JSON 独立复算） |
| 转换 PyTorch checkpoint 本机 CPU 评估 | 0.480477 | 0.656151 | 0.519500 | 0.307267 | 0.514806 | 0.639255 | 已验证（固定 JSON 独立复算） |
| M3 checkpoint（官方转换权重后再训练 1 epoch） | 0.468 | 0.643 | 0.504 | 0.302 | 0.501 | 0.624 | 已验证，但不是同权重基线 |

### PyTorch 复现命令

```bash
COCO_ROOT=/path/to/coco2017
CUDA_VISIBLE_DEVICES=0 .venv/bin/rtdetrv3-eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth \
  --anno_file "$COCO_ROOT/annotations/instances_val2017.json" \
  --image_dir "$COCO_ROOT/val2017" \
  --batch_size 16 --num_workers 4 --device cuda \
  --output-dir output/m4-r18-official-val2017/pytorch
```

**已验证**：两次 PyTorch 命令都处理 5000 图/313 batch 并完成 COCO API 统计，模型前向约 76–77 秒。第一次 AP/AP50/AP75 为 `0.480/0.656/0.520`，第二次为 `0.481/0.656/0.519`；其余三项 AP 均为 `0.307/0.515/0.640`。这是 CUDA 运行在三位小数边界的直接观察，不能把单次舍入值冒充逐预测确定性。第二次保留的 `bbox.json` 为 `236,335,539` 字节，SHA-256 `7bfd1a4cf0e32561ef1d74d1aa39e617ceb1b2a6a0d76dea34f3dddce015f054`；对该固定 JSON 独立复算得到未舍入 AP `0.480502167075`、AP50 `0.656089305298`、AP75 `0.519446428999`，作为后续逐图比较的固定 PyTorch 侧输入。

**已验证**：同权重 CPU 全量评估耗时 `1233.57s`，保留的 `bbox.json` 为 `236,327,405` 字节，SHA-256 `bb29e359cb521dacc32235f3eafe727d0693d8eb5d76d7fb5885ac9fb0b07b40`；独立复算 AP/AP50/AP75 为 `0.480477134768/0.656151392882/0.519500286057`。CPU 与固定 CUDA JSON 的 AP 绝对差仅 `0.000025032307`，但这只证明同一 PyTorch 实现的设备稳定性，不是 Paddle/PyTorch 等价证据。

**观察到**：CPU/CUDA 按排名直接对齐会把临界 score 交换误认为大框偏差，因此另使用“同图、同类、score 阈值、坐标 L∞ 容差 `1px`”匹配。score `>=0.5` 时两侧分别选出 `26243/26240` 个预测，匹配 `26218`个，左/右未匹配 `25/22`个；匹配框坐标 L∞ 差中位数 `0.002892px`、99 分位 `0.088248px`。该容差匹配用于描述稳定性，不替代 COCO API 指标。

两次 CUDA AP 都与发布参考相差不超过 `0.001`。AP50 相对发布参考约低 `0.006`；本机 Paddle 的 AP50 也为 `0.656152`，因此该发布表差异不支持“由 PyTorch 转换造成”的解释，但当前证据也不能确定发布结果的具体环境差异。

### Paddle 复现命令

```bash
COCO_ROOT=/path/to/coco2017
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 FLAGS_paddle_num_threads=8 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=third-party/RT-DETRv3-paddle .venv/bin/python \
  third-party/RT-DETRv3-paddle/tools/eval.py \
  -c third-party/RT-DETRv3-paddle/configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --output_eval output/m4-r18-official-val2017/paddle-cpu \
  -o weights=pretrained_models/paddle/rtdetrv3_r18vd_6x_coco.pdparams \
     EvalDataset.dataset_dir="$COCO_ROOT" EvalReader.batch_size=16 \
     worker_num=4 use_gpu=False
```

当前 Paddle 为 CPU build。固定 16 图、batch 16、8 线程探针观察到 `0.8263` 图/秒；batch 32、16 线程反而降至 `0.7888` 图/秒，因此完整评估使用 batch 16、8 线程和 4 workers。探针指标只用于估算耗时，子集 AP 不进入精度表。

**已验证**：完整评估处理 5000 图/313 batch，退出码为 0，总耗时 `6120.77s`，Paddle 报告平均 `0.831185` FPS，峰值 RSS `5,426,120 KiB`。保留的 `bbox.json` 为 `236,329,151` 字节，SHA-256 `50ad6ceebb052f8ff7e826fa9109dc60e058404203338c29905b2fe3da0ce873`；独立复算得到 AP/AP50/AP75/APs/APm/APl `0.480477300367/0.656152367330/0.519499977301/0.307266486593/0.514806586690/0.639255472633`。

### 同设备双框架比较

| 指标 | Paddle CPU | PyTorch CPU | 绝对差 |
|---|---:|---:|---:|
| AP | 0.480477300367 | 0.480477134768 | 0.000000165599 |
| AP50 | 0.656152367330 | 0.656151392882 | 0.000000974448 |
| AP75 | 0.519499977301 | 0.519500286057 | 0.000000308756 |
| APs | 0.307266486593 | 0.307266567713 | 0.000000081120 |
| APm | 0.514806586690 | 0.514806052609 | 0.000000534081 |
| APl | 0.639255472633 | 0.639255477387 | 0.000000004754 |

**已验证**：主 AP 绝对差为 `1.65599e-7`，即 `0.0000165599` 个 COCO 百分点，远低于 `0.5` 点 gate。为排除汇总指标抵消，另按同图、同类和坐标 L∞ `<=1px` 匹配 prediction：score `>=0.3` 时两侧 `53780/53780` 个预测全部匹配；score `>=0.5` 时 `26243/26243` 全部匹配，坐标差中位数 `0.0000153px`、99 分位 `0.0001221px`、最大值 `0.0133057px`。score `>=0.1` 时左/右仅 `5/6` 个未匹配；top-10 按排名对齐的 2 个类别不同来自近乎相同 score 的排名交换，不是高置信度预测分歧。

**结论**：R18 官方同权重、同 CPU/FP32 的完整 val2017 gate 通过。该结论验证当前 R18 转换权重、预处理、模型前向、后处理和 COCO 评估链路；它不是 72 epoch 训练收敛、多 seed 稳定性或 R34/R50 精度的完成证据。

## R18 正式训练初始化与协议

官方 Paddle 配置使用 `ResNet18_vd_pretrained.pdparams`，不是随机初始化，也不是已训练好的 RT-DETRv3 COCO checkpoint。本机已验证：

- 源文件 `44,850,756` 字节，SHA-256 `68d7632cb67ad2c658fe67ab5837d8eb65466a7bc1574badc74860059ef5e7f0`。
- 目标感知严格转换得到 115/115 个 backbone tensor；0 未映射源 key、0 unexpected key、0 缺失 backbone key，所有 tensor 有限。
- 转换输出 SHA-256 `2483b5b00ed2b84192540bbd1bd1768e3e4422c2f8fa1598ae96e0c2d6f64db2`。转换文件带 session/timestamp 元数据，因此该输出 hash 是本次本机证据，不是跨次稳定发布 hash。
- 真实 2-GPU、16 图、AMP+EMA 协议烟测产生 1 次有效 optimizer/scheduler/EMA update；checkpoint 记录 seed 0、完整 pretrain path、EMA step 1。`rtdetrv3-eval --use-ema` 已从该 checkpoint 成功加载 EMA 并完成 4 图链路；4 图 AP 不作为准确率证据。

正式训练固定为 world size 2、每 rank batch 8、global batch 16、无梯度累积、AMP、EMA exponential `0.9999`、72 epoch、当前 PyTorch LR schedule、seed `0/1/2`、每 3 epoch 原子 checkpoint（final 强制保存），并对最终 EMA 权重做完整 val2017。即时模型可作诊断，但不替代 EMA 主结果。EMA checkpoint 烟测约 368 MB；按每 epoch保存三 seed 约需 79.5 GB，`snapshot_epoch=3` 将预算降至约 26.5 GB，代价是最多重跑 3 epoch。

```bash
COCO_ROOT=/path/to/coco2017
SEED=0
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 \
  .venv/bin/rtdetrv3-train --ddp --amp --seed "$SEED" \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  -o TrainDataset.dataset_dir="$COCO_ROOT" TrainReader.batch_size=8 \
     worker_num=4 epoch=72 log_iter=50 use_ema=True \
     accumulate_steps=1 snapshot_epoch=3 \
     pretrain_weights=pretrained_models/pytorch/ResNet18_vd_pretrained.pth \
     save_dir="output/m4-r18-seed-${SEED}"
```

恢复时保持全部参数不变，只增加 `--resume output/m4-r18-seed-${SEED}/epoch_<N>.pth`。训练完成后用下列命令评估主结果：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/rtdetrv3-eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint "output/m4-r18-seed-${SEED}/model_final.pth" --use-ema \
  --anno_file "$COCO_ROOT/annotations/instances_val2017.json" \
  --image_dir "$COCO_ROOT/val2017" \
  --batch_size 16 --num_workers 4 --device cuda
```

## 训练协议差异

| 项目 | Paddle 参考 | 当前 PyTorch | 处理方式 |
|---|---:|---:|---|
| epoch | 72 | 72 | 相同 |
| base LR | `4e-4` | `1e-4` | 保留差异并报告 |
| warmup | 2000 step | 1000 step | 保留差异并报告 |
| piecewise milestone | 100 | 60 | 保留差异并报告 |
| gamma | `1.0` | `0.1` | 保留差异并报告 |
| optimizer | AdamW | AdamW | 不要求逐元素更新一致 |
| weight decay | `1e-4` | `1e-4` | 配置相同，参数组语义另有 M3 测试 |
| gradient clip | norm `0.1` | norm `0.1` | M3 已验证调用位置 |

同权重推理 AP 用于隔离模型、预处理和后处理语义；各自 schedule 的训练结果用于评估训练库质量。两者不能混为单一“框架差异”。

## 多 seed 稳定性

| 模型 | seed | schedule | checkpoint SHA-256 | AP | AP50 | AP75 | 状态 |
|---|---:|---|---|---:|---:|---:|---|
| R18 | 0 | PyTorch 72 epoch | — | — | — | — | 计划 |
| R18 | 1 | PyTorch 72 epoch | — | — | — | — | 计划 |
| R18 | 2 | PyTorch 72 epoch | — | — | — | — | 计划 |

均值和标准差只能在三次训练均完成并核对配置后填写。M3 的单次 1 epoch 结果不能进入该统计。显式 seed 与 checkpoint 恢复已由 19 个定向测试覆盖 Python/NumPy/PyTorch RNG、单/多卡 sampler、epoch 和 DataLoader workers；另一个真实 2-GPU、32 图 smoke 记录 `seed=17`、8 次更新、`sampler_epoch=1` 和 90 项有效配置。在修复 DDP 只保存 rank 0 RNG 的缺口后，又用真实双卡烟测确认 checkpoint 收集了 2 份不同的 rank RNG state，且新单测会在恢复时选择当前 rank。这些只证明协议传播和恢复边界，不是多 seed 稳定性结果。

## 当前结论与局限

- M4 尚未完成；R18 官方同权重 CPU/FP32 完整 val2017 gate 已通过，主 AP 绝对差 `1.65599e-7`，score `>=0.3` 的 `53780` 个 prediction 全部匹配。
- 本机 Paddle/PyTorch AP50 均约为 `0.65615`，官方表的 `0.662` 仍是未定位的发布环境/评估差异；双框架本机结果一致，因此它不阻塞当前 PyTorch 长训。
- Paddle CPU build 使全量评估耗时超过 100 分钟，但未降低本次同设备双框架证据标准。
- R34/R50 必须等待 R18 的同权重门槛与训练协议稳定后再开始。

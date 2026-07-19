# 权重转换经验

本文整合历史权重转换规格与当前 `ppdet_pytorch.conversion` 实现。

## 当前可验证能力

- 加载 `.pdparams`，通过 NumPy 转为 PyTorch tensor。
- 按规则转换 BatchNorm、权重和 bias 名称，并允许 JSON 手工覆盖。
- conversion CLI 默认通过 `--config` 构建目标 `state_dict`，执行 key/shape 检查；只有显式 `--no-validate` 才跳过。
- 根据目标 `torch.nn.Linear` 模块决定二维权重转置，不再只依赖参数名称。
- 保存模型权重、显式 SHA-256、转换元数据、统计和参数映射报告。
- 对同一 config/架构支持单文件、目录或 glob 批量转换；每个文件独立记录结果，失败后继续，并可写 JSON summary。
- checkpoint 通过同目录临时文件原子发布；`--memory-efficient` 使用轻量目标 shape map，并按参数批次释放源 Paddle tensor。
- 小型 fixture 的单元和集成回归已纳入默认测试集。
- 官方 R18/R34/R50 checkpoint 的可选数值用例已覆盖参数、backbone、neck、transformer、head、后处理、受控 loss 和整体梯度方向。

## 参数名称映射

常见规则：

| Paddle | PyTorch | 说明 |
|---|---|---|
| `._mean` | `.running_mean` | BatchNorm 运行均值 |
| `._variance` | `.running_var` | BatchNorm 运行方差 |
| `._scale` | `.weight` | BatchNorm scale |
| `._offset` | `.bias` | BatchNorm offset |
| `.w_0` | `.weight` | 普通权重 |
| `.b_0` | `.bias` | 普通 bias |

规则匹配只说明名称候选合理，不证明该参数已正确对齐。必须同时检查目标 key、形状、dtype 和数值。自定义模块、不同层次命名或重构后的模型需要手工映射。

## 张量布局

- 卷积权重通常同为 `[out_channels, in_channels, H, W]`，不应因框架不同就无条件转置。
- Paddle Linear 权重通常为 `[in_features, out_features]`，PyTorch 为 `[out_features, in_features]`；转换时应以目标模块类型为准。
- R18 审计发现 65 个二维 `.weight` 中有 64 个属于 `torch.nn.Linear`、1 个属于 Embedding。原名称规则漏转置了 12 个 `[256, 256]` Linear 权重，包括 decoder `cross_attn` 的 `value_proj`/`output_proj`、`transformer.enc_output.*.0.weight` 和 `transformer.map_memory.0.weight`。
- 方阵 Linear 转置前后 shape 不变，所以“shape 全部匹配”无法发现这类错误。修复前 backbone/neck 通过，transformer 首次明显分歧；目标模块感知转置后 transformer 恢复到设定容差内。
- 只看元素数相等不能盲目 reshape。当前通用 shape 检查只把完全相同 shape 和二维反转 shape 视为可兼容候选；后者仍需由目标模块证明是 Linear transpose，其他等元素数变化必须显式验证数学语义。
- 默认保留源 dtype。FP16/BF16 权重应在转换后单独记录误差，不与 FP32 使用同一容差。

## 校验层级

1. **文件级**：后缀、可读性、大小、checksum、输出元数据。
2. **映射级**：源/目标 key 覆盖率、手工覆盖、未映射清单。
3. **参数级**：shape、dtype、transpose 规则和数值保留。
4. **模块级**：同输入下比较 backbone/neck/transformer/head 激活。
5. **模型级**：同一预处理后比较 boxes/logits 和 COCO 指标。

模型输出比较只接受两端同为字典或同为 tensor/list/tuple；一端为字典、另一端为 tensor 会显式报输出结构不兼容，不将结构错配伪装成数值误差。

当前官方 R18/R34/R50 已覆盖文件、映射、参数、eval 模块、后处理、受控训练 loss、整体梯度方向和同一 COCO 图片的统一渲染对比。R18 还通过了同权重完整 val2017 指标门禁；R34/R50 的完整 val2017 指标、三变体真实训练收敛和跨框架优化器状态迁移仍未验证。

## 三变体官方 checkpoint 汇总（2026-07-18）

| 变体 | 源文件大小 | 源 SHA-256 | 转换覆盖 | 未填充目标 key | 本次输出大小 / SHA-256 | wall / 最大 RSS |
|---|---:|---|---:|---|---|---|
| R18 | `91,945,530` | `f32dbd008bd7e5311c877d522f6d8c9e349795978c889f53823588b5e5d74a5f` | 571/571 | 75 个 BN counter + 2 个 auxiliary buffer | `92,075,629` / `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547` | `4.75 s` / `1,056,232 KiB` |
| R34 | `137,016,081` | `29b09c64d6c372cde46d94caee1b57a23cee0aae24bd7bd3e2937cf57e581a68` | 681/681 | 91 个 BN counter + 2 个 auxiliary buffer | `137,170,947` / `e69207749b37e493596086579f435d5f08e9f058b66322452456053b78a4f272` | `5.13 s` / `1,188,888 KiB` |
| R50 | `182,331,170` | `e8b1d5db3208ce0f9edba5a914f23c918141b608ab4cd409db9d9204f7ed4b08` | 789/789 | 103 个 BN counter + 2 个 auxiliary buffer | `182,510,207` / `5e3e34ac3d3d14f57ebf6100b146b5702f8dface24fbe57cbc993f59381b67f7` | `5.50 s` / `1,322,312 KiB` |

- **已验证**：三个变体均为 0 skipped、0 未映射源 key、0 unexpected key；加载时显式 missing key 仅为 `aux_o2m_head.anchor_points` 与 `aux_o2m_head.stride_tensor`。
- **已验证**：共 2,041 个转换 tensor 均按目标 `torch.nn.Linear` 布局转置或原样保留，并与 Paddle 源数组逐个精确相等。
- **说明**：输出文件包含时间戳和 session ID，上表输出 SHA-256 只是本次运行证据，不是稳定发布 checksum。最大 RSS 包含 Python、Paddle、PyTorch 和完整目标模型，不代表纯转换增量。
- **已验证**：三变体均通过 CPU/float32、PyTorch 单线程下的分层 eval、受控 loss 和整体梯度方向门槛。R18/R34 使用 `rtol=1e-4, atol=1e-5`；R50 使用 `rtol=3e-4, atol=1e-5`。
- **观察到**：R50 后处理 300 个 label 中有 2 个 top-k 离散边界差异；全部 score、298 个稳定候选坐标和 `bbox_num` 通过。该结果不应表述为逐候选完全一致。

## 可视化输出对比

可视化是对迁移结果的直观说明，但不能用两个框架各自默认的渲染图直接判定数值等价。字体、调色板、阈值、坐标取整和图像编码都可能制造与模型无关的视觉差异。推荐协议是：

1. 两侧固定同一 checkpoint 来源、原图、预处理、设备、dtype 和输出阈值。
2. 先保留两侧原始预测 JSON，按同图、同类和明确像素容差匹配，记录未匹配项、score 和坐标误差。
3. 再用同一个渲染器、颜色和取整规则从原始 JSON 画左右面板；可为可读性提高绘制阈值，但必须标注数值比较阈值。
4. 图像只是报告入口，同时提供机器可读数据、输入/checkpoint checksum 和复现命令。

R18/R34/R50 已按该协议在 COCO `000000000139.jpg` 上验证：`score >= 0.3` 时分别有 `30/30`、`31/31`、`28/28` 个预测匹配，最大 score 差均不超过 `3.79e-6`，最大框坐标差均不超过 `1.23e-4 px`。统一渲染图、JSON 和命令见[三变体预测可视化报告](../reports/prediction-visualization.md)。单图一致不能代替完整 val2017 AP 或训练收敛证据。

## R18 官方 checkpoint 实证（2026-07-18）

以 [`configs/checkpoints/rtdetrv3_coco.yml`](../../configs/checkpoints/rtdetrv3_coco.yml) 中的官方 R18 条目为准：

- **已验证**：源文件 `91,945,530` 字节，SHA-256 为 `f32dbd008bd7e5311c877d522f6d8c9e349795978c889f53823588b5e5d74a5f`。
- **已验证**：源 571 个 key 全部转换，0 skipped，0 个未映射源 key。目标 648 个 key 中有 77 个未由源文件填充：75 个 BatchNorm `num_batches_tracked`，以及 auxiliary head 的 `anchor_points`/`stride_tensor`。
- **已验证**：对 571 个转换 tensor 逐个与 Paddle 源值做精确数组比较；属于目标 `torch.nn.Linear` 的权重先按布局转置，其余 tensor 直接比较，全部完全相等。这分别验证了名称映射、布局决策和数值保留，而不只是 shape。
- **已验证**：`load_state_dict(strict=False)` 的 unexpected key 为 0，显式 missing key 仅为上述两个 auxiliary buffer；PyTorch 会按 BatchNorm 兼容规则处理缺失的 `num_batches_tracked`。
- **已验证**：Python 3.12.11、Paddle 3.3.0、PyTorch 2.5.1+cu121、CPU/float32/eval、NumPy seed 2026、输入 `[1, 3, 640, 640]`、PyTorch CPU 单线程下，backbone 和 neck 最大绝对误差分别不超过 `8.17e-6` 和 `5.25e-6`。transformer 四个张量在 `rtol=1e-4, atol=1e-5` 下通过；其中 decoder box/logit 最大绝对误差分别为 `1.54e-5` 和 `3.62e-5`。
- **已验证**：head box/logit 在同一容差下通过；后处理的 300 个 label 与 `bbox_num` 完全相等，score 通过上述容差，像素坐标最大绝对误差为 `0.00358 px`，归一化坐标在上述容差下通过。
- **观察到**：使用默认多线程 CPU reduction 时，一次运行中 decoder box 有 16/1200 个值超出容差，最大差异 `0.364`；固定 PyTorch CPU 单线程后重复用例通过。**推断**：并行 reduction 的微小差异可能改变边界 top-k 候选顺序；这需要后续用 top-k 前激活和候选 margin 进一步证实。

## R18-vd backbone 预训练权重（2026-07-18）

正式训练初始化与完整检测 checkpoint 是两种不同用途。官方 R18 配置从 ImageNet `ResNet18_vd_pretrained.pdparams` 启动；用已经训练完成的 COCO 检测权重做 72 epoch 训练会改变实验问题，随机初始化则偏离官方起点。

- 权威来源与 checksum 记录在 [`configs/checkpoints/rtdetrv3_coco.yml`](../../configs/checkpoints/rtdetrv3_coco.yml) 的 `pretraining.resnet18_vd`。
- 源 checkpoint 为 `44,850,756` 字节、115 个 key，SHA-256 `68d7632cb67ad2c658fe67ab5837d8eb65466a7bc1574badc74860059ef5e7f0`。
- 使用完整 R18 目标 config 做严格转换，得到 115/115 个 `backbone.*` tensor，0 未映射源 key、0 unexpected key、0 缺失 backbone key；其余 533 个目标 key 属于检测 neck/head/transformer，保持模型初始化值是预期行为。
- 本次转换输出 SHA-256 为 `2483b5b00ed2b84192540bbd1bd1768e3e4422c2f8fa1598ae96e0c2d6f64db2`。输出 metadata 含 timestamp/session ID，所以它是本次证据，不是跨次稳定 hash。
- 该训练初始化权重与三个检测权重使用同一发布合同，用户别名为 `r18-backbone`；发布前可用 `rtdetrv3-models verify r18-backbone` 校验本地文件，发布后使用固定 tag 的 HTTPS URL 下载。

```bash
curl --fail --location \
  --output pretrained_models/paddle/ResNet18_vd_pretrained.pdparams \
  https://paddledet.bj.bcebos.com/models/pretrained/ResNet18_vd_pretrained.pdparams
sha256sum pretrained_models/paddle/ResNet18_vd_pretrained.pdparams

uv run rtdetrv3-convert --strict \
  --input pretrained_models/paddle/ResNet18_vd_pretrained.pdparams \
  --output pretrained_models/pytorch/ResNet18_vd_pretrained.pth \
  --config configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --save-mapping pretrained_models/reports/ResNet18_vd_pretrained.mapping.json
```

下载后必须核对上述 manifest 中的 SHA-256，再执行转换；HTTP 成功不等于文件来源与内容已验证。

部分 backbone checkpoint 的“未填充目标 key”不能与完整检测 checkpoint 共用同一判据：前者要证明所有源 key 都映射且目标 backbone 完整，后者才要求除已知派生 buffer 外填充完整模型。

三变体可选回归命令（checkpoint 路径由本地环境提供，不进 Git）：

```bash
RTDETRV3_R18_PADDLE_CHECKPOINT=pretrained_models/paddle/rtdetrv3_r18vd_6x_coco.pdparams \
  uv run pytest -q -p no:cacheprovider tests/numerical/test_r18_official_checkpoint.py -k r18

RTDETRV3_R34_PADDLE_CHECKPOINT=pretrained_models/paddle/rtdetrv3_r34vd_6x_coco.pdparams \
  uv run pytest -q -p no:cacheprovider tests/numerical/test_r18_official_checkpoint.py -k r34

RTDETRV3_R50_PADDLE_CHECKPOINT=pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
  uv run pytest -q -p no:cacheprovider tests/numerical/test_r18_official_checkpoint.py -k r50
```

## 批量与低内存模式

批量模式适合同一模型架构的多个训练 checkpoint；一个命令只接受一个目标 config，不根据文件名猜测架构：

```bash
uv run rtdetrv3-convert \
  --batch \
  --input 'pretrained_models/paddle/r18-runs/*.pdparams' \
  --output pretrained_models/pytorch/r18-runs \
  --config configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --save-mapping pretrained_models/reports/r18-runs \
  --summary pretrained_models/reports/r18-runs.summary.json \
  --memory-efficient \
  --parameter-batch-size 64
```

- 输出文件使用源文件 stem 加 `.pth`；mapping 使用 stem 加 `.mapping.json`。
- 未指定 `--force` 时，已有 checkpoint 或已有同名 mapping 任一存在都会让该输入显式失败，不会因另一项尚不存在而静默覆盖。失败清理只删除本次新建的产物，不删除进入命令前已有的文件。
- 每个输入使用独立 conversion session。单文件失败会记录错误并继续后续输入；只要存在失败，CLI 最终退出码为 1。
- JSON summary 记录总数、成功/失败数、耗时，以及每个文件的 session、转换/跳过参数数和错误。
- `torch.save` 先写目标目录中的临时文件，成功后原子替换，因此中途失败不会破坏已有目标。
- `--memory-efficient` 会在构建目标模型后只保留 key/shape 和 Linear key 集合，并在转换过程中分批释放源 tensor。Paddle checkpoint 仍会整体加载，最终 PyTorch state dict 仍会整体驻留；这不是流式 checkpoint 格式。
- **已验证**：官方 R18 严格模式、batch size 64 完成 571/571，wall time `5.71 s`，最大 RSS `925,780 KiB`。此前普通路径观测为 `1,056,232 KiB`，下降约 `12.4%`；这是多项改动的组合观测，不外推到其他变体。

## 已遇到的框架边界

- 部分 Paddle 版本的 `paddle.load` 不接受 `pathlib.Path`，在边界使用 `paddle.load(str(path))`。
- `Path` 写入 JSON 元数据会触发序列化错误，应在构建元数据时转为 `str`。
- 官方 R18 eval 配置为 640 输入预计算 position embedding；直接换成 64 输入会在 Paddle neck 出现 token 数 4 与 400 不匹配。这是配置/输入契约，不是框架数值分歧。
- transformer 的 top-k 选择是不连续操作；在检验两框架的细小浮点差异时，应固定 CPU 线程并优先比较 top-k 前激活，不能把候选顺序跳变直接归因为权重错误。
- 转换优化器、调度器或 AMP scaler 状态风险很高；当前仅转换模型参数，转换后用 PyTorch 重新初始化训练状态。
- 分布式训练产物可能带 `module.` 等 wrapper 前缀，量化、剪枝和自定义算子元数据不能假设可跨框架保留。

## 尚未完成

- 建议性 shape fix、流式 checkpoint 读写，以及跨架构 manifest 驱动的批量转换。
- R34/R50 的完整 COCO val2017 AP 对照，以及三个变体的完整训练收敛；三变体真实 COCO 单图对比已经完成。
- 优化器参数组、ResNet LR multiplier、weight decay 排除、裁剪/EMA 顺序与恢复语义；项目不要求 AdamW 更新逐元素完全一致。

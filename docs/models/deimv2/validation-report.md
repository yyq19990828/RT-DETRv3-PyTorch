# DEIMv2 验证报告

验证日期:2026-08-14。本报告记录 DEIMv2 八个官方 COCO 变体在本仓库的模型级验收方法、环境、结论与限制。数值明细见[指标记录](metrics.md),可执行入口见[证据索引](evidence-index.md)。

## 结论

- 八个官方 checkpoint 全部以 identity mapping 零未知 key 严格加载,容器为 `{"model": state_dict}`。
- 完整 val2017 bbox AP 与官方值最大绝对误差 `0.000398`,全部满足预注册门槛 `≤ 0.001`。
- HGNetv2 分支激活与上游逐位一致;DINOv3 分支激活首分歧在浮点噪声量级,raw 输出差异由 topk 近平局翻转主导,已量化界定。
- 两分支代表变体通过 reduced train/resume:epoch 边界恢复后首步 loss 与 LR 逐位一致。
- 导出:八变体 TorchScript 逐值一致;七变体 ONNX 默认容差通过,`deimv2-x` ONNX 按预注册 family-specific 例外接受(证据见下)。

## 验证环境与范围

- Python 3.12,PyTorch 2.5.1+cu121,torchvision 0.20.1,onnx 1.22.0,onnxruntime 1.27.0(与上游 requirements 锁定的 torch/torchvision 版本一致)。
- CUDA GPU 完整 val2017;CPU/FP32 逐激活对齐与导出。
- 上游 checkout:`Intellindust-AI-Lab/DEIMv2@add5bcdb499bf7b8a366bfeac1a47d3dc278de27`,以 identity strict-load 官方 checkpoint 后逐激活对比。
- 官方 checkpoint:Google Drive 下载,SHA-256 记录于 manifest;`rtdetrv3-models --family deimv2 verify` 通过。

## 数值对齐

- 固定随机输入(seed 42/20260813)、CPU/FP32,对比 backbone/encoder 前向钩子激活与 `exclude_post_process` raw 输出。
- HGNetv2 分支(n/pico/femto/atto):backbone、encoder、logits、boxes 全部 max_abs = 0。
- DINOv3 分支(x/l/m/s):backbone[1]/[2] 逐位一致,backbone 第 0 级(stride-8 STA 融合路径)首分歧 `≤ 1.9e-6`(`rtol=1e-5, atol=1e-6` 内);encoder 放大至 `1e-5` 量级。随机输入下 raw logits 发散(s 最大 `2.7`,x)由 top-300 边界近平局翻转主导:对 encoder 施加 `3e-6` 随机扰动测得 logits 放大约 256 倍,且 s 与 femto 的官方自评对照实验证明该量级噪声不改变化真实图像结论。
- 上游自评对照:使用 pinned 上游代码 + pycocotools 在同一 checkpoint、同一 COCO val2017 上评估,`deimv2-s` 为 `0.508649`、`deimv2-femto` 为 `0.309858`,与本仓库管线结果(`0.508602`/`0.309933`)一致到 `5e-5`/`8e-5`,证明 README 公布值即该 checkpoint 权重的评估值。

## 完整 val2017

八个变体逐个以官方 checkpoint 评估,命令形如 `rtdetrv3-eval -c configs/deimv2/<variant>.yml --checkpoint <official>.pth --device cuda`。逐变体 AP 见[指标记录](metrics.md);annotation SHA 与 prediction JSON SHA 待发布级验收时补充进 metrics。

## 训练与恢复

- reduced train/resume 使用 val2017 前 96 图派生的 COCO 格式子集(`instances_reduced_train.json`,663 标注),epoch=2、snapshot_epoch=1,增强窗口前移(policy_epochs `[0,1,2]`、mixup `[0,1]`、copyblend `[0,2]`)、matcher 切换 epoch 设为 1,以在短程内激活 Copy-Blend 与 IoU 排序匹配分支。
- `deimv2_dinov3_s`:epoch 0 首步 loss 49.0650 → epoch 1 首步 loss 43.7886(LR 2.5e-5);从 `epoch_1.pth` 恢复后续训 epoch 1 首步 loss 43.7886,逐位一致。
- `deimv2_hgnetv2_atto`(Copy-Blend prob 0.5 激活):epoch 1 首步 loss 34.3674,恢复后同为 34.3674,逐位一致。
- 训练 checkpoint 为仓库 format-version-1(含 EMA/optimizer/scheduler/training_state),无 teacher/distill 键。
- 范围声明:reduced 训练只验证训练链路机制与恢复确定性,不构成任何精度收敛或完整 schedule 证据。

## 推理与部署

- 导出命令 `rtdetrv3-export -c <config> --checkpoint <official>.pth --format both`,固定空间尺寸取自各变体 TestReader(640/416/320),opset 17、动态 batch。
- atto/femto/pico/n/s/m/l:TorchScript score/box max_abs = 0.0(batch 1/4 检查与逐图匹配通过)。
- `deimv2-x`:TorchScript 逐值一致;ONNX 在验证器随机输入(torch.rand,seed 20260813)下 297/300 匹配,未匹配项 score 分别为 0.0017/0.0019/0.0021(全集最低分 0.0016),为 top-300 底部近平局交换;真实图像验证 3/4 张 300/300,1 张 298/300 且两处差异为值级漂移(score 2.7e-5 与 1.4e-4,box 一致)。按 DEIM-RTv2 X 先例预注册为 family-specific 例外,记录于本报告与 metrics,不作为其他变体放宽依据。
- 评估与导出的坐标还原、`bbox`/`bbox_num` 输出契约与既有家族一致;错误尺寸/动态高宽输入按导出合同拒绝。

## 负例与限制

- 非法 `gamma ≤ 0`、`fuse_op` 非 cat/sum、不支持的 encoder `version`、剪枝变体加载无匹配张量的 B0 权重、`DINOv3STAs.weights_path` 指向缺失文件均在构建期失败。
- `num_top_queries` 缺省 100 会使完整 val2017 系统性偏低约 `0.001-0.002`;deimv2 配置显式固定为上游的 300,该问题已在验收前修复并以八个变体 ≤ 0.001 的复测关闭。
- 完整 schedule(72-500 epoch)、多 seed、低精度与吞吐性能不在本次验收范围。
- 官方权重仅 Google Drive 托管,manifest 只支持 list/verify;下载需经上游链接人工完成。

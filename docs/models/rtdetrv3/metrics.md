# RT-DETRv3 指标记录

> 2026-08-14 当前快照。所有 AP 均为 0-1 标度；历史环境和完整命令见[验证报告](validation-report.md)与[归档报告](../../archive/rtdetrv3-v0.1.0/reports/README.md)。

## 官方权重

| 变体 | 源文件大小 | 源 SHA-256 | 转换 tensor | 发布 `.pth` 大小 | 发布 SHA-256 |
|---|---:|---|---:|---:|---|
| R18 | 91,945,530 | `f32dbd008bd7e5311c877d522f6d8c9e349795978c889f53823588b5e5d74a5f` | 571 | 92,075,629 | `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547` |
| R34 | 137,016,081 | `29b09c64d6c372cde46d94caee1b57a23cee0aae24bd7bd3e2937cf57e581a68` | 681 | 137,170,947 | `e69207749b37e493596086579f435d5f08e9f058b66322452456053b78a4f272` |
| R50 | 182,331,170 | `e8b1d5db3208ce0f9edba5a914f23c918141b608ab4cd409db9d9204f7ed4b08` | 789 | 182,510,207 | `5e3e34ac3d3d14f57ebf6100b146b5702f8dface24fbe57cbc993f59381b67f7` |

R18-vd 训练初始化资产另含 `115` 个 tensor：源文件 `44,850,756` bytes、SHA-256 `68d7632cb67ad2c658fe67ab5837d8eb65466a7bc1574badc74860059ef5e7f0`；转换产物 SHA-256 `2483b5b00ed2b84192540bbd1bd1768e3e4422c2f8fa1598ae96e0c2d6f64db2`。

## R18 COCO val2017

条件为 5000 图、CPU/FP32、eval、640x640、batch 16、同源官方权重。

| 来源 | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| 官方模型表 | 0.481 | 0.662 | - | - | - | - |
| Paddle 独立复算 | 0.480477300367 | 0.656152367330 | 0.519499977301 | 0.307266486593 | 0.514806586690 | 0.639255472633 |
| PyTorch CPU | 0.480477134768 | 0.656151392882 | 0.519500286057 | 0.307266567713 | 0.514806052609 | 0.639255477387 |
| PyTorch CUDA 固定 JSON | 0.480502167075 | 0.656089305298 | 0.519446428999 | 0.307272 | 0.514910 | 0.639605 |

- Paddle/PyTorch CPU 主 AP 绝对差：`1.65599e-7`。
- score `>=0.3`：`53,780/53,780` 个预测完成同图、同类、坐标 L-infinity `<=1 px` 匹配。
- score `>=0.5`：`26,243/26,243` 全匹配，最大坐标差 `0.0133057 px`。
- 官方 AP50 与本机双框架结果约差 `0.006`；因本机 Paddle/PyTorch 一致，不能归因于转换。

## 数值与部署

R18 整体梯度合同实测 relative L2 `0.00434`、cosine `0.999991`、符号分歧率 `0.1304%`；门分别为 `<0.01`、`>0.9999`、`<0.5%`。

| 比较 | 输入 | 最大 score 误差 | 最大 box 误差 |
|---|---|---:|---:|
| R18 ONNX CPU vs eager CPU | 单图 | 1.49012e-6 | 9.15527e-5 px |
| R18 TorchScript CPU vs eager CPU | 单图 | 0 | 0 px |
| R18 ONNX CPU vs eager CPU | 四图 batch 4 | 6.82473e-6 | 0.000183105 px |
| R18 ONNX CUDA vs eager CUDA | 四图 batch 4 | 6.06865e-4 | 0.0238647 px |
| R18 TorchScript CPU vs eager CPU | 四图 batch 4 | 1.90735e-6 | 9.15527e-5 px |
| R18 TorchScript CUDA vs eager CUDA | 四图 batch 4 | 2.79218e-4 | 0.00872803 px |
| R34 ONNX CPU vs eager CPU | 导出矩阵最大值 | 9.4771e-6 | 0.011780 px |
| R50 ONNX CPU vs eager CPU | 导出矩阵最大值 | 1.8962e-5 | 0.005615 px |
| R34 ONNX CUDA vs eager CUDA | 四图 batch 4 | 0.00141865 | 0.0375671 px |
| R50 ONNX CUDA vs eager CUDA | 四图 batch 4 | 0.000972390 | 0.0349426 px |

最后两行是保留的未通过结果，不得写成严格数值批准。

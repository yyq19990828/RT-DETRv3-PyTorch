# 模型结构:骨干网络

`detrs.modeling.backbones` 提供各模型族使用的骨干:PResNet、ResNet、HGNetV2、CSPDarkNet、CSPResNet、ViT-Tiny 以及 DEIMv2 的 DINOv3 骨干适配。

`backbones/dinov3/` 是按上游 DEIMv2 同样方式裁剪的 Meta DINOv3 前向代码,vendored 保留其 DINOv3 License 头,不按 Apache-2.0 处理;本项目不 vendor、不再分发任何 DINOv3 骨干初始权重。

::: detrs.modeling.backbones

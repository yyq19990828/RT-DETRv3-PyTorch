# DEIM

DEIM 在本仓库中包含两个经过独立验收的产品分支，而不是一个可互换 checkpoint
的单一图：

- [DEIM-D-FINE](../deim-dfine/README.md)：N/S/M/L/X，HGNetv2 与 D-FINE decoder。
- [DEIM-RT-DETRv2](../deim-rtdetrv2/README.md)：S/M/M*/L/X，PResNet 与受限 RT-DETRv2 decoder。

两者固定到 `Intellindust-AI-Lab/DEIM@09d35d53d39ee3145a1e61e3a989b28b9468d1dd`
（Apache-2.0），共享 MAL、Dense O2O、FlatCosine 和两阶段 EMA 协议，但配置、
checkpoint 和部署容差不可混用。官方资产由上游托管，不进入本项目 Release。

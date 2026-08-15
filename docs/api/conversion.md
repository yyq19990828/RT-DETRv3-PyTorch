# 权重转换

`detrs.conversion` 实现 Paddle checkpoint 到 PyTorch 的参数名映射、张量布局转换(Linear 权重转置等)与转换后校验。该子包属于 `dev` extra 场景,模块内部在需要时才惰性导入 Paddle。转换协议与校验层级见[权重转换协议](../migrations/weight-conversion.md)。

::: detrs.conversion

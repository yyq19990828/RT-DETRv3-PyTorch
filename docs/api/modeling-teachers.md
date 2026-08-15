# 模型结构:教师模型

`detrs.modeling.teachers` 实现训练期教师模型封装,当前用于 RT-DETRv4 的 DINOv3 特征蒸馏。teacher 只在训练时构造;student 的 eval、infer 与 export 不访问 teacher 资产。

::: detrs.modeling.teachers

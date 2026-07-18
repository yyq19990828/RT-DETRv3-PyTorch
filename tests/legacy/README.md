# Legacy tests

这里保留迁移早期针对 `rtdetrv3_pytorch.models`、分类 Registry、旧版构建器和旧模型构造参数编写的测试。

当前可安装包使用 `ppdet_pytorch`、`ppdet_pytorch.core.workspace` 以及 Paddle 风格的统一注册系统；旧测试的前提与现有实现不兼容，因此默认 pytest 不收集本目录。若后续需要恢复其中的覆盖场景，应按当前公开 API 重写后移回 `tests/unit`、`tests/integration` 或 `tests/numerical`，不要重新引入旧包兼容层。

## M1 场景去向（2026-07-18）

| 历史场景 | 当前证据 | 处理 |
|---|---|---|
| Registry/workspace 字典构建、命名配置、参数优先级 | `tests/unit/core/test_workspace.py` | 按当前 `workspace` API 重写 |
| R18-vd backbone 输出 shape、`out_shape`、梯度 | `tests/unit/modeling/test_r18_components.py` | 按当前 API 重写 |
| DINOv3 head 解码层选择、PPYOLOE DFL、DETR 匹配索引与 VFL 梯度 | `tests/unit/modeling/test_r18_components.py` 和 `test_training_losses.py` | 按 M1 训练路径重写 |
| DETR post-process top-k、类别和坐标缩放 | `tests/unit/modeling/test_r18_components.py` | 按当前 `DETRPostProcess` API 重写 |
| R18 config 模型注入、batch、loss、backward、optimizer 和 5-step 训练 | `tests/integration/test_rtdetrv3_training_chain.py` | 按当前 Trainer 链路重写 |
| R34/R50/R101、冻结边界、可变输入、空 GT、Eval/Infer 和保存/恢复 | `ROADMAP.md` M2、M3、M5 | 非 M1 必要场景，保留待重写 |
| 历史 numerical 用例 | 尚无同 checkpoint/输入的双框架证据 | 不计为已迁移，待 M2 |

本轮不删除历史文件：当前活跃测试只替代 M1 直接所需的合同，并未一对一覆盖所有边界。

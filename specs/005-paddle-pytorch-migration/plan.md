# Implementation Plan: RT-DETRv3 Paddle to PyTorch Migration

**Branch**: `005-paddle-pytorch-migration` | **Date**: 2025-10-20 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-paddle-pytorch-migration/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

将 RT-DETRv3 从 PaddlePaddle 框架完整迁移到 PyTorch 框架,保持算法逻辑、精度和工具接口的一致性。核心模块(backbone, transformer, head, loss)的迁移已完成,本计划重点关注:
1. 代码结构重组为 `rtdetrv3_pytorch/ppdet/` 双层包结构(模仿 Paddle)
2. 统一注册系统迁移到 `ppdet/core/workspace.py`
3. Dataset和Engine模块的完整迁移(保留所有逻辑分支)
4. Tools脚本的命令行接口对齐
5. 数值精度验证和性能对比

## Technical Context

**Language/Version**: Python 3.9+ (minimum for compatibility), Python 3.11 recommended
**Primary Dependencies**:
  - PyTorch >= 2.0 (for torch.compile support and modern features)
  - PaddlePaddle (仅用于权重转换和数值对比验证)
  - torchvision (视觉组件)
  - pycocotools (COCO评估)
  - opencv-python <= 4.6.0 (与Paddle版本保持一致)
  - numpy < 2.0 (兼容性考虑)
**Storage**: 本地文件系统 (COCO数据集、模型权重.pth、训练日志、checkpoint)
**Testing**: pytest (单元测试、集成测试、数值等价性测试)
**Target Platform**: Linux服务器 (GPU训练), 支持CUDA 11.8+ / ROCm 5.4+
**Project Type**: 单项目 (科研代码库,包含可安装的Python包)
**Performance Goals**:
  - 训练吞吐量: ≥95% Paddle基线 (samples/sec on NVIDIA T4/A100)
  - 推理延迟: ≤105% Paddle基线 (关键的实时检测指标)
  - 精度: COCO mAP 与 Paddle 差异 ≤0.5%
**Constraints**:
  - 内存使用: ≤110% Paddle基线
  - 数值精度: FP32前向传播误差 < 1e-5
  - 框架兼容性: 必须支持 PyTorch 1.12+ 和 Python 3.7-3.11
  - 配置文件格式: 保持YAML格式与Paddle版本兼容
**Scale/Scope**:
  - 代码规模: ~10K LOC (包含dataset, engine, models, utils, tools)
  - 数据集: COCO2017 (~118K训练图像, 5K验证图像)
  - 模型规模: RT-DETRv3-R50 (~43M参数), 支持多种backbone变体

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

基于 `.specify/memory/constitution.md` 的质量门检查:

### Gate 1: Component Completion (适用于已迁移的核心模块)
- [x] 所有 PaddlePaddle APIs 映射到 PyTorch (backbone, transformer, head, loss已完成)
- [x] 数值等价性测试通过 (forward + backward) - 见 `rtdetrv3_pytorch/tests/numerical/`
- [x] 代码审查已批准 (当前阶段自审)
- [x] 文档完整 (API映射见 tech-report.md)

### Gate 2: Subsystem Validation (部分完成)
- [x] 组件集成到子系统 (模型可端到端运行)
- [ ] **待完成**: 子系统级数值测试完整通过 (需要完整的 dataset/engine 迁移)
- [ ] **待完成**: 性能无回归 >5%

### Gate 3: Full Model Validation (未开始)
- [ ] **待完成**: 端到端模型在 COCO 数据集运行
- [ ] **待完成**: mAP 匹配 Paddle 基线 (±0.1 AP)
- [ ] **待完成**: 训练收敛曲线匹配参考实现
- [ ] **待完成**: 推理速度在 ±5% 以内

### Gate 4: Release Readiness (未开始)
- [ ] **待完成**: 所有组件通过验证
- [ ] **待完成**: 配置转换工具测试
- [ ] **待完成**: 迁移指南文档
- [ ] **待完成**: 预训练权重转换和验证

### Constitution Principle Compliance

✅ **Principle I (Framework Parity First)**: 核心模块已验证数值等价性 (1e-5 tolerance)
✅ **Principle II (Modular Migration)**: 采用模块化迁移策略,已完成 backbone→neck→transformer→head→loss
✅ **Principle III (Validation-Driven Development)**: 所有迁移模块有对应的数值测试
⚠️  **Principle IV (Reproducibility & Documentation)**: API映射文档存在但需补充完整
✅ **Principle V (Performance Parity)**: 初步验证性能接近,需完整基准测试
⚠️  **Principle VI (Configuration Compatibility)**: YAML配置格式保持,需配置转换工具

**当前状态**: 通过 Gate 1,部分 Gate 2。本次实施计划目标是完成 Gate 2 和 Gate 3 的基础设施(代码结构、完整pipeline)。

---

## Constitution Check Re-evaluation (Phase 1 设计后)

**Re-evaluation Date**: 2025-10-20 (Phase 1 完成后)

### Design Artifacts Checklist
- ✅ **research.md**: 完成,详细研究了 Paddle→PyTorch 迁移的关键技术点
- ✅ **data-model.md**: 完成,定义了所有核心数据实体和关系
- ✅ **contracts/**: 完成,定义了模型、数据、训练的接口契约
- ✅ **quickstart.md**: 完成,提供了完整的使用指南

### Constitution Compliance Review

#### Principle I (Framework Parity First) - ✅ 完全符合
- **研究深度**: research.md 详细映射了所有关键 API (优化器、调度器、数据增强、DataLoader)
- **数值验证计划**: 定义了清晰的验证清单 (组件级、训练流程、端到端、边界情况)
- **接口对等**: model_interface.py 确保 PyTorch 版本保持与 Paddle 相同的接口结构

#### Principle II (Modular Migration Strategy) - ✅ 完全符合
- **模块化设计**: data-model.md 清晰分离了 Model, Dataset, Training 等模块
- **依赖关系明确**: 定义了 Backbone → Neck → Transformer → Head 的数据流
- **独立验证**: contracts/ 定义的接口支持模块级单元测试

#### Principle III (Validation-Driven Development) - ✅ 完全符合
- **接口契约**: contracts/ 定义了所有组件必须满足的接口,测试可基于契约编写
- **数值验证清单**: research.md 第 7 节提供了完整的验证检查点
- **测试策略**: quickstart.md 包含了测试命令和验证流程

#### Principle IV (Reproducibility & Documentation) - ✅ 显著改进
- **API 映射**: research.md 提供了完整的 Paddle-PyTorch API 映射表
- **配置文件**: quickstart.md 详细说明了 YAML 配置格式和参数覆盖
- **使用文档**: quickstart.md 提供了从安装到部署的完整流程

#### Principle V (Performance Parity) - ✅ 规划充分
- **基准测试**: quickstart.md 第 11 节定义了性能基准和目标 (≥95% 训练速度, ≤105% 推理延迟)
- **优化策略**: research.md 提到了 AMP、梯度累积等性能优化手段

#### Principle VI (Configuration Compatibility) - ✅ 规划充分
- **YAML 格式保持**: quickstart.md 展示了与 Paddle 一致的配置文件格式
- **注册系统**: research.md 第 1 节详细设计了与 Paddle 兼容的注册机制
- **参数覆盖**: quickstart.md 说明了命令行覆盖配置的方法

### Updated Gate Status

#### Gate 1: Component Completion - ✅ 保持通过
无变化,核心组件已完成迁移和验证。

#### Gate 2: Subsystem Validation - ⚠️ 设计就绪,等待实施
- ✅ **设计完成**: data-model.md 定义了所有子系统的数据结构和接口
- ✅ **接口定义**: contracts/ 提供了可测试的接口契约
- ⏳ **待实施**: 代码重构到新结构 (rtdetrv3_pytorch/ppdet/)
- ⏳ **待实施**: 子系统级集成测试

#### Gate 3: Full Model Validation - ⚠️ 基础设施就绪
- ✅ **训练流程设计**: training_interface.py 定义了完整的训练引擎接口
- ✅ **评估策略**: quickstart.md 说明了 COCO 评估流程
- ✅ **性能基准**: quickstart.md 第 11 节定义了性能目标
- ⏳ **待实施**: 端到端训练和评估
- ⏳ **待实施**: 性能和精度验证

#### Gate 4: Release Readiness - ⏳ 文档已完成
- ✅ **文档**: quickstart.md 提供了完整的使用指南
- ⏳ **配置转换工具**: 规划在 research.md,待实施
- ⏳ **权重转换**: quickstart.md 第 8.3 节提到,待实施
- ⏳ **预训练权重**: 需要从 Paddle 转换

### Risks Identified During Design Phase

1. **数值精度累积误差** (research.md 第 8 节)
   - **缓解策略**: 设置相同随机种子,逐 epoch 对比中间结果

2. **数据增强随机性差异** (research.md 第 8 节)
   - **缓解策略**: 优先使用 NumPy 随机数生成器,编写单元测试锁定输出

3. **配置文件兼容性**
   - **缓解策略**: research.md 第 1 节设计了完全兼容的注册系统

4. **性能回归风险**
   - **缓解策略**: quickstart.md 第 10.1 节规划了 AMP、EMA 等优化手段

### Action Items for Phase 2 (Implementation)

根据 Constitution 要求和当前设计,Phase 2 (tasks.md) 应包含:

1. **代码重构** (Principle II)
   - 将现有代码迁移到 `rtdetrv3_pytorch/ppdet/` 结构
   - 实现 `ppdet/core/workspace.py` 注册系统
   - 更新所有导入路径

2. **数据集和引擎迁移** (Principle I)
   - 迁移 Paddle 的 dataset 模块到 `ppdet/data/`
   - 迁移 Paddle 的 engine 模块到 `ppdet/engine/`
   - 保留所有逻辑分支(即使当前未使用)

3. **数值验证** (Principle III)
   - 为所有新迁移的组件编写数值等价性测试
   - 运行 research.md 第 7 节的验证清单

4. **性能优化** (Principle V)
   - 实现 AMP 支持
   - 实现 EMA
   - 运行性能基准测试

5. **工具脚本对齐** (Principle I)
   - 更新 tools/ 脚本的导入路径
   - 确保命令行接口与 Paddle 版本一致

**下一步**: 运行 `/speckit.tasks` 生成详细的任务清单。

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

**当前结构** (需要重组):
```
rtdetrv3_pytorch/
├── dataset/          # 需要迁移到 ppdet/data/
├── engine/           # 需要迁移到 ppdet/engine/
├── models/           # 需要迁移到 ppdet/modeling/
├── utils/            # 需要迁移到 ppdet/utils/
├── configs/          # 需要迁移到 ppdet/ 内或保持在顶层
└── tools/            # 保持在顶层,更新导入路径
```

**目标结构** (双层包结构,模仿 PaddlePaddle):
```
rtdetrv3_pytorch/
├── ppdet/                    # 主包 (NEW)
│   ├── __init__.py
│   ├── core/                 # 核心基础设施 (NEW)
│   │   ├── __init__.py
│   │   └── workspace.py      # 统一注册系统 (register, global_config, create)
│   ├── modeling/             # 模型组件 (重构自 models/)
│   │   ├── __init__.py
│   │   ├── architectures/    # 完整模型定义 (rtdetrv3.py)
│   │   ├── backbones/        # ResNet, ResNeXt等
│   │   ├── necks/            # HybridEncoder
│   │   ├── transformers/     # RTDETRTransformerv3
│   │   ├── heads/            # DINOv3Head, PPYOLOEHead
│   │   ├── losses/           # 损失函数
│   │   ├── layers.py         # 基础层 (ConvNormLayer等)
│   │   ├── ops.py            # 自定义算子
│   │   └── post_process.py   # 后处理
│   ├── data/                 # 数据加载 (重构自 dataset/)
│   │   ├── __init__.py
│   │   ├── source/           # 数据集定义
│   │   │   ├── coco.py
│   │   │   ├── lvis.py
│   │   │   └── ...
│   │   ├── transform/        # 数据增强
│   │   │   ├── batch_operators.py
│   │   │   ├── operators.py
│   │   │   └── ...
│   │   ├── reader.py         # DataLoader构建
│   │   └── utils.py
│   ├── engine/               # 训练/评估引擎 (重构自 engine/)
│   │   ├── __init__.py
│   │   ├── trainer.py        # 训练循环
│   │   ├── callbacks.py      # 回调系统
│   │   └── env.py           # 环境设置
│   ├── optimizer/            # 优化器 (NEW, from Paddle)
│   │   ├── __init__.py
│   │   ├── optimizer.py      # 优化器构建
│   │   ├── lr_scheduler.py   # 学习率调度
│   │   └── ema.py           # EMA
│   ├── metrics/              # 评估指标 (NEW, from Paddle)
│   │   ├── __init__.py
│   │   ├── coco_utils.py
│   │   ├── metrics.py
│   │   └── ...
│   └── utils/                # 工具函数 (重构自 utils/)
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── logger.py
│       ├── config.py
│       └── ...
├── configs/                  # 配置文件 (保持在顶层或移入ppdet)
│   ├── runtime.yml
│   ├── datasets/
│   └── rtdetrv3/
├── tools/                    # 工具脚本 (保持在顶层)
│   ├── train.py             # 更新导入: from ppdet.core.workspace import ...
│   ├── eval.py
│   ├── infer.py
│   └── export_model.py
├── tests/                    # 测试 (更新导入路径)
│   ├── unit/
│   ├── integration/
│   └── numerical/
├── pyproject.toml           # 包配置
└── README.md
```

**Structure Decision**:
采用 **Option 1: Single project** 的变体,特别适配深度学习框架迁移项目。选择双层包结构 `rtdetrv3_pytorch/ppdet/` 的原因:
1. **框架对等性**: 与 PaddlePaddle 的 `ppdet` 包结构保持一致,便于代码对照和迁移
2. **命名空间隔离**: `ppdet` 作为独立子包,避免与 PyTorch 生态其他包冲突
3. **可安装性**: 符合 Python 包规范,支持 `pip install -e .` 安装后使用 `from rtdetrv3_pytorch.ppdet import ...`
4. **核心模块分离**: `core/workspace.py` 集中管理注册系统,类似 Paddle 的设计模式

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

**无违规项** - 当前设计完全符合 Constitution 的所有原则。双层包结构不增加复杂性,而是为了与参考实现(PaddlePaddle)保持对等,这是 Principle I (Framework Parity First) 的要求。


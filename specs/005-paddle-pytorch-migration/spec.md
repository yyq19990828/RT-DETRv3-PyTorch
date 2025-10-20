# Feature Specification: RT-DETRv3 Paddle to PyTorch Migration

**Feature Branch**: `005-paddle-pytorch-migration`
**Created**: 2025-10-20
**Status**: Draft
**Input**: User description: "codebase的任务是迁移RT-DETRv3的整个pipeline从paddle版本到pytorch版本, 目前已经迁移了大部分核心代码.
更多的需求:
1) 优化代码结构与paddle版本一致. rtdetrv3_pytorch
rtdetrv3_pytorch/engine
rtdetrv3_pytorch/dataset
rtdetrv3_pytorch/models
rtdetrv3_pytorch/utils
应该放到一个可以作为后续安装包的文件夹里
2) rtdetrv3_pytorch/dataset
rtdetrv3_pytorch/engine
中的文件构建组件的方式应该和paddle保持一致, 虽然有很多逻辑分支没有用到 但应该保留,便于后续的扩展
3) RT-DETRv3-paddle/tools 中的工具脚本应该和paddle版本对应的脚本完全一致"

## Clarifications

### Session 2025-10-20

- Q: 具体的包结构层次应采用什么方案? → A: 双层结构 `rtdetrv3_pytorch/ppdet/{engine,dataset,models,utils}`,模仿Paddle的ppdet子包结构
- Q: 迁移模块范围是否需要完整迁移所有核心模块? → A: 已实现迁移,不需要额外操作
- Q: 当前代码重组到新结构的策略? → A: 直接就地重构,同步更新测试文件
- Q: 配置文件中的导入路径是否需要更新? → A: 配置文件应该不受路径影响
- Q: 工具脚本的调整范围? → A: 仅更新内部导入语句以适配新包结构,保持命令行接口和参数完全不变
- Q: 注册机制的统一位置? → A: 移动到`ppdet/core/workspace.py`,类似PaddlePaddle的core模块
- Q: 需要注册哪些组件类型? → A: 与Paddle保持一致,使用统一的`@register`装饰器和`global_config`,支持所有组件(models, optimizer, data transforms等)的注册
- Q: 注册机制重构后的向后兼容性策略? → A: 完全迁移到统一`@register`装饰器,移除旧的分类注册表(BACKBONE_REGISTRY等),更新所有现有组件

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 完整的PyTorch训练流程 (Priority: P1)

研究人员需要使用PyTorch框架训练RT-DETRv3模型,完成从数据加载、模型训练到模型评估的完整流程。

**Why this priority**: 这是核心功能,直接决定了迁移项目是否可用。训练流程是所有后续功能的基础。

**Independent Test**: 通过运行训练脚本(如`tools/train.py`)并成功完成一个epoch的训练,验证模型能正确加载数据、前向传播、计算损失、反向传播并更新参数。

**Acceptance Scenarios**:

1. **Given** 配置文件和COCO数据集已准备, **When** 用户运行`python tools/train.py -c configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml`, **Then** 训练流程正常启动,模型正确加载,第一个epoch成功完成并输出loss值
2. **Given** 训练进行中, **When** 用户中断训练并从checkpoint恢复, **Then** 训练从正确的epoch继续,loss值保持连续性
3. **Given** 训练完成, **When** 用户运行评估命令, **Then** 输出COCO mAP指标,与Paddle版本的精度差异在±0.5%以内

---

### User Story 2 - 代码结构标准化与可安装性 (Priority: P2)

开发者需要将PyTorch版本的代码组织成与Paddle版本一致的结构,并支持作为Python包安装,以便于分发和集成。

**Why this priority**: 代码结构一致性保证了长期可维护性,可安装包功能便于用户使用和集成到其他项目。

**Independent Test**: 运行`pip install -e .`成功安装包,并能通过`import rtdetrv3_pytorch`导入核心模块,验证目录结构符合Python包规范。

**Acceptance Scenarios**:

1. **Given** PyTorch实现代码已完成, **When** 开发者检查目录结构, **Then** 所有核心模块(core, engine, dataset, models, utils)位于`rtdetrv3_pytorch/ppdet/`包目录下,采用与Paddle相同的双层结构,其中core模块包含workspace.py统一注册系统
2. **Given** 包含setup.py/pyproject.toml的安装配置, **When** 用户执行`pip install -e .`, **Then** 安装成功,可以在任意位置导入`rtdetrv3_pytorch`模块
3. **Given** 安装完成的包, **When** 用户导入`from rtdetrv3_pytorch.ppdet.models import RTDETRV3`, **Then** 成功导入模型类,无模块缺失错误

---

### User Story 3 - 数据集与引擎组件的完整迁移 (Priority: P2)

开发者需要将Paddle版本的dataset和engine模块完整迁移到PyTorch,保留所有逻辑分支(即使当前未使用),确保未来可扩展性。

**Why this priority**: 保留完整逻辑分支确保未来功能扩展时无需重写基础组件,减少技术债务。

**Independent Test**: 对比Paddle和PyTorch版本的dataset/engine模块,验证所有公共接口和配置选项均已实现,未使用的分支通过单元测试覆盖。

**Acceptance Scenarios**:

1. **Given** Paddle版本的COCODataset类支持多种数据增强选项, **When** 检查PyTorch版本的实现, **Then** 所有数据增强选项(包括未默认启用的)均已实现并可通过配置文件控制
2. **Given** Paddle版本的Trainer支持多种训练策略(如AMP、梯度累积), **When** 对比PyTorch版本, **Then** 所有训练策略均已实现,即使默认未启用
3. **Given** PyTorch版本的组件构建方式, **When** 通过配置文件创建组件实例, **Then** 使用与Paddle版本相同的注册机制(如`@register`装饰器)和工厂函数

---

### User Story 4 - 工具脚本功能一致性 (Priority: P3)

用户需要使用与Paddle版本完全一致的工具脚本(如train.py, eval.py, export_model.py),确保命令行接口和使用习惯保持一致。

**Why this priority**: 工具脚本一致性降低用户学习成本,保证从Paddle迁移到PyTorch的平滑过渡。

**Independent Test**: 对比tools目录下所有脚本,验证命令行参数、配置文件格式、输出日志格式完全一致。

**Acceptance Scenarios**:

1. **Given** Paddle版本的train.py支持参数`-c`, `--eval`, `-o`, **When** 检查PyTorch版本, **Then** 支持完全相同的命令行参数,帮助信息一致,仅内部导入路径已更新适配新包结构
2. **Given** 相同的配置文件, **When** 分别在Paddle和PyTorch版本运行训练, **Then** 输出日志格式一致(epoch信息、loss名称、评估指标格式)
3. **Given** 训练完成的模型, **When** 运行export_model.py导出, **Then** PyTorch版本支持导出为ONNX/TorchScript格式(对应Paddle的导出格式)

---

### Edge Cases

- 当配置文件中指定了Paddle专有的优化器(如Momentum with L2Decay)时,PyTorch版本如何映射到等效的PyTorch优化器?
- 当模型使用了Paddle特有的层(如某些归一化层)时,如何保证数值精度在迁移后一致?
- 当用户在不同Python版本(3.7-3.11)和PyTorch版本(1.12-2.0)下运行时,如何确保兼容性?
- 当训练使用分布式多GPU时,PyTorch的DDP与Paddle的分布式API行为差异如何处理?
- 当数据集路径包含特殊字符或软链接时,数据加载器能否正确处理?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统必须基于已迁移的RT-DETRv3核心模块(已完成的backbone, transformer, head, loss等组件),保持算法逻辑与Paddle版本一致,无需额外迁移
- **FR-002**: 系统必须通过就地重构将现有代码组织为可安装的Python包,采用双层结构`rtdetrv3_pytorch/ppdet/{core,engine,dataset,models,utils}`,其中core模块包含workspace.py等核心基础设施,模仿Paddle的ppdet子包组织方式,同时更新所有相关测试文件的导入路径
- **FR-003**: 系统必须在dataset模块中实现Paddle版本的所有数据加载和预处理逻辑,包括当前未使用的数据增强分支
- **FR-004**: 系统必须在engine模块中实现完整的训练/评估引擎,支持Paddle版本的所有训练策略(AMP、梯度累积、学习率调度、EMA等)
- **FR-005**: 系统必须保持tools目录下工具脚本的命令行接口与Paddle版本完全一致,包括参数名称、默认值、帮助信息,仅更新脚本内部的导入语句以适配新包结构
- **FR-006**: 系统必须将现有的多注册表机制(BACKBONE_REGISTRY, HEAD_REGISTRY等)完全重构为与Paddle一致的统一注册系统,迁移到`ppdet/core/workspace.py`,使用单一的`@register`装饰器和`global_config`字典,移除所有旧的分类注册表常量,更新所有现有组件使用新的注册机制,支持所有组件类型(models, optimizer, lr_scheduler, data transforms等)的注册和动态创建,保持`__inject__`和`__shared__`注解的依赖注入功能
- **FR-007**: 系统必须验证迁移后的模型在COCO数据集上的精度与Paddle版本的差异在±0.5% mAP以内
- **FR-008**: 系统必须支持从Paddle checkpoint转换权重到PyTorch格式,保证数值一致性
- **FR-009**: 系统必须提供与Paddle版本相同的配置文件格式(YAML),支持所有配置项的映射
- **FR-010**: 系统必须保留Paddle版本中所有未使用但已实现的逻辑分支,通过配置文件控制启用/禁用

### Key Entities

- **Model Architecture**: RT-DETRv3完整模型,包括backbone(ResNet/ResNeXt), neck(HybridEncoder), transformer(RTDETRTransformerv3), head(DINOv3Head), auxiliary head(PPYOLOEHead)
- **Training Engine**: 训练循环管理器,负责epoch迭代、优化器更新、学习率调度、checkpoint保存、日志记录
- **Dataset**: COCO数据集加载器,包含图像读取、标注解析、数据增强(随机裁剪、翻转、颜色抖动、Mosaic、Mixup等)
- **Loss Function**: 多组损失函数,包括主分支loss(VFL+L1+GIoU)、辅助分支loss、o2m分支loss、denoising loss
- **Configuration**: 配置管理系统,支持YAML配置文件解析、参数覆盖、组件注册表查询
- **Checkpoint**: 模型权重和训练状态,包含模型参数、优化器状态、学习率、epoch数、EMA权重
- **Evaluation Metrics**: COCO评估指标,包括mAP、AP50、AP75、APs、APm、APl等

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 使用PyTorch版本在COCO val2017上训练RT-DETRv3-R50,达到与Paddle版本相差不超过0.5% mAP的精度(Paddle版本为53.6% AP)
- **SC-002**: 训练速度与Paddle版本相差不超过10%(在相同硬件和batch size下,以iterations/second衡量)
- **SC-003**: 所有工具脚本(train.py, eval.py, export_model.py)支持与Paddle版本相同的命令行参数,通过参数兼容性测试100%通过
- **SC-004**: 代码结构通过Python包规范检查,成功通过`pip install -e .`安装并可导入所有核心模块
- **SC-005**: Dataset和engine模块的代码覆盖率达到90%以上,包括未默认启用的逻辑分支,所有测试文件的导入路径已同步更新到新的包结构
- **SC-006**: 从Paddle权重转换到PyTorch后,前向传播输出的数值误差在1e-5以内(使用相同输入)
- **SC-007**: 支持PyTorch 1.12+和Python 3.7+版本,通过多版本兼容性测试矩阵(至少覆盖3个PyTorch版本和3个Python版本)

## Assumptions

1. **框架差异处理**: 假设Paddle和PyTorch的核心操作(卷积、归一化、激活函数)在数值上等价,对于差异较大的操作(如某些初始化方法)需要手动对齐
2. **依赖库可用性**: 假设PyTorch生态中存在Paddle依赖库的等效替代(如pycocotools, opencv, yaml解析库)
3. **配置文件兼容**: 假设可以使用相同的YAML配置文件格式,仅需调整框架特定的参数(如优化器名称、学习率调度器名称)
4. **分布式训练**: 假设使用PyTorch DDP替代Paddle的分布式API,行为上保持一致(如梯度同步、reduce操作)
5. **数据增强库**: 假设使用Albumentations或torchvision.transforms实现与Paddle相同的数据增强效果
6. **权重转换**: 假设Paddle的权重文件(.pdparams)可以通过脚本解析并映射到PyTorch的state_dict格式
7. **默认配置保留**: 假设所有Paddle版本中未使用但已实现的配置选项(如某些数据增强、训练策略)在PyTorch版本中默认禁用但代码保留
8. **测试数据一致**: 假设使用相同的COCO数据集版本和划分进行精度验证
9. **随机种子控制**: 假设通过设置相同的随机种子,PyTorch和Paddle版本在相同初始化下产生可比较的训练轨迹
10. **导出格式映射**: 假设Paddle的inference模型格式可以映射到PyTorch的ONNX或TorchScript格式,满足部署需求

## Dependencies

- **Paddle源代码**: 需要访问完整的RT-DETRv3-paddle代码库,作为迁移的参考实现
- **技术报告**: tech-report.md提供了Paddle版本的算法实现细节和代码映射关系
- **PyTorch框架**: 依赖PyTorch>=1.12,支持Deformable Attention等高级操作
- **COCO数据集**: 需要COCO2017数据集用于训练和验证
- **转换工具**: 需要开发或使用现有工具将Paddle权重转换为PyTorch格式
- **测试基础设施**: 需要单元测试框架(pytest)和精度验证脚本

## Out of Scope

- **模型架构改进**: 本次迁移不包含对RT-DETRv3算法本身的改进或优化,仅复现原有功能
- **新功能开发**: 不添加Paddle版本中不存在的新功能(如新的数据增强方法、新的损失函数)
- **性能优化**: 不进行PyTorch特定的性能优化(如算子融合、图优化),仅保证功能对齐
- **部署推理优化**: 不包含TensorRT、ONNX Runtime等推理引擎的专门优化
- **文档撰写**: 不包含用户手册、API文档的完整编写(仅保留必要的README和代码注释)
- **Web界面**: 不提供可视化训练监控界面或Web演示
- **多模态扩展**: 不扩展到视频目标检测或其他模态
- **AutoML集成**: 不集成自动超参数搜索或NAS功能

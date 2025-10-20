# Feature Specification: RT-DETRv3 Paddle to PyTorch Migration Completion

**Feature Branch**: `004-paddle-pytorch-migration`
**Created**: 2025-10-17
**Status**: Draft
**Input**: User description: "完善RT-DETRv3组件构建从paddle到pytorch的迁移, 参考文档 PADDLE_MIGRATION_SUMMARY.md 和 PADDLE_STYLE_MIGRATION.md"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Complete Component Registration System (Priority: P1)

开发者需要确保所有RT-DETRv3核心组件都已正确注册到相应的注册表中,使其能够通过PaddlePaddle风格的配置系统进行实例化。这是整个迁移工作的基础。

**Why this priority**: 这是系统架构的核心,没有完整的注册系统,后续的配置驱动构建和依赖注入都无法实现。

**Independent Test**: 可以通过运行验证脚本 `verify_paddle_migration.py` 来独立测试,验证所有8个核心组件是否都正确注册到各自的注册表中。

**Acceptance Scenarios**:

1. **Given** 所有核心组件类已定义, **When** 导入组件模块, **Then** 所有组件应自动注册到对应的注册表(ARCHITECTURE_REGISTRY, BACKBONE_REGISTRY, NECK_REGISTRY, TRANSFORMER_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY)
2. **Given** 组件已注册, **When** 调用 `REGISTRY.list()`, **Then** 应返回包含该组件名称的列表
3. **Given** 组件已注册, **When** 调用 `REGISTRY.get('ComponentName')`, **Then** 应返回组件类引用

---

### User Story 2 - Implement Dependency Injection Chain (Priority: P1)

开发者需要实现完整的依赖注入链,使得backbone、neck、transformer、head之间能够自动传递形状信息和关键属性,无需手动配置中间参数。

**Why this priority**: 依赖注入是PaddlePaddle构建模式的核心特性,直接影响用户体验和代码简洁性。

**Independent Test**: 可以通过创建配置字典并调用 `RTDETRv3.from_config(config)` 来测试,验证各组件是否正确接收上游组件的输出形状。

**Acceptance Scenarios**:

1. **Given** backbone配置, **When** 创建backbone实例, **Then** backbone应提供 `out_shape` 属性
2. **Given** backbone实例和neck配置, **When** 使用from_config创建neck, **Then** neck应自动接收backbone.out_shape作为input_shape
3. **Given** neck实例和transformer配置, **When** 创建transformer, **Then** transformer应自动接收neck的输出形状信息
4. **Given** transformer实例和head配置, **When** 创建head, **Then** head应自动接收transformer的hidden_dim等属性

---

### User Story 3 - Enable Config-Driven Model Building (Priority: P2)

开发者需要能够通过YAML配置文件或Python字典来构建完整的RT-DETRv3模型,而无需手动实例化每个组件。

**Why this priority**: 这是PaddlePaddle风格的标志性特性,极大提升了配置的可读性和可维护性,但可以在核心注册和注入功能完成后实现。

**Independent Test**: 可以通过加载示例配置文件并调用 `create('RTDETRv3', global_config=config, **config['RTDETRv3'])` 来测试,验证模型是否成功构建。

**Acceptance Scenarios**:

1. **Given** 包含所有组件配置的字典, **When** 调用create()函数, **Then** 应返回完整配置的RTDETRv3模型实例
2. **Given** 全局配置(num_classes, hidden_dim), **When** 创建模型, **Then** 共享配置应自动传递给所有需要的组件
3. **Given** 部分配置缺失, **When** 尝试创建模型, **Then** 系统应使用合理的默认值或报错说明缺失的必需参数

---

### User Story 4 - Maintain Backward Compatibility (Priority: P2)

现有代码使用直接实例化方式创建模型,迁移完成后这些代码应继续正常工作,无需修改。

**Why this priority**: 保证现有代码不受破坏,降低迁移风险,但优先级低于核心功能实现。

**Independent Test**: 运行现有的测试套件或示例代码,验证直接实例化方式仍然有效。

**Acceptance Scenarios**:

1. **Given** 旧代码使用 `model = RTDETRv3(num_classes=80)`, **When** 运行代码, **Then** 模型应成功创建且功能正常
2. **Given** 旧代码手动创建各组件并传递, **When** 运行代码, **Then** 应与原来行为一致
3. **Given** 新注册系统已实现, **When** 使用旧方式实例化, **Then** 不应产生任何警告或错误

---

### User Story 5 - Add Comprehensive Validation Tools (Priority: P3)

开发者需要验证工具来检查迁移的完整性和正确性,包括组件注册状态、依赖注入功能、配置加载等。

**Why this priority**: 验证工具有助于发现问题,但不是核心功能,可以在主要功能完成后添加。

**Independent Test**: 运行验证脚本,查看输出报告,确认所有检查项通过。

**Acceptance Scenarios**:

1. **Given** 验证脚本, **When** 运行脚本, **Then** 应报告所有已注册组件的列表
2. **Given** 组件缺失__category__或__inject__注解, **When** 运行验证, **Then** 应给出警告信息
3. **Given** 依赖注入链断裂, **When** 运行验证, **Then** 应识别并报告问题位置

---

### Edge Cases

- What happens when 配置中指定的组件类型不存在? 系统应抛出清晰的KeyError错误信息,说明哪个组件未注册以及可用的组件列表。
- How does system handle 循环依赖? 系统应在from_config()执行时检测循环依赖并抛出错误,防止无限递归。
- What happens when backbone没有out_shape属性? neck应使用默认的输入形状参数,或在无法推断时提示用户手动指定input_shape。
- How does system handle 全局配置与局部配置冲突? 应优先使用局部配置(组件级配置),全局配置作为默认值。
- What happens when from_config返回的kwargs与__init__参数不匹配? Python会在实例化时抛出TypeError,提示意外的关键字参数。
- What happens when 组件被多次注册? Registry应允许覆盖注册(可选警告),或抛出错误防止意外覆盖。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统MUST为每个组件类别提供独立的注册表(ARCHITECTURE_REGISTRY, BACKBONE_REGISTRY, NECK_REGISTRY, TRANSFORMER_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY)
- **FR-002**: 所有核心组件MUST使用@register装饰器注册到对应的注册表
- **FR-003**: 所有组件MUST定义__category__属性以标识其类别
- **FR-004**: 需要依赖注入的组件MUST定义__inject__属性列出需要注入的字段
- **FR-005**: RTDETRv3类MUST实现from_config()类方法以支持配置驱动构建
- **FR-006**: from_config()方法MUST实现依赖注入链:backbone.out_shape → neck, neck输出 → transformer, transformer属性 → head
- **FR-007**: Registry类MUST提供create()方法支持通过名称和配置实例化组件
- **FR-008**: 系统MUST提供全局create()函数作为PaddlePaddle风格的入口点
- **FR-009**: 组件MUST在直接实例化(旧方式)和通过注册表实例化(新方式)下行为一致
- **FR-010**: backbone组件MUST提供out_shape属性以供下游组件使用
- **FR-011**: 系统MUST支持全局配置(global_config)参数在组件间共享
- **FR-012**: 组件配置MUST支持嵌套结构(如backbone: {type: ResNet, depth: 50})
- **FR-013**: 系统MUST在组件未注册时抛出清晰的KeyError错误
- **FR-014**: 所有已实现的核心组件(RTDETRv3, ResNet, HybridEncoder, RTDETRTransformerv3, DINOv3Head, PPYOLOEHead, DINOv3Loss)MUST完成迁移
- **FR-015**: 验证脚本MUST能够列出所有已注册组件并报告迁移状态

### Key Entities

- **Registry**: 组件注册表,管理特定类别的所有组件类,提供注册、查询、实例化功能
- **Component**: 可注册的模型组件,包含__category__, __inject__, __shared__等元数据
- **Configuration**: 配置字典,包含组件类型(type)和构造参数,支持嵌套结构
- **Global Config**: 全局配置字典,包含所有组件共享的参数(如num_classes, hidden_dim)
- **Dependency Chain**: 依赖链,描述组件间的依赖关系(backbone → neck → transformer → head)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 所有8个核心组件(RTDETRv3, ResNet, HybridEncoder, RTDETRTransformerv3, DINOv3Head, PPYOLOEHead, DINOv3Loss, 1个额外组件)都成功注册到对应注册表
- **SC-002**: 开发者可以通过配置字典创建完整的RT-DETRv3模型,代码行数减少至少60%(从手动实例化的~50行减少到配置驱动的~20行)
- **SC-003**: 依赖注入链完整工作,backbone的out_shape能够自动传递到neck,neck的输出能够自动传递到transformer,无需手动传递参数
- **SC-004**: 现有的直接实例化代码(如`model = RTDETRv3(num_classes=80)`)继续正常工作,通过率100%
- **SC-005**: 验证脚本能够在2秒内完成所有组件的注册状态检查并输出完整报告
- **SC-006**: 代码结构与PaddlePaddle实现保持100%概念对应(注册、注入、共享机制一一对应)
- **SC-007**: 文档完整性达到100%,包括迁移指南、使用示例、API参考,每个核心概念都有配套说明

## Assumptions *(optional)*

- 假设所有核心组件已经在代码库中实现,只需要添加注册和注入功能
- 假设现有的组件构造函数签名不需要大幅修改
- 假设开发者熟悉Python装饰器和类方法的概念
- 假设配置文件使用YAML格式或Python字典,不需要支持其他格式(如JSON, TOML)
- 假设依赖注入链是单向的(无循环依赖),从backbone流向head
- 假设out_shape等属性在组件初始化后立即可用(不需要延迟计算)

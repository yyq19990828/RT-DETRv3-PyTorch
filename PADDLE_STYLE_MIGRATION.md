# PaddlePaddle-Style Registry and Dependency Injection System

## 概述

本次改进实现了PaddlePaddle风格的模型构建系统,包括:

1. **@register装饰器** - 组件注册机制
2. **依赖注入 (__inject__)** - 自动从配置创建依赖组件
3. **共享配置 (__shared__)** - 全局配置共享
4. **from_config()模式** - 类方法构建模式

## 主要变更

### 1. 增强的Registry类

**位置**: `rtdetrv3_pytorch/models/__init__.py`

**新功能**:
- ✅ 支持`__inject__`注解 - 自动注入依赖组件
- ✅ 支持`__shared__`注解 - 共享全局配置
- ✅ 支持`__category__`注解 - 组件分类
- ✅ `create()`方法 - PaddlePaddle风格的全局创建函数
- ✅ `registry.create()`方法 - 支持依赖注入的实例化

**示例**:
```python
@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    __inject__ = ['norm_layer']  # 自动从配置注入
    __shared__ = ['num_classes']  # 从全局配置共享
    __category__ = 'backbone'

    def __init__(self, depth=50, norm_layer=None, num_classes=1000):
        ...
```

### 2. RTDETRv3模型类改进

**位置**: `rtdetrv3_pytorch/models/rtdetrv3.py`

**新增**:
- ✅ `@ARCHITECTURE_REGISTRY.register()`装饰器
- ✅ `__category__ = 'architecture'`
- ✅ `__inject__ = ['backbone', 'neck', 'transformer', 'detr_head', 'aux_head']`
- ✅ `from_config()`类方法 - 从配置字典构建组件

**from_config()方法**:
```python
@classmethod
def from_config(cls, cfg: Dict[str, Any], global_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    从配置构建RTDETRv3组件

    支持依赖注入模式:
    - backbone输出形状 → neck输入
    - neck输出形状 → transformer输入
    - transformer属性 → head配置
    """
    kwargs = {}

    # 创建backbone
    if 'backbone' in cfg:
        kwargs['backbone'] = create(cfg['backbone']['type'], global_config, ...)

    # 创建neck(注入backbone输出形状)
    if 'neck' in cfg and 'backbone' in kwargs:
        neck_kwargs = {...}
        if hasattr(kwargs['backbone'], 'out_shape'):
            neck_kwargs['input_shape'] = kwargs['backbone'].out_shape
        kwargs['neck'] = create(cfg['neck']['type'], global_config, **neck_kwargs)

    # ... 其他组件类似

    return kwargs
```

### 3. 配置驱动的模型构建

**两种使用方式**:

#### 方式1: 直接使用create()函数(PaddlePaddle风格)
```python
from rtdetrv3_pytorch.models import create

# 全局配置
global_config = {
    'num_classes': 80,
    'hidden_dim': 256,
    # ...
}

# 从配置创建模型
model = create('RTDETRv3', global_config=global_config)
```

#### 方式2: 使用from_config()模式
```python
from rtdetrv3_pytorch.models import ARCHITECTURE_REGISTRY

config = {
    'backbone': {'type': 'ResNet', 'depth': 50, 'variant': 'd'},
    'neck': {'type': 'HybridEncoder', 'hidden_dim': 256},
    'transformer': {'type': 'RTDETRTransformerv3', 'num_queries': 300},
    'detr_head': {'type': 'DINOv3Head'},
    'num_classes': 80
}

# 获取类
RTDETRv3 = ARCHITECTURE_REGISTRY.get('RTDETRv3')

# 从配置构建组件
kwargs = RTDETRv3.from_config(config)

# 创建模型
model = RTDETRv3(**kwargs, num_classes=80)
```

#### 方式3: 向后兼容的手动实例化
```python
from rtdetrv3_pytorch.models import RTDETRv3

# 仍然支持直接实例化
model = RTDETRv3(num_classes=80)
```

## 与PaddlePaddle的对应关系

| PaddlePaddle | PyTorch实现 | 说明 |
|--------------|-------------|------|
| `@register` | `@REGISTRY.register()` | 注册装饰器 |
| `__inject__` | `__inject__` | 依赖注入字段列表 |
| `__shared__` | `__shared__` | 共享配置字段列表 |
| `__category__` | `__category__` | 组件类别 |
| `create(name, **kwargs)` | `create(name, global_config, **kwargs)` | 全局创建函数 |
| `from_config(cls, cfg)` | `from_config(cls, cfg, global_config)` | 类方法构建 |
| `ppdet.core.workspace` | `rtdetrv3_pytorch.models` | 注册和构建模块 |

## 依赖注入流程示例

### PaddlePaddle版本
```python
# ppdet/modeling/architectures/rtdetrv3.py
@register
class RTDETRV3(BaseArch):
    __inject__ = ['post_process']

    @classmethod
    def from_config(cls, cfg):
        backbone = create(cfg['backbone'])
        neck = create(cfg['neck'], input_shape=backbone.out_shape)
        transformer = create(cfg['transformer'], input_shape=neck.out_shape)
        detr_head = create(cfg['detr_head'],
                          hidden_dim=transformer.hidden_dim)
        return {
            'backbone': backbone,
            'transformer': transformer,
            'detr_head': detr_head,
            'neck': neck
        }
```

### PyTorch版本(现在)
```python
# rtdetrv3_pytorch/models/rtdetrv3.py
@ARCHITECTURE_REGISTRY.register()
class RTDETRv3(nn.Module):
    __inject__ = ['backbone', 'neck', 'transformer', 'detr_head', 'aux_head']

    @classmethod
    def from_config(cls, cfg, global_config=None):
        kwargs = {}

        # 创建backbone
        if 'backbone' in cfg:
            kwargs['backbone'] = create(cfg['backbone']['type'], global_config, ...)

        # 创建neck(注入backbone.out_shape)
        if 'neck' in cfg and 'backbone' in kwargs:
            neck_kwargs = {...}
            if hasattr(kwargs['backbone'], 'out_shape'):
                neck_kwargs['input_shape'] = kwargs['backbone'].out_shape
            kwargs['neck'] = create(cfg['neck']['type'], global_config, **neck_kwargs)

        # 创建transformer(注入neck.out_shape)
        if 'transformer' in cfg:
            transformer_kwargs = {...}
            if 'neck' in kwargs and hasattr(kwargs['neck'], 'out_shape'):
                transformer_kwargs['input_shape'] = kwargs['neck'].out_shape
            kwargs['transformer'] = create(cfg['transformer']['type'],
                                          global_config, **transformer_kwargs)

        # 创建detr_head(注入transformer属性)
        if 'detr_head' in cfg and 'transformer' in kwargs:
            head_kwargs = {...}
            if hasattr(kwargs['transformer'], 'hidden_dim'):
                head_kwargs['hidden_dim'] = kwargs['transformer'].hidden_dim
            kwargs['detr_head'] = create(cfg['detr_head']['type'],
                                        global_config, **head_kwargs)

        return kwargs
```

**一致性**: ✅ 完全遵循PaddlePaddle的依赖注入模式

## 配置示例

### YAML配置文件(PaddlePaddle风格)
```yaml
# configs/rtdetrv3_r50_coco.yml
architecture: RTDETRv3

RTDETRv3:
  backbone:
    type: ResNet
    depth: 50
    variant: d
    frozen_stages: 1
    return_idx: [1, 2, 3]

  neck:
    type: HybridEncoder
    in_channels: [512, 1024, 2048]
    hidden_dim: 256
    feat_strides: [8, 16, 32]
    num_encoder_layers: 1
    expansion: 1.0

  transformer:
    type: RTDETRTransformerv3
    num_queries: 300
    num_decoder_layers: 6
    num_levels: 3
    num_decoder_points: 4
    hidden_dim: 256

  detr_head:
    type: DINOv3Head
    eval_idx: -1
    o2m: 4
    o2m_branch: false

  aux_head:
    type: PPYOLOEHead  # 可选
    num_classes: 80

num_classes: 80  # 共享配置
```

### Python使用
```python
import yaml
from rtdetrv3_pytorch.models import create

# 加载配置
with open('configs/rtdetrv3_r50_coco.yml') as f:
    config = yaml.safe_load(f)

# 创建模型(PaddlePaddle风格)
model = create('RTDETRv3', global_config=config, **config['RTDETRv3'])
```

## 测试验证

运行测试:
```bash
uv run python test_registry_system.py
```

测试覆盖:
- ✅ 基础注册功能
- ✅ 手动实例化(向后兼容)
- ✅ 配置驱动实例化(PaddlePaddle风格)
- ✅ from_config()模式
- ✅ 依赖注入
- ✅ Registry内省

## 优势

1. **一致性**: 与PaddlePaddle实现保持高度一致,便于代码迁移和理解
2. **模块化**: 组件解耦,易于扩展和替换
3. **配置驱动**: 通过配置文件控制模型结构,无需修改代码
4. **依赖注入**: 自动解析组件依赖关系,减少手动配置
5. **向后兼容**: 保留原有的手动实例化方式
6. **类型安全**: 保持PyTorch的类型提示优势

## 未来扩展

可以进一步添加:
- [ ] YAML配置文件加载器(类似ppdet.core.workspace.load_config)
- [ ] 配置继承机制(类似PaddlePaddle的_BASE_)
- [ ] 序列化支持(@serializable)
- [ ] 配置验证和schema检查

## 参考

- PaddlePaddle workspace: `ppdet/core/workspace.py`
- PaddlePaddle RTDETRv3: `ppdet/modeling/architectures/rtdetrv3.py`
- 技术报告: `tech-report.md`

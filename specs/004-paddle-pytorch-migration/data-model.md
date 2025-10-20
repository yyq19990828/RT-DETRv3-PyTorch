# Data Model: Component Metadata & Configuration Schema

**Feature**: 004-paddle-pytorch-migration
**Date**: 2025-10-17
**Purpose**: Define the structure of component metadata, registry entries, and configuration dictionaries

## Overview

This document describes the data structures used in the RT-DETRv3 registry system for component registration, dependency injection, and config-driven model building.

## Entity 1: Registry

A Registry manages a collection of registered components of a specific category (e.g., backbones, heads, losses).

### Attributes

| Attribute | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `_name` | `str` | Registry category name | Required, immutable after init |
| `_registry` | `Dict[str, Type]` | Map of component names to classes | Keys unique within registry |
| `_config_cache` | `Dict[str, Dict]` | Cache of component configs | Optional, for optimization |

### Methods

| Method | Signature | Purpose | Return Value |
|--------|-----------|---------|--------------|
| `register(name=None)` | `(name: Optional[str]) -> Callable` | Decorator to register component | Decorated class (unchanged) |
| `get(name)` | `(name: str) -> Type` | Retrieve component by name | Component class |
| `create(name, global_config, **kwargs)` | `(name: str, global_config: Optional[Dict], **kwargs) -> Any` | Instantiate component with dependency injection | Component instance |
| `list()` | `() -> List[str]` | List all registered component names | List of names |
| `__contains__(name)` | `(name: str) -> bool` | Check if component registered | Boolean |

### Relationships

- **Composition**: Contains multiple Component classes
- **Hierarchy**: All registries aggregated in `ALL_REGISTRIES` list for cross-registry lookups

### State Transitions

```
Empty Registry
    ↓ register(component)
Registry with Components
    ↓ create(name)
Instantiated Component
```

### Invariants

1. Component names must be unique within a registry
2. Registered component must be a class (not instance)
3. Registry._name is immutable after construction

## Entity 2: Component (Registered Class)

A Component is any class registered with a Registry, annotated with metadata for dependency injection.

### Metadata Attributes (Class-Level)

| Attribute | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `__category__` | `str` | Component category | Registry name | `'backbone'`, `'head'` |
| `__inject__` | `List[str]` | Fields to inject from config | `[]` | `['backbone', 'neck']` |
| `__shared__` | `List[str]` | Fields from global config | `[]` | `['num_classes', 'hidden_dim']` |

### Constructor Contract

```python
def __init__(self, <injected_fields>, <shared_fields>, **kwargs):
    """
    Component constructor receives:
    - Injected fields: Instantiated dependencies (from __inject__)
    - Shared fields: Global config values (from __shared__)
    - **kwargs: Additional explicit parameters
    """
```

### Class Method Contract

```python
@classmethod
def from_config(cls, cfg: Dict[str, Any], global_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Build constructor kwargs from config dict.

    Args:
        cfg: Component-specific config (may include nested component configs)
        global_config: Global configuration for shared/inject resolution

    Returns:
        Dict of kwargs to pass to __init__

    Example:
        cfg = {'backbone': {'type': 'ResNet', 'depth': 50}, 'num_classes': 80}
        kwargs = RTDETRv3.from_config(cfg, global_config)
        # kwargs = {'backbone': <ResNet instance>, 'num_classes': 80}
    """
```

### Relationships

- **Dependency**: Component may depend on other components (via __inject__)
- **Registration**: Component belongs to one Registry
- **Instantiation**: Component can be created via direct call or Registry.create()

### Lifecycle States

```
Defined (class exists in code)
    ↓ @REGISTRY.register()
Registered (added to registry)
    ↓ REGISTRY.create(name) or direct call
Instantiated (instance created)
```

### Invariants

1. __inject__ fields must exist as __init__ parameters
2. __shared__ fields must exist as __init__ parameters with defaults
3. from_config() must return dict compatible with __init__ signature

## Entity 3: Component Configuration

A Configuration is a dictionary describing how to build a component, including its type and constructor parameters.

### Structure

```yaml
# Flat component config (no dependencies)
type: ResNet
depth: 50
variant: d
frozen_stages: 1

# Nested component config (with dependencies)
type: RTDETRv3
backbone:
  type: ResNet
  depth: 50
neck:
  type: HybridEncoder
  hidden_dim: 256
num_classes: 80
```

### Schema

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `type` | `str` | Yes | Component class name | Must be registered |
| `<param>` | `Any` | No | Constructor parameters | Type depends on component |
| `<dependency>` | `Dict` | No | Nested component config | Must have 'type' key |

### Validation Rules

1. **Required 'type' key**: All component configs must have 'type' field
2. **Type exists**: 'type' value must be a registered component name
3. **Nested configs**: Dependency configs must also have 'type' key
4. **No circular refs**: Dependency graph must be acyclic

### Example Transformations

**Input (YAML)**:
```yaml
architecture: RTDETRv3
RTDETRv3:
  backbone:
    type: ResNet
    depth: 50
  num_classes: 80
```

**Parsed (Python Dict)**:
```python
{
    'architecture': 'RTDETRv3',
    'RTDETRv3': {
        'backbone': {'type': 'ResNet', 'depth': 50},
        'num_classes': 80
    }
}
```

**After Resolution** (instances created):
```python
{
    'backbone': ResNet(depth=50),  # Instantiated
    'num_classes': 80
}
```

## Entity 4: Global Configuration

Global configuration contains shared parameters that multiple components need.

### Structure

```python
global_config = {
    'num_classes': 80,              # Shared by heads, losses
    'hidden_dim': 256,               # Shared by neck, transformer, head
    'num_queries': 300,              # Transformer-specific but global
    'backbone': {...},               # Component configs for injection
    'neck': {...},
    'transformer': {...}
}
```

### Fields

| Field | Type | Purpose | Access Pattern |
|-------|------|---------|----------------|
| Scalar values | `int`, `float`, `str`, `bool` | Shared config | Via __shared__ |
| Component configs | `Dict` with 'type' | Dependency specs | Via __inject__ |
| Lists/tuples | `List`, `Tuple` | Multi-value params | Direct access |

### Resolution Priority

When a parameter appears in multiple places:

1. **Explicit kwargs** (highest priority): `create('ResNet', depth=101)`
2. **Global config (__shared__)**: `global_config['num_classes']`
3. **Component config**: `config['RTDETRv3']['num_classes']`
4. **Constructor defaults** (lowest priority): `def __init__(self, num_classes=80)`

**Formula**: `final_value = explicit ?? shared ?? component_cfg ?? default`

## Entity 5: Dependency Chain

Dependency Chain describes the relationships between components in a model.

### Graph Structure

```
RTDETRv3 (root)
  ├─ backbone (ResNet)
  │    └─ out_shape → [used by neck]
  ├─ neck (HybridEncoder)
  │    ├─ input_shape ← backbone.out_shape
  │    └─ out_shape → [used by transformer]
  ├─ transformer (RTDETRTransformerv3)
  │    ├─ input_shape ← neck.out_shape
  │    └─ hidden_dim → [used by head]
  ├─ detr_head (DINOv3Head)
  │    └─ hidden_dim ← transformer.hidden_dim
  └─ aux_head (PPYOLOEHead, optional)
       └─ num_classes ← global_config
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `node` | `str` | Component name |
| `dependencies` | `List[str]` | Components this depends on (from __inject__) |
| `provides` | `Dict[str, Any]` | Attributes available for injection (e.g., out_shape) |
| `depth` | `int` | Depth in dependency graph (0 = no deps) |

### Topological Sort

Components must be instantiated in dependency order:

**Correct Order**:
1. backbone (depth 0, no dependencies)
2. neck (depth 1, depends on backbone)
3. transformer (depth 2, depends on neck)
4. head (depth 3, depends on transformer)

**Incorrect Order** (would fail):
1. head ❌ (needs transformer which doesn't exist yet)

### Cycle Detection

Circular dependencies are forbidden:

```python
# ❌ Invalid: A depends on B, B depends on A
class ComponentA:
    __inject__ = ['b']

class ComponentB:
    __inject__ = ['a']
```

**Detection**: Perform topological sort; if cycle exists, raise error before instantiation.

## Data Flow Diagrams

### 1. Component Registration Flow

```
Python Module Import
    ↓
@REGISTRY.register() decorator executes
    ↓
Extract __category__, __inject__, __shared__ from class
    ↓
Add class to Registry._registry dict
    ↓
Return class unchanged (non-invasive)
```

### 2. Config-Driven Instantiation Flow

```
Load YAML config
    ↓
Parse to Python dict
    ↓
REGISTRY.create(component_name, global_config, **cfg)
    ↓
Resolve __shared__ fields from global_config
    ↓
Resolve __inject__ dependencies (recursive create)
    ↓
Call from_config() if defined (build dependency chain)
    ↓
Call __init__ with merged kwargs
    ↓
Return component instance
```

### 3. Dependency Injection Flow

```
create('RTDETRv3', global_config={...})
    ↓
RTDETRv3.from_config(cfg, global_config)
    ↓
    create('ResNet', ...)
        ↓
        ResNet instance with out_shape
    ↓
    create('HybridEncoder', input_shape=backbone.out_shape)
        ↓
        HybridEncoder instance with out_shape
    ↓
    create('RTDETRTransformerv3', input_shape=neck.out_shape)
        ↓
        Transformer instance with hidden_dim
    ↓
    create('DINOv3Head', hidden_dim=transformer.hidden_dim)
        ↓
        Head instance
    ↓
Return {'backbone': ..., 'neck': ..., 'transformer': ..., 'head': ...}
    ↓
RTDETRv3(**kwargs) instantiated
```

## Validation Schema

### Registry Entry Validation

```python
def validate_registry_entry(cls: Type) -> None:
    """Validate component meets registry requirements"""
    # Check 1: Has __init__ method
    if not hasattr(cls, '__init__'):
        raise TypeError(f"{cls.__name__} must have __init__ method")

    # Check 2: __inject__ fields exist in __init__ signature
    if hasattr(cls, '__inject__'):
        init_params = inspect.signature(cls.__init__).parameters
        for field in cls.__inject__:
            if field not in init_params:
                raise ValueError(
                    f"{cls.__name__}.__inject__ contains '{field}' "
                    f"but __init__ has no such parameter"
                )

    # Check 3: __shared__ fields have defaults in __init__
    if hasattr(cls, '__shared__'):
        init_params = inspect.signature(cls.__init__).parameters
        for field in cls.__shared__:
            if field not in init_params:
                raise ValueError(
                    f"{cls.__name__}.__shared__ contains '{field}' "
                    f"but __init__ has no such parameter"
                )
            param = init_params[field]
            if param.default == inspect.Parameter.empty:
                raise ValueError(
                    f"{cls.__name__}.__shared__ field '{field}' "
                    f"must have default value in __init__"
                )
```

### Config Structure Validation

```python
def validate_component_config(cfg: Dict[str, Any]) -> None:
    """Validate component config structure"""
    # Check 1: Has 'type' key
    if 'type' not in cfg:
        raise ValueError(f"Component config missing 'type' key: {cfg}")

    # Check 2: Case sensitivity
    if 'Type' in cfg or 'TYPE' in cfg:
        raise ValueError("Use lowercase 'type' key")

    # Check 3: Nested configs also have 'type'
    for key, value in cfg.items():
        if isinstance(value, dict) and key != 'type':
            # This might be a nested component config
            # Recursively validate
            if 'type' in value or any(k.lower() == 'type' for k in value.keys()):
                validate_component_config(value)
```

## Usage Examples

### Example 1: Simple Component (No Dependencies)

```python
@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    __category__ = 'backbone'
    __inject__ = []
    __shared__ = []

    def __init__(self, depth=50, variant='d', **kwargs):
        super().__init__()
        self.depth = depth
        # ... implementation

    def _setup_out_shape(self):
        self.out_shape = [...]  # Provide for downstream

# Usage
backbone = BACKBONE_REGISTRY.create('ResNet', depth=50)
# or
backbone = create('ResNet', depth=50)
```

### Example 2: Component with Dependencies

```python
@ARCHITECTURE_REGISTRY.register()
class RTDETRv3(nn.Module):
    __category__ = 'architecture'
    __inject__ = ['backbone', 'neck', 'transformer', 'detr_head']
    __shared__ = ['num_classes']

    def __init__(self, backbone, neck, transformer, detr_head, num_classes=80, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        # ...

    @classmethod
    def from_config(cls, cfg, global_config=None):
        kwargs = {}

        # Create backbone
        if 'backbone' in cfg:
            kwargs['backbone'] = create(
                cfg['backbone']['type'],
                global_config,
                **{k: v for k, v in cfg['backbone'].items() if k != 'type'}
            )

        # Create neck with injected backbone shape
        if 'neck' in cfg and 'backbone' in kwargs:
            neck_cfg = cfg['neck'].copy()
            if hasattr(kwargs['backbone'], 'out_shape'):
                neck_cfg['input_shape'] = kwargs['backbone'].out_shape
            kwargs['neck'] = create(neck_cfg['type'], global_config, **neck_cfg)

        # ...
        return kwargs

# Usage
config = {
    'backbone': {'type': 'ResNet', 'depth': 50},
    'neck': {'type': 'HybridEncoder', 'hidden_dim': 256},
    # ...
}
model = create('RTDETRv3', global_config=config)
```

## Migration Mapping

### PaddlePaddle → PyTorch

| PaddlePaddle Concept | PyTorch Equivalent | Notes |
|----------------------|-------------------|-------|
| `@register` | `@REGISTRY.register()` | Decorator pattern same |
| `__inject__` | `__inject__` | Identical |
| `__shared__` | `__shared__` | Identical |
| `__category__` | `__category__` | Identical |
| `create(name)` | `create(name)` or `REGISTRY.create(name)` | Global function available |
| `from_config(cfg)` | `from_config(cfg, global_config)` | Added global_config parameter |
| `workspace.py` | `models/__init__.py` | Module location different |

## Glossary

- **Registry**: Collection of registered components by category
- **Component**: Registered class (backbone, head, loss, etc.)
- **Configuration**: Dictionary describing component type and parameters
- **Global Config**: Shared parameters across multiple components
- **Dependency Injection**: Automatic instantiation of component dependencies
- **Dependency Chain**: Graph of component relationships
- **from_config()**: Class method to build kwargs from config dict
- **__inject__**: Class attribute listing dependencies to auto-instantiate
- **__shared__**: Class attribute listing global config fields to use

---

**Document Status**: ✅ Complete
**Next Step**: Generate contracts/

# Contract: Component Protocol

**Feature**: 004-paddle-pytorch-migration
**Date**: 2025-10-17
**Purpose**: Define the protocol (interface) that registered components must follow

## Overview

This document specifies the requirements for components to participate in the registry system with dependency injection support.

## Component Requirements

### Requirement 1: Class-Based (MANDATORY)

Components MUST be classes (not functions or instances).

```python
# ✅ Valid
@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    pass

# ❌ Invalid
@BACKBONE_REGISTRY.register()
def create_resnet():  # Functions not supported
    pass
```

### Requirement 2: Constructor Signature (MANDATORY)

Components MUST have a`__init__` method.

```python
# ✅ Valid
class ResNet(nn.Module):
    def __init__(self, depth=50, **kwargs):
        super().__init__()
        self.depth = depth

# ❌ Invalid (no __init__)
class InvalidComponent(nn.Module):
    pass  # Missing __init__
```

### Requirement 3: Annotation Attributes (OPTIONAL)

Components MAY define these class-level attributes:

#### `__category__` (str, optional)

Specifies component category. Defaults to registry name if not provided.

```python
class ResNet(nn.Module):
    __category__ = 'backbone'  # Optional, defaults to registry name
```

**Contract**:
- Type: `str`
- Default: Registry name (e.g., 'backbone', 'head')
- Purpose: Categorize component for introspection

#### `__inject__` (List[str], optional)

Lists __init__ parameters that should be automatically created from config.

```python
class RTDETRv3(nn.Module):
    __inject__ = ['backbone', 'neck', 'transformer']  # Auto-create these

    def __init__(self, backbone, neck, transformer, **kwargs):
        # backbone, neck, transformer will be instances (not dicts)
        self.backbone = backbone
```

**Contract**:
- Type: `List[str]`
- Default: `[]` (empty list)
- Constraints:
  - All items MUST exist as `__init__` parameters
  - Registry.create() will recursively instantiate these from config
  - Parameters may be required (no default) or optional (with default)

**Error if violated**:
```python
class BadComponent(nn.Module):
    __inject__ = ['nonexistent']  # ❌ Not in __init__

    def __init__(self):
        pass

# Raises ValueError during registration or validation
```

#### `__shared__` (List[str], optional)

Lists __init__ parameters that should be populated from global_config.

```python
class DINOv3Head(nn.Module):
    __shared__ = ['num_classes', 'hidden_dim']  # From global config

    def __init__(self, num_classes=80, hidden_dim=256, **kwargs):
        # If global_config has these keys, use them; otherwise use defaults
        self.num_classes = num_classes
```

**Contract**:
- Type: `List[str]`
- Default: `[]` (empty list)
- Constraints:
  - All items MUST exist as `__init__` parameters
  - Parameters MUST have default values (since global_config may not provide them)
  - Registry.create() copies values from global_config to kwargs

**Error if violated**:
```python
class BadComponent(nn.Module):
    __shared__ = ['num_classes']

    def __init__(self, num_classes):  # ❌ No default value
        pass

# Raises ValueError during validation
```

### Requirement 4: from_config() Class Method (OPTIONAL)

Components MAY implement `from_config()` to customize instantiation from config.

#### Signature

```python
@classmethod
def from_config(
    cls,
    cfg: Dict[str, Any],
    global_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build constructor kwargs from config dict.

    Args:
        cfg: Component-specific configuration
        global_config: Global configuration for dependency resolution

    Returns:
        Dict of kwargs to pass to __init__

    Note:
        - Returned kwargs are MERGED with existing kwargs (explicit > from_config)
        - Use this for complex dependency chains (e.g., backbone → neck → head)
        - Recursive component creation happens here
    """
```

#### Example Implementation

```python
@ARCHITECTURE_REGISTRY.register()
class RTDETRv3(nn.Module):
    @classmethod
    def from_config(cls, cfg, global_config=None):
        from rtdetrv3_pytorch.models import create

        kwargs = {}

        # Create backbone
        if 'backbone' in cfg:
            backbone_cfg = cfg['backbone'].copy()
            backbone_type = backbone_cfg.pop('type')
            kwargs['backbone'] = create(backbone_type, global_config, **backbone_cfg)

        # Create neck with injected backbone shape
        if 'neck' in cfg and 'backbone' in kwargs:
            neck_cfg = cfg['neck'].copy()
            neck_type = neck_cfg.pop('type')
            # Inject dependency
            if hasattr(kwargs['backbone'], 'out_shape'):
                neck_cfg['input_shape'] = kwargs['backbone'].out_shape
            kwargs['neck'] = create(neck_type, global_config, **neck_cfg)

        return kwargs
```

**Contract**:
- **Input**: `cfg` (component config), `global_config` (global config)
- **Output**: `Dict[str, Any]` compatible with `__init__` signature
- **Behavior**:
  - Called by Registry.create() after resolving __inject__ and __shared__
  - Returned kwargs merged with existing kwargs (explicit args take priority)
  - Should NOT call `__init__` directly (Registry does that)

**When to use**:
- ✅ Complex dependency chains (backbone → neck → transformer)
- ✅ Need to inject intermediate values (e.g., backbone.out_shape)
- ✅ Custom config transformations
- ❌ Simple components with no dependencies (unnecessary overhead)

### Requirement 5: Dependency Provision (OPTIONAL)

Components that are dependencies of others SHOULD provide necessary attributes.

#### Example: Backbone Providing out_shape

```python
class ResNet(nn.Module):
    def __init__(self, depth=50, **kwargs):
        super().__init__()
        self.depth = depth
        self._setup_out_shape()  # Provide for downstream components

    def _setup_out_shape(self):
        """Compute output shape for each feature level"""
        if self.depth == 50:
            self.out_shape = [
                {'channels': 512, 'stride': 8},
                {'channels': 1024, 'stride': 16},
                {'channels': 2048, 'stride': 32}
            ]
```

**Contract**:
- Attribute name: Domain-specific (e.g., `out_shape`, `hidden_dim`)
- Type: Any (list, int, dict, etc.)
- Purpose: Downstream components access via `component.attribute_name`

**Best practices**:
- Set attribute in `__init__` or call helper method
- Document expected structure (e.g., list of dicts with 'channels', 'stride')
- Make it available immediately after construction

## Component Lifecycle

```
1. Class Definition
   ↓
2. @REGISTRY.register() decorator
   ↓ (sets __category__, __inject__, __shared__)
3. Class added to registry
   ↓
4. [Later] REGISTRY.create(name, ...)
   ↓
5. Resolve __shared__ from global_config
   ↓
6. Resolve __inject__ (recursive create)
   ↓
7. Call from_config() if defined
   ↓
8. Merge all kwargs
   ↓
9. Call __init__(**kwargs)
   ↓
10. Instance created
```

## Naming Conventions

### Component Names

**Registry Names** (registered as):
- PascalCase: `ResNet`, `DINOv3Head`, `RTDETRv3`
- Match class name (or use custom name in `@register('CustomName')`)
- No version suffixes in default registration (`ResNet` not `ResNet50`)

**Config Type Keys**:
```yaml
# Use exact registered name
backbone:
  type: ResNet  # Matches @BACKBONE_REGISTRY.register() on class ResNet
  depth: 50
```

### Parameter Names

**Convention**: snake_case

```python
# ✅ Good
def __init__(self, num_classes=80, hidden_dim=256):
    pass

# ❌ Bad
def __init__(self, NumClasses=80, hiddenDim=256):
    pass
```

### Attribute Names for Dependency Injection

**Common Attributes**:
- `out_shape`: Output shape info (backbones, necks)
- `hidden_dim`: Feature dimension (transformers)
- `num_queries`: Number of object queries (transformers)
- `in_channels`: Input channel count (necks, heads)

## Validation Rules

### Rule 1: __inject__ Parameters Must Exist

```python
def validate_inject(cls):
    init_params = inspect.signature(cls.__init__).parameters
    for field in getattr(cls, '__inject__', []):
        assert field in init_params, \
            f"__inject__ field '{field}' not in __init__ signature"
```

### Rule 2: __shared__ Parameters Must Have Defaults

```python
def validate_shared(cls):
    init_params = inspect.signature(cls.__init__).parameters
    for field in getattr(cls, '__shared__', []):
        assert field in init_params, \
            f"__shared__ field '{field}' not in __init__ signature"
        param = init_params[field]
        assert param.default != inspect.Parameter.empty, \
            f"__shared__ field '{field}' must have default value"
```

### Rule 3: from_config() Must Return Dict

```python
def validate_from_config(cls):
    if hasattr(cls, 'from_config'):
        result = cls.from_config({}, {})
        assert isinstance(result, dict), \
            "from_config() must return dict, got {type(result)}"
```

## Migration Checklist

When migrating a component to use the registry system:

- [ ] **Step 1**: Add `@REGISTRY.register()` decorator
- [ ] **Step 2**: Define `__category__` (if different from registry name)
- [ ] **Step 3**: Identify dependencies → add to `__inject__`
- [ ] **Step 4**: Identify global params → add to `__shared__`
- [ ] **Step 5**: If complex dependencies, implement `from_config()`
- [ ] **Step 6**: Provide necessary attributes (e.g., `out_shape`)
- [ ] **Step 7**: Test both instantiation modes (direct + registry)
- [ ] **Step 8**: Verify numerical equivalence (registered == direct)

## Examples

### Example 1: Simple Component (No Dependencies)

```python
@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    __category__ = 'backbone'
    # No __inject__ or __shared__ needed

    def __init__(self, depth=50, variant='d', **kwargs):
        super().__init__()
        self.depth = depth
        self.variant = variant
        self._setup_out_shape()

    def _setup_out_shape(self):
        # Provide for downstream
        self.out_shape = [...]
```

### Example 2: Component with Shared Config

```python
@HEAD_REGISTRY.register()
class DINOv3Head(nn.Module):
    __category__ = 'head'
    __shared__ = ['num_classes', 'hidden_dim']

    def __init__(self, num_classes=80, hidden_dim=256, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
```

### Example 3: Component with Dependencies

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
        self.transformer = transformer
        self.detr_head = detr_head

    @classmethod
    def from_config(cls, cfg, global_config=None):
        # Complex dependency chain
        return build_components_with_injection(cfg, global_config)
```

## Anti-Patterns (What NOT to Do)

### ❌ Anti-Pattern 1: Calling __init__ in from_config

```python
# WRONG
@classmethod
def from_config(cls, cfg, global_config):
    backbone = create('ResNet', **cfg['backbone'])
    return cls(backbone=backbone)  # ❌ Don't call __init__!

# CORRECT
@classmethod
def from_config(cls, cfg, global_config):
    backbone = create('ResNet', **cfg['backbone'])
    return {'backbone': backbone}  # ✅ Return kwargs dict
```

### ❌ Anti-Pattern 2: Modifying global_config

```python
# WRONG
@classmethod
def from_config(cls, cfg, global_config):
    global_config['my_param'] = 123  # ❌ Don't mutate!
    return {}

# CORRECT
@classmethod
def from_config(cls, cfg, global_config):
    local_cfg = global_config.copy()  # ✅ Copy if needed
    local_cfg['my_param'] = 123
    # Use local_cfg...
    return {}
```

### ❌ Anti-Pattern 3: Hard-Coding Dependency Types

```python
# WRONG
@classmethod
def from_config(cls, cfg, global_config):
    backbone = ResNet(depth=50)  # ❌ Hard-coded type!
    return {'backbone': backbone}

# CORRECT
@classmethod
def from_config(cls, cfg, global_config):
    backbone = create(cfg['backbone']['type'], ...)  # ✅ Config-driven!
    return {'backbone': backbone}
```

---

**Document Status**: ✅ Complete
**Related Contracts**: registry-api.md, config-schema.yaml

# Quickstart: Migrating Components to PaddlePaddle-Style Registry

**Feature**: 004-paddle-pytorch-migration
**Date**: 2025-10-17
**Audience**: Developers migrating RT-DETRv3 components to the registry system

## Goal

Migrate existing RT-DETRv3 components to support PaddlePaddle-style config-driven instantiation while maintaining 100% backward compatibility with direct instantiation.

**Time to Complete**: ~15-30 minutes per component

## Prerequisites

- [ ] Python 3.9+ environment with PyTorch ≥2.5.1
- [ ] RT-DETRv3 codebase cloned and dependencies installed (`uv sync`)
- [ ] Familiarity with Python decorators and class methods
- [ ] Understanding of component's role in model architecture

## Overview: 5-Step Migration Process

```
1. Add @register decorator    (2 min)
    ↓
2. Define metadata attributes  (3 min)
    ↓
3. Implement from_config()     (10-15 min, if needed)
    ↓
4. Test both modes             (5-10 min)
    ↓
5. Verify equivalence          (5 min)
```

**Total**: ~25-35 minutes for complex components, ~10 minutes for simple ones

## Step 1: Add `@register` Decorator (2 minutes)

### 1.1 Import the Registry

Add import at top of file:

```python
# For backbones
from rtdetrv3_pytorch.models import BACKBONE_REGISTRY

# For necks
from rtdetrv3_pytorch.models import NECK_REGISTRY

# For transformers
from rtdetrv3_pytorch.models import TRANSFORMER_REGISTRY

# For heads
from rtdetrv3_pytorch.models import HEAD_REGISTRY

# For losses
from rtdetrv3_pytorch.models import LOSS_REGISTRY

# For top-level models
from rtdetrv3_pytorch.models import ARCHITECTURE_REGISTRY
```

### 1.2 Add Decorator to Class

**Before**:
```python
class ResNet(nn.Module):
    def __init__(self, depth=50, ...):
        super().__init__()
```

**After**:
```python
@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    def __init__(self, depth=50, ...):
        super().__init__()
```

**Result**: Component now registered and discoverable via `BACKBONE_REGISTRY.list()`

**Verify**:
```python
>>> from rtdetrv3_pytorch.models import BACKBONE_REGISTRY
>>> BACKBONE_REGISTRY.list()
['ResNet', ...]  # Should include your component
```

## Step 2: Define Metadata Attributes (3 minutes)

Add class-level attributes to specify category, dependencies, and shared config.

### 2.1 Define `__category__` (Optional)

```python
@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    __category__ = 'backbone'  # Defaults to registry name if omitted
```

**When to define**: Only if you want to override the default (rarely needed)

### 2.2 Define `__inject__` (For Components with Dependencies)

List `__init__` parameters that should be automatically created from nested configs.

**Example**: RTDETRv3 depends on backbone, neck, transformer, head

```python
@ARCHITECTURE_REGISTRY.register()
class RTDETRv3(nn.Module):
    __inject__ = ['backbone', 'neck', 'transformer', 'detr_head']

    def __init__(self, backbone, neck, transformer, detr_head, ...):
        # backbone, neck, etc. will be instances (not dicts)
        self.backbone = backbone
```

**When to use**:
- ✅ Component needs other components as constructor parameters
- ✅ Want config-driven creation of dependencies
- ❌ Component has no dependencies (leave as `[]` or omit)

### 2.3 Define `__shared__` (For Components Using Global Config)

List `__init__` parameters that should be populated from global configuration.

**Example**: Head uses num_classes from global config

```python
@HEAD_REGISTRY.register()
class DINOv3Head(nn.Module):
    __shared__ = ['num_classes', 'hidden_dim']

    def __init__(self, num_classes=80, hidden_dim=256, ...):
        # If global_config has these, use them; otherwise use defaults
        self.num_classes = num_classes
```

**When to use**:
- ✅ Parameter appears in multiple components (avoid duplication)
- ✅ Parameter has project-wide significance (num_classes, hidden_dim)
- ❌ Parameter is component-specific (use regular config instead)

**Important**: __shared__ parameters MUST have default values in `__init__`!

## Step 3: Implement `from_config()` (10-15 minutes, optional)

### 3.1 When to Implement from_config()

**Implement if**:
- ✅ Component has dependencies that need intermediate values injected
  - Example: neck needs backbone.out_shape
- ✅ Complex construction logic (multiple dependent components)
- ✅ Need to transform config values before passing to `__init__`

**Skip if**:
- ❌ Component has no dependencies
- ❌ Simple mapping from config → constructor args

### 3.2 from_config() Template

```python
@classmethod
def from_config(cls, cfg: Dict[str, Any], global_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Build constructor kwargs from config.

    Args:
        cfg: Component-specific config (may include nested component configs)
        global_config: Global configuration for shared/inject resolution

    Returns:
        Dict of kwargs to pass to __init__
    """
    from rtdetrv3_pytorch.models import create

    kwargs = {}

    # Create dependencies
    if 'dependency_name' in cfg:
        dep_cfg = cfg['dependency_name'].copy()
        dep_type = dep_cfg.pop('type')
        kwargs['dependency_name'] = create(dep_type, global_config, **dep_cfg)

    # Inject intermediate values
    if 'downstream_component' in cfg and 'dependency_name' in kwargs:
        downstream_cfg = cfg['downstream_component'].copy()
        downstream_type = downstream_cfg.pop('type')
        # Inject dependency's output
        if hasattr(kwargs['dependency_name'], 'out_shape'):
            downstream_cfg['input_shape'] = kwargs['dependency_name'].out_shape
        kwargs['downstream_component'] = create(downstream_type, global_config, **downstream_cfg)

    return kwargs
```

### 3.3 Real Example: RTDETRv3.from_config()

```python
@ARCHITECTURE_REGISTRY.register()
class RTDETRv3(nn.Module):
    @classmethod
    def from_config(cls, cfg, global_config=None):
        from rtdetrv3_pytorch.models import create

        kwargs = {}

        # Step 1: Create backbone
        if 'backbone' in cfg:
            backbone_cfg = cfg['backbone'].copy()
            backbone_type = backbone_cfg.pop('type')
            kwargs['backbone'] = create(backbone_type, global_config, **backbone_cfg)

        # Step 2: Create neck (inject backbone.out_shape)
        if 'neck' in cfg and 'backbone' in kwargs:
            neck_cfg = cfg['neck'].copy()
            neck_type = neck_cfg.pop('type')
            # Inject dependency
            if hasattr(kwargs['backbone'], 'out_shape'):
                neck_cfg['input_shape'] = kwargs['backbone'].out_shape
            kwargs['neck'] = create(neck_type, global_config, **neck_cfg)

        # Step 3: Create transformer (inject neck.out_shape)
        if 'transformer' in cfg and 'neck' in kwargs:
            transformer_cfg = cfg['transformer'].copy()
            transformer_type = transformer_cfg.pop('type')
            if hasattr(kwargs['neck'], 'out_shape'):
                transformer_cfg['input_shape'] = kwargs['neck'].out_shape
            kwargs['transformer'] = create(transformer_type, global_config, **transformer_cfg)

        # Step 4: Create head (inject transformer.hidden_dim)
        if 'detr_head' in cfg and 'transformer' in kwargs:
            head_cfg = cfg['detr_head'].copy()
            head_type = head_cfg.pop('type')
            if hasattr(kwargs['transformer'], 'hidden_dim'):
                head_cfg['hidden_dim'] = kwargs['transformer'].hidden_dim
            kwargs['detr_head'] = create(head_type, global_config, **head_cfg)

        return kwargs
```

## Step 4: Test Both Instantiation Modes (5-10 minutes)

### 4.1 Test Direct Instantiation (Backward Compatibility)

```python
# Old way (must still work!)
from rtdetrv3_pytorch.models.backbones.resnet import ResNet

backbone = ResNet(depth=50, variant='d')
print(backbone.depth)  # Should work as before
```

**Expected**: No errors, identical behavior to before migration

### 4.2 Test Registry-Based Instantiation

```python
# New way (via registry)
from rtdetrv3_pytorch.models import BACKBONE_REGISTRY

backbone = BACKBONE_REGISTRY.create('ResNet', depth=50, variant='d')
print(backbone.depth)  # Should work identically
```

**Expected**: Same result as direct instantiation

### 4.3 Test Config-Driven Instantiation

```python
# PaddlePaddle-style config
from rtdetrv3_pytorch.models import create

config = {
    'type': 'ResNet',
    'depth': 50,
    'variant': 'd'
}

backbone = create(config['type'], **{k: v for k, v in config.items() if k != 'type'})
print(backbone.depth)  # Should be 50
```

**Expected**: Same result as previous modes

### 4.4 Test Dependency Injection (If Applicable)

```python
# For components with __inject__
global_config = {
    'num_classes': 80,
    'backbone': {'type': 'ResNet', 'depth': 50},
    'neck': {'type': 'HybridEncoder', 'hidden_dim': 256}
}

model = create('RTDETRv3', global_config=global_config)
print(model.backbone.depth)  # Should be 50 (auto-injected)
```

**Expected**: Dependencies automatically created and passed to constructor

## Step 5: Verify Numerical Equivalence (5 minutes)

Ensure registered component produces identical outputs to direct instantiation.

### 5.1 Create Test Script

```python
import torch
from rtdetrv3_pytorch.models import create
from rtdetrv3_pytorch.models.backbones.resnet import ResNet

# Create via direct instantiation
direct = ResNet(depth=50, variant='d')
direct.eval()

# Create via registry
registered = create('ResNet', depth=50, variant='d')
registered.eval()

# Copy weights (ensure same initialization)
registered.load_state_dict(direct.state_dict())

# Test with random input
x = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    out_direct = direct(x)
    out_registered = registered(x)

# Check equivalence
for i, (d, r) in enumerate(zip(out_direct, out_registered)):
    diff = (d - r).abs().max().item()
    print(f"Level {i}: max diff = {diff:.2e}")
    assert diff < 1e-5, f"Outputs differ at level {i}!"

print("✅ Numerical equivalence verified!")
```

**Expected**: All differences < 1e-5 (numerical tolerance)

## Checklist: Component Migration Complete

- [ ] Added `@REGISTRY.register()` decorator
- [ ] Defined `__category__` (if needed)
- [ ] Defined `__inject__` (if has dependencies)
- [ ] Defined `__shared__` (if uses global config)
- [ ] Implemented `from_config()` (if complex dependencies)
- [ ] Tested direct instantiation (backward compat)
- [ ] Tested registry instantiation
- [ ] Tested config-driven instantiation
- [ ] Verified numerical equivalence
- [ ] Updated component documentation

## Common Issues & Solutions

### Issue 1: "Component not found in registry"

**Symptom**:
```
KeyError: "backbone 'ResNet' not found in registry. Available: []"
```

**Cause**: Component file not imported

**Solution**: Add import to `rtdetrv3_pytorch/models/backbones/__init__.py`:
```python
from .resnet import ResNet  # Triggers @register decorator
```

### Issue 2: "__init__() got unexpected keyword argument 'foo'"

**Symptom**:
```
TypeError: __init__() got unexpected keyword argument 'input_shape'
```

**Cause**: Config contains parameter not in `__init__` signature

**Solutions**:
1. Add parameter to `__init__`: `def __init__(self, input_shape=None, ...)`
2. Add `**kwargs` to absorb extra params: `def __init__(self, ..., **kwargs)`
3. Remove parameter from config

### Issue 3: "__shared__ field must have default value"

**Symptom**:
```
ValueError: __shared__ field 'num_classes' must have default value
```

**Cause**: __shared__ parameter lacks default in `__init__`

**Solution**: Add default value:
```python
# Before
def __init__(self, num_classes):  # ❌ No default

# After
def __init__(self, num_classes=80):  # ✅ Has default
```

### Issue 4: "Dependency 'UnknownComponent' not found"

**Symptom**:
```
ValueError: Dependency 'HybridEncoder' not found in any registry
```

**Cause**: Dependency not registered or not imported

**Solution**:
1. Ensure dependency has `@REGISTRY.register()` decorator
2. Ensure dependency module is imported in `__init__.py`
3. Check spelling of 'type' value in config

### Issue 5: Numerical outputs differ

**Symptom**: `assert diff < 1e-5` fails

**Possible Causes**:
1. Different random seed → Set `torch.manual_seed(42)` before both
2. Different initialization → Copy weights: `registered.load_state_dict(direct.state_dict())`
3. Different device → Ensure both on same device (CPU or GPU)
4. Different precision → Use same dtype (FP32 for validation)

**Solution**: Isolate the difference by testing component in isolation, then check integration

## Examples by Component Type

### Example 1: Simple Backbone (No Dependencies)

```python
# File: rtdetrv3_pytorch/models/backbones/resnet.py
from rtdetrv3_pytorch.models import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    __category__ = 'backbone'

    def __init__(self, depth=50, variant='d', **kwargs):
        super().__init__()
        self.depth = depth
        # ... implementation
        self._setup_out_shape()

    def _setup_out_shape(self):
        """Provide out_shape for downstream components"""
        self.out_shape = [...]
```

**Migration time**: ~5 minutes (no from_config needed)

### Example 2: Head with Shared Config

```python
# File: rtdetrv3_pytorch/models/heads/detr_head.py
from rtdetrv3_pytorch.models import HEAD_REGISTRY

@HEAD_REGISTRY.register()
class DINOv3Head(nn.Module):
    __category__ = 'head'
    __shared__ = ['num_classes', 'hidden_dim']

    def __init__(self, num_classes=80, hidden_dim=256, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        # ... implementation
```

**Migration time**: ~10 minutes (simple shared config)

### Example 3: Complex Model with Dependencies

```python
# File: rtdetrv3_pytorch/models/rtdetrv3.py
from rtdetrv3_pytorch.models import ARCHITECTURE_REGISTRY, create

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
        # Complex dependency chain (see Step 3.3 for full example)
        kwargs = {}
        # ... create components with injection
        return kwargs
```

**Migration time**: ~25 minutes (complex from_config logic)

## Next Steps

After migrating a component:

1. **Update Tests**: Add test cases for registry instantiation
2. **Update Docs**: Document PaddlePaddle-style usage in component docstring
3. **Update Examples**: Add config-driven example to README or examples/
4. **Migrate Dependents**: If other components depend on this one, migrate them next

## Additional Resources

- **API Contracts**: See `contracts/registry-api.md`, `contracts/component-protocol.md`
- **Data Model**: See `data-model.md` for entity relationships
- **Config Schema**: See `contracts/config-schema.yaml` for YAML structure
- **Research**: See `research.md` for design rationale and patterns

---

**Document Status**: ✅ Complete
**Feedback**: Report issues or suggestions to project maintainers

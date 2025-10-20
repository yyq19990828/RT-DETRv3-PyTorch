# Contract: Registry API

**Feature**: 004-paddle-pytorch-migration
**Date**: 2025-10-17
**Purpose**: Define the public API contract for the Registry class

## Overview

The `Registry` class provides component registration and instantiation with dependency injection support, following PaddlePaddle's workspace pattern.

## Class Signature

```python
class Registry:
    """Registry for model components with dependency injection support."""

    def __init__(self, name: str) -> None:
        """
        Initialize registry for a specific component category.

        Args:
            name: Category name (e.g., 'backbone', 'head', 'loss')

        Post-conditions:
            - self._name == name
            - self._registry == {}
            - self._config_cache == {}
        """

    def register(self, name: Optional[str] = None) -> Callable[[Type], Type]:
        """
        Decorator to register a component class.

        Args:
            name: Optional custom registration name. If None, uses cls.__name__

        Returns:
            Decorator function that returns the class unchanged

        Side Effects:
            - Adds class to self._registry
            - Sets __category__, __inject__, __shared__ attributes on class
            - Logs warning if overwriting existing registration

        Example:
            @BACKBONE_REGISTRY.register()
            class ResNet(nn.Module):
                pass

            @BACKBONE_REGISTRY.register('ResNet50')
            class CustomResNet(nn.Module):
                pass
        """

    def get(self, name: str) -> Type:
        """
        Retrieve registered component class by name.

        Args:
            name: Registered component name

        Returns:
            Component class (not instance)

        Raises:
            KeyError: If name not found in registry. Error message includes
                     list of available component names.

        Example:
            ResNetClass = BACKBONE_REGISTRY.get('ResNet')
            backbone = ResNetClass(depth=50)
        """

    def create(
        self,
        name: str,
        global_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """
        Create component instance with dependency injection.

        Instantiation process:
        1. Look up component class by name
        2. Resolve __shared__ fields from global_config
        3. Resolve __inject__ dependencies (recursive create)
        4. Call from_config() class method if defined
        5. Merge all kwargs and call __init__

        Parameter resolution order (highest to lowest priority):
        1. Explicit kwargs passed to create()
        2. __shared__ fields from global_config
        3. __inject__ dependencies created from global_config
        4. Constructor default values

        Args:
            name: Registered component name
            global_config: Global configuration for shared/inject resolution
            **kwargs: Additional parameters passed to __init__

        Returns:
            Component instance

        Raises:
            KeyError: If name not registered
            ValueError: If dependency not found in any registry
            TypeError: If kwargs incompatible with __init__ signature

        Example:
            # Simple creation
            backbone = BACKBONE_REGISTRY.create('ResNet', depth=50)

            # With global config
            global_cfg = {'num_classes': 80}
            head = HEAD_REGISTRY.create('DINOv3Head', global_cfg, hidden_dim=256)

            # With dependency injection
            model_cfg = {
                'backbone': {'type': 'ResNet', 'depth': 50},
                'num_classes': 80
            }
            model = ARCHITECTURE_REGISTRY.create('RTDETRv3', model_cfg)
        """

    def list(self) -> List[str]:
        """
        List all registered component names.

        Returns:
            Sorted list of component names

        Example:
            >>> BACKBONE_REGISTRY.list()
            ['ResNet', 'ResNeXt', 'CSPDarkNet']
        """

    def __contains__(self, name: str) -> bool:
        """
        Check if component is registered.

        Args:
            name: Component name to check

        Returns:
            True if registered, False otherwise

        Example:
            if 'ResNet' in BACKBONE_REGISTRY:
                backbone = BACKBONE_REGISTRY.create('ResNet')
        """

    def __repr__(self) -> str:
        """
        String representation showing registry name and contents.

        Returns:
            String like "Registry(name=backbone, items=['ResNet', 'ResNeXt'])"
        """
```

## Global Functions

### `create(name, global_config=None, **kwargs)`

```python
def create(
    name: str,
    global_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Any:
    """
    Create component from any registry (PaddlePaddle-style global function).

    Searches all registries for component name and creates instance.

    Args:
        name: Component class name
        global_config: Global configuration for dependency injection
        **kwargs: Additional constructor arguments

    Returns:
        Component instance

    Raises:
        ValueError: If name not found in any registry. Error message lists
                   all available registries.

    Example:
        # Equivalent to BACKBONE_REGISTRY.create('ResNet', depth=50)
        backbone = create('ResNet', depth=50)

        # Works across registries
        model = create('RTDETRv3', global_config={...})
    """
```

### `build_from_config(cfg, registry, global_config=None, **kwargs)`

```python
def build_from_config(
    cfg: Dict[str, Any],
    registry: Registry,
    global_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Any:
    """
    Build component from config dict using specified registry.

    Config must contain 'type' key specifying component name.
    Other keys are treated as constructor parameters.

    Args:
        cfg: Config dict with 'type' key
        registry: Registry to look up component
        global_config: Global configuration
        **kwargs: Additional parameters (merged with cfg)

    Returns:
        Component instance

    Raises:
        ValueError: If 'type' key missing from cfg

    Example:
        cfg = {'type': 'ResNet', 'depth': 50, 'variant': 'd'}
        backbone = build_from_config(cfg, BACKBONE_REGISTRY)
    """
```

## Registry Instances

### Global Registries

```python
# Available in rtdetrv3_pytorch.models
BACKBONE_REGISTRY = Registry('backbone')
NECK_REGISTRY = Registry('neck')
TRANSFORMER_REGISTRY = Registry('transformer')
HEAD_REGISTRY = Registry('head')
LOSS_REGISTRY = Registry('loss')
ARCHITECTURE_REGISTRY = Registry('architecture')

# List of all registries for cross-registry lookups
ALL_REGISTRIES = [
    BACKBONE_REGISTRY,
    NECK_REGISTRY,
    TRANSFORMER_REGISTRY,
    HEAD_REGISTRY,
    LOSS_REGISTRY,
    ARCHITECTURE_REGISTRY
]
```

## Error Handling

### Error: Component Not Registered

```python
# Input
BACKBONE_REGISTRY.get('NonExistent')

# Output
KeyError: "backbone 'NonExistent' not found in registry. Available: ['ResNet', 'ResNeXt']"
```

### Error: Missing 'type' Key

```python
# Input
cfg = {'depth': 50}  # Missing 'type'
build_from_config(cfg, BACKBONE_REGISTRY)

# Output
ValueError: "Config must contain 'type' key: {'depth': 50}"
```

### Error: Dependency Not Found

```python
# Input
global_config = {'backbone': {'type': 'UnknownBackbone'}}
ARCHITECTURE_REGISTRY.create('RTDETRv3', global_config)

# Output
ValueError: "Dependency 'UnknownBackbone' not found in any registry. Available registries: ['backbone', 'neck', ...]"
```

## Performance Guarantees

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `register()` | O(1) | Dict insertion |
| `get(name)` | O(1) | Dict lookup |
| `create(name)` | O(D) | D = depth of dependency tree |
| `list()` | O(N log N) | N = number of components, sorted |
| `__contains__` | O(1) | Dict membership test |

**Space Complexity**: O(N) where N is number of registered components

## Thread Safety

**Current Status**: NOT thread-safe

**Assumptions**:
- Component registration happens at module import time (single-threaded)
- Component instantiation may happen concurrently (read-only registry access)
- Python GIL provides protection for dict reads

**Future Enhancement** (if needed):
```python
import threading

class Registry:
    def __init__(self, name: str):
        self._name = name
        self._registry = {}
        self._lock = threading.RLock()  # Re-entrant lock

    def register(self, name=None):
        def decorator(cls):
            with self._lock:
                self._registry[...] = cls
            return cls
        return decorator
```

## Backward Compatibility

**Guaranteed**:
- Decorated classes are returned unchanged (non-invasive)
- Direct instantiation (`MyClass(args)`) works identically to before registration
- Registry API additive (no breaking changes to existing code)

**Example**:
```python
# Before registration
class ResNet(nn.Module):
    def __init__(self, depth=50):
        pass

backbone = ResNet(depth=50)  # Works

# After registration
@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    def __init__(self, depth=50):
        pass

backbone = ResNet(depth=50)  # Still works (backward compatible)
backbone = BACKBONE_REGISTRY.create('ResNet', depth=50)  # Also works (new feature)
```

## Version Compatibility

| PyTorch Version | Python Version | Status |
|-----------------|----------------|--------|
| ≥2.5.1 | 3.9+ | ✅ Supported |
| 2.0-2.4 | 3.8+ | ⚠️ Untested (likely works) |
| <2.0 | Any | ❌ Not supported |

## Changelog

### Version 1.0 (2025-10-17)
- Initial registry API based on PaddlePaddle workspace pattern
- Support for `__inject__`, `__shared__`, `__category__` annotations
- Global `create()` function for cross-registry instantiation
- Six registry categories: architecture, backbone, neck, transformer, head, loss

---

**Document Status**: ✅ Complete
**Related Contracts**: component-protocol.md, config-schema.yaml

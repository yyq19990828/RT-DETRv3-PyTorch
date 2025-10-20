"""
Unified workspace module for RT-DETRv3 PyTorch

Provides PaddlePaddle-compatible registration system for dynamic component instantiation.
Implements register() decorator, global_config dict, and create() factory function.

References:
    - PaddlePaddle ppdet/core/workspace.py
    - RT-DETRv3 research.md Section 1: Registration System Migration
"""

import logging
from typing import Any, Callable, Dict, Optional, Type
import copy


logger = logging.getLogger(__name__)


# Global configuration dictionary (PaddlePaddle pattern)
global_config = {}


def register(cls: Optional[Type] = None, *, name: Optional[str] = None) -> Callable:
    """
    Register decorator for component classes.

    Supports PaddlePaddle-style annotations:
    - __inject__: List of constructor parameters to inject from config
    - __shared__: List of parameters to inject from global shared config
    - __category__: Component category (optional metadata)

    Args:
        cls: Class to register (when used without parentheses)
        name: Optional custom name. Defaults to class.__name__

    Returns:
        Decorated class or decorator function

    Example:
        >>> @register
        >>> class ResNet(nn.Module):
        >>>     __inject__ = ['norm_layer']
        >>>     __shared__ = ['num_classes']
        >>>     def __init__(self, depth, num_classes=80, norm_layer='BatchNorm2d'):
        >>>         pass
        >>>
        >>> # Or with custom name:
        >>> @register(name='ResNet50')
        >>> class ResNet:
        >>>     pass
    """
    def decorator(target_cls: Type) -> Type:
        register_name = name if name is not None else target_cls.__name__

        if register_name in global_config:
            logger.warning(
                f"Component '{register_name}' already registered in global_config, overwriting"
            )

        # Store class in global_config
        global_config[register_name] = target_cls

        # Extract metadata from class (PaddlePaddle pattern)
        if not hasattr(target_cls, '__inject__'):
            target_cls.__inject__ = []
        if not hasattr(target_cls, '__shared__'):
            target_cls.__shared__ = []
        if not hasattr(target_cls, '__category__'):
            target_cls.__category__ = 'component'

        logger.debug(f"Registered '{register_name}' to global_config")

        return target_cls

    # Support both @register and @register(name='CustomName')
    if cls is None:
        # Called with arguments: @register(name='...')
        return decorator
    else:
        # Called without arguments: @register
        return decorator(cls)


def create(cfg_or_name, global_cfg: Optional[Dict] = None, **kwargs) -> Any:
    """
    Factory function to create component instances with dependency injection.

    Follows PaddlePaddle's create() pattern:
    1. Look up class from global_config by name or cfg['type']
    2. Resolve __inject__ dependencies recursively
    3. Resolve __shared__ fields from global config
    4. Instantiate with merged parameters

    Args:
        cfg_or_name: Either:
            - str: Class name (e.g., 'ResNet')
            - dict: Config dict with 'type' key (e.g., {'type': 'ResNet', 'depth': 50})
        global_cfg: Global configuration for shared/__inject__ resolution.
                    If None, uses module-level global_config.
        **kwargs: Additional arguments to pass to constructor

    Returns:
        Instantiated component

    Raises:
        KeyError: If class name not found in global_config
        ValueError: If cfg dict missing 'type' key

    Example:
        >>> # Direct instantiation
        >>> backbone = create('ResNet', depth=50)
        >>>
        >>> # From config dict
        >>> cfg = {'type': 'ResNet', 'depth': 50, 'norm_layer': {'type': 'BatchNorm2d'}}
        >>> backbone = create(cfg, global_cfg={'num_classes': 80})
    """
    # Use module-level global_config if not provided
    if global_cfg is None:
        global_cfg = global_config

    # Parse cfg_or_name
    if isinstance(cfg_or_name, str):
        # Simple string name
        cls_name = cfg_or_name
        cls_kwargs = kwargs.copy()
    elif isinstance(cfg_or_name, dict):
        # Config dictionary
        cfg = cfg_or_name.copy()
        if 'type' not in cfg:
            raise ValueError(f"Config dict must contain 'type' key: {cfg}")

        cls_name = cfg.pop('type')
        # Merge config and kwargs (kwargs take precedence)
        cls_kwargs = {**cfg, **kwargs}
    else:
        raise TypeError(
            f"cfg_or_name must be str or dict, got {type(cfg_or_name)}"
        )

    # Look up class in global_config
    if cls_name not in global_cfg:
        raise KeyError(
            f"Class '{cls_name}' not found in global_config. "
            f"Available: {sorted(global_cfg.keys())}"
        )

    cls = global_cfg[cls_name]

    # Resolve __shared__ fields
    if hasattr(cls, '__shared__'):
        shared_cfg = global_cfg.get('__shared__', {})
        for field in cls.__shared__:
            if field in shared_cfg and field not in cls_kwargs:
                cls_kwargs[field] = shared_cfg[field]
                logger.debug(f"Shared config {field}={shared_cfg[field]} for {cls_name}")

    # Resolve __inject__ fields (recursive dependency injection)
    if hasattr(cls, '__inject__'):
        for field in cls.__inject__:
            if field in cls_kwargs:
                # Check if it's a config dict that needs instantiation
                field_value = cls_kwargs[field]
                if isinstance(field_value, dict) and 'type' in field_value:
                    # Recursively create dependency
                    dep_instance = create(field_value, global_cfg)
                    cls_kwargs[field] = dep_instance
                    logger.debug(f"Injected {field}={field_value['type']} for {cls_name}")
                # else: already provided as instance, keep as-is
            elif field in global_cfg:
                # Look up from global config
                field_config = global_cfg[field]
                if isinstance(field_config, dict) and 'type' in field_config:
                    dep_instance = create(field_config, global_cfg)
                    cls_kwargs[field] = dep_instance
                    logger.debug(f"Injected {field} from global_config for {cls_name}")
                else:
                    cls_kwargs[field] = field_config

    logger.debug(f"Creating {cls_name} with kwargs: {list(cls_kwargs.keys())}")

    return cls(**cls_kwargs)


def merge_config(cfg: Dict[str, Any]) -> None:
    """
    Merge YAML config into global_config.

    PaddlePaddle pattern: Configuration files define component configs
    that get merged into global_config dict for later lookup.

    Args:
        cfg: Configuration dictionary to merge

    Example:
        >>> yaml_cfg = {
        >>>     'ResNet': {'depth': 50, 'freeze_at': 0},
        >>>     '__shared__': {'num_classes': 80},
        >>> }
        >>> merge_config(yaml_cfg)
        >>> # Now global_config contains these configs
    """
    global global_config

    # Deep merge to avoid modifying original config
    for key, value in cfg.items():
        if key in global_config and isinstance(value, dict) and isinstance(global_config[key], dict):
            # Merge dict values
            global_config[key] = {**global_config[key], **value}
        else:
            # Overwrite
            global_config[key] = copy.deepcopy(value)

    logger.debug(f"Merged config keys: {list(cfg.keys())}")


def reset_global_config() -> None:
    """
    Reset global_config to empty state.

    Useful for testing and isolation between config loads.
    """
    global global_config
    global_config.clear()
    logger.debug("Reset global_config")


def get_registered_classes() -> Dict[str, Type]:
    """
    Get all registered classes (excluding config entries).

    Returns:
        Dict mapping class names to class objects
    """
    return {
        name: obj for name, obj in global_config.items()
        if isinstance(obj, type) and name not in ('__shared__',)
    }


__all__ = [
    'global_config',
    'register',
    'create',
    'merge_config',
    'reset_global_config',
    'get_registered_classes',
]

"""
Model registry system for RT-DETRv3 PyTorch

Provides dynamic component instantiation from config using registry pattern.
Following PaddlePaddle's workspace design for consistency.
"""

import logging
from typing import Any, Callable, Dict, Optional, Type, List
import inspect


logger = logging.getLogger(__name__)


class Registry:
    """
    Registry for model components (backbones, necks, transformers, heads, losses).

    Supports dependency injection and shared configuration like PaddlePaddle's workspace.

    Example:
        >>> BACKBONE_REGISTRY = Registry('backbone')
        >>> @BACKBONE_REGISTRY.register()
        >>> class ResNet(nn.Module):
        >>>     __inject__ = ['norm_layer']  # auto-inject from config
        >>>     __shared__ = ['num_classes']  # shared global config
        >>>     pass
        >>>
        >>> backbone = BACKBONE_REGISTRY.create('ResNet', depth=50)
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}
        self._config_cache: Dict[str, Dict] = {}  # Store config for each registered class

    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a class.

        Supports PaddlePaddle-style annotations:
        - __inject__: List of fields to be injected from config
        - __shared__: List of fields shared across modules
        - __category__: Category of the module (e.g., 'architecture', 'backbone')

        Args:
            name: Optional custom name. If None, uses class name.

        Returns:
            Decorator function
        """
        def decorator(cls: Type) -> Type:
            register_name = name if name is not None else cls.__name__

            if register_name in self._registry:
                logger.warning(
                    f"{self._name} '{register_name}' already registered, overwriting"
                )

            # Extract metadata from class
            cls.__category__ = getattr(cls, '__category__', self._name)
            cls.__inject__ = getattr(cls, '__inject__', [])
            cls.__shared__ = getattr(cls, '__shared__', [])

            self._registry[register_name] = cls
            logger.debug(f"Registered {self._name}: {register_name}")

            return cls

        return decorator

    def get(self, name: str) -> Type:
        """
        Get registered class by name.

        Args:
            name: Registered class name

        Returns:
            Registered class

        Raises:
            KeyError if name not found
        """
        if name not in self._registry:
            raise KeyError(
                f"{self._name} '{name}' not found in registry. "
                f"Available: {list(self._registry.keys())}"
            )

        return self._registry[name]

    def create(self, name: str, global_config: Optional[Dict] = None, **kwargs) -> Any:
        """
        Create instance with dependency injection support.

        Following PaddlePaddle's create() pattern:
        1. Resolve __inject__ dependencies from config
        2. Resolve __shared__ values from global config
        3. Call from_config() classmethod if available

        Args:
            name: Registered class name
            global_config: Global configuration dict for shared/inject resolution
            **kwargs: Additional arguments to pass to constructor

        Returns:
            Instantiated module
        """
        cls = self.get(name)
        global_config = global_config or {}

        # Start with provided kwargs
        cls_kwargs = kwargs.copy()

        # Resolve __shared__ fields
        if hasattr(cls, '__shared__'):
            for field in cls.__shared__:
                if field in global_config and field not in cls_kwargs:
                    cls_kwargs[field] = global_config[field]
                    logger.debug(f"Shared config {field}={global_config[field]} for {name}")

        # Resolve __inject__ fields (create dependent modules)
        if hasattr(cls, '__inject__'):
            for field in cls.__inject__:
                if field in cls_kwargs:
                    # Already provided, skip
                    continue

                # Look up dependency in config
                if field in global_config:
                    dep_config = global_config[field]
                    if isinstance(dep_config, dict) and 'type' in dep_config:
                        # Recursively create dependency
                        dep_type = dep_config['type']
                        dep_kwargs = {k: v for k, v in dep_config.items() if k != 'type'}
                        # Find appropriate registry for dependency
                        dep_instance = self._create_dependency(dep_type, dep_kwargs, global_config)
                        cls_kwargs[field] = dep_instance
                        logger.debug(f"Injected {field}={dep_type} for {name}")
                    else:
                        # Direct value injection
                        cls_kwargs[field] = dep_config

        # Call from_config() classmethod if available (PaddlePaddle pattern)
        if hasattr(cls, 'from_config') and callable(getattr(cls, 'from_config')):
            from_config_kwargs = cls.from_config(cls_kwargs, global_config)
            cls_kwargs.update(from_config_kwargs)

        return cls(**cls_kwargs)

    def _create_dependency(self, dep_type: str, dep_kwargs: Dict, global_config: Dict) -> Any:
        """Helper to create dependency from any registry"""
        # Try all registries to find the dependency
        for registry in ALL_REGISTRIES:
            if dep_type in registry:
                return registry.create(dep_type, global_config, **dep_kwargs)

        # If not found in any registry, raise error
        raise ValueError(
            f"Dependency '{dep_type}' not found in any registry. "
            f"Available registries: {[r._name for r in ALL_REGISTRIES]}"
        )

    def list(self) -> list:
        """List all registered names"""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._registry.keys())})"


# Create global registries for different component types
BACKBONE_REGISTRY = Registry('backbone')
NECK_REGISTRY = Registry('neck')
TRANSFORMER_REGISTRY = Registry('transformer')
HEAD_REGISTRY = Registry('head')
LOSS_REGISTRY = Registry('loss')
ARCHITECTURE_REGISTRY = Registry('architecture')  # For top-level models

# Global list of all registries for dependency resolution
ALL_REGISTRIES = [
    BACKBONE_REGISTRY,
    NECK_REGISTRY,
    TRANSFORMER_REGISTRY,
    HEAD_REGISTRY,
    LOSS_REGISTRY,
    ARCHITECTURE_REGISTRY
]


def build_from_config(cfg: Dict[str, Any], registry: Registry, global_config: Optional[Dict] = None, **kwargs) -> Any:
    """
    Build module from config dict using registry with dependency injection.

    Following PaddlePaddle's pattern:
    - Supports 'type' key for class name
    - Supports dependency injection via __inject__
    - Supports shared config via __shared__

    Args:
        cfg: Config dict with 'type' key specifying class name
        registry: Registry to look up class
        global_config: Global configuration for dependency injection
        **kwargs: Additional arguments to pass to constructor

    Returns:
        Instantiated module

    Example:
        >>> cfg = {'type': 'ResNet', 'depth': 50}
        >>> backbone = build_from_config(cfg, BACKBONE_REGISTRY)
    """
    cfg = cfg.copy()  # Don't modify original config

    if 'type' not in cfg:
        raise ValueError(f"Config must contain 'type' key: {cfg}")

    module_type = cfg.pop('type')

    # Merge config and kwargs
    module_kwargs = {**cfg, **kwargs}

    logger.debug(f"Building {registry._name} '{module_type}' with args: {module_kwargs}")

    # Use registry.create() for dependency injection support
    return registry.create(module_type, global_config=global_config, **module_kwargs)


def create(name: str, global_config: Optional[Dict] = None, **kwargs) -> Any:
    """
    Create instance from any registry (PaddlePaddle-style global create function).

    Args:
        name: Class name to instantiate
        global_config: Global configuration for dependency injection
        **kwargs: Additional arguments

    Returns:
        Instantiated module

    Example:
        >>> backbone = create('ResNet', global_config={'num_classes': 80}, depth=50)
    """
    # Try to find the class in any registry
    for registry in ALL_REGISTRIES:
        if name in registry:
            return registry.create(name, global_config, **kwargs)

    raise ValueError(
        f"Class '{name}' not found in any registry. "
        f"Available registries: {[r._name for r in ALL_REGISTRIES]}"
    )


# Convenience functions for building components (backward compatibility)
def build_backbone(cfg: Dict[str, Any], global_config: Optional[Dict] = None, **kwargs):
    """Build backbone from config"""
    return build_from_config(cfg, BACKBONE_REGISTRY, global_config, **kwargs)


def build_neck(cfg: Dict[str, Any], global_config: Optional[Dict] = None, **kwargs):
    """Build neck from config"""
    return build_from_config(cfg, NECK_REGISTRY, global_config, **kwargs)


def build_transformer(cfg: Dict[str, Any], global_config: Optional[Dict] = None, **kwargs):
    """Build transformer from config"""
    return build_from_config(cfg, TRANSFORMER_REGISTRY, global_config, **kwargs)


def build_head(cfg: Dict[str, Any], global_config: Optional[Dict] = None, **kwargs):
    """Build detection head from config"""
    return build_from_config(cfg, HEAD_REGISTRY, global_config, **kwargs)


def build_loss(cfg: Dict[str, Any], global_config: Optional[Dict] = None, **kwargs):
    """Build loss function from config"""
    return build_from_config(cfg, LOSS_REGISTRY, global_config, **kwargs)


# Import main model
from .rtdetrv3 import RTDETRv3, build_rtdetrv3


__all__ = [
    # Registry system
    'Registry',
    'BACKBONE_REGISTRY',
    'NECK_REGISTRY',
    'TRANSFORMER_REGISTRY',
    'HEAD_REGISTRY',
    'LOSS_REGISTRY',
    'ARCHITECTURE_REGISTRY',
    'ALL_REGISTRIES',
    # Builder functions
    'build_from_config',
    'create',  # PaddlePaddle-style global create
    'build_backbone',
    'build_neck',
    'build_transformer',
    'build_head',
    'build_loss',
    # Main model
    'RTDETRv3',
    'build_rtdetrv3',
]

"""
Model registry system for RT-DETRv3 PyTorch

Provides dynamic component instantiation from config using registry pattern.
"""

import logging
from typing import Any, Callable, Dict, Optional, Type


logger = logging.getLogger(__name__)


class Registry:
    """
    Registry for model components (backbones, necks, transformers, heads, losses).
    
    Example:
        >>> BACKBONE_REGISTRY = Registry('backbone')
        >>> @BACKBONE_REGISTRY.register()
        >>> class ResNet(nn.Module):
        >>>     pass
        >>> 
        >>> backbone = BACKBONE_REGISTRY.get('ResNet')(depth=50)
    """
    
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a class.
        
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


def build_from_config(cfg: Dict[str, Any], registry: Registry, **kwargs) -> Any:
    """
    Build module from config dict using registry.
    
    Args:
        cfg: Config dict with 'type' key specifying class name
        registry: Registry to look up class
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
    module_cls = registry.get(module_type)
    
    # Merge config and kwargs
    module_kwargs = {**cfg, **kwargs}
    
    logger.debug(f"Building {registry._name} '{module_type}' with args: {module_kwargs}")
    
    return module_cls(**module_kwargs)


# Convenience functions for building components
def build_backbone(cfg: Dict[str, Any], **kwargs):
    """Build backbone from config"""
    return build_from_config(cfg, BACKBONE_REGISTRY, **kwargs)


def build_neck(cfg: Dict[str, Any], **kwargs):
    """Build neck from config"""
    return build_from_config(cfg, NECK_REGISTRY, **kwargs)


def build_transformer(cfg: Dict[str, Any], **kwargs):
    """Build transformer from config"""
    return build_from_config(cfg, TRANSFORMER_REGISTRY, **kwargs)


def build_head(cfg: Dict[str, Any], **kwargs):
    """Build detection head from config"""
    return build_from_config(cfg, HEAD_REGISTRY, **kwargs)


def build_loss(cfg: Dict[str, Any], **kwargs):
    """Build loss function from config"""
    return build_from_config(cfg, LOSS_REGISTRY, **kwargs)


# Import main model
from .rtdetrv3 import RTDETRv3, build_rtdetrv3


__all__ = [
    'Registry',
    'BACKBONE_REGISTRY',
    'NECK_REGISTRY',
    'TRANSFORMER_REGISTRY',
    'HEAD_REGISTRY',
    'LOSS_REGISTRY',
    'build_from_config',
    'build_backbone',
    'build_neck',
    'build_transformer',
    'build_head',
    'build_loss',
    # Main model
    'RTDETRv3',
    'build_rtdetrv3',
]

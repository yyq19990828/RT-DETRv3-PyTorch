"""
RT-DETRv3 PyTorch Workspace Management
Component registration and dependency injection system
"""

import inspect
import importlib
import functools
from collections import defaultdict
from typing import Any, Dict, Optional, List, Union

__all__ = ['register', 'create', 'GLOBAL_CONFIG', 'extract_schema']

# Global configuration registry
GLOBAL_CONFIG = defaultdict(dict)


def register(dct: Any = GLOBAL_CONFIG, name: str = None, force: bool = False):
    """
    Register a class or function to the global configuration.
    
    Args:
        dct: Target dictionary or class to register to
        name: Custom name for registration (uses class/function name if None)
        force: Whether to force registration even if already exists
    
    Returns:
        Decorator function
    """
    def decorator(cls_or_func):
        register_name = cls_or_func.__name__ if name is None else name
        
        # Check if already registered
        if not force:
            if inspect.isclass(dct):
                assert not hasattr(dct, register_name), \
                    f'Module {dct.__name__} already has {register_name}'
            else:
                assert register_name not in dct, \
                    f'{register_name} has already been registered'
        
        # Register function
        if inspect.isfunction(cls_or_func):
            @functools.wraps(cls_or_func)
            def wrap_func(*args, **kwargs):
                return cls_or_func(*args, **kwargs)
            
            if isinstance(dct, dict):
                dct[register_name] = wrap_func
            elif inspect.isclass(dct):
                setattr(dct, register_name, wrap_func)
            else:
                raise AttributeError(f'Unsupported registration target: {type(dct)}')
            
            return wrap_func
        
        # Register class
        elif inspect.isclass(cls_or_func):
            schema = extract_schema(cls_or_func)
            dct[register_name] = schema
            return cls_or_func
        
        else:
            raise ValueError(f'Unsupported registration type: {type(cls_or_func)}')
    
    return decorator


def extract_schema(cls: type) -> dict:
    """
    Extract schema information from a class for dependency injection.
    
    Args:
        cls: Class to extract schema from
        
    Returns:
        Dictionary containing schema information
    """
    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']
    num_defaults = len(argspec.defaults) if argspec.defaults is not None else 0
    num_required = len(arg_names) - num_defaults
    
    schema = {}
    schema['_name'] = cls.__name__
    schema['_pymodule'] = importlib.import_module(cls.__module__)
    schema['_inject'] = getattr(cls, '__inject__', [])
    schema['_share'] = getattr(cls, '__share__', [])
    schema['_kwargs'] = {}
    
    for i, name in enumerate(arg_names):
        if name in schema['_share']:
            assert i >= num_required, 'Shared config must have default value'
            value = argspec.defaults[i - num_required]
        elif i >= num_required:
            value = argspec.defaults[i - num_required]
        else:
            value = None
        
        schema[name] = value
        schema['_kwargs'][name] = value
    
    return schema


def create(type_or_name: Union[type, str], global_cfg: dict = None, **kwargs):
    """
    Create an instance from registered configuration.
    
    Args:
        type_or_name: Class type or registered name
        global_cfg: Global configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Created instance
    """
    if global_cfg is None:
        global_cfg = GLOBAL_CONFIG
    
    assert isinstance(type_or_name, (type, str)), \
        'create() expects a class type or registered name'
    
    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__
    
    # Check if already instantiated
    if name in global_cfg:
        if hasattr(global_cfg[name], '__dict__'):
            return global_cfg[name]
    else:
        raise ValueError(f'Module {name} is not registered')
    
    cfg = global_cfg[name]
    
    # Handle type redirection
    if isinstance(cfg, dict) and 'type' in cfg:
        target_cfg = global_cfg[cfg['type']]
        
        # Clean existing args
        keys_to_remove = [k for k in target_cfg.keys() if not k.startswith('_')]
        for key in keys_to_remove:
            del target_cfg[key]
        
        # Restore defaults and update with config
        target_cfg.update(target_cfg['_kwargs'])
        target_cfg.update(cfg)
        target_cfg.update(kwargs)
        
        target_name = target_cfg.pop('type')
        return create(target_name, global_cfg)
    
    # Get the actual class
    module = getattr(cfg['_pymodule'], name)
    module_kwargs = {}
    module_kwargs.update(cfg)
    
    # Handle shared variables
    for key in cfg['_share']:
        if key in global_cfg:
            module_kwargs[key] = global_cfg[key]
        else:
            module_kwargs[key] = cfg[key]
    
    # Handle dependency injection
    for key in cfg['_inject']:
        inject_config = cfg[key]
        
        if inject_config is None:
            continue
        
        if isinstance(inject_config, str):
            # Direct injection by name
            if inject_config not in global_cfg:
                raise ValueError(f'Missing inject config: {inject_config}')
            
            inject_cfg = global_cfg[inject_config]
            
            if isinstance(inject_cfg, dict):
                module_kwargs[key] = create(inject_cfg['_name'], global_cfg)
            else:
                module_kwargs[key] = inject_cfg
        
        elif isinstance(inject_config, dict):
            # Injection with type specification
            if 'type' not in inject_config:
                raise ValueError('Missing "type" in inject config')
            
            inject_type = str(inject_config['type'])
            if inject_type not in global_cfg:
                raise ValueError(f'Missing {inject_type} in global config')
            
            # Similar to type redirection above
            inject_cfg = global_cfg[inject_type]
            keys_to_remove = [k for k in inject_cfg.keys() if not k.startswith('_')]
            for key_to_remove in keys_to_remove:
                del inject_cfg[key_to_remove]
            
            inject_cfg.update(inject_cfg['_kwargs'])
            inject_cfg.update(inject_config)
            inject_name = inject_cfg.pop('type')
            
            module_kwargs[key] = create(inject_name, global_cfg)
        
        else:
            raise ValueError(f'Unsupported inject type: {type(inject_config)}')
    
    # Filter out private keys
    module_kwargs = {k: v for k, v in module_kwargs.items() if not k.startswith('_')}
    
    # Create instance
    return module(**module_kwargs)


def list_registered(pattern: str = None) -> List[str]:
    """
    List all registered components.
    
    Args:
        pattern: Optional pattern to filter names
        
    Returns:
        List of registered component names
    """
    if pattern is None:
        return list(GLOBAL_CONFIG.keys())
    
    import re
    return [name for name in GLOBAL_CONFIG.keys() if re.match(pattern, name)]


def get_registered(name: str) -> dict:
    """
    Get registration information for a component.
    
    Args:
        name: Component name
        
    Returns:
        Registration schema
    """
    if name not in GLOBAL_CONFIG:
        raise ValueError(f'Component {name} not found')
    
    return GLOBAL_CONFIG[name]


def clear_registry():
    """Clear all registered components"""
    GLOBAL_CONFIG.clear()


# Registry for common PyTorch components
def register_torch_optimizers():
    """Register common PyTorch optimizers"""
    import torch.optim as optim
    
    optimizers = [
        optim.Adam, optim.AdamW, optim.SGD, optim.RMSprop
    ]
    
    for optimizer_cls in optimizers:
        GLOBAL_CONFIG[optimizer_cls.__name__] = {
            '_name': optimizer_cls.__name__,
            '_pymodule': optim,
            '_inject': [],
            '_share': [],
            '_kwargs': {}
        }


def register_torch_schedulers():
    """Register common PyTorch learning rate schedulers"""
    import torch.optim.lr_scheduler as lr_scheduler
    
    schedulers = [
        lr_scheduler.StepLR, lr_scheduler.MultiStepLR, 
        lr_scheduler.ExponentialLR, lr_scheduler.CosineAnnealingLR,
        lr_scheduler.ReduceLROnPlateau
    ]
    
    for scheduler_cls in schedulers:
        GLOBAL_CONFIG[scheduler_cls.__name__] = {
            '_name': scheduler_cls.__name__,
            '_pymodule': lr_scheduler,
            '_inject': [],
            '_share': [],
            '_kwargs': {}
        }


# Auto-register common components
register_torch_optimizers()
register_torch_schedulers()
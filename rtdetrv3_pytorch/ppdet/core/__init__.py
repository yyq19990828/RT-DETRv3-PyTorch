"""
Core module for RT-DETRv3 PyTorch

Provides the unified registration system and workspace utilities.
This module implements PaddlePaddle-compatible component registration and factory patterns.
"""

from .workspace import (
    global_config,
    register,
    create,
    merge_config,
    reset_global_config,
    get_registered_classes,
)

__all__ = [
    'global_config',
    'register',
    'create',
    'merge_config',
    'reset_global_config',
    'get_registered_classes',
]

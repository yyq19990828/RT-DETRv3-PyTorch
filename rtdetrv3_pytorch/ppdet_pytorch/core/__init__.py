"""
Core module for RT-DETRv3 PyTorch

Migrated from PaddlePaddle RT-DETRv3/ppdet/core
Provides the unified registration system and workspace utilities.

Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Adapted for PyTorch by RT-DETRv3 PyTorch Team.
"""

from .workspace import (
    global_config,
    load_config,
    merge_config,
    get_registered_modules,
    create,
    register,
    serializable,
    dump_value,
)

from .config import (
    SchemaDict,
    SchemaValue,
    SharedConfig,
    extract_schema,
    Callable,
)

__all__ = [
    # workspace functions
    'global_config',
    'load_config',
    'merge_config',
    'get_registered_modules',
    'create',
    'register',
    'serializable',
    'dump_value',
    # config classes
    'SchemaDict',
    'SchemaValue',
    'SharedConfig',
    'extract_schema',
    'Callable',
]

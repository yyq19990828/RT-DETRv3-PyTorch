"""
Configuration system for RT-DETRv3 PyTorch

Provides PaddlePaddle-compatible configuration schema and YAML serialization.
"""

from .schema import SchemaValue, SchemaDict, SharedConfig, extract_schema
from .yaml_helpers import serializable, Callable

__all__ = [
    'SchemaValue',
    'SchemaDict',
    'SharedConfig',
    'extract_schema',
    'serializable',
    'Callable',
]

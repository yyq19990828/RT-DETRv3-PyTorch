"""Core configuration and workspace management for RT-DETRv3"""

from .config import RTDETRConfig, YAMLConfig, load_config, merge_config, merge_dict, parse_cli
from .workspace import register, create, GLOBAL_CONFIG, extract_schema

__all__ = [
    'RTDETRConfig', 'YAMLConfig', 'load_config', 'merge_config', 'merge_dict', 'parse_cli',
    'register', 'create', 'GLOBAL_CONFIG', 'extract_schema'
]
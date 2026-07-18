"""
Configuration system for RT-DETRv3 PyTorch

Adapted from PaddlePaddle's ppdet.core.workspace to maintain compatibility
with PaddlePaddle config format including _BASE_ inheritance.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


logger = logging.getLogger(__name__)


BASE_KEY = '_BASE_'


class AttrDict(dict):
    """Single level attribute dict, NOT recursive (matches PaddlePaddle)"""
    
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)
    
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def copy(self):
        new_dict = AttrDict()
        for k, v in self.items():
            new_dict.update({k: v})
        return new_dict


global_config = AttrDict()


def dict_merge(dct: Dict, merge_dct: Dict) -> Dict:
    """
    Recursive dict merge. Inspired by dict.update(), instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The merge_dct is merged into dct.
    
    Args:
        dct: dict onto which the merge is executed
        merge_dct: dict merged into dct
        
    Returns:
        dct (merged)
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and 
                isinstance(merge_dct[k], dict)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def _load_config_with_base(file_path: str) -> Dict:
    """
    Parse and load _BASE_ recursively.
    
    Args:
        file_path: Path to YAML config file
        
    Returns:
        Loaded config dict with _BASE_ configs merged
    """
    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)
    
    # NOTE: cfgs outside have higher priority than cfgs in _BASE_
    if BASE_KEY in file_cfg:
        all_base_cfg = AttrDict()
        base_ymls = list(file_cfg[BASE_KEY])
        
        for base_yml in base_ymls:
            # Expand user home directory
            if base_yml.startswith("~"):
                base_yml = os.path.expanduser(base_yml)
            
            # Convert relative path to absolute
            if not base_yml.startswith('/'):
                base_yml = os.path.join(os.path.dirname(file_path), base_yml)
            
            # Recursively load base config
            base_cfg = _load_config_with_base(base_yml)
            all_base_cfg = merge_config(base_cfg, all_base_cfg)
        
        # Remove _BASE_ key after processing
        del file_cfg[BASE_KEY]
        
        # Merge current config over base configs (current has higher priority)
        return merge_config(file_cfg, all_base_cfg)
    
    return file_cfg


def load_config(file_path: str) -> AttrDict:
    """
    Load config from file (matches PaddlePaddle API).
    
    Args:
        file_path: Path of the config file to be loaded
        
    Returns:
        global_config (AttrDict)
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    
    # Load config from file and merge into global config
    cfg = _load_config_with_base(file_path)
    cfg['filename'] = os.path.splitext(os.path.split(file_path)[-1])[0]
    merge_config(cfg)
    
    logger.info(f"Loaded config from {file_path}")
    
    return global_config


def merge_config(config: Dict, another_cfg: Optional[Dict] = None) -> Dict:
    """
    Merge config into global config or another_cfg.
    
    Args:
        config: Config to be merged
        another_cfg: Optional target dict (defaults to global_config)
        
    Returns:
        Merged config dict
    """
    global global_config
    dct = another_cfg if another_cfg is not None else global_config
    return dict_merge(dct, config)


def apply_overrides(config: AttrDict, overrides: List[str]) -> AttrDict:
    """
    Apply command-line config overrides.
    
    Args:
        config: Base config AttrDict
        overrides: List of "key=value" or "key.nested=value" strings
        
    Returns:
        Modified config
    """
    for override in overrides:
        if '=' not in override:
            logger.warning(f"Invalid override format (expected key=value): {override}")
            continue
        
        key_path, value_str = override.split('=', 1)
        keys = key_path.split('.')
        
        # Parse value (try int, float, bool, then string)
        value = parse_value(value_str)
        
        # Navigate to nested dict and set value
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        logger.info(f"Override: {key_path} = {value}")
    
    return config


def parse_value(value_str: str) -> Any:
    """
    Parse string value to appropriate Python type.
    
    Args:
        value_str: String representation of value
        
    Returns:
        Parsed value (int, float, bool, or string)
    """
    # Boolean
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # None
    if value_str.lower() in ('none', 'null'):
        return None
    
    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # List (comma-separated)
    if ',' in value_str:
        return [parse_value(v.strip()) for v in value_str.split(',')]
    
    # String
    return value_str


def save_config(config: AttrDict, save_path: str):
    """
    Save config to YAML file.
    
    Args:
        config: Config AttrDict
        save_path: Output YAML file path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.safe_dump(dict(config), f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved config to {save_path}")


def parse_args():
    """
    Parse command-line arguments for training/evaluation scripts.
    
    Returns:
        argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(description="RT-DETRv3 PyTorch")
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('-o', '--override', nargs='*', default=[],
                       help='Config overrides (e.g., epoch=100 LearningRate.base_lr=0.0002)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation only')
    
    args = parser.parse_args()
    return args


# Backward compatibility aliases
Config = AttrDict


__all__ = [
    'AttrDict',
    'Config',
    'global_config',
    'load_config',
    'merge_config',
    'apply_overrides',
    'parse_value',
    'save_config',
    'parse_args',
    'BASE_KEY',
]

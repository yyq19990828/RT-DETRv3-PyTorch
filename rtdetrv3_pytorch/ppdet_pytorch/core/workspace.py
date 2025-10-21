"""
Unified workspace module for RT-DETRv3 PyTorch

Migrated from PaddlePaddle RT-DETRv3/ppdet/core/workspace.py
Provides PaddlePaddle-compatible registration system for dynamic component instantiation.

Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Adapted for PyTorch by RT-DETRv3 PyTorch Team.
"""

import importlib
import os
import sys
import yaml
import collections

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

from .config.schema import SchemaDict, SharedConfig, extract_schema
from .config.yaml_helpers import serializable

__all__ = [
    'global_config',
    'load_config',
    'merge_config',
    'get_registered_modules',
    'create',
    'register',
    'serializable',
    'dump_value',
]


def dump_value(value):
    """
    Dump value to string representation for logging.

    Args:
        value: Value to dump

    Returns:
        str: String representation
    """
    if hasattr(value, '__dict__') or isinstance(value, (dict, tuple, list)):
        value = yaml.dump(value, default_flow_style=True)
        value = value.replace('\n', '')
        value = value.replace('...', '')
        return f"'{value}'"
    else:
        # primitive types
        return str(value)


class AttrDict(dict):
    """
    Single level attribute dict, NOT recursive.

    Allows accessing dict keys as attributes: cfg.batch_size instead of cfg['batch_size']
    """

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


# Global configuration storage
global_config = AttrDict()

BASE_KEY = '_BASE_'


def _load_config_with_base(file_path):
    """
    Parse and load _BASE_ recursively.

    Args:
        file_path (str): Path to YAML config file

    Returns:
        dict: Loaded config with base configs merged
    """
    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)

    # NOTE: cfgs outside have higher priority than cfgs in _BASE_
    if BASE_KEY in file_cfg:
        all_base_cfg = AttrDict()
        base_ymls = list(file_cfg[BASE_KEY])
        for base_yml in base_ymls:
            if base_yml.startswith("~"):
                base_yml = os.path.expanduser(base_yml)
            if not base_yml.startswith('/'):
                base_yml = os.path.join(os.path.dirname(file_path), base_yml)

            base_cfg = _load_config_with_base(base_yml)
            all_base_cfg = merge_config(base_cfg, all_base_cfg)

        del file_cfg[BASE_KEY]
        return merge_config(file_cfg, all_base_cfg)

    return file_cfg


def load_config(file_path):
    """
    Load config from file.

    Args:
        file_path (str): Path of the config file to be loaded.

    Returns:
        AttrDict: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"

    # load config from file and merge into global config
    cfg = _load_config_with_base(file_path)
    cfg['filename'] = os.path.splitext(os.path.split(file_path)[-1])[0]
    merge_config(cfg)

    return global_config


def dict_merge(dct, merge_dct):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        dct: dict onto which the merge is executed
        merge_dct: dct merged into dct

    Returns:
        dct
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and
                isinstance(merge_dct[k], collectionsAbc.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def merge_config(config, another_cfg=None):
    """
    Merge config into global config or another_cfg.

    Args:
        config (dict): Config to be merged.
        another_cfg (dict, optional): Another config to merge into.
                                       If None, merges into global_config.

    Returns:
        dict: Merged config
    """
    global global_config
    dct = another_cfg or global_config
    return dict_merge(dct, config)


def get_registered_modules():
    """
    Get all registered modules (those with SchemaDict).

    Returns:
        dict: Mapping of module name to SchemaDict
    """
    return {k: v for k, v in global_config.items() if isinstance(v, SchemaDict)}


def make_partial(cls):
    """
    Make partial function for operator classes.

    Args:
        cls: Class with __op__ attribute

    Returns:
        cls: Modified class with partial application
    """
    op_module = importlib.import_module(cls.__op__.__module__)
    op = getattr(op_module, cls.__op__.__name__)
    cls.__category__ = getattr(cls, '__category__', None) or 'op'

    def partial_apply(self, *args, **kwargs):
        kwargs_ = self.__dict__.copy()
        kwargs_.update(kwargs)
        return op(*args, **kwargs_)

    if getattr(cls, '__append_doc__', True):
        if sys.version_info[0] > 2:
            cls.__doc__ = f"Wrapper for `{op.__name__}` OP"
            cls.__init__.__doc__ = op.__doc__
            cls.__call__ = partial_apply
            cls.__call__.__doc__ = op.__doc__
        else:
            # Python 2 compatibility
            partial_apply.__doc__ = op.__doc__
            cls.__call__ = partial_apply
    return cls


def register(cls):
    """
    Register a given module class.

    Args:
        cls (type): Module class to be registered.

    Returns:
        cls: The registered class

    Raises:
        ValueError: If module class already registered
    """
    if cls.__name__ in global_config:
        raise ValueError(f"Module class already registered: {cls.__name__}")

    if hasattr(cls, '__op__'):
        cls = make_partial(cls)

    global_config[cls.__name__] = extract_schema(cls)
    return cls


def create(cls_or_name, **kwargs):
    """
    Create an instance of given module class.

    Args:
        cls_or_name (type or str): Class of which to create instance.
        **kwargs: Additional keyword arguments to pass to constructor

    Returns:
        Instance of type `cls_or_name`

    Raises:
        ValueError: If module not registered
    """
    assert type(cls_or_name) in [type, str], "should be a class or name of a class"
    name = cls_or_name if type(cls_or_name) == str else cls_or_name.__name__

    if name in global_config:
        if isinstance(global_config[name], SchemaDict):
            pass
        elif hasattr(global_config[name], "__dict__"):
            # support instance return directly
            return global_config[name]
        else:
            raise ValueError(f"The module {name} is not registered")
    else:
        raise ValueError(f"The module {name} is not registered")

    config = global_config[name]
    cls = getattr(config.pymodule, name)
    cls_kwargs = {}
    cls_kwargs.update(global_config[name])

    # parse `shared` annotation of registered modules
    if getattr(config, 'shared', None):
        for k in config.shared:
            target_key = config[k]
            shared_conf = config.schema[k].default
            assert isinstance(shared_conf, SharedConfig)
            if target_key is not None and not isinstance(target_key, SharedConfig):
                continue  # value is given for the module
            elif shared_conf.key in global_config:
                # `key` is present in config
                cls_kwargs[k] = global_config[shared_conf.key]
            else:
                cls_kwargs[k] = shared_conf.default_value

    # parse `inject` annotation of registered modules
    if getattr(cls, 'from_config', None):
        cls_kwargs.update(cls.from_config(config, **kwargs))

    if getattr(config, 'inject', None):
        for k in config.inject:
            target_key = config[k]
            # optional dependency
            if target_key is None:
                continue

            if isinstance(target_key, dict) or hasattr(target_key, '__dict__'):
                if 'name' not in target_key.keys():
                    continue
                inject_name = str(target_key['name'])
                if inject_name not in global_config:
                    raise ValueError(
                        f"Missing injection name {k} and check it's name in cfg file"
                    )
                target = global_config[inject_name]
                for i, v in target_key.items():
                    if i == 'name':
                        continue
                    target[i] = v
                if isinstance(target, SchemaDict):
                    cls_kwargs[k] = create(inject_name)
            elif isinstance(target_key, str):
                if target_key not in global_config:
                    raise ValueError(f"Missing injection config: {target_key}")
                target = global_config[target_key]
                if isinstance(target, SchemaDict):
                    cls_kwargs[k] = create(target_key)
                elif hasattr(target, '__dict__'):  # serialized object
                    cls_kwargs[k] = target
            else:
                raise ValueError(f"Unsupported injection type: {target_key}")

    # prevent modification of global config values of reference types
    # (e.g., list, dict) from within the created module instances
    return cls(**cls_kwargs)

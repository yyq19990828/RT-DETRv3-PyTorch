# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import importlib
import os
import sys
from collections.abc import Mapping

import yaml

from .config.schema import SchemaDict, SharedConfig, extract_schema
from .config.yaml_helpers import serializable

__all__ = [
    "global_config",
    "load_config",
    "merge_config",
    "get_registered_modules",
    "create",
    "register",
    "serializable",
    "dump_value",
]


def dump_value(value):
    # XXX this is hackish, but collections.abc is not available in python 2
    if hasattr(value, "__dict__") or isinstance(value, (dict, tuple, list)):
        value = yaml.dump(value, default_flow_style=True)
        value = value.replace("\n", "")
        value = value.replace("...", "")
        return "'{}'".format(value)
    else:
        # primitive types
        return str(value)


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        new_dict = AttrDict()
        for k, v in self.items():
            new_dict.update({k: v})
        return new_dict


global_config = AttrDict()

BASE_KEY = "_BASE_"


# parse and load _BASE_ recursively
def _load_config_with_base(file_path):
    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)

    # NOTE: cfgs outside have higher priority than cfgs in _BASE_
    if BASE_KEY in file_cfg:
        all_base_cfg = AttrDict()
        base_ymls = list(file_cfg[BASE_KEY])
        for base_yml in base_ymls:
            if base_yml.startswith("~"):
                base_yml = os.path.expanduser(base_yml)
            if not base_yml.startswith("/"):
                base_yml = os.path.join(os.path.dirname(file_path), base_yml)

            with open(base_yml) as f:
                base_cfg = _load_config_with_base(base_yml)
                all_base_cfg = merge_config(base_cfg, all_base_cfg)

        del file_cfg[BASE_KEY]
        return merge_config(file_cfg, all_base_cfg)

    return file_cfg


def load_config(file_path):
    """
    Load one isolated config from file.

    Registered component schemas are preserved, while runtime values from a
    previous call are removed before the newly parsed config is merged.
    ``_BASE_`` files are still merged as one config before that replacement.

    Args:
        file_path (str): Path of the config file to be loaded.

    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"

    # Parse first so a malformed file does not destroy the active workspace.
    cfg = _load_config_with_base(file_path)
    cfg["filename"] = os.path.splitext(os.path.split(file_path)[-1])[0]
    _reset_runtime_config()
    merge_config(cfg)

    return global_config


def _reset_runtime_config():
    """Keep registrations while removing values from earlier config loads."""
    registered = {
        name: extract_schema(config.cls)
        for name, config in global_config.items()
        if isinstance(config, SchemaDict)
    }
    global_config.clear()
    global_config.update(registered)


def dict_merge(dct, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        dct: dict onto which the merge is executed
        merge_dct: dct merged into dct

    Returns: dct
    """
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], Mapping):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def merge_config(config, another_cfg=None):
    """
    Merge config into global config or another_cfg.

    Args:
        config (dict): Config to be merged.

    Returns: global config
    """
    global global_config
    dct = global_config if another_cfg is None else another_cfg
    return dict_merge(dct, config)


def get_registered_modules():
    return {k: v for k, v in global_config.items() if isinstance(v, SchemaDict)}


def make_partial(cls):
    op_module = importlib.import_module(cls.__op__.__module__)
    op = getattr(op_module, cls.__op__.__name__)
    cls.__category__ = getattr(cls, "__category__", None) or "op"

    def partial_apply(self, *args, **kwargs):
        kwargs_ = self.__dict__.copy()
        kwargs_.update(kwargs)
        return op(*args, **kwargs_)

    if getattr(cls, "__append_doc__", True):  # XXX should default to True?
        if sys.version_info[0] > 2:
            cls.__doc__ = "Wrapper for `{}` OP".format(op.__name__)
            cls.__init__.__doc__ = op.__doc__
            cls.__call__ = partial_apply
            cls.__call__.__doc__ = op.__doc__
        else:
            # XXX work around for python 2
            partial_apply.__doc__ = op.__doc__
            cls.__call__ = partial_apply
    return cls


def register(cls):
    """
    Register a given module class.

    Args:
        cls (type): Module class to be registered.

    Returns: cls
    """
    if cls.__name__ in global_config:
        raise ValueError("Module class already registered: {}".format(cls.__name__))
    if hasattr(cls, "__op__"):
        cls = make_partial(cls)
    global_config[cls.__name__] = extract_schema(cls)
    return cls


def create(cls_or_name, **kwargs):
    """
    Create an instance from a registered class or configuration block.

    Args:
        cls_or_name (type, str or mapping): Registered class, registered name,
            or a configuration containing a ``name``/``type`` field.
        **kwargs: Explicit constructor values and context consumed by
            ``from_config`` (for example ``input_shape``).

    Returns: instance of type `cls_or_name`
    """
    component_config = {}
    if isinstance(cls_or_name, Mapping):
        component_config = dict(cls_or_name)
        name = component_config.pop("name", None)
        if name is None:
            name = component_config.pop("type", None)
        else:
            component_config.pop("type", None)
        if name is None:
            raise ValueError("Component config must contain a 'name' or 'type' field")
    elif isinstance(cls_or_name, str):
        name = cls_or_name
    elif isinstance(cls_or_name, type):
        name = cls_or_name.__name__
    else:
        raise TypeError(
            "cls_or_name must be a class, registered name, or config mapping"
        )

    if name not in global_config:
        raise ValueError("The module {} is not registered".format(name))

    registered = global_config[name]
    if not isinstance(registered, SchemaDict):
        if isinstance(registered, Mapping):
            return create(registered, **kwargs)
        if hasattr(registered, "__dict__"):
            return registered
        raise ValueError("The module {} is not registered".format(name))

    # Work on a local schema copy so nested creation cannot mutate the global
    # registered configuration shared by later tests or commands.
    config = registered.copy()
    config.update(component_config)
    cls = config.cls
    if cls is None:
        if config.pymodule is None:
            raise ValueError("The module {} has no registered class".format(name))
        cls = getattr(config.pymodule, name)
    cls_kwargs = {}
    cls_kwargs.update(config)

    explicit_constructor_kwargs = {
        key: value for key, value in kwargs.items() if key in config.schema
    }
    from_config_context = {
        key: value for key, value in kwargs.items() if key not in config.schema
    }

    # parse `shared` annoation of registered modules
    if getattr(config, "shared", None):
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

    # parse `inject` annoation of registered modules
    if getattr(cls, "from_config", None):
        cls_kwargs.update(cls.from_config(config, **from_config_context))

    inject_fields = set(getattr(config, "inject", None) or [])
    if inject_fields:
        for k in config.inject:
            target_key = explicit_constructor_kwargs.get(k, config[k])
            # optional dependency
            if target_key is None:
                if k in explicit_constructor_kwargs:
                    cls_kwargs[k] = None
                continue

            if isinstance(target_key, Mapping):
                if "name" in target_key or "type" in target_key:
                    cls_kwargs[k] = create(target_key)
                else:
                    cls_kwargs[k] = target_key
            elif isinstance(target_key, str):
                if target_key not in global_config:
                    raise ValueError("Missing injection config: {}".format(target_key))
                cls_kwargs[k] = create(target_key)
            else:
                cls_kwargs[k] = target_key

    # Explicit values have the highest priority. Injected fields were already
    # resolved above and must not be replaced by their raw string/dict input.
    for key, value in explicit_constructor_kwargs.items():
        if key not in inject_fields:
            cls_kwargs[key] = value
    # prevent modification of global config values of reference types
    # (e.g., list, dict) from within the created module instances
    # kwargs = copy.deepcopy(kwargs)
    return cls(**cls_kwargs)

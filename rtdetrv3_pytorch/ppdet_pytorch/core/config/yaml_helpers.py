"""
YAML serialization helpers for RT-DETRv3 PyTorch

Provides PaddlePaddle-compatible YAML serialization for configuration objects.
Migrated from PaddlePaddle RT-DETRv3/ppdet/core/config/yaml_helpers.py

Key features:
- @serializable decorator for YAML-serializable classes
- Callable helper for dynamic function invocation from YAML
- Custom YAML constructors and representers
"""

import importlib
import inspect
from typing import Any, Type

import yaml

from .schema import SharedConfig

__all__ = ['serializable', 'Callable']


def represent_dictionary_order(self, dict_data):
    """Represent OrderedDict in YAML with preserved order"""
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())


def setup_orderdict():
    """Setup OrderedDict representation in YAML"""
    from collections import OrderedDict
    yaml.add_representer(OrderedDict, represent_dictionary_order)


def _make_python_constructor(cls: Type) -> callable:
    """
    Create a YAML constructor for a given class.

    Args:
        cls: Class to create constructor for

    Returns:
        Constructor function for YAML loader
    """
    def python_constructor(loader, node):
        if isinstance(node, yaml.SequenceNode):
            # Construct from list: !ClassName [arg1, arg2, ...]
            args = loader.construct_sequence(node, deep=True)
            return cls(*args)
        else:
            # Construct from dict: !ClassName {key1: val1, key2: val2}
            kwargs = loader.construct_mapping(node, deep=True)
            try:
                return cls(**kwargs)
            except Exception as ex:
                print(f"Error when construct {cls.__name__} instance from yaml config")
                raise ex

    return python_constructor


def _make_python_representer(cls: Type) -> callable:
    """
    Create a YAML representer for a given class.

    Args:
        cls: Class to create representer for

    Returns:
        Representer function for YAML dumper
    """
    # Get constructor argspec
    if hasattr(inspect, 'getfullargspec'):
        argspec = inspect.getfullargspec(cls.__init__)
    else:
        # Python 2 compatibility (not needed for PyTorch, but kept for completeness)
        argspec = inspect.getargspec(cls.__init__)

    argnames = [arg for arg in argspec.args if arg != 'self']

    def python_representer(dumper, obj):
        # Extract attributes corresponding to constructor arguments
        if argnames:
            data = {name: getattr(obj, name, None) for name in argnames}
        else:
            # Fallback to all attributes
            data = obj.__dict__.copy()

        # Remove internal _id field if present
        if '_id' in data:
            del data['_id']

        # Represent as !ClassName {key: value, ...}
        return dumper.represent_mapping(f'!{cls.__name__}', data)

    return python_representer


def serializable(cls: Type) -> Type:
    """
    Add YAML loader and dumper for given class.

    Marks a class as "trivially serializable" - can be reconstructed from
    its constructor arguments. Adds YAML constructor and representer.

    Args:
        cls: Class to be serialized

    Returns:
        cls (unchanged, but registered with YAML)

    Example:
        >>> @serializable
        >>> class COCODataSet:
        >>>     def __init__(self, dataset_dir, image_dir, anno_path):
        >>>         self.dataset_dir = dataset_dir
        >>>         self.image_dir = image_dir
        >>>         self.anno_path = anno_path
        >>>
        >>> # Can now serialize/deserialize:
        >>> yaml_str = yaml.dump(dataset)
        >>> loaded = yaml.load(yaml_str, Loader=yaml.Loader)
    """
    # Add constructor: YAML -> Python object
    yaml.add_constructor(
        f'!{cls.__name__}',
        _make_python_constructor(cls),
        Loader=yaml.Loader
    )

    # Add representer: Python object -> YAML
    yaml.add_representer(
        cls,
        _make_python_representer(cls),
        Dumper=yaml.Dumper
    )

    return cls


# Add SharedConfig representer
yaml.add_representer(
    SharedConfig,
    lambda dumper, obj: dumper.represent_data(obj.default_value),
    Dumper=yaml.Dumper
)


@serializable
class Callable:
    """
    Helper to be used in YAML for creating arbitrary class objects.

    Allows invoking any function/class from YAML configuration by specifying
    its full module path.

    Args:
        full_type (str): The full module path to target function (e.g., 'torch.optim.AdamW')
        args (list): Positional arguments for the function
        kwargs (dict): Keyword arguments for the function

    Example YAML:
        ```yaml
        optimizer: !Callable
          full_type: torch.optim.AdamW
          kwargs:
            lr: 0.0001
            weight_decay: 0.0001
        ```

    Example Python:
        >>> callable_obj = Callable('torch.optim.AdamW', kwargs={'lr': 0.0001})
        >>> optimizer = callable_obj()  # Returns AdamW instance
    """

    def __init__(self, full_type: str, args: list = None, kwargs: dict = None):
        super(Callable, self).__init__()
        self.full_type = full_type
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

    def __call__(self) -> Any:
        """
        Invoke the function/class specified by full_type.

        Returns:
            Result of calling the function/class
        """
        # Parse module and function name
        if '.' in self.full_type:
            idx = self.full_type.rfind('.')
            module = importlib.import_module(self.full_type[:idx])
            func_name = self.full_type[idx + 1:]
        else:
            # Builtin function (no module path)
            try:
                module = importlib.import_module('builtins')
            except Exception:
                # Python 2 compatibility
                module = importlib.import_module('__builtin__')
            func_name = self.full_type

        # Get function/class
        func = getattr(module, func_name)

        # Invoke with args and kwargs
        return func(*self.args, **self.kwargs)

"""
Configuration schema system for RT-DETRv3 PyTorch

Provides PaddlePaddle-compatible schema validation for configuration objects.
Migrated from PaddlePaddle RT-DETRv3/ppdet/core/config/schema.py

Key features:
- SchemaValue: Individual configuration value with type and default
- SchemaDict: Dictionary with schema validation
- SharedConfig: Shared configuration injection
- extract_schema: Extract schema from class definition
"""

import importlib
import inspect
import re
from typing import Any, Optional, Type

# Optional dependencies
try:
    from docstring_parser import parse as doc_parse
except ImportError:
    def doc_parse(*args):
        """Fallback when docstring_parser not available"""
        return None

try:
    from typeguard import check_type
except ImportError:
    def check_type(*args):
        """Fallback when typeguard not available"""
        pass


__all__ = ['SchemaValue', 'SchemaDict', 'SharedConfig', 'extract_schema']


class SchemaValue:
    """
    Schema definition for a single configuration value.

    Attributes:
        name (str): Parameter name
        doc (str): Documentation string
        type: Expected type (for validation)
        default: Default value (if has_default() is True)
    """

    def __init__(self, name: str, doc: str = '', type_: Optional[Type] = None):
        super(SchemaValue, self).__init__()
        self.name = name
        self.doc = doc
        self.type = type_

    def set_default(self, value: Any) -> None:
        """Set default value for this parameter"""
        self.default = value

    def has_default(self) -> bool:
        """Check if this parameter has a default value"""
        return hasattr(self, 'default')


class SchemaDict(dict):
    """
    Dictionary with schema validation.

    Extends dict to support:
    - Schema-based validation (required/optional params, type checking)
    - Default values from schema
    - Nested SchemaDict updates
    - Strict mode (disallow extra keys)

    Attributes:
        schema (dict): Mapping of key -> SchemaValue
        strict (bool): If True, reject keys not in schema
        doc (str): Documentation for this schema
        name (str): Name of the class this schema describes
    """

    def __init__(self, **kwargs):
        super(SchemaDict, self).__init__()
        self.schema = {}
        self.strict = False
        self.doc = ""
        self.update(kwargs)

    def __setitem__(self, key, value):
        """Update value, merging SchemaDict if key already exists"""
        if isinstance(value, dict) and key in self and isinstance(self[key], SchemaDict):
            # Merge nested SchemaDict
            self[key].update(value)
        else:
            super(SchemaDict, self).__setitem__(key, value)

    def __missing__(self, key):
        """Return default value or schema when key not found"""
        if self.has_default(key):
            return self.schema[key].default
        elif key in self.schema:
            return self.schema[key]
        else:
            raise KeyError(key)

    def copy(self):
        """Deep copy of SchemaDict"""
        newone = SchemaDict()
        newone.__dict__.update(self.__dict__)
        newone.update(self)
        return newone

    def set_schema(self, key: str, value: SchemaValue) -> None:
        """Set schema for a key"""
        assert isinstance(value, SchemaValue), f"Schema value must be SchemaValue, got {type(value)}"
        self.schema[key] = value

    def set_strict(self, strict: bool) -> None:
        """Enable/disable strict mode (reject extra keys)"""
        self.strict = strict

    def has_default(self, key: str) -> bool:
        """Check if key has a default value in schema"""
        return key in self.schema and self.schema[key].has_default()

    def is_default(self, key: str) -> bool:
        """Check if current value is the default value"""
        if not self.has_default(key):
            return False
        if hasattr(self[key], '__dict__'):
            # Object with __dict__ is considered default if it has default value
            return True
        else:
            return key not in self or self[key] == self.schema[key].default

    def find_default_keys(self):
        """Find all keys using default values"""
        return [
            k for k in list(self.keys()) + list(self.schema.keys())
            if self.is_default(k)
        ]

    def mandatory(self) -> bool:
        """Check if any schema key is mandatory (no default)"""
        return any([k for k in self.schema.keys() if not self.has_default(k)])

    def find_missing_keys(self):
        """Find required keys that are missing or placeholder values"""
        missing = [
            k for k in self.schema.keys()
            if k not in self and not self.has_default(k)
        ]
        placeholders = [k for k in self if self[k] in ('<missing>', '<value>')]
        return missing + placeholders

    def find_extra_keys(self):
        """Find keys in dict but not in schema"""
        return list(set(self.keys()) - set(self.schema.keys()))

    def find_mismatch_keys(self):
        """Find keys with values that don't match schema type"""
        mismatch_keys = []
        for arg in self.schema.values():
            if arg.type is not None and arg.name in self:
                try:
                    check_type(f"{self.name}.{arg.name}", self[arg.name], arg.type)
                except Exception:
                    mismatch_keys.append(arg.name)
        return mismatch_keys

    def validate(self) -> None:
        """
        Validate configuration against schema.

        Raises:
            ValueError: If required keys are missing or extra keys in strict mode
            TypeError: If value types don't match schema
        """
        missing_keys = self.find_missing_keys()
        if missing_keys:
            raise ValueError(
                f"Missing param for class<{self.name}>: {', '.join(missing_keys)}"
            )

        extra_keys = self.find_extra_keys()
        if extra_keys and self.strict:
            raise ValueError(
                f"Extraneous param for class<{self.name}>: {', '.join(extra_keys)}"
            )

        mismatch_keys = self.find_mismatch_keys()
        if mismatch_keys:
            raise TypeError(
                f"Wrong param type for class<{self.name}>: {', '.join(mismatch_keys)}"
            )


class SharedConfig:
    """
    Representation for `__shared__` annotations.

    Shared config allows parameter injection from global config with fallback:
    1. If `key` is set for the module in config, use that value
    2. If `key` is not set but present in global config, use global value
    3. Otherwise, use the provided `default_value`

    Args:
        key (str): Config key to inject (e.g., 'num_classes')
        default_value: Fallback value if key not found

    Example:
        >>> class Detector:
        >>>     __shared__ = ['num_classes']
        >>>     def __init__(self, backbone, num_classes=80):
        >>>         pass
        >>>
        >>> # If global config has num_classes=91, it will be injected
        >>> # Otherwise, defaults to 80
    """

    def __init__(self, key: str, default_value: Any = None):
        super(SharedConfig, self).__init__()
        self.key = key
        self.default_value = default_value


def extract_schema(cls: Type) -> SchemaDict:
    """
    Extract configuration schema from a class.

    Analyzes class constructor to build a SchemaDict with:
    - Parameter names, types, and defaults
    - Documentation from docstrings
    - __inject__ and __shared__ annotations
    - Category metadata

    Args:
        cls (type): Class from which to extract schema

    Returns:
        SchemaDict: Extracted schema with validation rules

    Example:
        >>> @register
        >>> class ResNet:
        >>>     __inject__ = ['norm_layer']
        >>>     __shared__ = ['num_classes']
        >>>     def __init__(self, depth: int, num_classes: int = 80):
        >>>         '''ResNet backbone
        >>>
        >>>         Args:
        >>>             depth: Network depth (18, 34, 50, 101, 152)
        >>>             num_classes: Number of output classes
        >>>         '''
        >>>         pass
        >>>
        >>> schema = extract_schema(ResNet)
        >>> # schema.name = 'ResNet'
        >>> # schema.schema['depth'] = SchemaValue(name='depth', type=int, no default)
        >>> # schema.schema['num_classes'] = SchemaValue(name='num_classes', type=int, default=SharedConfig('num_classes', 80))
    """
    ctor = cls.__init__

    # Get constructor signature
    if hasattr(inspect, 'getfullargspec'):
        argspec = inspect.getfullargspec(ctor)
        annotations = argspec.annotations
        has_kwargs = argspec.varkw is not None
    else:
        # Python 2 compatibility (not needed for PyTorch, but kept for completeness)
        argspec = inspect.getargspec(ctor)
        annotations = getattr(ctor, '__annotations__', {})
        has_kwargs = argspec.keywords is not None

    names = [arg for arg in argspec.args if arg != 'self']
    defaults = argspec.defaults
    num_defaults = len(argspec.defaults) if argspec.defaults is not None else 0
    num_required = len(names) - num_defaults

    # Parse docstring
    docs = cls.__doc__
    if docs is None and getattr(cls, '__category__', None) == 'op':
        docs = cls.__call__.__doc__

    try:
        docstring = doc_parse(docs)
    except Exception:
        docstring = None

    # Extract parameter comments from docstring
    if docstring is None:
        comments = {}
    else:
        comments = {}
        if hasattr(docstring, 'params'):
            for p in docstring.params:
                match_obj = re.match(r'^([a-zA-Z_]+[a-zA-Z_0-9]*).*', p.arg_name)
                if match_obj is not None:
                    comments[match_obj.group(1)] = p.description

    # Build schema
    schema = SchemaDict()
    schema.name = cls.__name__
    schema.doc = ""

    # Extract class docstring summary
    if docs is not None:
        start_pos = 1 if docs[0] == '\n' else 0
        schema.doc = docs[start_pos:].split("\n")[0].strip()

    # Handle PaddlePaddle's weird doc convention (**text**)
    if schema.doc.startswith('**') and schema.doc.endswith('**'):
        schema.doc = schema.doc[2:-2].strip()

    schema.category = getattr(cls, '__category__', 'module')
    schema.strict = not has_kwargs
    schema.pymodule = importlib.import_module(cls.__module__)
    schema.inject = getattr(cls, '__inject__', [])
    schema.shared = getattr(cls, '__shared__', [])

    # Build schema for each parameter
    for idx, name in enumerate(names):
        comment = comments.get(name, name)

        # Injected parameters don't have type validation
        if name in schema.inject:
            type_ = None
        else:
            type_ = annotations.get(name, None)

        value_schema = SchemaValue(name, comment, type_)

        # Handle shared config
        if name in schema.shared:
            assert idx >= num_required, f"Shared config '{name}' must have default value"
            default = defaults[idx - num_required]
            value_schema.set_default(SharedConfig(name, default))
        elif idx >= num_required:
            # Regular default value
            default = defaults[idx - num_required]
            value_schema.set_default(default)

        schema.set_schema(name, value_schema)

    return schema

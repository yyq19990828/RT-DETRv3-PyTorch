"""
RT-DETRv3 PyTorch Detection Library

Main package for RT-DETRv3 object detection framework, migrated from PaddlePaddle.
Provides unified registration system and modular components.

Package Structure:
    - core: Registration system and workspace utilities
    - modeling: Model architectures, backbones, necks, transformers, heads, losses
    - data: Dataset loaders and data transformations
    - engine: Training and evaluation engine
    - optimizer: Optimizers and learning rate schedulers
    - metrics: Evaluation metrics (COCO, etc.)
    - utils: Helper utilities

Example:
    >>> from rtdetrv3_pytorch.ppdet.core.workspace import register, create
    >>> from rtdetrv3_pytorch.ppdet import modeling
    >>>
    >>> # Create model from config
    >>> model_cfg = {'type': 'RTDETRV3', 'backbone': {'type': 'ResNet', 'depth': 50}}
    >>> model = create(model_cfg)
"""

from .core import workspace

__version__ = '1.0.0'

__all__ = [
    'workspace',
    '__version__',
]

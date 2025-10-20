"""
RT-DETRv3 PyTorch Modeling Module

Contains all model components:
- architectures: Complete model definitions (RTDETRV3)
- backbones: Feature extraction networks (ResNet, etc.)
- necks: Feature fusion networks (HybridEncoder)
- transformers: Transformer modules (RTDETRTransformerv3)
- heads: Detection heads (DINOv3Head, PPYOLOEHead)
- losses: Loss functions (DETRLoss, VFLLoss, etc.)

All components are registered using ppdet.core.workspace.register decorator.
"""

# Import all submodules to trigger registration
from . import backbones
from . import necks
from . import transformers
from . import heads
from . import losses
from . import architectures

# Import main model
from .architectures.rtdetrv3 import RTDETRV3

__all__ = [
    'backbones',
    'necks',
    'transformers',
    'heads',
    'losses',
    'architectures',
    'RTDETRV3',
]

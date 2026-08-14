"""
RT-DETRv3 PyTorch Modeling Module

Contains all model components:
- architectures: Complete model definitions (RTDETRV3)
- backbones: Feature extraction networks (ResNet, etc.)
- necks: Feature fusion networks (HybridEncoder)
- transformers: Transformer modules (RTDETRTransformerv3)
- heads: Detection heads (DINOv3Head, PPYOLOEHead)
- losses: Loss functions (DETRLoss, VFLLoss, etc.)

All components are registered using detrs.core.workspace.register decorator.
"""

# Import all submodules to trigger registration
from . import (
    architectures,
    backbones,
    heads,
    losses,
    necks,
    post_process,
    teachers,
    transformers,
)

# Import main model
from .architectures import DEIM, DEIMV2, DFINE, RTDETRV3, RTDETRV4

__all__ = [
    "backbones",
    "necks",
    "transformers",
    "heads",
    "losses",
    "post_process",
    "architectures",
    "teachers",
    "RTDETRV3",
    "DEIM",
    "DFINE",
    "RTDETRV4",
]

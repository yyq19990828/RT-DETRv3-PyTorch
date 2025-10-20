"""
Neck Networks for RT-DETRv3

Available necks:
- HybridEncoder: FPN-PAN neck for multi-scale feature fusion
"""

from .hybrid_encoder import HybridEncoder

__all__ = ['HybridEncoder']

"""
Neck Networks for RT-DETRv3

Available necks:
- HybridEncoder: FPN-PAN neck for multi-scale feature fusion
"""

from .hybrid_encoder import HybridEncoder, build_hybrid_encoder

__all__ = ['HybridEncoder', 'build_hybrid_encoder']

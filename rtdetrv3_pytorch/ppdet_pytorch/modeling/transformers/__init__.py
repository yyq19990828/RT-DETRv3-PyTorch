"""
Transformer Components for RT-DETRv3

Available components:
- MSDeformableAttention: Multi-scale deformable attention
- PositionEmbedding: Unified position embedding (sine or learned)
- MLP: Multi-layer perceptron
- TransformerDecoderLayer: Single decoder layer
- TransformerDecoder: Stack of decoder layers
- MultiHeadAttention: Standard multi-head attention
- RTDETRTransformerv3: RT-DETRv3 transformer decoder
- PPMSDeformableAttention: Pyramid Pooling Multi-scale Deformable Attention
- HybridEncoder: Hybrid encoder with FPN + PAN + Transformer
- TransformerLayer: Single transformer encoder layer
- CSPRepLayer: CSP repeated block layer
"""

from .attention import MSDeformableAttention, deformable_attention_core_func
from .utils import (
    MLP,
    get_sine_pos_embed,
    inverse_sigmoid
)
from .position_encoding import PositionEmbedding

from .rtdetr_transformerv3 import (
    RTDETRTransformerv3,
    PPMSDeformableAttention
)
from .hybrid_encoder import (
    HybridEncoder,
    MaskHybridEncoder,
    TransformerLayer,
    CSPRepLayer
)

__all__ = [
    'MSDeformableAttention',
    'deformable_attention_core_func',
    'PositionEmbedding',
    'MLP',
    'get_sine_pos_embed',
    'inverse_sigmoid',
    'TransformerDecoderLayer',
    'TransformerDecoder',
    'MultiHeadAttention',
    'RTDETRTransformerv3',
    'PPMSDeformableAttention',
    'HybridEncoder',
    'MaskHybridEncoder',
    'TransformerLayer',
    'CSPRepLayer'
]

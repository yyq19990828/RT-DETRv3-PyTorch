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
from .dfine_decoder import (
    LQE,
    DFINETransformer,
    Integral,
)
from .dfine_decoder import (
    TransformerDecoder as DFINETransformerDecoder,
)
from .dfine_hybrid_encoder import DFINEHybridEncoder, RTDETRV2HybridEncoder
from .dfine_utils import bbox2distance, distance2bbox, weighting_function
from .hybrid_encoder import (
    CSPRepLayer,
    HybridEncoder,
    MaskHybridEncoder,
    TransformerLayer,
)
from .matchers import HungarianMatcher
from .position_encoding import PositionEmbedding
from .rtdetr_transformerv2 import RTDETRTransformerv2
from .rtdetr_transformerv3 import PPMSDeformableAttention, RTDETRTransformerv3
from .utils import MLP, bbox_cxcywh_to_xyxy, get_sine_pos_embed, inverse_sigmoid

__all__ = [
    "MSDeformableAttention",
    "deformable_attention_core_func",
    "PositionEmbedding",
    "MLP",
    "get_sine_pos_embed",
    "inverse_sigmoid",
    "bbox_cxcywh_to_xyxy",
    "TransformerDecoderLayer",
    "TransformerDecoder",
    "MultiHeadAttention",
    "RTDETRTransformerv3",
    "RTDETRTransformerv2",
    "PPMSDeformableAttention",
    "HybridEncoder",
    "MaskHybridEncoder",
    "TransformerLayer",
    "CSPRepLayer",
    "HungarianMatcher",
    "Integral",
    "LQE",
    "DFINETransformerDecoder",
    "DFINETransformer",
    "DFINEHybridEncoder",
    "RTDETRV2HybridEncoder",
    "weighting_function",
    "distance2bbox",
    "bbox2distance",
]

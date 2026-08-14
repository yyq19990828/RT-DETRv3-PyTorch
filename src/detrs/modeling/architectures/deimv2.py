"""DEIMv2 detection architecture on the shared detector graph."""

from detrs.core.workspace import register

from .deim import DEIM

__all__ = ["DEIMV2"]


@register
class DEIMV2(DEIM):
    """Compose DEIMv2 backbones, encoders and the DEIMTransformer decoder.

    The component wiring, batch contracts, deploy conversion and epoch-aware
    criterion call are inherited from the DEIM/DFINE graph; DEIMv2 only
    contributes its registered components (DINOv3STAs, LiteEncoder,
    DEIMTransformer, DEIMv2Criterion/DEIMv2HungarianMatcher).
    """

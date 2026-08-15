"""DEIM architecture adapter using the shared detector graph."""

from detrs.core.workspace import register

from .dfine import DFINE

__all__ = ["DEIM"]


@register
class DEIM(DFINE):
    """Apply DEIM training losses without adding an inference-time branch.

    Wiring and constructor arguments are shared with `DFINE`; only the
    criterion (DEIM MAL/Dense O2O losses) differs.
    """

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        components = super().from_config(cfg, *args, **kwargs)
        if (
            type(components["decoder"]).__name__ == "RTDETRTransformerv2"
            and "local" in components["criterion"].losses
        ):
            raise ValueError("DEIM-RT-DETRv2 does not support D-FINE local loss")
        return components

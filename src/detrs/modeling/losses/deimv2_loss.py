"""DEIMv2 criterion: DEIM MAL losses with epoch-aware matcher switching."""

from __future__ import annotations

from detrs.core.workspace import register, serializable
from detrs.modeling.transformers.dfine_support import (
    DEIMv2HungarianMatcher,
)

from .deim_loss import DEIMCriterion

__all__ = ["DEIMv2Criterion"]


@register
@serializable
class DEIMv2Criterion(DEIMCriterion):
    """DEIM MAL criterion with configurable gamma and matcher epoch hooks.

    Upstream DEIMv2 keeps the DEIM loss set (``mal``/``boxes``/``local``) with
    ``gamma: 1.5`` in every published config but allows other values (class
    default 2.0), and forwards the training epoch into the matcher so the
    ``change_matcher`` IoU-ordered cost can activate at its scheduled epoch.
    """

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
    ):
        if not isinstance(gamma, (int, float)) or gamma <= 0:
            raise ValueError("DEIMv2 MAL gamma must be a positive number")
        # Bypass the DEIM gamma==1.5 assertion (parent validates everything
        # else) and restore the configured gamma on the instance afterwards.
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=1.5,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
            mal_alpha=mal_alpha,
            use_uni_set=use_uni_set,
        )
        self.gamma = float(gamma)

    def _match_indices(self, output, targets, epoch=None):
        if isinstance(self.matcher, DEIMv2HungarianMatcher):
            return self.matcher(
                output, targets, epoch=0 if epoch is None else int(epoch)
            )["indices"]
        return super()._match_indices(output, targets, epoch)

    def forward(self, outputs, targets, epoch=0, **kwargs):
        return super().forward(outputs, targets, epoch=epoch, **kwargs)

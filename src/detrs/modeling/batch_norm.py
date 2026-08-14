"""BatchNorm layers with layout-stable backward behavior."""

import torch.nn as nn


def _make_grad_output_contiguous(_module, grad_output):
    return tuple(
        value.contiguous() if value is not None else None for value in grad_output
    )


class ContiguousGradBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d that normalizes grad-output layout before backward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_full_backward_pre_hook(_make_grad_output_contiguous)

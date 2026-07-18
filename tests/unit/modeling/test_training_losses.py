import torch

from ppdet_pytorch.modeling.heads.ppyoloe_head import PPYOLOEHead
from ppdet_pytorch.modeling.losses.detr_loss import DETRLoss
from ppdet_pytorch.modeling.transformers.utils import (
    varifocal_loss_with_logits,
)


def test_varifocal_loss_preserves_gradient_through_dynamic_weight():
    logits = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.1, 0.5, -0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    scores = labels * torch.tensor([[0.8], [0.6]])

    loss = varifocal_loss_with_logits(logits, scores, labels)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == logits.numel()


def test_distribution_focal_loss_uses_last_dimension_as_class_axis():
    head = PPYOLOEHead(
        in_channels=[8],
        num_classes=2,
        reg_max=4,
        static_assigner=None,
        assigner=None,
        nms=None,
        use_shared_conv=False,
    )
    pred_dist = torch.randn(2, 4, 5, requires_grad=True)
    target = torch.tensor(
        [[0.25, 1.50, 2.75, 3.25], [1.25, 2.50, 0.75, 3.50]],
        dtype=torch.float32,
    )

    loss = head._df_loss(pred_dist, target)
    loss.sum().backward()

    assert loss.shape == (2, 1)
    assert torch.isfinite(loss).all()
    assert pred_dist.grad is not None
    assert torch.isfinite(pred_dist.grad).all()


def test_detr_loss_selects_matches_per_batch():
    loss = DETRLoss(
        num_classes=2,
        matcher=None,
        aux_loss=False,
        use_focal_loss=True,
    )
    source = [
        torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        torch.tensor([[4.0, 40.0], [5.0, 50.0]]),
    ]
    target = [
        torch.tensor([[11.0, 110.0], [12.0, 120.0]]),
        torch.tensor([[13.0, 130.0], [14.0, 140.0]]),
    ]
    matches = [
        (torch.tensor([2, 0]), torch.tensor([1, 0])),
        (torch.tensor([1]), torch.tensor([0])),
    ]

    source_assign, target_assign = loss._get_src_target_assign(
        source,
        target,
        matches,
    )

    assert torch.equal(
        source_assign,
        torch.tensor([[3.0, 30.0], [1.0, 10.0], [5.0, 50.0]]),
    )
    assert torch.equal(
        target_assign,
        torch.tensor([[12.0, 120.0], [11.0, 110.0], [13.0, 130.0]]),
    )

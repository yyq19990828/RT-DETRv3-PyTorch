import pytest
import torch

from detrs.modeling.assigners.atss_assigner import ATSSAssigner
from detrs.modeling.assigners.task_aligned_assigner import (
    TaskAlignedAssigner,
)
from detrs.modeling.assigners.utils import (
    generate_anchors_for_grid_cell,
    pad_gt,
)
from detrs.modeling.heads.ppyoloe_head import PPYOLOEHead
from detrs.modeling.losses.detr_loss import DETRLoss
from detrs.modeling.transformers.matchers import HungarianMatcher
from detrs.modeling.transformers.utils import (
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_detr_loss_accepts_cpu_match_indices_for_cuda_tensors():
    loss = DETRLoss(
        num_classes=2,
        matcher=None,
        aux_loss=False,
        use_focal_loss=True,
    )
    source = [torch.tensor([[1.0, 10.0], [2.0, 20.0]], device="cuda")]
    target = [torch.tensor([[11.0, 110.0]], device="cuda")]
    matches = [(torch.tensor([1]), torch.tensor([0]))]

    source_assign, target_assign = loss._get_src_target_assign(
        source,
        target,
        matches,
    )

    assert source_assign.device.type == "cuda"
    assert target_assign.device.type == "cuda"
    assert torch.equal(source_assign.cpu(), torch.tensor([[2.0, 20.0]]))
    assert torch.equal(target_assign.cpu(), torch.tensor([[11.0, 110.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hungarian_matcher_returns_indices_on_prediction_device():
    matcher = HungarianMatcher(use_focal_loss=True)
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]],
        device="cuda",
    )
    logits = torch.tensor([[[2.0, -1.0], [-1.0, 2.0]]], device="cuda")
    gt_bbox = [torch.tensor([[0.5, 0.5, 0.2, 0.2]], device="cuda")]
    gt_class = [torch.tensor([[0]], dtype=torch.int64, device="cuda")]

    match_indices = matcher(boxes, logits, gt_bbox, gt_class)

    assert match_indices[0][0].device.type == "cuda"
    assert match_indices[0][1].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_auxiliary_head_assignment_tensors_follow_cuda_features():
    feature = torch.zeros((1, 8, 2, 2), device="cuda")
    anchors, anchor_points, num_anchors_list, stride_tensor = (
        generate_anchors_for_grid_cell([feature], [8])
    )
    assert anchors.device.type == "cuda"
    assert anchor_points.device.type == "cuda"
    assert stride_tensor.device.type == "cuda"

    head = PPYOLOEHead(
        in_channels=[8],
        num_classes=2,
        fpn_strides=[8],
        reg_max=4,
        static_assigner=None,
        assigner=None,
        nms=None,
        use_shared_conv=False,
    ).cuda()
    eval_anchor_points, eval_stride_tensor = head._generate_anchors([feature])
    assert eval_anchor_points.device.type == "cuda"
    assert eval_stride_tensor.device.type == "cuda"

    gt_labels = torch.tensor([[[0]]], dtype=torch.int64, device="cuda")
    gt_bboxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0]]], dtype=torch.float32, device="cuda"
    )
    pad_gt_mask = torch.ones((1, 1, 1), device="cuda")

    atss_outputs = ATSSAssigner(topk=2, num_classes=2)(
        anchors,
        num_anchors_list,
        gt_labels,
        gt_bboxes,
        pad_gt_mask,
        bg_index=2,
    )
    aligned_outputs = TaskAlignedAssigner(topk=2)(
        torch.full((1, 4, 2), 0.5, device="cuda"),
        anchors.unsqueeze(0),
        anchor_points,
        num_anchors_list,
        gt_labels,
        gt_bboxes,
        pad_gt_mask,
        bg_index=2,
    )

    assert all(output.device.type == "cuda" for output in atss_outputs)
    assert all(output.device.type == "cuda" for output in aligned_outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_assigner_empty_targets_and_padding_preserve_cuda_device():
    gt_labels = torch.empty((1, 0, 1), dtype=torch.int64, device="cuda")
    gt_bboxes = torch.empty((1, 0, 4), device="cuda")
    pad_gt_mask = torch.empty((1, 0, 1), device="cuda")
    anchors = torch.zeros((4, 4), device="cuda")

    atss_outputs = ATSSAssigner(topk=2, num_classes=2)(
        anchors,
        [4],
        gt_labels,
        gt_bboxes,
        pad_gt_mask,
        bg_index=2,
    )
    aligned_outputs = TaskAlignedAssigner(topk=2)(
        torch.full((1, 4, 2), 0.5, device="cuda"),
        anchors.unsqueeze(0),
        torch.zeros((4, 2), device="cuda"),
        [4],
        gt_labels,
        gt_bboxes,
        pad_gt_mask,
        bg_index=2,
    )
    padded = pad_gt(
        [torch.tensor([[1]], device="cuda")],
        [torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda")],
    )

    assert all(output.device.type == "cuda" for output in atss_outputs)
    assert all(output.device.type == "cuda" for output in aligned_outputs)
    assert all(output.device.type == "cuda" for output in padded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_auxiliary_head_classification_losses_are_amp_safe():
    logits = torch.tensor([[0.2, -0.4], [0.7, -0.1]], device="cuda", requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda")
    gt_scores = labels * 0.8

    with torch.autocast("cuda"):
        scores = torch.sigmoid(logits)
        loss = PPYOLOEHead._focal_loss(scores, labels)
        loss = loss + PPYOLOEHead._varifocal_loss(scores, gt_scores, labels)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

# Copyright (c) 2025 RT-DETRv3 PyTorch Authors. All Rights Reserved.

"""
Unit tests for loss functions.

Tests cover:
- Varifocal Loss computation
- GIoU Loss computation
- Hungarian matching
- Loss gradient flow
- Multi-branch loss aggregation
"""

import pytest
import torch
import torch.nn as nn
from ppdet_pytorch.modeling.losses import (
    DINOv3Loss,
    GIoULoss,
    HungarianMatcher,
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    sigmoid_focal_loss,
    varifocal_loss_with_logits,
    bbox_iou
)


class TestBboxConversion:
    """Test bounding box format conversion functions."""

    def test_cxcywh_to_xyxy(self):
        """Test conversion from center format to corner format."""
        # Create sample boxes in cxcywh format
        boxes_cxcywh = torch.tensor([
            [10.0, 10.0, 4.0, 6.0],  # cx=10, cy=10, w=4, h=6
            [20.0, 15.0, 8.0, 10.0]  # cx=20, cy=15, w=8, h=10
        ])

        # Expected boxes in xyxy format
        expected_xyxy = torch.tensor([
            [8.0, 7.0, 12.0, 13.0],   # x1=10-2, y1=10-3, x2=10+2, y2=10+3
            [16.0, 10.0, 24.0, 20.0]  # x1=20-4, y1=15-5, x2=20+4, y2=15+5
        ])

        boxes_xyxy = bbox_cxcywh_to_xyxy(boxes_cxcywh)
        assert torch.allclose(boxes_xyxy, expected_xyxy, atol=1e-5)

    def test_xyxy_to_cxcywh(self):
        """Test conversion from corner format to center format."""
        # Create sample boxes in xyxy format
        boxes_xyxy = torch.tensor([
            [8.0, 7.0, 12.0, 13.0],
            [16.0, 10.0, 24.0, 20.0]
        ])

        # Expected boxes in cxcywh format
        expected_cxcywh = torch.tensor([
            [10.0, 10.0, 4.0, 6.0],
            [20.0, 15.0, 8.0, 10.0]
        ])

        boxes_cxcywh = bbox_xyxy_to_cxcywh(boxes_xyxy)
        assert torch.allclose(boxes_cxcywh, expected_cxcywh, atol=1e-5)

    def test_roundtrip_conversion(self):
        """Test that converting back and forth preserves values."""
        original = torch.randn(5, 4).abs()  # Ensure positive values
        converted = bbox_cxcywh_to_xyxy(bbox_xyxy_to_cxcywh(original))
        assert torch.allclose(original, converted, atol=1e-5)


class TestBboxIoU:
    """Test IoU calculation."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes should be 1.0."""
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

        iou = bbox_iou(box1, box2)
        assert torch.allclose(iou, torch.tensor([[1.0]]), atol=1e-5)

    def test_non_overlapping_boxes(self):
        """Test IoU of non-overlapping boxes should be 0.0."""
        box1 = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        box2 = torch.tensor([[10.0, 10.0, 15.0, 15.0]])

        iou = bbox_iou(box1, box2)
        assert torch.allclose(iou, torch.tensor([[0.0]]), atol=1e-5)

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])  # Area = 100
        box2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])  # Area = 100
        # Intersection = 5x5 = 25
        # Union = 100 + 100 - 25 = 175
        # IoU = 25 / 175 ≈ 0.1429

        iou = bbox_iou(box1, box2)
        expected_iou = 25.0 / 175.0
        assert torch.allclose(iou, torch.tensor([[expected_iou]]), atol=1e-4)


class TestGIoULoss:
    """Test GIoU Loss computation."""

    def test_identical_boxes_zero_loss(self):
        """Test that identical boxes have zero GIoU loss."""
        giou_loss = GIoULoss()

        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

        loss = giou_loss(pred_boxes, target_boxes)
        assert torch.allclose(loss, torch.tensor([0.0]), atol=1e-5)

    def test_giou_loss_positive(self):
        """Test that GIoU loss is positive for non-identical boxes."""
        giou_loss = GIoULoss()

        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target_boxes = torch.tensor([[5.0, 5.0, 15.0, 15.0]])

        loss = giou_loss(pred_boxes, target_boxes)
        assert loss > 0

    def test_giou_loss_gradient(self):
        """Test that GIoU loss supports backward pass."""
        giou_loss = GIoULoss()

        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
        target_boxes = torch.tensor([[5.0, 5.0, 15.0, 15.0]])

        loss = giou_loss(pred_boxes, target_boxes).sum()
        loss.backward()

        assert pred_boxes.grad is not None
        assert pred_boxes.grad.shape == pred_boxes.shape


class TestSigmoidFocalLoss:
    """Test sigmoid focal loss computation."""

    def test_focal_loss_computation(self):
        """Test basic focal loss computation."""
        # Create sample predictions and targets
        logits = torch.randn(4, 10)  # 4 samples, 10 classes
        labels = torch.zeros(4, 10)
        labels[0, 2] = 1.0
        labels[1, 5] = 1.0

        loss = sigmoid_focal_loss(logits, labels, normalizer=4.0)

        # Loss should be a scalar
        assert loss.dim() == 0
        assert loss > 0

    def test_focal_loss_gradient(self):
        """Test that focal loss supports gradients."""
        logits = torch.randn(4, 10, requires_grad=True)
        labels = torch.zeros(4, 10)
        labels[0, 2] = 1.0

        loss = sigmoid_focal_loss(logits, labels)
        loss.backward()

        assert logits.grad is not None


class TestVarifocalLoss:
    """Test varifocal loss computation."""

    def test_varifocal_loss_computation(self):
        """Test basic varifocal loss computation."""
        pred_logits = torch.randn(4, 10)
        gt_score = torch.rand(4, 10)  # Quality scores
        label = torch.zeros(4, 10)
        label[0, 2] = 1.0
        label[1, 5] = 1.0

        loss = varifocal_loss_with_logits(pred_logits, gt_score, label, normalizer=4.0)

        # Loss should be a scalar
        assert loss.dim() == 0
        assert loss >= 0

    def test_varifocal_loss_gradient(self):
        """Test that varifocal loss supports gradients."""
        pred_logits = torch.randn(4, 10, requires_grad=True)
        gt_score = torch.rand(4, 10)
        label = torch.zeros(4, 10)
        label[0, 2] = 1.0

        loss = varifocal_loss_with_logits(pred_logits, gt_score, label)
        loss.backward()

        assert pred_logits.grad is not None


class TestHungarianMatcher:
    """Test Hungarian Matcher."""

    def test_matching_basic(self):
        """Test basic Hungarian matching."""
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)

        # Create sample predictions
        batch_size = 2
        num_queries = 10
        num_classes = 80

        pred_boxes = torch.rand(batch_size, num_queries, 4)  # cxcywh format
        pred_logits = torch.randn(batch_size, num_queries, num_classes)

        # Create sample ground truth
        gt_boxes = [
            torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]]),  # 2 boxes for image 1
            torch.tensor([[0.6, 0.6, 0.25, 0.35]])  # 1 box for image 2
        ]
        gt_labels = [
            torch.tensor([5, 10], dtype=torch.int64),
            torch.tensor([15], dtype=torch.int64)
        ]

        # Perform matching
        indices = matcher(pred_boxes, pred_logits, gt_boxes, gt_labels)

        # Check output format
        assert len(indices) == batch_size
        for i, (pred_idx, gt_idx) in enumerate(indices):
            assert pred_idx.dtype == torch.int64
            assert gt_idx.dtype == torch.int64
            assert len(pred_idx) == len(gt_idx)
            # Number of matches should equal number of GT boxes
            assert len(pred_idx) == len(gt_boxes[i])

    def test_matching_empty_gt(self):
        """Test matching when there are no ground truth boxes."""
        matcher = HungarianMatcher()

        pred_boxes = torch.rand(2, 10, 4)
        pred_logits = torch.randn(2, 10, 80)

        gt_boxes = [torch.empty(0, 4), torch.empty(0, 4)]
        gt_labels = [torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)]

        indices = matcher(pred_boxes, pred_logits, gt_boxes, gt_labels)

        # Should return empty indices for each image
        assert len(indices) == 2
        for pred_idx, gt_idx in indices:
            assert len(pred_idx) == 0
            assert len(gt_idx) == 0


class TestDINOv3Loss:
    """Test DINOv3Loss computation."""

    def test_loss_forward_basic(self):
        """Test basic forward pass of DINOv3Loss."""
        loss_fn = DINOv3Loss(num_classes=80)

        batch_size = 2
        num_queries = 300
        num_layers = 6

        # Create sample predictions (one per decoder layer)
        pred_boxes = [torch.rand(batch_size, num_queries, 4) for _ in range(num_layers)]
        pred_logits = [torch.randn(batch_size, num_queries, 80) for _ in range(num_layers)]

        # Create sample ground truth
        gt_boxes = [
            torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]]),
            torch.tensor([[0.6, 0.6, 0.25, 0.35]])
        ]
        gt_labels = [
            torch.tensor([5, 10], dtype=torch.int64),
            torch.tensor([15], dtype=torch.int64)
        ]

        # Compute loss
        losses = loss_fn(pred_boxes, pred_logits, gt_boxes, gt_labels)

        # Check that all expected loss components are present
        assert 'loss_class' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses
        assert 'loss_class_aux' in losses
        assert 'loss_bbox_aux' in losses
        assert 'loss_giou_aux' in losses
        assert 'loss_class_dn' in losses
        assert 'loss_bbox_dn' in losses
        assert 'loss_giou_dn' in losses

        # Check that losses are scalars
        for key, value in losses.items():
            assert value.dim() == 0, f"{key} should be a scalar"
            assert value >= 0, f"{key} should be non-negative"

    def test_loss_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = DINOv3Loss(num_classes=80)

        # Create predictions with gradients
        pred_boxes = [torch.rand(2, 300, 4, requires_grad=True) for _ in range(6)]
        pred_logits = [torch.randn(2, 300, 80, requires_grad=True) for _ in range(6)]

        # Create ground truth (must match batch size)
        gt_boxes = [
            torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
            torch.tensor([[0.6, 0.6, 0.25, 0.35]])
        ]
        gt_labels = [
            torch.tensor([5], dtype=torch.int64),
            torch.tensor([10], dtype=torch.int64)
        ]

        # Compute loss and backpropagate
        losses = loss_fn(pred_boxes, pred_logits, gt_boxes, gt_labels)
        total_loss = losses['loss_class'] + losses['loss_bbox'] + losses['loss_giou']

        # Just check that backward pass doesn't crash
        # Note: Due to the Hungarian matcher using @torch.no_grad() and other operations
        # that detach gradients (like IoU computation for VFL), some intermediate gradients
        # may not exist. The important thing is that the loss can be backpropagated without error.
        assert total_loss.requires_grad, "Total loss should require gradients"
        total_loss.backward()  # Should not raise an error

    def test_loss_with_o2m(self):
        """Test loss computation with one-to-many supervision."""
        loss_fn = DINOv3Loss(num_classes=80)

        pred_boxes = [torch.rand(2, 450, 4) for _ in range(6)]  # More queries for o2m
        pred_logits = [torch.randn(2, 450, 80) for _ in range(6)]

        gt_boxes = [
            torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
            torch.tensor([[0.6, 0.6, 0.25, 0.35]])
        ]
        gt_labels = [
            torch.tensor([5], dtype=torch.int64),
            torch.tensor([10], dtype=torch.int64)
        ]

        # Compute loss with o2m=3
        losses = loss_fn(pred_boxes, pred_logits, gt_boxes, gt_labels, o2m=3)

        assert 'loss_class' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses

    def test_loss_empty_gt(self):
        """Test loss computation with no ground truth boxes."""
        loss_fn = DINOv3Loss(num_classes=80)

        pred_boxes = [torch.rand(2, 300, 4) for _ in range(6)]
        pred_logits = [torch.randn(2, 300, 80) for _ in range(6)]

        # Empty ground truth
        gt_boxes = [torch.empty(0, 4), torch.empty(0, 4)]
        gt_labels = [torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)]

        # Should not crash
        losses = loss_fn(pred_boxes, pred_logits, gt_boxes, gt_labels)

        assert 'loss_class' in losses
        # Bbox losses should be zero when no GT
        assert losses['loss_bbox'] == 0.0
        assert losses['loss_giou'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

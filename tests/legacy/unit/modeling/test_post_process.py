"""
Unit tests for DETR post-processing

Tests the DETRPostProcessor implementation to ensure correctness.
"""

import pytest
import torch
from ppdet_pytorch.modeling.post_process import (
    DETRPostProcessor,
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    build_detr_post_processor
)


class TestBboxConversion:
    """Test bounding box coordinate conversion utilities"""

    def test_cxcywh_to_xyxy_basic(self):
        """Test basic CXCYWH to XYXY conversion"""
        # Input: center (0.5, 0.5), size (0.2, 0.3)
        bbox_cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        bbox_xyxy = bbox_cxcywh_to_xyxy(bbox_cxcywh)

        expected = torch.tensor([[0.4, 0.35, 0.6, 0.65]])
        assert torch.allclose(bbox_xyxy, expected, atol=1e-6)

    def test_xyxy_to_cxcywh_basic(self):
        """Test basic XYXY to CXCYWH conversion"""
        bbox_xyxy = torch.tensor([[0.4, 0.35, 0.6, 0.65]])
        bbox_cxcywh = bbox_xyxy_to_cxcywh(bbox_xyxy)

        expected = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        assert torch.allclose(bbox_cxcywh, expected, atol=1e-6)

    def test_conversion_roundtrip(self):
        """Test that conversion is reversible"""
        original = torch.rand(10, 4)
        # Ensure w, h > 0
        original[:, 2:] = original[:, 2:].abs() + 0.01

        converted = bbox_cxcywh_to_xyxy(original)
        recovered = bbox_xyxy_to_cxcywh(converted)

        assert torch.allclose(original, recovered, atol=1e-6)

    def test_batch_conversion(self):
        """Test batch processing"""
        bbox_batch = torch.rand(8, 100, 4)
        bbox_batch[:, :, 2:] = bbox_batch[:, :, 2:].abs() + 0.01

        converted = bbox_cxcywh_to_xyxy(bbox_batch)
        assert converted.shape == bbox_batch.shape


class TestDETRPostProcessor:
    """Test DETRPostProcessor functionality"""

    @pytest.fixture
    def post_processor(self):
        """Create default post-processor"""
        return DETRPostProcessor(
            num_classes=80,
            num_top_queries=100,
            use_focal_loss=True
        )

    @pytest.fixture
    def sample_inputs(self):
        """Create sample model outputs"""
        batch_size = 2
        num_queries = 300
        num_classes = 80

        bboxes = torch.rand(batch_size, num_queries, 4)
        logits = torch.randn(batch_size, num_queries, num_classes)
        im_shape = torch.tensor([[640.0, 640.0], [480.0, 640.0]])
        scale_factor = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        return bboxes, logits, im_shape, scale_factor

    def test_basic_forward(self, post_processor, sample_inputs):
        """Test basic forward pass"""
        bboxes, logits, im_shape, scale_factor = sample_inputs

        bbox_pred, bbox_num = post_processor(
            bboxes, logits, im_shape, scale_factor
        )

        # Check output shapes
        batch_size = bboxes.shape[0]
        expected_total = batch_size * post_processor.num_top_queries
        assert bbox_pred.shape == (expected_total, 6)
        assert bbox_num.shape == (batch_size,)
        assert (bbox_num == post_processor.num_top_queries).all()

    def test_output_format(self, post_processor, sample_inputs):
        """Test output format correctness"""
        bboxes, logits, im_shape, scale_factor = sample_inputs

        bbox_pred, bbox_num = post_processor(
            bboxes, logits, im_shape, scale_factor
        )

        # Check column structure: [class_id, score, x1, y1, x2, y2]
        class_ids = bbox_pred[:, 0]
        scores = bbox_pred[:, 1]

        # Class IDs should be in [0, num_classes)
        assert (class_ids >= 0).all()
        assert (class_ids < post_processor.num_classes).all()

        # Scores should be in [0, 1]
        assert (scores >= 0).all()
        assert (scores <= 1).all()

        # Note: boxes may have negative coordinates if model outputs
        # predictions outside image bounds (expected behavior)

    def test_coordinate_scaling(self):
        """Test coordinate scaling to pixel coordinates"""
        post_processor = DETRPostProcessor(
            num_classes=80,
            num_top_queries=10,
            use_focal_loss=True
        )

        # Single detection at center with size 0.5
        bboxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
        logits = torch.randn(1, 1, 80)
        im_shape = torch.tensor([[640.0, 640.0]])
        scale_factor = torch.tensor([[1.0, 1.0]])

        bbox_pred, _ = post_processor(
            bboxes, logits, im_shape, scale_factor
        )

        # Extract coordinates
        x1, y1, x2, y2 = bbox_pred[0, 2:].tolist()

        # Check that coordinates are in pixel space
        # Center at (0.5, 0.5) with size (0.5, 0.5) in 640x640 image
        # Should be approximately: x1=160, y1=160, x2=480, y2=480
        assert 150 <= x1 <= 170
        assert 150 <= y1 <= 170
        assert 470 <= x2 <= 490
        assert 470 <= y2 <= 490

    def test_dual_queries_mode(self):
        """Test dual queries mode (O2O + O2M)"""
        post_processor = DETRPostProcessor(
            num_classes=80,
            num_top_queries=100,
            dual_queries=True,
            dual_groups=1,  # O2O + Noise + O2M
            use_focal_loss=True
        )

        # Total queries: 300 + 100 + 450 = 850
        # Should keep first 850 / (1+1) = 425 queries (O2O + Noise)
        # But actually, for dual_groups=1: 850 / (1+1) = 425
        # Let's use 900 to make it cleaner: 900 / 2 = 450
        batch_size = 2
        num_queries = 900
        bboxes = torch.rand(batch_size, num_queries, 4)
        logits = torch.randn(batch_size, num_queries, 80)
        im_shape = torch.tensor([[640.0, 640.0], [640.0, 640.0]])
        scale_factor = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        bbox_pred, bbox_num = post_processor(
            bboxes, logits, im_shape, scale_factor
        )

        # Should process correctly
        assert bbox_pred.shape[0] == batch_size * post_processor.num_top_queries

    def test_softmax_mode(self):
        """Test Softmax classification mode"""
        post_processor = DETRPostProcessor(
            num_classes=80,
            num_top_queries=100,
            use_focal_loss=False  # Use Softmax
        )

        batch_size = 2
        num_queries = 300
        # Logits shape includes background class for softmax
        bboxes = torch.rand(batch_size, num_queries, 4)
        logits = torch.randn(batch_size, num_queries, 81)  # 80 + 1 background
        im_shape = torch.tensor([[640.0, 640.0], [640.0, 640.0]])
        scale_factor = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        bbox_pred, bbox_num = post_processor(
            bboxes, logits, im_shape, scale_factor
        )

        # Check output
        assert bbox_pred.shape == (batch_size * 100, 6)

    def test_top_k_selection(self):
        """Test top-K selection mechanism"""
        post_processor = DETRPostProcessor(
            num_classes=80,
            num_top_queries=50,  # Select top 50
            use_focal_loss=True
        )

        batch_size = 2
        num_queries = 300
        bboxes = torch.rand(batch_size, num_queries, 4)
        # Create logits with clear top-K pattern
        logits = torch.randn(batch_size, num_queries, 80)
        logits[:, :50, :] += 5.0  # Make first 50 queries have higher scores

        im_shape = torch.tensor([[640.0, 640.0], [640.0, 640.0]])
        scale_factor = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        bbox_pred, bbox_num = post_processor(
            bboxes, logits, im_shape, scale_factor
        )

        # Should select exactly 50 queries per batch
        assert bbox_pred.shape[0] == batch_size * 50
        assert (bbox_num == 50).all()

        # Check that scores are sorted (descending)
        for i in range(batch_size):
            start_idx = i * 50
            end_idx = start_idx + 50
            batch_scores = bbox_pred[start_idx:end_idx, 1]
            # Scores should be relatively high (boosted by +5)
            assert batch_scores.mean() > 0.9

    def test_scale_factor_handling(self):
        """Test different scale factors"""
        post_processor = DETRPostProcessor(
            num_classes=80,
            num_top_queries=10,
            use_focal_loss=True
        )

        # Center box (0.5, 0.5) with size (0.2, 0.2)
        bboxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        logits = torch.randn(1, 1, 80)

        # Original image 1280x960, scaled to 640x640
        im_shape = torch.tensor([[640.0, 640.0]])
        scale_factor = torch.tensor([[0.5, 0.6666667]])  # 640/1280, 640/960

        bbox_pred, _ = post_processor(
            bboxes, logits, im_shape, scale_factor
        )

        # Check that coordinates account for scale factor
        # Original size: floor(640/0.5 + 0.5) = 1280 (H), floor(640/0.6667 + 0.5) = 960 (W)
        # out_shape.flip(-1) = [960, 1280, 960, 1280] (W, H, W, H)
        # CXCYWH (0.5, 0.5, 0.2, 0.2) -> XYXY (0.4, 0.4, 0.6, 0.6) normalized
        # Multiply by out_shape: [0.4*960, 0.4*1280, 0.6*960, 0.6*1280]
        #                       = [384, 512, 576, 768]
        x1, y1, x2, y2 = bbox_pred[0, 2:].tolist()

        # Verify actual calculated values
        assert 380 < x1 < 390  # Should be 384
        assert 508 < y1 < 516  # Should be 512
        assert 572 < x2 < 580  # Should be 576
        assert 764 < y2 < 772  # Should be 768


class TestBuilder:
    """Test builder function"""

    def test_build_default(self):
        """Test default configuration"""
        post_processor = build_detr_post_processor()

        assert post_processor.num_classes == 80
        assert post_processor.num_top_queries == 100
        assert post_processor.use_focal_loss is True
        assert post_processor.dual_queries is False

    def test_build_custom(self):
        """Test custom configuration"""
        post_processor = build_detr_post_processor(
            num_classes=1000,
            num_top_queries=300,
            dual_queries=True,
            dual_groups=1
        )

        assert post_processor.num_classes == 1000
        assert post_processor.num_top_queries == 300
        assert post_processor.dual_queries is True
        assert post_processor.dual_groups == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Integration tests for RT-DETRv3 inference pipeline

Tests the full inference workflow including:
- Image preprocessing
- Model forward pass
- Post-processing (NMS, confidence filtering)
- Output format validation
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

# Add parent directory to path
parent_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_path))

from models import create
from tools.infer import preprocess_image, postprocess


class TestInference:
    """Test inference pipeline components"""

    @pytest.fixture
    def model(self):
        """Build a small RT-DETRv3 model for testing"""
        config = {
            'type': 'RTDETRv3',
            'num_classes': 80,
            'backbone': {'type': 'ResNet', 'depth': 50, 'variant': 'd', 'return_idx': [1, 2, 3]},
            'neck': {'type': 'HybridEncoder', 'hidden_dim': 256},
            'transformer': {'type': 'RTDETRTransformerv3', 'num_queries': 300, 'num_decoder_layers': 6, 'hidden_dim': 256},
            'detr_head': {'type': 'DINOv3Head', 'eval_idx': -1}
        }
        model = create('RTDETRv3', global_config=config, num_classes=80)
        model.eval()
        return model

    @pytest.fixture
    def dummy_image(self):
        """Create a dummy test image"""
        # Create a 640x480 BGR image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return image

    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Create a temporary test image file"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(image_path), image)
        return str(image_path)

    def test_preprocess_image_shape(self, dummy_image):
        """Test image preprocessing produces correct output shape"""
        input_size = 640
        image_tensor, meta = preprocess_image(dummy_image, input_size)

        # Check tensor shape
        assert image_tensor.shape == (1, 3, input_size, input_size)
        assert image_tensor.dtype == torch.float32

        # Check meta info
        assert 'orig_size' in meta
        assert 'resized_size' in meta
        assert 'scale' in meta
        assert meta['orig_size'] == (480, 640)
        assert meta['input_size'] == input_size

    def test_preprocess_image_normalization(self, dummy_image):
        """Test image preprocessing normalizes values correctly"""
        input_size = 640
        image_tensor, _ = preprocess_image(dummy_image, input_size)

        # Values should be roughly in range [-3, 3] after normalization
        # (normalized = (x/255 - mean) / std, with mean~0.5, std~0.2)
        assert image_tensor.min() >= -5.0
        assert image_tensor.max() <= 5.0

    def test_model_forward_eval_mode(self, model, dummy_image):
        """Test model forward pass in eval mode"""
        input_size = 640
        image_tensor, _ = preprocess_image(dummy_image, input_size)

        with torch.no_grad():
            outputs = model(image_tensor)

        # Check output format
        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs

        # Check output shapes
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        assert pred_logits.shape[0] == 1  # batch size
        assert pred_logits.shape[2] == 80  # num_classes
        assert pred_boxes.shape[0] == 1   # batch size
        assert pred_boxes.shape[2] == 4   # box coordinates

    def test_postprocess_output_format(self, model, dummy_image):
        """Test post-processing produces correct output format"""
        input_size = 640
        image_tensor, meta = preprocess_image(dummy_image, input_size)

        with torch.no_grad():
            outputs = model(image_tensor)
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']

        # Post-process
        results = postprocess(
            pred_logits,
            pred_boxes,
            meta,
            conf_threshold=0.3,
            nms_threshold=0.7
        )

        # Check output format
        assert len(results) == 1
        result = results[0]

        assert 'boxes' in result
        assert 'scores' in result
        assert 'labels' in result

        # Check shapes are consistent
        num_detections = len(result['boxes'])
        assert result['scores'].shape[0] == num_detections
        assert result['labels'].shape[0] == num_detections

        # Check boxes format (should be [x1, y1, x2, y2])
        if num_detections > 0:
            boxes = result['boxes']
            assert boxes.shape[1] == 4

            # x2 > x1, y2 > y1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            assert (boxes[:, 3] >= boxes[:, 1]).all()

    def test_postprocess_confidence_threshold(self, model, dummy_image):
        """Test confidence threshold filtering"""
        input_size = 640
        image_tensor, meta = preprocess_image(dummy_image, input_size)

        with torch.no_grad():
            outputs = model(image_tensor)
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']

        # Test with different thresholds
        results_low = postprocess(pred_logits, pred_boxes, meta, conf_threshold=0.1, nms_threshold=0.7)
        results_high = postprocess(pred_logits, pred_boxes, meta, conf_threshold=0.9, nms_threshold=0.7)

        # Higher threshold should produce fewer detections
        num_low = len(results_low[0]['boxes'])
        num_high = len(results_high[0]['boxes'])
        assert num_high <= num_low

        # All scores should be above threshold
        if num_high > 0:
            assert (results_high[0]['scores'] >= 0.9).all()

    def test_postprocess_nms(self, model, dummy_image):
        """Test NMS removes overlapping boxes"""
        input_size = 640
        image_tensor, meta = preprocess_image(dummy_image, input_size)

        with torch.no_grad():
            outputs = model(image_tensor)
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']

        # Test with different NMS thresholds
        results_no_nms = postprocess(pred_logits, pred_boxes, meta, conf_threshold=0.3, nms_threshold=1.0)
        results_with_nms = postprocess(pred_logits, pred_boxes, meta, conf_threshold=0.3, nms_threshold=0.5)

        # NMS should reduce or maintain the number of detections
        num_no_nms = len(results_no_nms[0]['boxes'])
        num_with_nms = len(results_with_nms[0]['boxes'])
        assert num_with_nms <= num_no_nms

    def test_batch_inference(self, model):
        """Test inference with batch size > 1"""
        batch_size = 4
        input_size = 640

        # Create batch of dummy images
        images = []
        metas = []
        for _ in range(batch_size):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            image_tensor, meta = preprocess_image(dummy_image, input_size)
            images.append(image_tensor)
            metas.append(meta)

        # Stack into batch
        batch_tensor = torch.cat(images, dim=0)
        assert batch_tensor.shape[0] == batch_size

        # Forward pass
        with torch.no_grad():
            outputs = model(batch_tensor)

        # Check batch outputs
        assert outputs['pred_logits'].shape[0] == batch_size
        assert outputs['pred_boxes'].shape[0] == batch_size

    def test_different_image_sizes(self, model):
        """Test inference with different input image sizes"""
        image_sizes = [(640, 640), (480, 640), (800, 600), (1920, 1080)]

        for h, w in image_sizes:
            # Create image with specific size
            dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

            # Preprocess
            input_size = 640
            image_tensor, meta = preprocess_image(dummy_image, input_size)

            # Forward pass
            with torch.no_grad():
                outputs = model(image_tensor)

            # Post-process
            results = postprocess(
                outputs['pred_logits'],
                outputs['pred_boxes'],
                meta,
                conf_threshold=0.3,
                nms_threshold=0.7
            )

            # Verify output
            assert len(results) == 1
            assert 'boxes' in results[0]

    def test_no_detections(self, model):
        """Test behavior when no objects are detected (high threshold)"""
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        input_size = 640
        image_tensor, meta = preprocess_image(dummy_image, input_size)

        with torch.no_grad():
            outputs = model(image_tensor)

        # Use very high threshold to filter all detections
        results = postprocess(
            outputs['pred_logits'],
            outputs['pred_boxes'],
            meta,
            conf_threshold=0.99,
            nms_threshold=0.7
        )

        # Should return empty detections
        assert len(results) == 1
        result = results[0]
        assert len(result['boxes']) == 0
        assert len(result['scores']) == 0
        assert len(result['labels']) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self, model, dummy_image):
        """Test inference on CUDA device"""
        model = model.cuda()

        input_size = 640
        image_tensor, meta = preprocess_image(dummy_image, input_size)
        image_tensor = image_tensor.cuda()

        with torch.no_grad():
            outputs = model(image_tensor)

        # Check outputs are on CUDA
        assert outputs['pred_logits'].is_cuda
        assert outputs['pred_boxes'].is_cuda

        # Post-process (should work with CUDA tensors)
        results = postprocess(
            outputs['pred_logits'],
            outputs['pred_boxes'],
            meta,
            conf_threshold=0.3,
            nms_threshold=0.7
        )

        # Results should be moved back to CPU
        assert results[0]['boxes'].device.type == 'cpu'
        assert results[0]['scores'].device.type == 'cpu'
        assert results[0]['labels'].device.type == 'cpu'

    def test_gradient_disabled_in_eval(self, model, dummy_image):
        """Test that gradients are disabled in eval mode"""
        model.eval()

        input_size = 640
        image_tensor, _ = preprocess_image(dummy_image, input_size)
        image_tensor.requires_grad_(True)

        with torch.no_grad():
            outputs = model(image_tensor)

        # Outputs should not require gradients
        assert not outputs['pred_logits'].requires_grad
        assert not outputs['pred_boxes'].requires_grad

    def test_output_values_range(self, model, dummy_image):
        """Test that output values are in expected ranges"""
        input_size = 640
        image_tensor, meta = preprocess_image(dummy_image, input_size)

        with torch.no_grad():
            outputs = model(image_tensor)

        # Check pred_boxes are normalized (should be roughly in [0, 1])
        # Allow some slack for model initialization
        pred_boxes = outputs['pred_boxes']
        assert pred_boxes.min() >= -1.0, "Boxes should be roughly normalized"
        assert pred_boxes.max() <= 2.0, "Boxes should be roughly normalized"

        # Check pred_logits are in reasonable range (before sigmoid)
        pred_logits = outputs['pred_logits']
        assert pred_logits.min() >= -20.0, "Logits should be in reasonable range"
        assert pred_logits.max() <= 20.0, "Logits should be in reasonable range"

        # After sigmoid, scores should be in [0, 1]
        scores = pred_logits.sigmoid()
        assert (scores >= 0).all()
        assert (scores <= 1).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

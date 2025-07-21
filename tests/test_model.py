"""Unit tests for RT-DETRv3 model components."""

import os
import sys
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nn.backbone.resnet import ResNet, ResNetVD
from src.nn.neck.hybrid_encoder import HybridEncoder
from src.nn.transformer.rtdetr_transformerv3 import RTDETRTransformerV3
from src.nn.head.rtdetr_head import RTDETRHead
from src.nn.criterion.rtdetr_criterion import RTDETRCriterion, HungarianMatcher
from src.zoo.rtdetrv3.rtdetrv3 import RTDETRv3


class TestResNetBackbone:
    """Test ResNet backbone."""
    
    def test_resnet50_forward(self):
        """Test ResNet50 forward pass."""
        model = ResNet(depth=50, return_idx=[1, 2, 3])
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        
        outputs = model(input_tensor)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert outputs[0].shape == (batch_size, 512, 80, 80)
        assert outputs[1].shape == (batch_size, 1024, 40, 40)
        assert outputs[2].shape == (batch_size, 2048, 20, 20)
    
    def test_resnet18_forward(self):
        """Test ResNet18 forward pass."""
        model = ResNet(depth=18, return_idx=[1, 2, 3])
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        
        outputs = model(input_tensor)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert outputs[0].shape == (batch_size, 128, 80, 80)
        assert outputs[1].shape == (batch_size, 256, 40, 40)
        assert outputs[2].shape == (batch_size, 512, 20, 20)
    
    def test_resnet_vd_forward(self):
        """Test ResNet-VD forward pass."""
        model = ResNetVD(depth=50, return_idx=[1, 2, 3])
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        
        outputs = model(input_tensor)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert outputs[0].shape == (batch_size, 512, 80, 80)
        assert outputs[1].shape == (batch_size, 1024, 40, 40)
        assert outputs[2].shape == (batch_size, 2048, 20, 20)
    
    def test_resnet_different_return_indices(self):
        """Test ResNet with different return indices."""
        model = ResNet(depth=50, return_idx=[0, 1, 2, 3])
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 640, 640)
        
        outputs = model(input_tensor)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 4
        assert outputs[0].shape == (batch_size, 256, 160, 160)
        assert outputs[1].shape == (batch_size, 512, 80, 80)
        assert outputs[2].shape == (batch_size, 1024, 40, 40)
        assert outputs[3].shape == (batch_size, 2048, 20, 20)


class TestHybridEncoder:
    """Test HybridEncoder neck."""
    
    def test_hybrid_encoder_forward(self):
        """Test HybridEncoder forward pass."""
        model = HybridEncoder(
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            nhead=8,
            dim_feedforward=1024
        )
        
        batch_size = 2
        feat1 = torch.randn(batch_size, 512, 80, 80)
        feat2 = torch.randn(batch_size, 1024, 40, 40)
        feat3 = torch.randn(batch_size, 2048, 20, 20)
        
        outputs = model([feat1, feat2, feat3])
        
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert outputs[0].shape == (batch_size, 256, 80, 80)
        assert outputs[1].shape == (batch_size, 256, 40, 40)
        assert outputs[2].shape == (batch_size, 256, 20, 20)
    
    def test_hybrid_encoder_different_config(self):
        """Test HybridEncoder with different configuration."""
        model = HybridEncoder(
            in_channels=[256, 512, 1024],
            feat_strides=[8, 16, 32],
            hidden_dim=128,
            use_encoder_idx=[1, 2],
            num_encoder_layers=2,
            nhead=4,
            dim_feedforward=512
        )
        
        batch_size = 2
        feat1 = torch.randn(batch_size, 256, 80, 80)
        feat2 = torch.randn(batch_size, 512, 40, 40)
        feat3 = torch.randn(batch_size, 1024, 20, 20)
        
        outputs = model([feat1, feat2, feat3])
        
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert outputs[0].shape == (batch_size, 128, 80, 80)
        assert outputs[1].shape == (batch_size, 128, 40, 40)
        assert outputs[2].shape == (batch_size, 128, 20, 20)


class TestRTDETRTransformerV3:
    """Test RT-DETR Transformer v3."""
    
    def test_transformer_forward(self):
        """Test transformer forward pass."""
        model = RTDETRTransformerV3(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            num_decoder_layers=6,
            nhead=8,
            dim_feedforward=1024,
            num_denoising=100
        )
        
        batch_size = 2
        feat1 = torch.randn(batch_size, 256, 80, 80)
        feat2 = torch.randn(batch_size, 256, 40, 40)
        feat3 = torch.randn(batch_size, 256, 20, 20)
        
        outputs = model([feat1, feat2, feat3])
        
        assert isinstance(outputs, list)
        assert len(outputs) == 6  # 6 decoder layers
        
        for output in outputs:
            assert isinstance(output, dict)
            assert 'pred_logits' in output
            assert 'pred_boxes' in output
            assert output['pred_logits'].shape == (batch_size, 300, 80)
            assert output['pred_boxes'].shape == (batch_size, 300, 4)
    
    def test_transformer_training_mode(self):
        """Test transformer in training mode with targets."""
        model = RTDETRTransformerV3(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            num_decoder_layers=6,
            nhead=8,
            dim_feedforward=1024,
            num_denoising=100
        )
        model.train()
        
        batch_size = 2
        feat1 = torch.randn(batch_size, 256, 80, 80)
        feat2 = torch.randn(batch_size, 256, 40, 40)
        feat3 = torch.randn(batch_size, 256, 20, 20)
        
        # Mock targets
        targets = [
            {'labels': torch.tensor([1, 2]), 'boxes': torch.randn(2, 4)},
            {'labels': torch.tensor([3]), 'boxes': torch.randn(1, 4)}
        ]
        
        outputs = model([feat1, feat2, feat3], targets=targets)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 6
        
        for output in outputs:
            assert isinstance(output, dict)
            assert 'pred_logits' in output
            assert 'pred_boxes' in output


class TestRTDETRHead:
    """Test RT-DETR detection head."""
    
    def test_head_forward(self):
        """Test detection head forward pass."""
        model = RTDETRHead(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            num_decoder_layers=6
        )
        
        batch_size = 2
        hidden_states = torch.randn(batch_size, 300, 256)
        
        pred_logits, pred_boxes = model(hidden_states)
        
        assert pred_logits.shape == (batch_size, 300, 80)
        assert pred_boxes.shape == (batch_size, 300, 4)
    
    def test_head_aux_loss(self):
        """Test detection head with auxiliary loss."""
        model = RTDETRHead(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            num_decoder_layers=6,
            aux_loss=True
        )
        
        batch_size = 2
        num_layers = 6
        hidden_states = [torch.randn(batch_size, 300, 256) for _ in range(num_layers)]
        
        outputs = model(hidden_states)
        
        assert isinstance(outputs, list)
        assert len(outputs) == num_layers
        
        for output in outputs:
            assert isinstance(output, dict)
            assert 'pred_logits' in output
            assert 'pred_boxes' in output
            assert output['pred_logits'].shape == (batch_size, 300, 80)
            assert output['pred_boxes'].shape == (batch_size, 300, 4)


class TestRTDETRCriterion:
    """Test RT-DETR loss criterion."""
    
    def test_hungarian_matcher(self):
        """Test Hungarian matcher."""
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=5.0,
            cost_giou=2.0
        )
        
        batch_size = 2
        num_queries = 300
        num_classes = 80
        
        outputs = [
            {
                'pred_logits': torch.randn(batch_size, num_queries, num_classes),
                'pred_boxes': torch.randn(batch_size, num_queries, 4)
            }
        ]
        
        targets = [
            {
                'labels': torch.tensor([1, 2]),
                'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])
            },
            {
                'labels': torch.tensor([3]),
                'boxes': torch.tensor([[0.7, 0.7, 0.3, 0.3]])
            }
        ]
        
        indices = matcher(outputs, targets)
        
        assert isinstance(indices, list)
        assert len(indices) == batch_size
        
        for idx in indices:
            assert isinstance(idx, tuple)
            assert len(idx) == 2  # (pred_indices, target_indices)
    
    def test_criterion_loss(self):
        """Test criterion loss computation."""
        criterion = RTDETRCriterion(
            num_classes=80,
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
            weight_dict={'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0},
            losses=['labels', 'boxes'],
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        batch_size = 2
        num_queries = 300
        num_classes = 80
        
        outputs = [
            {
                'pred_logits': torch.randn(batch_size, num_queries, num_classes),
                'pred_boxes': torch.randn(batch_size, num_queries, 4)
            }
        ]
        
        targets = [
            {
                'labels': torch.tensor([1, 2]),
                'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])
            },
            {
                'labels': torch.tensor([3]),
                'boxes': torch.tensor([[0.7, 0.7, 0.3, 0.3]])
            }
        ]
        
        losses = criterion(outputs, targets)
        
        assert isinstance(losses, dict)
        assert 'loss_ce' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses
        
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.requires_grad


class TestRTDETRv3:
    """Test complete RT-DETRv3 model."""
    
    def test_rtdetrv3_forward(self):
        """Test RT-DETRv3 forward pass."""
        model = RTDETRv3(
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            num_classes=80,
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6}
        )
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640)
        
        outputs = model(images)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 6  # 6 decoder layers
        
        for output in outputs:
            assert isinstance(output, dict)
            assert 'pred_logits' in output
            assert 'pred_boxes' in output
            assert output['pred_logits'].shape == (batch_size, 300, 80)
            assert output['pred_boxes'].shape == (batch_size, 300, 4)
    
    def test_rtdetrv3_with_targets(self):
        """Test RT-DETRv3 with training targets."""
        model = RTDETRv3(
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            num_classes=80,
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6}
        )
        model.train()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640)
        
        targets = [
            {'labels': torch.tensor([1, 2]), 'boxes': torch.randn(2, 4)},
            {'labels': torch.tensor([3]), 'boxes': torch.randn(1, 4)}
        ]
        
        outputs = model(images, targets=targets)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 6
        
        for output in outputs:
            assert isinstance(output, dict)
            assert 'pred_logits' in output
            assert 'pred_boxes' in output
    
    def test_rtdetrv3_different_variants(self):
        """Test different RT-DETRv3 variants."""
        # Test ResNet18 variant
        model_r18 = RTDETRv3(
            backbone_name='ResNet18',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            num_classes=80,
            backbone_args={'depth': 18, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [128, 256, 512], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6}
        )
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640)
        
        outputs = model_r18(images)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 6
        
        for output in outputs:
            assert isinstance(output, dict)
            assert 'pred_logits' in output
            assert 'pred_boxes' in output
            assert output['pred_logits'].shape == (batch_size, 300, 80)
            assert output['pred_boxes'].shape == (batch_size, 300, 4)


class TestNumericalPrecision:
    """Test numerical precision and consistency."""
    
    def test_model_deterministic(self):
        """Test model produces consistent results."""
        model = RTDETRv3(
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            num_classes=80,
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6}
        )
        model.eval()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640)
        
        with torch.no_grad():
            torch.manual_seed(42)
            outputs1 = model(images)
            
            torch.manual_seed(42)
            outputs2 = model(images)
        
        # Check that outputs are identical
        for out1, out2 in zip(outputs1, outputs2):
            assert torch.allclose(out1['pred_logits'], out2['pred_logits'], atol=1e-6)
            assert torch.allclose(out1['pred_boxes'], out2['pred_boxes'], atol=1e-6)
    
    def test_gradient_computation(self):
        """Test gradient computation."""
        model = RTDETRv3(
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            num_classes=80,
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 300, 'num_decoder_layers': 6}
        )
        model.train()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640, requires_grad=True)
        
        targets = [
            {'labels': torch.tensor([1, 2]), 'boxes': torch.randn(2, 4)},
            {'labels': torch.tensor([3]), 'boxes': torch.randn(1, 4)}
        ]
        
        # Forward pass
        outputs = model(images, targets=targets)
        
        # Compute loss
        criterion = RTDETRCriterion(
            num_classes=80,
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
            weight_dict={'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0},
            losses=['labels', 'boxes'],
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        losses = criterion(outputs, targets)
        total_loss = sum(losses.values())
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        assert images.grad is not None
        
        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
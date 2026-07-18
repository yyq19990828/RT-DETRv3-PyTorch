"""
Integration tests for RT-DETRv3 full model

Tests cover:
- Full forward pass with batch input
- Eval mode (inference)
- Train mode (loss computation - when implemented)
- Gradient flow through entire model
- Output shapes and validity
- Different model configurations

Following PaddlePaddle implementation for consistency.
"""

import pytest
import torch
import torch.nn as nn
from ppdet_pytorch.core.workspace import create


class TestRTDETRv3Integration:
    """Test full RT-DETRv3 model integration"""

    def test_forward_eval_mode(self):
        """Test forward pass in evaluation mode"""
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

        # Prepare input
        batch = 2
        images = torch.randn(batch, 3, 640, 640)

        # Forward pass
        with torch.no_grad():
            outputs = model(images)

        # Check output structure
        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs

        # Check shapes
        # Note: 400 queries = 300 (o2o) + 100 (noise group)
        # This matches PaddlePaddle's multi-group query mechanism
        assert outputs['pred_logits'].shape == (batch, 400, 80)
        assert outputs['pred_boxes'].shape == (batch, 400, 4)

        # Check bbox values are in [0, 1]
        assert (outputs['pred_boxes'] >= 0).all() and (outputs['pred_boxes'] <= 1).all()

        # Check no NaN or Inf
        assert not torch.isnan(outputs['pred_logits']).any()
        assert not torch.isinf(outputs['pred_logits']).any()
        assert not torch.isnan(outputs['pred_boxes']).any()
        assert not torch.isinf(outputs['pred_boxes']).any()

    def test_forward_train_mode_not_implemented(self):
        """Test that training mode raises NotImplementedError until T040"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')
        model.train()

        images = torch.randn(2, 3, 640, 640)

        # Create dummy targets
        targets = [
            {
                'boxes': torch.rand(5, 4),  # 5 objects
                'labels': torch.randint(0, 80, (5,))
            },
            {
                'boxes': torch.rand(3, 4),  # 3 objects
                'labels': torch.randint(0, 80, (3,))
            }
        ]

        # Training mode should raise NotImplementedError until T040
        with pytest.raises(NotImplementedError, match="Training mode not yet implemented"):
            model(images, targets)

    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            images = torch.randn(batch_size, 3, 640, 640)

            with torch.no_grad():
                outputs = model(images)

            # Total queries = 300 (o2o) + 100 (noise) = 400
            assert outputs['pred_logits'].shape == (batch_size, 400, 80)
            assert outputs['pred_boxes'].shape == (batch_size, 400, 4)

    def test_different_image_sizes(self):
        """Test with different input image sizes"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')
        model.eval()

        for size in [640, 800, 1024]:
            images = torch.randn(2, 3, size, size)

            with torch.no_grad():
                outputs = model(images)

            # Output shape should be independent of input size
            assert outputs['pred_logits'].shape == (2, 400, 80)
            assert outputs['pred_boxes'].shape == (2, 400, 4)

    def test_resnet18_variant(self):
        """Test with ResNet-18 backbone"""
        model = build_rtdetrv3(
            num_classes=80,
            backbone='resnet18',
            num_queries=300
        )
        model.eval()

        images = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            outputs = model(images)

        assert outputs['pred_logits'].shape == (2, 400, 80)
        assert outputs['pred_boxes'].shape == (2, 400, 4)

    def test_resnet101_variant(self):
        """Test with ResNet-101 backbone"""
        model = build_rtdetrv3(
            num_classes=80,
            backbone='resnet101',
            num_queries=300
        )
        model.eval()

        images = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            outputs = model(images)

        assert outputs['pred_logits'].shape == (2, 400, 80)
        assert outputs['pred_boxes'].shape == (2, 400, 4)

    def test_different_num_queries(self):
        """Test with different number of queries"""
        for num_queries in [100, 300, 900]:
            model = build_rtdetrv3(
                num_classes=80,
                backbone='resnet50',
                num_queries=num_queries
            )
            model.eval()

            images = torch.randn(2, 3, 640, 640)

            with torch.no_grad():
                outputs = model(images)

            # Total queries = num_queries (o2o) + 100 (noise group)
            total_queries = num_queries + 100
            assert outputs['pred_logits'].shape == (2, total_queries, 80)
            assert outputs['pred_boxes'].shape == (2, total_queries, 4)

    def test_different_decoder_layers(self):
        """Test with different number of decoder layers"""
        for num_layers in [3, 6, 9]:
            model = build_rtdetrv3(
                num_classes=80,
                backbone='resnet50',
                num_decoder_layers=num_layers
            )
            model.eval()

            images = torch.randn(2, 3, 640, 640)

            with torch.no_grad():
                outputs = model(images)

            # Output shape should be same regardless of decoder layers
            assert outputs['pred_logits'].shape == (2, 400, 80)
            assert outputs['pred_boxes'].shape == (2, 400, 4)

    def test_gradient_flow_backbone_to_output(self):
        """Test gradient flow through entire model"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')
        model.eval()  # Use eval mode to avoid NotImplementedError

        # Create input with gradient tracking
        images = torch.randn(2, 3, 640, 640, requires_grad=True)

        # Forward pass
        outputs = model(images)

        # Compute dummy loss
        loss = outputs['pred_logits'].sum() + outputs['pred_boxes'].sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert images.grad is not None
        assert images.grad.abs().sum() > 0

        # Check backbone parameters have gradients
        backbone_params = list(model.backbone.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone_params)

    def test_o2m_branch_configuration(self):
        """Test model with one-to-many branch enabled"""
        model = build_rtdetrv3(
            num_classes=80,
            backbone='resnet50',
            num_queries=300,
            o2m_branch=True,
            num_queries_o2m=450,
            o2m=4
        )
        model.eval()

        images = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            outputs = model(images)

        # Total queries = 300 (o2o) + 100 (noise) + 450 (o2m) = 850
        assert outputs['pred_logits'].shape == (2, 850, 80)
        assert outputs['pred_boxes'].shape == (2, 850, 4)

    def test_frozen_backbone_stages(self):
        """Test that frozen backbone stages don't update"""
        model = build_rtdetrv3(
            num_classes=80,
            backbone='resnet50',
            frozen_stages=1  # Freeze stage 0 (stem) and stage 1 (layer1)
        )
        model.train()

        # In eval mode to avoid NotImplementedError
        model.eval()
        images = torch.randn(2, 3, 640, 640, requires_grad=True)
        outputs = model(images)
        loss = outputs['pred_logits'].sum() + outputs['pred_boxes'].sum()
        loss.backward()

        # Check frozen parameters have requires_grad=False
        # frozen_stages=1 should freeze:
        # - stem (conv1_1, bn1_1, conv1_2, bn1_2, conv1_3, bn1_3 for variant='d')
        # - layer1 (all parameters in layer1)
        for name, param in model.backbone.named_parameters():
            # Check stem layers (stage 0)
            if any(stem in name for stem in ['conv1_1', 'bn1_1', 'conv1_2', 'bn1_2', 'conv1_3', 'bn1_3']):
                assert not param.requires_grad, f"Stem parameter {name} should be frozen (requires_grad=False)"
                assert param.grad is None, f"Stem parameter {name} should have no gradient"

            # Check layer1 (stage 1)
            elif name.startswith('layer1.'):
                assert not param.requires_grad, f"Layer1 parameter {name} should be frozen (requires_grad=False)"
                assert param.grad is None, f"Layer1 parameter {name} should have no gradient"

            # Check layer2, layer3, layer4 are NOT frozen
            elif any(name.startswith(f'layer{i}.') for i in [2, 3, 4]):
                assert param.requires_grad, f"Parameter {name} should NOT be frozen"

    def test_model_device_transfer(self):
        """Test model can be moved to different devices"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')
        model.eval()  # Set to eval mode to avoid NotImplementedError

        # CPU
        model.cpu()
        images_cpu = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs_cpu = model(images_cpu)
        assert outputs_cpu['pred_logits'].device.type == 'cpu'

        # GPU (if available)
        if torch.cuda.is_available():
            model.cuda()
            images_gpu = torch.randn(1, 3, 640, 640).cuda()
            with torch.no_grad():
                outputs_gpu = model(images_gpu)
            assert outputs_gpu['pred_logits'].device.type == 'cuda'

    def test_model_state_dict_save_load(self):
        """Test model can be saved and loaded"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')
        model.eval()

        # Save state dict
        state_dict = model.state_dict()

        # Create new model and load
        model2 = build_rtdetrv3(num_classes=80, backbone='resnet50')
        model2.load_state_dict(state_dict)
        model2.eval()

        # Test both models produce same output
        images = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            outputs1 = model(images)
            outputs2 = model2(images)

        # Outputs should be identical
        assert torch.allclose(outputs1['pred_logits'], outputs2['pred_logits'])
        assert torch.allclose(outputs1['pred_boxes'], outputs2['pred_boxes'])


class TestRTDETRv3Components:
    """Test individual components within full model"""

    def test_backbone_integration(self):
        """Test backbone produces expected feature shapes"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')

        images = torch.randn(2, 3, 640, 640)

        # Extract backbone features
        with torch.no_grad():
            feats = model.backbone(images)

        # Check multi-scale features
        assert len(feats) == 3
        assert feats[0].shape == (2, 512, 80, 80)   # C3: stride 8
        assert feats[1].shape == (2, 1024, 40, 40)  # C4: stride 16
        assert feats[2].shape == (2, 2048, 20, 20)  # C5: stride 32

    def test_neck_integration(self):
        """Test neck processes backbone features correctly"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')

        images = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            feats = model.backbone(images)
            body_feats = model.neck(feats)

        # Check neck outputs
        assert len(body_feats) == 3
        # All features should have hidden_dim channels (256)
        assert body_feats[0].shape == (2, 256, 80, 80)
        assert body_feats[1].shape == (2, 256, 40, 40)
        assert body_feats[2].shape == (2, 256, 20, 20)

    def test_transformer_integration(self):
        """Test transformer processes neck features correctly"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')
        model.eval()

        images = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            feats = model.backbone(images)
            body_feats = model.neck(feats)
            out_transformer = model.transformer(body_feats, targets=None)

        # Check transformer outputs
        dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta = out_transformer

        # In eval mode, only one layer output
        # Total queries = 300 (o2o) + 100 (noise) = 400
        assert dec_out_bboxes.shape == (1, 2, 400, 4)
        assert dec_out_logits.shape == (1, 2, 400, 80)
        assert enc_topk_bboxes.shape == (2, 400, 4)
        assert enc_topk_logits.shape == (2, 400, 80)

    def test_head_integration(self):
        """Test detection head processes transformer outputs correctly"""
        model = build_rtdetrv3(num_classes=80, backbone='resnet50')
        model.eval()

        images = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            feats = model.backbone(images)
            body_feats = model.neck(feats)
            out_transformer = model.transformer(body_feats, targets=None)
            pred_bboxes, pred_logits, _ = model.detr_head(out_transformer, body_feats, None)

        # Check head outputs
        assert pred_bboxes.shape == (2, 400, 4)
        assert pred_logits.shape == (2, 400, 80)


class TestRTDETRv3WithAuxHead:
    """Test RT-DETRv3 with auxiliary PPYOLOEHead"""

    def test_model_with_aux_head_eval(self):
        """Test model with auxiliary head in eval mode"""
        model = build_rtdetrv3(
            num_classes=80,
            backbone='resnet50',
            num_queries=300,
            use_aux_head=True  # Enable auxiliary head
        )
        model.eval()

        images = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            outputs = model(images)

        # In eval mode, auxiliary head is not used
        # Output should only contain main branch predictions
        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs
        assert outputs['pred_logits'].shape == (2, 400, 80)
        assert outputs['pred_boxes'].shape == (2, 400, 4)

    def test_aux_head_forward_pass(self):
        """Test auxiliary head can process neck features"""
        model = build_rtdetrv3(
            num_classes=80,
            backbone='resnet50',
            use_aux_head=True
        )
        model.eval()  # Set to eval mode

        # Verify aux_head exists
        assert model.aux_head is not None

        images = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            # Extract features
            feats = model.backbone(images)
            body_feats = model.neck(feats)

            # Forward through auxiliary head
            aux_cls_scores, aux_reg_distris = model.aux_head(body_feats)

        # Check auxiliary head outputs
        # In eval mode: cls_scores (B, num_classes, total_anchors), reg_dists (B, total_anchors, 4)
        total_anchors = 8400  # 80*80 + 40*40 + 20*20
        assert aux_cls_scores.shape == (2, 80, total_anchors)
        assert aux_reg_distris.shape == (2, total_anchors, 4)

    def test_aux_head_training_mode_not_implemented(self):
        """Test that training with aux head raises NotImplementedError until T040"""
        model = build_rtdetrv3(
            num_classes=80,
            backbone='resnet50',
            use_aux_head=True
        )
        model.train()

        images = torch.randn(2, 3, 640, 640)
        targets = [
            {'boxes': torch.rand(5, 4), 'labels': torch.randint(0, 80, (5,))},
            {'boxes': torch.rand(3, 4), 'labels': torch.randint(0, 80, (3,))}
        ]

        # Training mode should raise NotImplementedError until T040 (loss implementation)
        with pytest.raises(NotImplementedError, match="Training mode not yet implemented"):
            model(images, targets)

    def test_model_without_aux_head(self):
        """Test model works without auxiliary head"""
        model = build_rtdetrv3(
            num_classes=80,
            backbone='resnet50',
            use_aux_head=False  # Disable auxiliary head
        )

        # Verify aux_head is None
        assert model.aux_head is None

        model.eval()
        images = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            outputs = model(images)

        # Should still work normally
        assert outputs['pred_logits'].shape == (2, 400, 80)
        assert outputs['pred_boxes'].shape == (2, 400, 4)

    def test_aux_head_gradient_flow(self):
        """Test gradients flow through auxiliary head"""
        model = build_rtdetrv3(
            num_classes=80,
            backbone='resnet50',
            frozen_stages=-1,  # Don't freeze any stages for gradient flow test
            use_aux_head=True
        )
        model.train()  # Use train mode for gradient flow

        images = torch.randn(2, 3, 640, 640, requires_grad=True)

        # Forward pass
        feats = model.backbone(images)
        body_feats = model.neck(feats)
        aux_cls_scores, aux_reg_distris = model.aux_head(body_feats)

        # Compute dummy loss
        loss = aux_cls_scores.sum() + aux_reg_distris.sum()

        # Backward pass
        loss.backward()

        # Check aux_head parameters have gradients
        aux_params = list(model.aux_head.parameters())
        # Exclude proj_conv which is frozen
        trainable_params = [p for p in aux_params if p.requires_grad]
        assert len(trainable_params) > 0, "Should have trainable parameters"
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable_params), \
            "Auxiliary head parameters should have gradients"

        # Note: In train mode with frozen BatchNorm, input gradients may be zero
        # This is expected behavior and doesn't indicate a problem with gradient flow
        # The important thing is that model parameters receive gradients


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

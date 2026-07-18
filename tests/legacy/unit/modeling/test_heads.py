"""
Unit tests for Detection Heads

Tests cover:
- DINOv3Head forward pass in eval mode
- PPYOLOEHead forward pass in training and eval modes
- Output shapes and validity
- Builder functions
- Gradient flow

Following PaddlePaddle implementation for numerical equivalence.

Note: Training mode tests for DINOv3Head will be added when loss computation is implemented (T040).
"""

import pytest
import torch
import torch.nn as nn
from ppdet_pytorch.modeling.heads.detr_head import DINOv3Head
from ppdet_pytorch.modeling.heads.ppyoloe_head import PPYOLOEHead


class TestDINOv3Head:
    """Test DINOv3Head"""

    def test_forward_eval_mode(self):
        """Test forward pass in evaluation mode"""
        head = DINOv3Head(eval_idx=-1)
        head.eval()

        # Prepare transformer outputs
        # (num_layers, batch, num_queries, ...)
        num_layers, batch, num_queries, num_classes = 6, 2, 300, 80

        dec_out_bboxes = torch.rand(num_layers, batch, num_queries, 4)  # Already sigmoid-ed in [0, 1]
        dec_out_logits = torch.randn(num_layers, batch, num_queries, num_classes)
        enc_topk_bboxes = torch.rand(batch, num_queries, 4)
        enc_topk_logits = torch.randn(batch, num_queries, num_classes)
        dn_meta = None

        out_transformer = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

        # Forward pass
        pred_bboxes, pred_logits, aux_outputs = head(out_transformer)

        # Check shapes
        assert pred_bboxes.shape == (batch, num_queries, 4)
        assert pred_logits.shape == (batch, num_queries, num_classes)
        assert aux_outputs is None  # No auxiliary outputs in eval mode

        # Check bbox values are in [0, 1]
        assert (pred_bboxes >= 0).all() and (pred_bboxes <= 1).all()

    def test_eval_idx_selection(self):
        """Test that eval_idx correctly selects decoder layer"""
        num_layers, batch, num_queries, num_classes = 6, 2, 300, 80

        dec_out_bboxes = torch.rand(num_layers, batch, num_queries, 4)
        dec_out_logits = torch.randn(num_layers, batch, num_queries, num_classes)
        enc_topk_bboxes = torch.rand(batch, num_queries, 4)
        enc_topk_logits = torch.randn(batch, num_queries, num_classes)
        dn_meta = None

        out_transformer = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

        # Test with different eval_idx values
        for eval_idx in [-1, -2, 0, 3]:
            head = DINOv3Head(eval_idx=eval_idx)
            head.eval()

            pred_bboxes, pred_logits, _ = head(out_transformer)

            # Check that output matches the selected layer
            expected_bboxes = dec_out_bboxes[eval_idx]
            expected_logits = dec_out_logits[eval_idx]

            assert torch.equal(pred_bboxes, expected_bboxes)
            assert torch.equal(pred_logits, expected_logits)

    def test_different_num_queries(self):
        """Test with different numbers of queries"""
        head = DINOv3Head(eval_idx=-1)
        head.eval()

        num_layers, batch, num_classes = 6, 2, 80

        for num_queries in [100, 300, 450, 900]:
            dec_out_bboxes = torch.rand(num_layers, batch, num_queries, 4)
            dec_out_logits = torch.randn(num_layers, batch, num_queries, num_classes)
            enc_topk_bboxes = torch.rand(batch, num_queries, 4)
            enc_topk_logits = torch.randn(batch, num_queries, num_classes)
            dn_meta = None

            out_transformer = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

            pred_bboxes, pred_logits, _ = head(out_transformer)

            assert pred_bboxes.shape == (batch, num_queries, 4)
            assert pred_logits.shape == (batch, num_queries, num_classes)

    def test_batch_size_one(self):
        """Test with batch size 1"""
        head = DINOv3Head(eval_idx=-1)
        head.eval()

        num_layers, batch, num_queries, num_classes = 6, 1, 300, 80

        dec_out_bboxes = torch.rand(num_layers, batch, num_queries, 4)
        dec_out_logits = torch.randn(num_layers, batch, num_queries, num_classes)
        enc_topk_bboxes = torch.rand(batch, num_queries, 4)
        enc_topk_logits = torch.randn(batch, num_queries, num_classes)
        dn_meta = None

        out_transformer = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

        pred_bboxes, pred_logits, _ = head(out_transformer)

        assert pred_bboxes.shape == (1, num_queries, 4)
        assert pred_logits.shape == (1, num_queries, num_classes)

    def test_training_mode_not_implemented(self):
        """Test that training mode raises NotImplementedError"""
        head = DINOv3Head(eval_idx=-1)
        head.train()

        num_layers, batch, num_queries, num_classes = 6, 2, 300, 80

        dec_out_bboxes = torch.rand(num_layers, batch, num_queries, 4)
        dec_out_logits = torch.randn(num_layers, batch, num_queries, num_classes)
        enc_topk_bboxes = torch.rand(batch, num_queries, 4)
        enc_topk_logits = torch.randn(batch, num_queries, num_classes)
        dn_meta = None

        out_transformer = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

        # Training mode should raise NotImplementedError until T040 is complete
        with pytest.raises(NotImplementedError, match="Training mode not yet implemented"):
            head(out_transformer)

    def test_no_nan_inf(self):
        """Test that outputs don't contain NaN or Inf"""
        head = DINOv3Head(eval_idx=-1)
        head.eval()

        num_layers, batch, num_queries, num_classes = 6, 2, 300, 80

        dec_out_bboxes = torch.rand(num_layers, batch, num_queries, 4)
        dec_out_logits = torch.randn(num_layers, batch, num_queries, num_classes)
        enc_topk_bboxes = torch.rand(batch, num_queries, 4)
        enc_topk_logits = torch.randn(batch, num_queries, num_classes)
        dn_meta = None

        out_transformer = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

        pred_bboxes, pred_logits, _ = head(out_transformer)

        assert not torch.isnan(pred_bboxes).any()
        assert not torch.isinf(pred_bboxes).any()
        assert not torch.isnan(pred_logits).any()
        assert not torch.isinf(pred_logits).any()


class TestBuildDINOv3Head:
    """Test builder function"""

    def test_build_from_config(self):
        """Test building head from config"""
        head = build_dinov3_head(
            eval_idx=-1,
            o2m=4,
            o2m_branch=False,
            num_queries_o2m=450
        )

        assert isinstance(head, DINOv3Head)
        assert head.eval_idx == -1
        assert head.o2m == 4
        assert head.o2m_branch == False
        assert head.num_queries_o2m == 450

    def test_build_with_defaults(self):
        """Test building head with default values"""
        head = build_dinov3_head()

        assert isinstance(head, DINOv3Head)
        assert head.eval_idx == -1  # Default
        assert head.o2m == 4        # Default


class TestIntegration:
    """Integration tests with decoder outputs"""

    def test_with_decoder_output_format(self):
        """Test with realistic decoder output format"""
        head = DINOv3Head(eval_idx=-1)
        head.eval()

        # Simulate decoder outputs
        # These would come from TransformerDecoder.forward()
        num_decoder_layers = 6
        batch_size = 2
        num_queries = 300
        num_classes = 80

        # Decoder outputs (stacked across layers)
        dec_out_bboxes = torch.rand(num_decoder_layers, batch_size, num_queries, 4)
        dec_out_logits = torch.randn(num_decoder_layers, batch_size, num_queries, num_classes)

        # Encoder top-k selections
        enc_topk_bboxes = torch.rand(batch_size, num_queries, 4)
        enc_topk_logits = torch.randn(batch_size, num_queries, num_classes)

        # Denoising metadata (None for simple case)
        dn_meta = None

        # Pack as transformer output
        out_transformer = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

        # Forward through head
        pred_bboxes, pred_logits, aux_outputs = head(out_transformer)

        # Verify output format
        assert pred_bboxes.shape == (batch_size, num_queries, 4)
        assert pred_logits.shape == (batch_size, num_queries, num_classes)
        assert aux_outputs is None

        # Verify bbox format (should be in [0, 1] as cx, cy, w, h)
        assert (pred_bboxes >= 0).all()
        assert (pred_bboxes <= 1).all()


class TestPPYOLOEHead:
    """Test PPYOLOEHead auxiliary detection head"""

    @pytest.fixture
    def multi_scale_features(self):
        """Create dummy multi-scale features from neck."""
        batch_size = 2
        # Three scales with different spatial resolutions
        feats = [
            torch.randn(batch_size, 256, 80, 80),   # stride=8, 640/8=80
            torch.randn(batch_size, 256, 40, 40),   # stride=16, 640/16=40
            torch.randn(batch_size, 256, 20, 20),   # stride=32, 640/32=20
        ]
        return feats

    def test_output_shapes_training(self, multi_scale_features):
        """Test output shapes in training mode."""
        head = PPYOLOEHead(
            in_channels=[256, 256, 256],
            num_classes=80,
            fpn_strides=(8, 16, 32),
            reg_max=16
        )
        head.train()

        cls_scores, reg_distris = head(multi_scale_features)

        batch_size = 2
        # Total anchors: 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
        total_anchors = 8400

        # Verify shapes
        assert cls_scores.shape == (batch_size, total_anchors, 80), \
            f"Expected cls_scores shape {(batch_size, total_anchors, 80)}, got {cls_scores.shape}"
        assert reg_distris.shape == (batch_size, total_anchors, 4 * 17), \
            f"Expected reg_distris shape {(batch_size, total_anchors, 68)}, got {reg_distris.shape}"

        # Verify value ranges
        assert torch.all((cls_scores >= 0) & (cls_scores <= 1)), \
            "Classification scores should be in [0, 1]"

    def test_output_shapes_eval(self, multi_scale_features):
        """Test output shapes in eval mode with DFL projection."""
        head = PPYOLOEHead(
            in_channels=[256, 256, 256],
            num_classes=80,
            fpn_strides=(8, 16, 32),
            reg_max=16
        )
        head.eval()

        with torch.no_grad():
            cls_scores, reg_dists = head(multi_scale_features)

        batch_size = 2
        total_anchors = 8400

        # In eval mode, cls_scores has different shape
        assert cls_scores.shape == (batch_size, 80, total_anchors), \
            f"Expected cls_scores shape {(batch_size, 80, total_anchors)}, got {cls_scores.shape}"
        # reg_dists are projected to distances (4 values per anchor)
        assert reg_dists.shape == (batch_size, total_anchors, 4), \
            f"Expected reg_dists shape {(batch_size, total_anchors, 4)}, got {reg_dists.shape}"

    def test_gradient_flow(self, multi_scale_features):
        """Test that gradients flow through the head."""
        head = PPYOLOEHead(
            in_channels=[256, 256, 256],
            num_classes=80,
            fpn_strides=(8, 16, 32),
            reg_max=16
        )

        # Set requires_grad for inputs
        for feat in multi_scale_features:
            feat.requires_grad = True

        head.train()
        cls_scores, reg_distris = head(multi_scale_features)

        # Dummy loss
        loss = cls_scores.sum() + reg_distris.sum()
        loss.backward()

        # Check gradients exist
        for i, feat in enumerate(multi_scale_features):
            assert feat.grad is not None, f"Gradients should flow to feature {i}"

        # Check head parameters have gradients
        assert head.pred_cls[0].weight.grad is not None, "Classification head should have gradients"
        assert head.pred_reg[0].weight.grad is not None, "Regression head should have gradients"

    def test_ese_attention(self, multi_scale_features):
        """Test ESE attention mechanism."""
        head = PPYOLOEHead(
            in_channels=[256, 256, 256],
            num_classes=80,
            fpn_strides=(8, 16, 32),
            reg_max=16
        )

        # Get one feature map
        feat = multi_scale_features[0]
        batch_size, channels, h, w = feat.shape

        # Apply attention
        avg_feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        attn_out = head.stem_cls[0](feat, avg_feat)

        # Verify output shape matches input
        assert attn_out.shape == feat.shape, \
            f"Attention output shape {attn_out.shape} should match input {feat.shape}"

    def test_projection_layer_frozen(self):
        """Test that projection layer weights are frozen."""
        head = PPYOLOEHead(
            in_channels=[256, 256, 256],
            num_classes=80,
            fpn_strides=(8, 16, 32),
            reg_max=16
        )

        # Projection layer should not have gradients
        assert not head.proj_conv.weight.requires_grad, \
            "Projection layer should be frozen"

        # Verify projection values are linear spacing
        expected = torch.linspace(0, 16, 17).view(1, -1, 1, 1)
        assert torch.allclose(head.proj_conv.weight, expected), \
            "Projection weights should be linear spacing from 0 to reg_max"

    def test_variable_input_sizes(self):
        """Test with different input sizes."""
        head = PPYOLOEHead(
            in_channels=[256, 256, 256],
            num_classes=80,
            fpn_strides=(8, 16, 32),
            reg_max=16
        )

        batch_size = 1
        # Different resolution inputs
        feats_small = [
            torch.randn(batch_size, 256, 40, 40),
            torch.randn(batch_size, 256, 20, 20),
            torch.randn(batch_size, 256, 10, 10),
        ]

        head.eval()
        with torch.no_grad():
            cls_scores, reg_dists = head(feats_small)

        # Total anchors: 40*40 + 20*20 + 10*10 = 1600 + 400 + 100 = 2100
        total_anchors = 2100
        assert cls_scores.shape == (batch_size, 80, total_anchors)
        assert reg_dists.shape == (batch_size, total_anchors, 4)

    def test_different_reg_max(self):
        """Test with different reg_max values."""
        for reg_max in [7, 12, 16]:
            head = PPYOLOEHead(
                in_channels=[256, 256, 256],
                num_classes=80,
                fpn_strides=(8, 16, 32),
                reg_max=reg_max
            )

            batch_size = 2
            feats = [
                torch.randn(batch_size, 256, 80, 80),
                torch.randn(batch_size, 256, 40, 40),
                torch.randn(batch_size, 256, 20, 20),
            ]

            head.train()
            cls_scores, reg_distris = head(feats)

            # reg_channels = reg_max + 1
            expected_reg_channels = reg_max + 1
            assert reg_distris.shape[2] == 4 * expected_reg_channels, \
                f"Expected {4 * expected_reg_channels} reg channels for reg_max={reg_max}"


class TestBuildPPYOLOEHead:
    """Test builder function for PPYOLOEHead"""

    def test_build_from_config(self):
        """Test building head using direct instantiation"""
        head = PPYOLOEHead(
            in_channels=[128, 256, 512],
            num_classes=91,
            fpn_strides=[8, 16, 32],
            reg_max=7,
            act='relu'
        )

        assert isinstance(head, PPYOLOEHead)
        assert head.num_classes == 91
        assert head.reg_max == 7
        assert head.in_channels == [128, 256, 512]

    def test_build_with_defaults(self):
        """Test building head with default values"""
        head = PPYOLOEHead()

        assert isinstance(head, PPYOLOEHead)
        assert head.num_classes == 80  # Default
        assert head.in_channels == [256, 256, 256]  # Default


class TestHeadInteraction:
    """Test interaction between DINOv3Head and PPYOLOEHead"""

    def test_both_heads_independent(self):
        """Test that both heads can operate independently."""
        batch_size = 2

        # Create DINOv3Head
        dinov3_head = DINOv3Head(eval_idx=-1)

        # Create PPYOLOEHead
        ppyoloe_head = PPYOLOEHead(
            in_channels=[256, 256, 256],
            num_classes=80,
            fpn_strides=(8, 16, 32)
        )

        # Prepare inputs
        # DINOv3Head: transformer outputs
        num_layers, num_queries, num_classes = 6, 300, 80
        dec_out_bboxes = torch.rand(num_layers, batch_size, num_queries, 4)
        dec_out_logits = torch.randn(num_layers, batch_size, num_queries, num_classes)
        enc_topk_bboxes = torch.rand(batch_size, num_queries, 4)
        enc_topk_logits = torch.randn(batch_size, num_queries, num_classes)
        transformer_out = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, None)

        # PPYOLOEHead: neck features
        feats = [
            torch.randn(batch_size, 256, 80, 80),
            torch.randn(batch_size, 256, 40, 40),
            torch.randn(batch_size, 256, 20, 20),
        ]

        # Forward both heads in eval mode
        dinov3_head.eval()
        ppyoloe_head.eval()

        with torch.no_grad():
            # Main branch
            main_bboxes, main_logits, _ = dinov3_head(transformer_out)

            # Auxiliary branch
            aux_scores, aux_dists = ppyoloe_head(feats)

        # Verify outputs are independent and have correct shapes
        assert main_bboxes.shape == (batch_size, num_queries, 4)
        assert main_logits.shape == (batch_size, num_queries, num_classes)
        assert aux_scores.shape == (batch_size, 80, 8400)
        assert aux_dists.shape == (batch_size, 8400, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

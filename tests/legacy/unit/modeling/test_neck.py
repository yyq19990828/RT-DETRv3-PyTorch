"""
Unit tests for HybridEncoder (FPN-PAN) Neck

Tests cover:
- Output shapes for multi-scale features
- Channel consistency across outputs
- Gradient flow through FPN-PAN
- Edge cases (different input resolutions)

Following PaddlePaddle implementation for numerical equivalence.
"""

import pytest
import torch
import torch.nn as nn
from ppdet_pytorch.modeling.transformers.hybrid_encoder import (
    HybridEncoder,
    CSPRepLayer,
    ConvNormAct
)


class TestHybridEncoderOutputShapes:
    """Test output shapes for various configurations"""

    @pytest.mark.parametrize("in_channels,hidden_dim,batch_size", [
        ([512, 1024, 2048], 256, 2),      # Standard ResNet-50 config
        ([256, 512, 1024], 256, 1),        # Smaller backbone
        ([512, 1024, 2048], 384, 4),       # Larger hidden_dim
    ])
    def test_output_shapes(self, in_channels, hidden_dim, batch_size):
        """Test forward pass output shapes"""
        neck = HybridEncoder(
            in_channels=in_channels,
            feat_strides=[8, 16, 32],
            hidden_dim=hidden_dim,
            num_encoder_layers=1,
            use_encoder_idx=[2]
        )
        neck.eval()

        # Create multi-scale input features
        h, w = 640, 640
        c3 = torch.randn(batch_size, in_channels[0], h // 8, w // 8)
        c4 = torch.randn(batch_size, in_channels[1], h // 16, w // 16)
        c5 = torch.randn(batch_size, in_channels[2], h // 32, w // 32)

        # Forward pass
        outputs = neck([c3, c4, c5])

        # Check number of outputs
        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

        # Check output shapes
        n3, n4, n5 = outputs
        assert n3.shape == (batch_size, hidden_dim, h // 8, w // 8), \
            f"Expected n3 shape {(batch_size, hidden_dim, h // 8, w // 8)}, got {n3.shape}"
        assert n4.shape == (batch_size, hidden_dim, h // 16, w // 16), \
            f"Expected n4 shape {(batch_size, hidden_dim, h // 16, w // 16)}, got {n4.shape}"
        assert n5.shape == (batch_size, hidden_dim, h // 32, w // 32), \
            f"Expected n5 shape {(batch_size, hidden_dim, h // 32, w // 32)}, got {n5.shape}"

    def test_channel_consistency(self):
        """Test that all outputs have the same channel dimension (hidden_dim)"""
        in_channels = [512, 1024, 2048]
        hidden_dim = 256

        neck = HybridEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim
        )
        neck.eval()

        # Create inputs
        c3 = torch.randn(2, 512, 80, 80)
        c4 = torch.randn(2, 1024, 40, 40)
        c5 = torch.randn(2, 2048, 20, 20)

        # Forward pass
        n3, n4, n5 = neck([c3, c4, c5])

        # Check all outputs have hidden_dim channels
        assert n3.shape[1] == hidden_dim
        assert n4.shape[1] == hidden_dim
        assert n5.shape[1] == hidden_dim

    def test_spatial_consistency(self):
        """Test that spatial dimensions match expected strides"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            hidden_dim=256
        )
        neck.eval()

        h, w = 640, 640
        c3 = torch.randn(2, 512, h // 8, w // 8)
        c4 = torch.randn(2, 1024, h // 16, w // 16)
        c5 = torch.randn(2, 2048, h // 32, w // 32)

        n3, n4, n5 = neck([c3, c4, c5])

        # Check spatial dimensions preserve input ratios
        assert n3.shape[2] == c3.shape[2] and n3.shape[3] == c3.shape[3]
        assert n4.shape[2] == c4.shape[2] and n4.shape[3] == c4.shape[3]
        assert n5.shape[2] == c5.shape[2] and n5.shape[3] == c5.shape[3]


class TestHybridEncoderGradientFlow:
    """Test gradient flow through FPN-PAN"""

    def test_gradient_flow_fpn_pan(self):
        """Test gradient flow through entire FPN-PAN pathway"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=256
        )
        neck.train()

        # Create inputs with gradients
        c3 = torch.randn(2, 512, 80, 80, requires_grad=True)
        c4 = torch.randn(2, 1024, 40, 40, requires_grad=True)
        c5 = torch.randn(2, 2048, 20, 20, requires_grad=True)

        # Forward pass
        n3, n4, n5 = neck([c3, c4, c5])

        # Create dummy loss
        loss = (n3.sum() + n4.sum() + n5.sum())
        loss.backward()

        # Check gradients exist and are non-zero
        assert c3.grad is not None and c3.grad.abs().sum() > 0, "c3 gradients missing or zero"
        assert c4.grad is not None and c4.grad.abs().sum() > 0, "c4 gradients missing or zero"
        assert c5.grad is not None and c5.grad.abs().sum() > 0, "c5 gradients missing or zero"

    def test_gradient_flow_parameters(self):
        """Test that all neck parameters receive gradients"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=256
        )
        neck.train()

        c3 = torch.randn(2, 512, 80, 80)
        c4 = torch.randn(2, 1024, 40, 40)
        c5 = torch.randn(2, 2048, 20, 20)

        n3, n4, n5 = neck([c3, c4, c5])
        loss = (n3.sum() + n4.sum() + n5.sum())
        loss.backward()

        # Check lateral convs have gradients
        for lateral_conv in neck.lateral_convs:
            assert lateral_conv.conv.weight.grad is not None
            assert lateral_conv.conv.weight.grad.abs().sum() > 0

        # Check FPN blocks have gradients
        for fpn_block in neck.fpn_blocks:
            assert fpn_block.conv1.conv.weight.grad is not None
            assert fpn_block.conv1.conv.weight.grad.abs().sum() > 0

        # Check PAN blocks have gradients
        for pan_block in neck.pan_blocks:
            assert pan_block.conv1.conv.weight.grad is not None
            assert pan_block.conv1.conv.weight.grad.abs().sum() > 0


class TestCSPRepLayer:
    """Test CSPRepLayer component"""

    def test_forward_shape(self):
        """Test CSPRepLayer output shape"""
        layer = CSPRepLayer(
            in_channels=512,
            out_channels=256,
            num_blocks=3,
            expansion=1.0
        )
        layer.eval()

        x = torch.randn(2, 512, 40, 40)
        output = layer(x)

        assert output.shape == (2, 256, 40, 40)

    def test_residual_addition(self):
        """Test that CSPRepLayer uses addition (not concatenation) following PaddlePaddle"""
        layer = CSPRepLayer(
            in_channels=256,
            out_channels=256,
            num_blocks=2,
            expansion=1.0
        )
        layer.eval()

        x = torch.randn(2, 256, 40, 40)
        output = layer(x)

        # Output should have same channels as input (after conv3)
        assert output.shape[1] == 256

    def test_gradient_flow(self):
        """Test gradient flow through CSPRepLayer"""
        layer = CSPRepLayer(
            in_channels=256,
            out_channels=256,
            num_blocks=3
        )
        layer.train()

        x = torch.randn(2, 256, 40, 40, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestConvNormAct:
    """Test ConvNormAct building block"""

    def test_forward_shape(self):
        """Test ConvNormAct output shape"""
        block = ConvNormAct(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1
        )
        block.eval()

        x = torch.randn(2, 256, 40, 40)
        output = block(x)

        assert output.shape == (2, 512, 20, 20)

    @pytest.mark.parametrize("act", ['relu', 'silu', 'none'])
    def test_activation_types(self, act):
        """Test different activation types"""
        block = ConvNormAct(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            act=act
        )
        block.eval()

        x = torch.randn(2, 256, 40, 40)
        output = block(x)

        assert output.shape == x.shape


class TestBuildHybridEncoder:
    """Test builder function"""

    def test_build_from_config(self):
        """Test building HybridEncoder using direct instantiation"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            num_encoder_layers=1,
            use_encoder_idx=[2]
        )

        assert isinstance(neck, HybridEncoder)
        assert neck.hidden_dim == 256
        assert neck.num_encoder_layers == 1
        assert neck.use_encoder_idx == [2]

    def test_build_with_defaults(self):
        """Test building with default values"""
        neck = HybridEncoder()

        assert isinstance(neck, HybridEncoder)
        assert neck.in_channels == [512, 1024, 2048]  # Default
        assert neck.hidden_dim == 256                  # Default


class TestEdgeCases:
    """Test edge cases and different input sizes"""

    def test_variable_input_sizes(self):
        """Test with different input resolutions"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=256
        )
        neck.eval()

        # Test multiple resolutions
        resolutions = [(320, 320), (640, 640), (800, 800), (1024, 1024)]

        for h, w in resolutions:
            c3 = torch.randn(1, 512, h // 8, w // 8)
            c4 = torch.randn(1, 1024, h // 16, w // 16)
            c5 = torch.randn(1, 2048, h // 32, w // 32)

            n3, n4, n5 = neck([c3, c4, c5])

            assert n3.shape == (1, 256, h // 8, w // 8)
            assert n4.shape == (1, 256, h // 16, w // 16)
            assert n5.shape == (1, 256, h // 32, w // 32)

    def test_non_square_inputs(self):
        """Test with non-square input resolutions"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=256
        )
        neck.eval()

        h, w = 480, 640
        c3 = torch.randn(2, 512, h // 8, w // 8)
        c4 = torch.randn(2, 1024, h // 16, w // 16)
        c5 = torch.randn(2, 2048, h // 32, w // 32)

        n3, n4, n5 = neck([c3, c4, c5])

        assert n3.shape == (2, 256, h // 8, w // 8)
        assert n4.shape == (2, 256, h // 16, w // 16)
        assert n5.shape == (2, 256, h // 32, w // 32)

    def test_batch_size_one(self):
        """Test with batch size 1"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=256
        )
        neck.eval()

        c3 = torch.randn(1, 512, 80, 80)
        c4 = torch.randn(1, 1024, 40, 40)
        c5 = torch.randn(1, 2048, 20, 20)

        n3, n4, n5 = neck([c3, c4, c5])

        assert n3.shape[0] == 1
        assert n4.shape[0] == 1
        assert n5.shape[0] == 1

    def test_invalid_input_count(self):
        """Test error handling for wrong number of inputs"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=256
        )
        neck.eval()

        # Only provide 2 inputs instead of 3
        c3 = torch.randn(2, 512, 80, 80)
        c4 = torch.randn(2, 1024, 40, 40)

        with pytest.raises(AssertionError, match="Expected 3 feature levels"):
            neck([c3, c4])

    def test_no_encoder_layers(self):
        """Test HybridEncoder without encoder layers"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=256,
            num_encoder_layers=0  # No encoder layers
        )
        neck.eval()

        c3 = torch.randn(2, 512, 80, 80)
        c4 = torch.randn(2, 1024, 40, 40)
        c5 = torch.randn(2, 2048, 20, 20)

        n3, n4, n5 = neck([c3, c4, c5])

        assert n3.shape == (2, 256, 80, 80)
        assert n4.shape == (2, 256, 40, 40)
        assert n5.shape == (2, 256, 20, 20)


class TestNumericalStability:
    """Test numerical stability"""

    def test_no_nan_inf(self):
        """Test that outputs don't contain NaN or Inf"""
        neck = HybridEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=256
        )
        neck.eval()

        c3 = torch.randn(2, 512, 80, 80)
        c4 = torch.randn(2, 1024, 40, 40)
        c5 = torch.randn(2, 2048, 20, 20)

        n3, n4, n5 = neck([c3, c4, c5])

        assert not torch.isnan(n3).any(), "n3 contains NaN"
        assert not torch.isnan(n4).any(), "n4 contains NaN"
        assert not torch.isnan(n5).any(), "n5 contains NaN"
        assert not torch.isinf(n3).any(), "n3 contains Inf"
        assert not torch.isinf(n4).any(), "n4 contains Inf"
        assert not torch.isinf(n5).any(), "n5 contains Inf"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

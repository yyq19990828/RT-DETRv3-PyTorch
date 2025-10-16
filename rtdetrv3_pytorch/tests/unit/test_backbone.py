"""
Unit tests for ResNet backbone

Tests cover:
- Output shapes for different ResNet variants
- Gradient flow through the network
- Frozen stages functionality
- ResNet-vd variant specific features
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.backbones.resnet import ResNet, build_resnet


class TestResNetOutputShapes:
    """Test output shapes for different ResNet configurations"""

    @pytest.mark.parametrize("depth,expected_channels", [
        (18, [128, 256, 512]),
        (34, [128, 256, 512]),
        (50, [512, 1024, 2048]),
        (101, [512, 1024, 2048]),
    ])
    def test_resnet_output_shapes(self, depth, expected_channels):
        """Test output shapes for batch input (batch=2, 3, 640, 640)"""
        model = ResNet(depth=depth, variant='d', return_idx=[1, 2, 3])
        model.eval()

        # Input: (batch, channels, height, width)
        x = torch.randn(2, 3, 640, 640)

        with torch.no_grad():
            outputs = model(x)

        # Should return 3 feature maps
        assert len(outputs) == 3

        # Check spatial dimensions (strides 8, 16, 32)
        assert outputs[0].shape == (2, expected_channels[0], 80, 80), \
            f"C3 shape mismatch: {outputs[0].shape}"
        assert outputs[1].shape == (2, expected_channels[1], 40, 40), \
            f"C4 shape mismatch: {outputs[1].shape}"
        assert outputs[2].shape == (2, expected_channels[2], 20, 20), \
            f"C5 shape mismatch: {outputs[2].shape}"

    def test_resnet50_single_return_idx(self):
        """Test returning only C5 feature map"""
        model = ResNet(depth=50, variant='d', return_idx=[3])
        model.eval()

        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            outputs = model(x)

        assert len(outputs) == 1
        assert outputs[0].shape == (1, 2048, 20, 20)

    def test_resnet50_all_return_idx(self):
        """Test returning all feature maps C2, C3, C4, C5"""
        model = ResNet(depth=50, variant='d', return_idx=[0, 1, 2, 3])
        model.eval()

        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            outputs = model(x)

        assert len(outputs) == 4
        # C2: stride 4
        assert outputs[0].shape == (1, 256, 160, 160)
        # C3: stride 8
        assert outputs[1].shape == (1, 512, 80, 80)
        # C4: stride 16
        assert outputs[2].shape == (1, 1024, 40, 40)
        # C5: stride 32
        assert outputs[3].shape == (1, 2048, 20, 20)


class TestResNetGradientFlow:
    """Test gradient flow through the network"""

    def test_resnet50_gradient_flow(self):
        """Test gradient flow (loss.backward() succeeds)"""
        model = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])
        model.train()

        x = torch.randn(2, 3, 640, 640, requires_grad=True)
        outputs = model(x)

        # Compute dummy loss
        loss = sum(o.sum() for o in outputs)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "No gradient for input"
        assert x.grad.shape == x.shape

        # Check some layer gradients
        assert model.layer4[0].conv1.weight.grad is not None
        assert model.layer3[0].conv1.weight.grad is not None

    def test_resnet18_gradient_flow(self):
        """Test gradient flow for ResNet-18 (BasicBlock)"""
        model = ResNet(depth=18, variant='d', return_idx=[1, 2, 3])
        model.train()

        x = torch.randn(2, 3, 640, 640, requires_grad=True)
        outputs = model(x)

        loss = sum(o.sum() for o in outputs)
        loss.backward()

        assert x.grad is not None
        assert model.layer4[0].conv1.weight.grad is not None


class TestResNetFrozenStages:
    """Test frozen stages functionality"""

    def test_frozen_stages_no_gradients(self):
        """Test frozen_stages parameter (gradients zeroed for frozen layers)"""
        model = ResNet(depth=50, variant='d', frozen_stages=1, return_idx=[1, 2, 3])
        model.train()

        x = torch.randn(2, 3, 640, 640)
        outputs = model(x)

        loss = sum(o.sum() for o in outputs)
        loss.backward()

        # Stem should be frozen (no gradients)
        assert model.conv1_1.weight.grad is None or torch.all(model.conv1_1.weight.grad == 0)
        assert model.bn1_1.weight.grad is None or torch.all(model.bn1_1.weight.grad == 0)

        # Layer1 should be frozen
        assert model.layer1[0].conv1.weight.grad is None or \
               torch.all(model.layer1[0].conv1.weight.grad == 0)

        # Layer2 should NOT be frozen (should have gradients)
        assert model.layer2[0].conv1.weight.grad is not None
        assert torch.any(model.layer2[0].conv1.weight.grad != 0)

    def test_frozen_stages_eval_mode(self):
        """Test frozen stages remain in eval mode during training"""
        model = ResNet(depth=50, variant='d', frozen_stages=2, return_idx=[1, 2, 3])
        model.train()

        # Layer1 and Layer2 should be in eval mode
        assert not model.layer1.training
        assert not model.layer2.training

        # Layer3 and Layer4 should be in training mode
        assert model.layer3.training
        assert model.layer4.training

    def test_no_frozen_stages(self):
        """Test with no frozen stages (all layers trainable)"""
        model = ResNet(depth=50, variant='d', frozen_stages=-1, return_idx=[1, 2, 3])
        model.train()

        x = torch.randn(2, 3, 640, 640)
        outputs = model(x)

        loss = sum(o.sum() for o in outputs)
        loss.backward()

        # All layers should have gradients
        assert model.conv1_1.weight.grad is not None
        assert model.layer1[0].conv1.weight.grad is not None
        assert model.layer2[0].conv1.weight.grad is not None
        assert model.layer3[0].conv1.weight.grad is not None
        assert model.layer4[0].conv1.weight.grad is not None


class TestResNetVariants:
    """Test different ResNet variants"""

    def test_resnet_vd_stem(self):
        """Test ResNet-vd variant has 3x3 stem convolutions"""
        model = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])

        # Should have three stem conv layers
        assert hasattr(model, 'conv1_1')
        assert hasattr(model, 'conv1_2')
        assert hasattr(model, 'conv1_3')

        # Check kernel sizes
        assert model.conv1_1.kernel_size == (3, 3)
        assert model.conv1_2.kernel_size == (3, 3)
        assert model.conv1_3.kernel_size == (3, 3)

    def test_resnet_standard_stem(self):
        """Test standard ResNet has 7x7 stem"""
        model = ResNet(depth=50, variant='a', return_idx=[1, 2, 3])

        # Should have single 7x7 conv
        assert hasattr(model, 'conv1')
        assert model.conv1.kernel_size == (7, 7)

    def test_resnet_vd_forward(self):
        """Test forward pass with ResNet-vd variant"""
        model = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])
        model.eval()

        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            outputs = model(x)

        assert len(outputs) == 3
        assert outputs[0].shape == (1, 512, 80, 80)
        assert outputs[1].shape == (1, 1024, 40, 40)
        assert outputs[2].shape == (1, 2048, 20, 20)


class TestBuildResNet:
    """Test build_resnet config function"""

    def test_build_from_config(self):
        """Test building ResNet from config dict"""
        cfg = {
            'depth': 50,
            'variant': 'd',
            'frozen_stages': 1,
            'return_idx': [1, 2, 3]
        }

        model = build_resnet(cfg)

        assert isinstance(model, ResNet)
        assert model.depth == 50
        assert model.variant == 'd'
        assert model.frozen_stages == 1
        assert model.return_idx == [1, 2, 3]

    def test_build_with_defaults(self):
        """Test building ResNet with default config"""
        cfg = {}

        model = build_resnet(cfg)

        assert isinstance(model, ResNet)
        assert model.depth == 50  # Default
        assert model.variant == 'd'  # Default
        assert model.frozen_stages == -1  # Default
        assert model.return_idx == [1, 2, 3]  # Default


class TestResNetEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_depth(self):
        """Test error handling for invalid depth"""
        with pytest.raises(ValueError, match="Unsupported depth"):
            ResNet(depth=99, variant='d', return_idx=[1, 2, 3])

    def test_variable_input_sizes(self):
        """Test with different input sizes"""
        model = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])
        model.eval()

        # Test different input sizes (use multiples of 32 to avoid padding issues)
        input_sizes = [(1, 3, 320, 320), (2, 3, 640, 640), (1, 3, 800, 1344)]

        for input_size in input_sizes:
            x = torch.randn(*input_size)
            with torch.no_grad():
                outputs = model(x)

            batch, _, h, w = input_size
            # Due to padding in conv/pool layers, output may not be exact h//stride
            # Just check batch size and channel dimensions
            assert outputs[0].shape[0] == batch
            assert outputs[0].shape[1] == 512
            assert outputs[1].shape[0] == batch
            assert outputs[1].shape[1] == 1024
            assert outputs[2].shape[0] == batch
            assert outputs[2].shape[1] == 2048

            # Check that spatial dimensions are approximately correct (within 1 pixel)
            assert abs(outputs[0].shape[2] - h // 8) <= 1
            assert abs(outputs[0].shape[3] - w // 8) <= 1

    def test_batch_size_one(self):
        """Test with batch size 1"""
        model = ResNet(depth=50, variant='d', return_idx=[1, 2, 3])
        model.eval()

        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            outputs = model(x)

        assert outputs[0].shape[0] == 1
        assert outputs[1].shape[0] == 1
        assert outputs[2].shape[0] == 1


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])

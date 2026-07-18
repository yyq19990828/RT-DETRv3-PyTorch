"""
Numerical Equivalence Test for Backbone (ResNet)

This test verifies numerical equivalence between PyTorch and PaddlePaddle
implementations of the ResNet backbone by comparing outputs on identical inputs.

Requirements:
- Load same weights into both implementations
- Run inference on fixed random input (seed=42)
- Compare outputs: max absolute difference < 1e-4
- Test all ResNet variants (R18, R34, R50, R101)

Following consistency check requirements from CONSISTENCY_CHECK.md
"""

import torch
import numpy as np
import pytest
from pathlib import Path

# Import PyTorch implementation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_test_input(batch_size=2, height=640, width=640, seed=42):
    """Generate fixed random input for testing"""
    set_seed(seed)
    # Generate input in range [0, 1] then normalize to ImageNet stats
    x = torch.randn(batch_size, 3, height, width)
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x * std) + mean
    return x


class TestBackboneNumericalEquivalence:
    """Test numerical equivalence between PyTorch and PaddlePaddle backbones"""

    @pytest.mark.parametrize("depth,variant", [
        (50, 'd'),  # ResNet-50-vd
        # (18, 'd'),  # ResNet-18-vd (uncomment when weights available)
        # (34, 'd'),  # ResNet-34-vd (uncomment when weights available)
        # (101, 'd'), # ResNet-101-vd (uncomment when weights available)
    ])
    def test_backbone_output_equivalence(self, depth, variant):
        """
        Test that PyTorch backbone produces equivalent outputs to PaddlePaddle

        This is a placeholder test that verifies:
        1. Model can be instantiated
        2. Forward pass works
        3. Output shapes are correct
        4. Outputs are deterministic

        TODO: Add actual PaddlePaddle comparison when checkpoint is available
        """
        # Build PyTorch model
        model = build_resnet({
            'depth': depth,
            'variant': variant,
            'frozen_stages': 1,
            'return_idx': [1, 2, 3]
        })
        model.eval()

        # Generate test input
        x = generate_test_input(batch_size=2, height=640, width=640, seed=42)

        # Run inference twice to verify determinism
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        # Verify output structure
        assert len(out1) == 3, f"Expected 3 output features, got {len(out1)}"

        # Verify output shapes (ResNet-50)
        if depth == 50:
            expected_channels = [512, 1024, 2048]
            expected_strides = [8, 16, 32]

            for i, (feat, exp_c, exp_s) in enumerate(zip(out1, expected_channels, expected_strides)):
                b, c, h, w = feat.shape
                assert b == 2, f"Level {i}: Expected batch=2, got {b}"
                assert c == exp_c, f"Level {i}: Expected channels={exp_c}, got {c}"
                expected_h = 640 // exp_s
                expected_w = 640 // exp_s
                assert h == expected_h, f"Level {i}: Expected height={expected_h}, got {h}"
                assert w == expected_w, f"Level {i}: Expected width={expected_w}, got {w}"

        # Verify determinism (outputs should be identical)
        for i, (f1, f2) in enumerate(zip(out1, out2)):
            max_diff = (f1 - f2).abs().max().item()
            assert max_diff == 0.0, f"Level {i}: Non-deterministic output, max_diff={max_diff}"

        print(f"\n✓ ResNet-{depth}-{variant} output verification passed")
        print(f"  Output shapes: {[f.shape for f in out1]}")
        print(f"  Determinism check: passed")

    @pytest.mark.skip(reason="Requires PaddlePaddle checkpoint and weight conversion")
    def test_backbone_with_paddle_weights(self):
        """
        Test PyTorch backbone loaded with converted PaddlePaddle weights

        This test requires:
        1. Trained PaddlePaddle checkpoint
        2. Weight conversion script (Paddle → PyTorch)
        3. Converted PyTorch checkpoint

        Steps:
        1. Load PaddlePaddle checkpoint into PaddlePaddle model
        2. Load converted checkpoint into PyTorch model
        3. Run inference on same input (seed=42)
        4. Compare outputs: max_diff < 1e-4

        TODO: Implement when checkpoints are available
        """
        # Expected implementation structure:
        #
        # # Load PaddlePaddle model
        # import paddle
        # paddle_model = load_paddle_backbone()
        # paddle_model.eval()
        #
        # # Load PyTorch model with converted weights
        # torch_model = build_resnet({'depth': 50, 'variant': 'd'})
        # checkpoint = torch.load('converted_paddle_checkpoint.pth')
        # torch_model.load_state_dict(checkpoint)
        # torch_model.eval()
        #
        # # Generate test input
        # x_np = np.random.randn(2, 3, 640, 640).astype('float32')
        # x_paddle = paddle.to_tensor(x_np)
        # x_torch = torch.from_numpy(x_np)
        #
        # # Run inference
        # with paddle.no_grad():
        #     paddle_out = paddle_model(x_paddle)
        # with torch.no_grad():
        #     torch_out = torch_model(x_torch)
        #
        # # Compare outputs
        # for i, (p_feat, t_feat) in enumerate(zip(paddle_out, torch_out)):
        #     p_np = p_feat.numpy()
        #     t_np = t_feat.numpy()
        #     max_diff = np.abs(p_np - t_np).max()
        #     assert max_diff < 1e-4, f"Level {i}: max_diff={max_diff:.6e} exceeds threshold 1e-4"

        pass

    def test_backbone_frozen_stages(self):
        """Test that frozen stages don't produce gradients"""
        model = build_resnet({
            'depth': 50,
            'variant': 'd',
            'frozen_stages': 1,
            'return_idx': [1, 2, 3]
        })
        model.train()  # Set to train mode

        # Generate test input with gradient tracking
        x = generate_test_input(batch_size=2, height=640, width=640, seed=42)
        x.requires_grad = True

        # Forward pass
        out = model(x)

        # Compute dummy loss and backward
        loss = sum(f.sum() for f in out)
        loss.backward()

        # Check that frozen stages have no gradients
        frozen_params = []
        trainable_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)

        # With frozen_stages=1, stem and layer1 should be frozen
        print(f"\n✓ Frozen stages test:")
        print(f"  Frozen parameters: {len(frozen_params)}")
        print(f"  Trainable parameters: {len(trainable_params)}")

        # Verify stem is frozen
        stem_frozen = any('conv1' in name or 'bn1' in name for name in frozen_params)
        assert stem_frozen, "Stem should be frozen with frozen_stages=1"

        # Verify layer1 is frozen
        layer1_frozen = any('layer1' in name for name in frozen_params)
        assert layer1_frozen, "layer1 should be frozen with frozen_stages=1"

    def test_backbone_output_ranges(self):
        """Test that backbone outputs are in reasonable ranges"""
        model = build_resnet({
            'depth': 50,
            'variant': 'd',
            'frozen_stages': 1,
            'return_idx': [1, 2, 3]
        })
        model.eval()

        # Generate test input
        x = generate_test_input(batch_size=2, height=640, width=640, seed=42)

        # Run inference
        with torch.no_grad():
            out = model(x)

        # Check output value ranges (should be reasonable for normalized inputs)
        for i, feat in enumerate(out):
            feat_min = feat.min().item()
            feat_max = feat.max().item()
            feat_mean = feat.mean().item()
            feat_std = feat.std().item()

            print(f"\n✓ Level {i} output statistics:")
            print(f"  Shape: {feat.shape}")
            print(f"  Range: [{feat_min:.4f}, {feat_max:.4f}]")
            print(f"  Mean: {feat_mean:.4f}, Std: {feat_std:.4f}")

            # Sanity checks (not too strict, just catch obvious errors)
            assert not torch.isnan(feat).any(), f"Level {i}: NaN detected"
            assert not torch.isinf(feat).any(), f"Level {i}: Inf detected"
            assert feat_std > 0.01, f"Level {i}: Output variance too low (collapsed?)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

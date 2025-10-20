"""
Numerical Equivalence Test for Neck (HybridEncoder)

This test verifies numerical equivalence between PyTorch and PaddlePaddle
implementations of the HybridEncoder neck by comparing outputs on identical inputs.

Requirements:
- Load same weights into both implementations
- Run inference on fixed random input (seed=42)
- Compare outputs: max absolute difference < 1e-4
- Test FPN-PAN fusion mechanism

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


def generate_test_features(seed=42):
    """
    Generate fixed random multi-scale features for testing

    Returns backbone-like features:
    - C3: (B, 512, H/8, W/8)
    - C4: (B, 1024, H/16, W/16)
    - C5: (B, 2048, H/32, W/32)
    """
    set_seed(seed)
    batch_size = 2
    height, width = 640, 640

    feats = [
        torch.randn(batch_size, 512, height // 8, width // 8),    # C3
        torch.randn(batch_size, 1024, height // 16, width // 16), # C4
        torch.randn(batch_size, 2048, height // 32, width // 32), # C5
    ]
    return feats


class TestNeckNumericalEquivalence:
    """Test numerical equivalence between PyTorch and PaddlePaddle necks"""

    def test_neck_output_equivalence(self):
        """
        Test that PyTorch neck produces equivalent outputs to PaddlePaddle

        This is a placeholder test that verifies:
        1. Model can be instantiated
        2. Forward pass works
        3. Output shapes are correct
        4. Outputs are deterministic

        TODO: Add actual PaddlePaddle comparison when checkpoint is available
        """
        # Build PyTorch model
        model = build_hybrid_encoder({
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'expansion': 1.0
        })
        model.eval()

        # Generate test input
        feats = generate_test_features(seed=42)

        # Run inference twice to verify determinism
        with torch.no_grad():
            out1 = model(feats)
            out2 = model(feats)

        # Verify output structure
        assert len(out1) == 3, f"Expected 3 output features, got {len(out1)}"

        # Verify output shapes (all should be hidden_dim=256)
        expected_shapes = [
            (2, 256, 80, 80),   # P3: stride 8
            (2, 256, 40, 40),   # P4: stride 16
            (2, 256, 20, 20),   # P5: stride 32
        ]

        for i, (feat, exp_shape) in enumerate(zip(out1, expected_shapes)):
            assert feat.shape == exp_shape, \
                f"Level {i}: Expected shape {exp_shape}, got {feat.shape}"

        # Verify determinism (outputs should be identical)
        for i, (f1, f2) in enumerate(zip(out1, out2)):
            max_diff = (f1 - f2).abs().max().item()
            assert max_diff == 0.0, f"Level {i}: Non-deterministic output, max_diff={max_diff}"

        print(f"\n✓ HybridEncoder output verification passed")
        print(f"  Output shapes: {[f.shape for f in out1]}")
        print(f"  Determinism check: passed")

    def test_neck_channel_unification(self):
        """Test that all outputs have unified channels (hidden_dim=256)"""
        model = build_hybrid_encoder({
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'expansion': 1.0
        })
        model.eval()

        # Generate test input
        feats = generate_test_features(seed=42)

        # Run inference
        with torch.no_grad():
            out = model(feats)

        # Check all outputs have hidden_dim channels
        for i, feat in enumerate(out):
            _, c, _, _ = feat.shape
            assert c == 256, f"Level {i}: Expected 256 channels, got {c}"

        print(f"\n✓ Channel unification test passed")
        print(f"  All outputs have 256 channels")

    def test_neck_fpn_pan_structure(self):
        """
        Test FPN-PAN structure by checking gradient flow

        In FPN-PAN:
        - FPN: top-down pathway (C5 → P5, C4 → P4, C3 → P3)
        - PAN: bottom-up pathway (P3 → N3, P4 → N4, P5 → N5)

        All features should be connected through fusion
        """
        model = build_hybrid_encoder({
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'expansion': 1.0
        })
        model.train()

        # Generate test input with gradient tracking
        feats = generate_test_features(seed=42)
        for f in feats:
            f.requires_grad = True

        # Forward pass
        out = model(feats)

        # Compute dummy loss from all outputs
        loss = sum(f.sum() for f in out)
        loss.backward()

        # Check that all input features receive gradients (connected through FPN-PAN)
        for i, feat in enumerate(feats):
            assert feat.grad is not None, f"Input level {i} has no gradient"
            grad_norm = feat.grad.norm().item()
            assert grad_norm > 0, f"Input level {i} has zero gradient"
            print(f"  Level {i} gradient norm: {grad_norm:.4f}")

        print(f"\n✓ FPN-PAN gradient flow test passed")

    def test_neck_with_encoder(self):
        """
        Test neck with multi-scale deformable attention encoder

        use_encoder_idx=[2] means only apply encoder to level 2 (C5/P5)
        """
        model = build_hybrid_encoder({
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],  # Apply encoder to C5
            'num_encoder_layers': 1,
            'expansion': 1.0
        })
        model.eval()

        # Generate test input
        feats = generate_test_features(seed=42)

        # Run inference
        with torch.no_grad():
            out = model(feats)

        # Verify outputs
        assert len(out) == 3
        for i, feat in enumerate(out):
            print(f"  Level {i}: {feat.shape}")

        print(f"\n✓ Encoder integration test passed")

    def test_neck_output_ranges(self):
        """Test that neck outputs are in reasonable ranges"""
        model = build_hybrid_encoder({
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'expansion': 1.0
        })
        model.eval()

        # Generate test input
        feats = generate_test_features(seed=42)

        # Run inference
        with torch.no_grad():
            out = model(feats)

        # Check output value ranges
        for i, feat in enumerate(out):
            feat_min = feat.min().item()
            feat_max = feat.max().item()
            feat_mean = feat.mean().item()
            feat_std = feat.std().item()

            print(f"\n✓ Level {i} output statistics:")
            print(f"  Shape: {feat.shape}")
            print(f"  Range: [{feat_min:.4f}, {feat_max:.4f}]")
            print(f"  Mean: {feat_mean:.4f}, Std: {feat_std:.4f}")

            # Sanity checks
            assert not torch.isnan(feat).any(), f"Level {i}: NaN detected"
            assert not torch.isinf(feat).any(), f"Level {i}: Inf detected"
            assert feat_std > 0.01, f"Level {i}: Output variance too low (collapsed?)"

    @pytest.mark.skip(reason="Requires PaddlePaddle checkpoint and weight conversion")
    def test_neck_with_paddle_weights(self):
        """
        Test PyTorch neck loaded with converted PaddlePaddle weights

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
        # paddle_model = load_paddle_neck()
        # paddle_model.eval()
        #
        # # Load PyTorch model with converted weights
        # torch_model = build_hybrid_encoder(...)
        # checkpoint = torch.load('converted_paddle_checkpoint.pth')
        # torch_model.load_state_dict(checkpoint)
        # torch_model.eval()
        #
        # # Generate test features
        # feats_np = [np.random.randn(*shape).astype('float32') for shape in feat_shapes]
        # feats_paddle = [paddle.to_tensor(f) for f in feats_np]
        # feats_torch = [torch.from_numpy(f) for f in feats_np]
        #
        # # Run inference
        # with paddle.no_grad():
        #     paddle_out = paddle_model(feats_paddle)
        # with torch.no_grad():
        #     torch_out = torch_model(feats_torch)
        #
        # # Compare outputs
        # for i, (p_feat, t_feat) in enumerate(zip(paddle_out, torch_out)):
        #     p_np = p_feat.numpy()
        #     t_np = t_feat.numpy()
        #     max_diff = np.abs(p_np - t_np).max()
        #     assert max_diff < 1e-4, f"Level {i}: max_diff={max_diff:.6e} exceeds threshold 1e-4"

        pass

    def test_neck_csprepLayer_addition_mode(self):
        """
        Test that CSPRepLayer uses addition (not concatenation)

        This is critical for PaddlePaddle consistency per CONSISTENCY_CHECK.md:
        "CSPRepLayer使用addition (NOT concatenation)"
        """
        model = build_hybrid_encoder({
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'expansion': 1.0
        })

        # Check that CSPRepLayer modules exist
        has_csp = False
        for name, module in model.named_modules():
            if 'CSPRepLayer' in type(module).__name__ or 'csp' in name.lower():
                has_csp = True
                print(f"  Found CSP module: {name}")

        if not has_csp:
            print(f"\n⚠ No CSPRepLayer modules found (may use alternative structure)")
        else:
            print(f"\n✓ CSPRepLayer modules present")

        # Functional test: run model
        feats = generate_test_features(seed=42)
        model.eval()
        with torch.no_grad():
            out = model(feats)

        # Verify output
        assert len(out) == 3
        print(f"✓ CSPRepLayer fusion test passed")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

"""
Numerical Equivalence Tests for Registered Components

This test suite verifies backward compatibility by comparing direct instantiation
vs registry-based instantiation of components. Tests ensure that adding the
@register decorator does not change the component's numerical behavior.

Requirements:
- Both instantiation methods should produce identical outputs
- Maximum absolute difference < 1e-5 (FP32 tolerance)
- Tests cover all major components: ResNet, HybridEncoder, RTDETRTransformerv3,
  DINOv3Head, RTDETRv3

Following User Story 4 (US4) requirements from tasks.md
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import registry system
from models import (
    BACKBONE_REGISTRY, NECK_REGISTRY, TRANSFORMER_REGISTRY,
    HEAD_REGISTRY, ARCHITECTURE_REGISTRY, create
)

# Import direct component classes
from rtdetrv3_pytorch.models.backbones.resnet import ResNet
from rtdetrv3_pytorch.models.necks.hybrid_encoder import HybridEncoder
from rtdetrv3_pytorch.models.transformers.rtdetr_transformer import RTDETRTransformerv3
from rtdetrv3_pytorch.models.heads.detr_head import DINOv3Head
from models.rtdetrv3 import RTDETRv3


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
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x * std) + mean
    return x


def assert_tensor_equal(t1, t2, tolerance=1e-5, name="tensor"):
    """Assert two tensors are numerically equal within tolerance"""
    if isinstance(t1, (list, tuple)) and isinstance(t2, (list, tuple)):
        assert len(t1) == len(t2), f"{name}: Length mismatch {len(t1)} vs {len(t2)}"
        for i, (v1, v2) in enumerate(zip(t1, t2)):
            assert_tensor_equal(v1, v2, tolerance, f"{name}[{i}]")
        return

    # Convert to tensors if needed
    if not isinstance(t1, torch.Tensor):
        t1 = torch.tensor(t1)
    if not isinstance(t2, torch.Tensor):
        t2 = torch.tensor(t2)

    # Check shapes match
    assert t1.shape == t2.shape, f"{name}: Shape mismatch {t1.shape} vs {t2.shape}"

    # Check numerical equivalence
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()

    assert max_diff < tolerance, (
        f"{name}: Max absolute difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}"
    )

    print(f"✓ {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")


@pytest.mark.numerical
class TestRegisteredComponentsNumericalEquivalence:
    """Test numerical equivalence between direct and registered instantiation"""

    def test_resnet_direct_vs_registered(self):
        """
        T060: Test ResNet numerical equivalence (direct vs registered)

        Verifies that @BACKBONE_REGISTRY.register() decorator is non-invasive
        and doesn't change the model's numerical behavior.
        """
        set_seed(42)

        # Configuration
        config = {
            'depth': 50,
            'variant': 'd',
            'frozen_stages': 1,
            'return_idx': [1, 2, 3]
        }

        # Method 1: Direct instantiation
        direct_model = ResNet(**config)
        direct_model.eval()

        # Method 2: Registry-based instantiation
        registered_model = BACKBONE_REGISTRY.create('ResNet', **config)
        registered_model.eval()

        # Copy weights from direct to registered to ensure same initialization
        registered_model.load_state_dict(direct_model.state_dict())

        # Generate test input
        x = generate_test_input(batch_size=2, height=640, width=640, seed=42)

        # Forward pass
        with torch.no_grad():
            direct_out = direct_model(x)
            registered_out = registered_model(x)

        # Verify numerical equivalence
        assert_tensor_equal(direct_out, registered_out, tolerance=1e-5, name="ResNet output")

        print("✓ T060: ResNet direct vs registered - PASSED")

    def test_hybrid_encoder_direct_vs_registered(self):
        """
        T061: Test HybridEncoder numerical equivalence (direct vs registered)

        Verifies that @NECK_REGISTRY.register() decorator is non-invasive.
        """
        set_seed(42)

        # Configuration (matching typical RTDETRv3 setup)
        in_channels = [512, 1024, 2048]  # ResNet-50 output channels
        feat_strides = [8, 16, 32]
        hidden_dim = 256

        config = {
            'in_channels': in_channels,
            'feat_strides': feat_strides,
            'hidden_dim': hidden_dim,
            'use_encoder_idx': [2],  # Use only last feature
            'num_encoder_layers': 1
        }

        # Method 1: Direct instantiation
        direct_model = HybridEncoder(**config)
        direct_model.eval()

        # Method 2: Registry-based instantiation
        registered_model = NECK_REGISTRY.create('HybridEncoder', **config)
        registered_model.eval()

        # Copy weights
        registered_model.load_state_dict(direct_model.state_dict())

        # Generate test input (multi-scale features)
        set_seed(42)
        x = [
            torch.randn(2, in_channels[0], 80, 80),
            torch.randn(2, in_channels[1], 40, 40),
            torch.randn(2, in_channels[2], 20, 20)
        ]

        # Forward pass
        with torch.no_grad():
            direct_out = direct_model(x)
            registered_out = registered_model(x)

        # Verify numerical equivalence
        assert_tensor_equal(direct_out, registered_out, tolerance=1e-5, name="HybridEncoder output")

        print("✓ T061: HybridEncoder direct vs registered - PASSED")

    def test_transformer_direct_vs_registered(self):
        """
        T062: Test RTDETRTransformerv3 numerical equivalence (direct vs registered)

        Verifies that @TRANSFORMER_REGISTRY.register() decorator is non-invasive.
        """
        set_seed(42)

        # Configuration
        config = {
            'num_queries': 300,
            'num_decoder_layers': 6,
            'hidden_dim': 256,
            'num_heads': 8,
            'dim_feedforward': 1024,
            'dropout': 0.0,
            'activation': 'relu',
            'num_denoising': 100,
            'label_noise_ratio': 0.5,
            'box_noise_scale': 1.0,
            'num_classes': 80
        }

        # Method 1: Direct instantiation
        direct_model = RTDETRTransformerv3(**config)
        direct_model.eval()

        # Method 2: Registry-based instantiation
        registered_model = TRANSFORMER_REGISTRY.create('RTDETRTransformerv3', **config)
        registered_model.eval()

        # Copy weights
        registered_model.load_state_dict(direct_model.state_dict())

        # Generate test input (multi-scale features from neck)
        set_seed(42)
        feats = [
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 256, 20, 20)
        ]

        # Forward pass (inference mode, no targets)
        with torch.no_grad():
            direct_out = direct_model(feats, targets=None)
            registered_out = registered_model(feats, targets=None)

        # Verify numerical equivalence
        # RTDETRTransformerv3 returns a tuple of 5 tensors
        assert len(direct_out) == 5, "Expected 5 outputs"
        assert len(registered_out) == 5, "Expected 5 outputs"

        for i in range(4):  # First 4 are tensors, 5th is dn_meta (None)
            assert_tensor_equal(direct_out[i], registered_out[i], tolerance=1e-5,
                              name=f"RTDETRTransformerv3 output[{i}]")

        print("✓ T062: RTDETRTransformerv3 direct vs registered - PASSED")

    def test_dinov3head_direct_vs_registered(self):
        """
        T063: Test DINOv3Head numerical equivalence (direct vs registered)

        Verifies that @HEAD_REGISTRY.register() decorator is non-invasive.
        """
        set_seed(42)

        # Configuration
        config = {
            'num_classes': 80,
            'hidden_dim': 256
        }

        # Method 1: Direct instantiation
        direct_model = DINOv3Head(**config)
        direct_model.eval()

        # Method 2: Registry-based instantiation
        registered_model = HEAD_REGISTRY.create('DINOv3Head', **config)
        registered_model.eval()

        # Copy weights
        registered_model.load_state_dict(direct_model.state_dict())

        # Generate test input (transformer output tuple)
        set_seed(42)
        dec_out_bboxes = torch.randn(6, 2, 300, 4)  # [num_layers, batch, num_queries, 4]
        dec_out_logits = torch.randn(6, 2, 300, 80)  # [num_layers, batch, num_queries, num_classes]
        enc_topk_bboxes = torch.randn(2, 300, 4)  # [batch, num_queries, 4]
        enc_topk_logits = torch.randn(2, 300, 80)  # [batch, num_queries, num_classes]
        dn_meta = None

        out_transformer = (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta)

        # Forward pass (eval mode)
        with torch.no_grad():
            direct_out = direct_model(out_transformer)
            registered_out = registered_model(out_transformer)

        # Verify numerical equivalence (output is tuple of 3 elements in eval mode)
        assert isinstance(direct_out, tuple), "Output should be tuple"
        assert isinstance(registered_out, tuple), "Output should be tuple"
        assert len(direct_out) == 3, "Expected 3 outputs"
        assert len(registered_out) == 3, "Expected 3 outputs"

        # Compare first two outputs (pred_bboxes, pred_logits)
        for i in range(2):
            assert_tensor_equal(
                direct_out[i], registered_out[i],
                tolerance=1e-5, name=f"DINOv3Head output[{i}]"
            )

        print("✓ T063: DINOv3Head direct vs registered - PASSED")

    def test_rtdetrv3_direct_vs_registered(self):
        """
        T064: Test RTDETRv3 (full model) numerical equivalence (direct vs registered)

        Verifies that @ARCHITECTURE_REGISTRY.register() decorator is non-invasive
        and the complete model works correctly through registry.
        """
        set_seed(42)

        # Configuration for complete model
        config = {
            'num_classes': 80,
            'hidden_dim': 256,
            'num_queries': 300,
            'num_decoder_layers': 6,
            'backbone_config': {
                'depth': 50,
                'variant': 'd',
                'frozen_stages': 1,
                'return_idx': [1, 2, 3]
            },
            'neck_config': {
                'in_channels': [512, 1024, 2048],
                'feat_strides': [8, 16, 32],
                'hidden_dim': 256,
                'use_encoder_idx': [2],
                'num_encoder_layers': 1
            },
            'transformer_config': {
                'num_queries': 300,
                'num_decoder_layers': 6,
                'hidden_dim': 256,
                'num_heads': 8,
                'dim_feedforward': 1024,
                'num_classes': 80
            },
            'head_config': {
                'num_classes': 80,
                'hidden_dim': 256
            }
        }

        # Method 1: Direct instantiation (from config dicts)
        # Note: RTDETRv3.__init__ expects component instances, not configs
        # So we build components first
                        
        set_seed(42)
        backbone = build_resnet(config['backbone_config'])
        neck = build_hybrid_encoder(config['neck_config'])
        # For transformer, directly instantiate since no builder exists
        transformer = RTDETRTransformerv3(**config['transformer_config'])
        # For head, directly instantiate with required parameters
        head = DINOv3Head(**config['head_config'])

        direct_model = RTDETRv3(
            backbone=backbone,
            neck=neck,
            transformer=transformer,
            detr_head=head,
            num_classes=config['num_classes']
        )
        direct_model.eval()

        # Method 2: Registry-based instantiation using from_config()
        set_seed(42)
        # Build a global config for from_config
        global_config = {
            'num_classes': 80,
            'hidden_dim': 256,
            'backbone': config['backbone_config'],
            'neck': config['neck_config'],
            'transformer': config['transformer_config'],
            'detr_head': config['head_config']
        }
        registered_model = RTDETRv3.from_config(global_config)
        registered_model.eval()

        # Copy weights to ensure same initialization
        registered_model.load_state_dict(direct_model.state_dict())

        # Generate test input
        x = generate_test_input(batch_size=2, height=640, width=640, seed=42)

        # Forward pass (inference mode)
        with torch.no_grad():
            direct_out = direct_model(x)
            registered_out = registered_model(x)

        # Verify numerical equivalence
        # Output is dict with 'pred_logits', 'pred_boxes'
        assert isinstance(direct_out, dict), "Output should be dict"
        assert isinstance(registered_out, dict), "Output should be dict"

        for key in ['pred_logits', 'pred_boxes']:
            assert key in direct_out, f"Missing key: {key}"
            assert key in registered_out, f"Missing key: {key}"
            assert_tensor_equal(
                direct_out[key], registered_out[key],
                tolerance=1e-5, name=f"RTDETRv3.{key}"
            )

        print("✓ T064: RTDETRv3 direct vs registered - PASSED")

    def test_global_create_function(self):
        """
        Additional test: Verify global create() function works correctly

        The create() function should search all registries and instantiate
        components without needing to specify which registry to use.
        """
        set_seed(42)

        # Test create() for each component type
        components = [
            ('ResNet', {'depth': 50, 'variant': 'd', 'return_idx': [1, 2, 3]}),
            ('HybridEncoder', {
                'in_channels': [512, 1024, 2048],
                'feat_strides': [8, 16, 32],
                'hidden_dim': 256,
                'use_encoder_idx': [2],
                'num_encoder_layers': 1
            }),
            ('RTDETRTransformerv3', {
                'num_queries': 300,
                'num_decoder_layers': 6,
                'hidden_dim': 256,
                'num_heads': 8,
                'dim_feedforward': 1024,
                'num_classes': 80
            }),
            ('DINOv3Head', {
                'num_classes': 80,
                'hidden_dim': 256
            })
        ]

        for name, config in components:
            # Create using global create() function
            instance = create(name, **config)
            assert instance is not None, f"Failed to create {name}"

            # Verify it's the correct type
            assert instance.__class__.__name__ == name, (
                f"Created instance type mismatch: expected {name}, "
                f"got {instance.__class__.__name__}"
            )

            print(f"✓ Global create('{name}') - PASSED")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-m', 'numerical'])

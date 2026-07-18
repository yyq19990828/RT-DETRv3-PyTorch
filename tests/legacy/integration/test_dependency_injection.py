"""
Integration tests for dependency injection chain (User Story 2)

Tests the complete dependency injection workflow:
- T032: backbone → neck injection
- T033: neck → transformer injection
- T034: transformer → head injection
- T035: end-to-end full dependency chain

Following TDD principle: Write tests FIRST, ensure they FAIL before implementation.
"""

import pytest
import torch
from rtdetrv3_pytorch.models import (
    BACKBONE_REGISTRY,
    NECK_REGISTRY,
    TRANSFORMER_REGISTRY,
    HEAD_REGISTRY,
    ARCHITECTURE_REGISTRY,
    create
)


class TestBackboneToNeckInjection:
    """T032: Test backbone → neck injection"""

    def test_resnet_provides_out_shape(self):
        """Test that ResNet provides out_shape attribute for downstream injection"""
        backbone = create('ResNet', depth=50, variant='d', return_idx=[1, 2, 3])

        # Verify out_shape exists
        assert hasattr(backbone, 'out_shape'), "ResNet must provide out_shape attribute"
        assert isinstance(backbone.out_shape, list), "out_shape must be a list"
        assert len(backbone.out_shape) == 3, "out_shape should have 3 items for return_idx=[1,2,3]"

        # Verify structure
        for i, shape_info in enumerate(backbone.out_shape):
            assert 'channels' in shape_info, f"out_shape[{i}] must have 'channels' key"
            assert 'stride' in shape_info, f"out_shape[{i}] must have 'stride' key"
            assert isinstance(shape_info['channels'], int), "channels must be int"
            assert isinstance(shape_info['stride'], int), "stride must be int"

        # Verify expected values for ResNet-50
        assert backbone.out_shape[0]['channels'] == 512
        assert backbone.out_shape[0]['stride'] == 8
        assert backbone.out_shape[1]['channels'] == 1024
        assert backbone.out_shape[1]['stride'] == 16
        assert backbone.out_shape[2]['channels'] == 2048
        assert backbone.out_shape[2]['stride'] == 32

    def test_backbone_to_neck_injection_via_config(self):
        """Test that neck receives backbone.out_shape when created via config"""
        global_config = {
            'backbone': {
                'type': 'ResNet',
                'depth': 50,
                'variant': 'd',
                'return_idx': [1, 2, 3]
            },
            'neck': {
                'type': 'HybridEncoder',
                'hidden_dim': 256
            }
        }

        # Create backbone - filter out 'type' from config
        backbone_cfg = {k: v for k, v in global_config['backbone'].items() if k != 'type'}
        backbone = create('ResNet', global_config=global_config, **backbone_cfg)

        # Neck should be able to receive input_shape
        # This will be tested once HybridEncoder has from_config()
        # For now, just verify backbone.out_shape exists
        assert hasattr(backbone, 'out_shape')


class TestNeckToTransformerInjection:
    """T033: Test neck → transformer injection"""

    @pytest.mark.skip(reason="Requires HybridEncoder from_config() implementation")
    def test_neck_provides_output_for_transformer(self):
        """Test that HybridEncoder provides necessary outputs for transformer"""
        # Will be implemented after neck gets from_config()
        pass

    @pytest.mark.skip(reason="Requires from_config() chain implementation")
    def test_neck_to_transformer_injection_via_config(self):
        """Test that transformer receives neck outputs when created via config"""
        # Will be implemented after injection chain is complete
        pass


class TestTransformerToHeadInjection:
    """T034: Test transformer → head injection"""

    @pytest.mark.skip(reason="Requires transformer from_config() implementation")
    def test_transformer_provides_hidden_dim_for_head(self):
        """Test that transformer provides hidden_dim for head"""
        # Will be implemented after transformer gets from_config()
        pass

    @pytest.mark.skip(reason="Requires from_config() chain implementation")
    def test_transformer_to_head_injection_via_config(self):
        """Test that head receives transformer.hidden_dim when created via config"""
        # Will be implemented after injection chain is complete
        pass


class TestEndToEndDependencyChain:
    """T035: Test complete dependency injection chain"""

    @pytest.mark.skip(reason="Requires RTDETRv3.from_config() implementation")
    def test_full_model_creation_from_config(self):
        """Test creating complete RTDETRv3 model from config with dependency injection"""
        config = {
            'type': 'RTDETRv3',
            'backbone': {
                'type': 'ResNet',
                'depth': 50,
                'variant': 'd',
                'return_idx': [1, 2, 3]
            },
            'neck': {
                'type': 'HybridEncoder',
                'hidden_dim': 256
            },
            'transformer': {
                'type': 'RTDETRTransformerv3',
                'num_queries': 300
            },
            'detr_head': {
                'type': 'DINOv3Head'
            },
            'num_classes': 80
        }

        # Create model
        model = create('RTDETRv3', global_config=config, **config)

        # Verify components were created
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'neck')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'detr_head')

        # Verify dependency chain worked
        # - neck should have received backbone.out_shape
        # - transformer should have received neck outputs
        # - head should have received transformer.hidden_dim

        # Test forward pass
        x = torch.randn(2, 3, 640, 640)
        outputs = model(x)
        assert outputs is not None

    @pytest.mark.skip(reason="Requires complete from_config() implementation")
    def test_dependency_chain_with_shared_config(self):
        """Test that __shared__ fields are properly distributed"""
        global_config = {
            'num_classes': 80,
            'hidden_dim': 256,
            'backbone': {'type': 'ResNet', 'depth': 50},
            'neck': {'type': 'HybridEncoder'},
            'transformer': {'type': 'RTDETRTransformerv3'},
            'detr_head': {'type': 'DINOv3Head'}
        }

        model = create('RTDETRv3', global_config=global_config)

        # Verify shared config was used
        assert model.num_classes == 80
        # Head should also have num_classes=80 from shared config
        assert model.detr_head.num_classes == 80

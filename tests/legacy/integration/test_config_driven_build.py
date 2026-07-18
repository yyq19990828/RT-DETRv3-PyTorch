"""
Integration tests for config-driven model building (User Story 3)

Tests the complete config-driven workflow:
- T050: YAML config loading
- T051: global_config parameter resolution
- T052: end-to-end model creation from config

Following TDD principle: Write tests FIRST, ensure they work with existing implementation.
"""

import pytest
import torch
import yaml
from pathlib import Path
from rtdetrv3_pytorch.models import (
    create,
    build_from_config,
    BACKBONE_REGISTRY,
    ARCHITECTURE_REGISTRY
)


class TestYAMLConfigLoading:
    """T050: Test YAML config loading"""

    def test_load_yaml_config_and_build_backbone(self, tmp_path):
        """Test loading YAML config and building backbone"""
        # Create temporary YAML config
        config = {
            'type': 'ResNet',
            'depth': 50,
            'variant': 'd',
            'return_idx': [1, 2, 3]
        }

        yaml_file = tmp_path / "resnet_config.yml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f)

        # Load and build
        with open(yaml_file, 'r') as f:
            loaded_config = yaml.safe_load(f)

        backbone = build_from_config(loaded_config, BACKBONE_REGISTRY)

        # Verify
        assert backbone is not None
        assert hasattr(backbone, 'out_shape')
        assert len(backbone.out_shape) == 3

    def test_load_nested_yaml_config(self, tmp_path):
        """Test loading nested YAML config with multiple components"""
        config = {
            'type': 'RTDETRv3',
            'num_classes': 80,
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

        yaml_file = tmp_path / "rtdetrv3_config.yml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f)

        # Load config
        with open(yaml_file, 'r') as f:
            loaded_config = yaml.safe_load(f)

        # Verify structure
        assert loaded_config['type'] == 'RTDETRv3'
        assert 'backbone' in loaded_config
        assert loaded_config['backbone']['type'] == 'ResNet'


class TestGlobalConfigParameterResolution:
    """T051: Test global_config parameter resolution"""

    def test_shared_config_resolution(self):
        """Test that __shared__ fields are resolved from global_config"""
        global_config = {
            'num_classes': 91,  # Non-default value
            'hidden_dim': 384   # Non-default value
        }

        # Create head using global_config
        from ppdet_pytorch.modeling.heads.detr_head import DINOv3Head
        head = create('DINOv3Head', global_config=global_config)

        # Verify shared config was used
        assert head.num_classes == 91
        assert head.hidden_dim == 384

    def test_explicit_parameter_overrides_shared(self):
        """Test that explicit parameters override shared config"""
        global_config = {
            'num_classes': 91
        }

        # Explicitly pass num_classes=80
        from ppdet_pytorch.modeling.heads.detr_head import DINOv3Head
        head = create('DINOv3Head', global_config=global_config, num_classes=80)

        # Explicit parameter should win
        assert head.num_classes == 80

    def test_parameter_resolution_priority(self):
        """Test parameter resolution priority: explicit > shared > default"""
        global_config = {
            'num_classes': 91,
            'hidden_dim': 384
        }

        from ppdet_pytorch.modeling.heads.detr_head import DINOv3Head

        # Case 1: Only defaults (num_classes=80, hidden_dim=256)
        head1 = create('DINOv3Head')
        assert head1.num_classes == 80
        assert head1.hidden_dim == 256

        # Case 2: Shared config overrides defaults
        head2 = create('DINOv3Head', global_config=global_config)
        assert head2.num_classes == 91
        assert head2.hidden_dim == 384

        # Case 3: Explicit overrides shared
        head3 = create('DINOv3Head', global_config=global_config, num_classes=100)
        assert head3.num_classes == 100
        assert head3.hidden_dim == 384  # Still from shared


class TestEndToEndConfigDrivenBuild:
    """T052: Test complete model creation from config"""

    def test_build_rtdetrv3_from_dict_config(self):
        """Test building complete RTDETRv3 from dict config"""
        config = {
            'type': 'RTDETRv3',
            'num_classes': 80,
            'backbone': {
                'type': 'ResNet',
                'depth': 50,
                'variant': 'd',
                'return_idx': [1, 2, 3]
            },
            'neck': {
                'type': 'HybridEncoder',
                'hidden_dim': 256,
                'in_channels': [512, 1024, 2048],
                'feat_strides': [8, 16, 32]
            },
            'transformer': {
                'type': 'RTDETRTransformerv3',
                'num_queries': 300,
                'hidden_dim': 256,
                'num_decoder_layers': 6
            },
            'detr_head': {
                'type': 'DINOv3Head',
                'eval_idx': -1
            }
        }

        # Build model via global create()
        # Note: Pass only non-component keys as kwargs, components will be auto-injected
        model_kwargs = {k: v for k, v in config.items() if k not in ['type', 'backbone', 'neck', 'transformer', 'detr_head', 'aux_head']}
        model = create('RTDETRv3', global_config=config, **model_kwargs)

        # Verify components were created
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'neck')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'detr_head')

        # Test forward pass
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(x)

        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs
        # Note: Default transformer uses 100 o2o + 300 o2m = 400 queries
        assert outputs['pred_logits'].shape[0] == 1
        assert outputs['pred_logits'].shape[2] == 80
        assert outputs['pred_boxes'].shape[0] == 1
        assert outputs['pred_boxes'].shape[2] == 4

    def test_build_with_minimal_config(self):
        """Test building with minimal config (relying on defaults)"""
        config = {
            'type': 'RTDETRv3',
            'num_classes': 80
        }

        # Should build with all defaults
        model = create('RTDETRv3', global_config=config, num_classes=80)

        assert model is not None
        assert model.num_classes == 80

    def test_config_with_yaml_file(self, tmp_path):
        """Test end-to-end: YAML file -> model creation -> inference"""
        config = {
            'type': 'RTDETRv3',
            'num_classes': 80,
            'backbone': {
                'type': 'ResNet',
                'depth': 18,  # Smaller model for faster test
                'variant': 'd',
                'return_idx': [1, 2, 3]
            },
            'neck': {
                'type': 'HybridEncoder',
                'hidden_dim': 256,
                'in_channels': [128, 256, 512],
                'feat_strides': [8, 16, 32]
            }
        }

        # Save to YAML
        yaml_file = tmp_path / "rtdetrv3_r18.yml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f)

        # Load from YAML
        with open(yaml_file, 'r') as f:
            loaded_config = yaml.safe_load(f)

        # Build model
        # Note: Pass only non-component keys as kwargs, components will be auto-injected
        model_kwargs = {k: v for k, v in loaded_config.items() if k not in ['type', 'backbone', 'neck', 'transformer', 'detr_head', 'aux_head']}
        model = create('RTDETRv3', global_config=loaded_config, **model_kwargs)

        # Quick inference test
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(x)

        assert outputs is not None
        assert 'pred_logits' in outputs

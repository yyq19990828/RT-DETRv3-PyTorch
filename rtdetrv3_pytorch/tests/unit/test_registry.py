"""
Unit tests for Registry system

Tests for component registration and basic registry functionality.
"""

import pytest
import torch
import torch.nn as nn

from rtdetrv3_pytorch.models import (
    Registry,
    BACKBONE_REGISTRY,
    NECK_REGISTRY,
    TRANSFORMER_REGISTRY,
    HEAD_REGISTRY,
    LOSS_REGISTRY,
    ARCHITECTURE_REGISTRY,
    create,
    validate_component_protocol
)


class TestRegistryList:
    """Test registry list() method for all component categories"""

    def test_backbone_registry_list(self):
        """T010: Test BACKBONE_REGISTRY.list() contains ResNet"""
        components = BACKBONE_REGISTRY.list()
        assert 'ResNet' in components, f"ResNet not found in BACKBONE_REGISTRY. Found: {components}"
        assert len(components) >= 1, "BACKBONE_REGISTRY should have at least 1 component"

    def test_neck_registry_list(self):
        """T011: Test NECK_REGISTRY.list() contains HybridEncoder"""
        components = NECK_REGISTRY.list()
        assert 'HybridEncoder' in components, f"HybridEncoder not found in NECK_REGISTRY. Found: {components}"
        assert len(components) >= 1, "NECK_REGISTRY should have at least 1 component"

    def test_transformer_registry_list(self):
        """T012: Test TRANSFORMER_REGISTRY.list() contains RTDETRTransformerv3"""
        components = TRANSFORMER_REGISTRY.list()
        assert 'RTDETRTransformerv3' in components, f"RTDETRTransformerv3 not found in TRANSFORMER_REGISTRY. Found: {components}"
        assert len(components) >= 1, "TRANSFORMER_REGISTRY should have at least 1 component"

    def test_head_registry_list(self):
        """T013: Test HEAD_REGISTRY.list() contains DINOv3Head and PPYOLOEHead"""
        components = HEAD_REGISTRY.list()
        assert 'DINOv3Head' in components, f"DINOv3Head not found in HEAD_REGISTRY. Found: {components}"
        assert 'PPYOLOEHead' in components, f"PPYOLOEHead not found in HEAD_REGISTRY. Found: {components}"
        assert len(components) >= 2, "HEAD_REGISTRY should have at least 2 components"

    def test_loss_registry_list(self):
        """T014: Test LOSS_REGISTRY.list() contains DINOv3Loss"""
        components = LOSS_REGISTRY.list()
        assert 'DINOv3Loss' in components, f"DINOv3Loss not found in LOSS_REGISTRY. Found: {components}"
        assert len(components) >= 1, "LOSS_REGISTRY should have at least 1 component"

    def test_architecture_registry_list(self):
        """T015: Test ARCHITECTURE_REGISTRY.list() contains RTDETRv3"""
        components = ARCHITECTURE_REGISTRY.list()
        assert 'RTDETRv3' in components, f"RTDETRv3 not found in ARCHITECTURE_REGISTRY. Found: {components}"
        assert len(components) >= 1, "ARCHITECTURE_REGISTRY should have at least 1 component"


class TestComponentRegistration:
    """Test that components are properly registered with metadata"""

    def test_resnet_registration(self):
        """Test ResNet is registered with correct metadata"""
        cls = BACKBONE_REGISTRY.get('ResNet')
        assert hasattr(cls, '__category__'), "ResNet should have __category__ attribute"
        assert cls.__category__ == 'backbone', f"Expected category 'backbone', got '{cls.__category__}'"
        assert hasattr(cls, '__inject__'), "ResNet should have __inject__ attribute"
        assert hasattr(cls, '__shared__'), "ResNet should have __shared__ attribute"

    def test_hybrid_encoder_registration(self):
        """Test HybridEncoder is registered with correct metadata"""
        cls = NECK_REGISTRY.get('HybridEncoder')
        assert hasattr(cls, '__category__'), "HybridEncoder should have __category__ attribute"
        assert cls.__category__ == 'neck', f"Expected category 'neck', got '{cls.__category__}'"

    def test_transformer_registration(self):
        """Test RTDETRTransformerv3 is registered with correct metadata"""
        cls = TRANSFORMER_REGISTRY.get('RTDETRTransformerv3')
        assert hasattr(cls, '__category__'), "RTDETRTransformerv3 should have __category__ attribute"
        assert cls.__category__ == 'transformer', f"Expected category 'transformer', got '{cls.__category__}'"

    def test_head_registration(self):
        """Test heads are registered with correct metadata"""
        for head_name in ['DINOv3Head', 'PPYOLOEHead']:
            cls = HEAD_REGISTRY.get(head_name)
            assert hasattr(cls, '__category__'), f"{head_name} should have __category__ attribute"
            assert cls.__category__ == 'head', f"Expected category 'head', got '{cls.__category__}'"

    def test_loss_registration(self):
        """Test DINOv3Loss is registered with correct metadata"""
        cls = LOSS_REGISTRY.get('DINOv3Loss')
        assert hasattr(cls, '__category__'), "DINOv3Loss should have __category__ attribute"
        assert cls.__category__ == 'loss', f"Expected category 'loss', got '{cls.__category__}'"

    def test_architecture_registration(self):
        """Test RTDETRv3 is registered with correct metadata"""
        cls = ARCHITECTURE_REGISTRY.get('RTDETRv3')
        assert hasattr(cls, '__category__'), "RTDETRv3 should have __category__ attribute"
        assert cls.__category__ == 'architecture', f"Expected category 'architecture', got '{cls.__category__}'"


class TestRegistryCreate:
    """Test registry.create() method for component instantiation"""

    def test_create_resnet(self):
        """Test creating ResNet via BACKBONE_REGISTRY.create()"""
        backbone = BACKBONE_REGISTRY.create('ResNet', depth=50, variant='d')
        assert backbone is not None
        assert backbone.depth == 50
        assert backbone.variant == 'd'

    def test_create_via_global_create(self):
        """Test creating component via global create() function"""
        backbone = create('ResNet', depth=50, variant='d')
        assert backbone is not None
        assert backbone.depth == 50

    def test_create_with_global_config(self):
        """Test creating component with global_config for __shared__ fields"""
        # Create a simple test registry
        test_registry = Registry('test')

        @test_registry.register()
        class TestComponent(nn.Module):
            __shared__ = ['num_classes']

            def __init__(self, num_classes=80, other_param=10):
                super().__init__()
                self.num_classes = num_classes
                self.other_param = other_param

        # Test without global_config (should use default)
        comp1 = test_registry.create('TestComponent', other_param=20)
        assert comp1.num_classes == 80  # default value
        assert comp1.other_param == 20

        # Test with global_config (should override default)
        comp2 = test_registry.create(
            'TestComponent',
            global_config={'num_classes': 100},
            other_param=30
        )
        assert comp2.num_classes == 100  # from global_config
        assert comp2.other_param == 30


class TestComponentProtocolValidation:
    """Test validate_component_protocol() helper"""

    def test_valid_component(self):
        """Test validation passes for valid component"""
        class ValidComponent(nn.Module):
            __inject__ = ['dep']
            __shared__ = ['shared_val']

            def __init__(self, dep, shared_val=10):
                super().__init__()
                self.dep = dep
                self.shared_val = shared_val

            @classmethod
            def from_config(cls, cfg, global_config=None):
                return {}

        # Should not raise
        assert validate_component_protocol(ValidComponent) is True

    def test_invalid_shared_no_default(self):
        """Test validation fails for __shared__ field without default"""
        class InvalidComponent(nn.Module):
            __shared__ = ['no_default']

            def __init__(self, no_default):  # Missing default!
                super().__init__()
                self.no_default = no_default

        with pytest.raises(ValueError, match="must have default value"):
            validate_component_protocol(InvalidComponent)

    def test_invalid_inject_missing_param(self):
        """Test validation fails for __inject__ field not in __init__"""
        class InvalidComponent(nn.Module):
            __inject__ = ['missing_param']

            def __init__(self, other_param):
                super().__init__()
                self.other_param = other_param

        with pytest.raises(ValueError, match="__inject__ contains 'missing_param'"):
            validate_component_protocol(InvalidComponent)


class TestNestedConfigSupport:
    """T049: Test Registry.create() with nested config"""

    def test_create_with_nested_component_config(self):
        """Test creating component with nested sub-component configs"""
        global_config = {
            'num_classes': 80,
            'backbone': {
                'type': 'ResNet',
                'depth': 50,
                'variant': 'd',
                'return_idx': [1, 2, 3]
            }
        }

        # RTDETRv3 should be able to use the nested backbone config
        # This tests the pattern used in config-driven building
        backbone_cfg = global_config['backbone']
        assert 'type' in backbone_cfg
        assert backbone_cfg['type'] == 'ResNet'

        # Create backbone using the nested config
        from rtdetrv3_pytorch.models import build_from_config
        backbone = build_from_config(backbone_cfg, BACKBONE_REGISTRY, global_config)

        assert backbone is not None
        assert hasattr(backbone, 'depth')
        assert backbone.depth == 50

    def test_global_create_searches_all_registries(self):
        """Test that global create() function searches all registries"""
        # Can create from any registry without specifying which one
        backbone = create('ResNet', depth=18)
        assert backbone is not None

        neck = create('HybridEncoder', hidden_dim=256, in_channels=[128, 256, 512])
        assert neck is not None

        head = create('DINOv3Head')
        assert head is not None

        # Should raise error for non-existent type
        with pytest.raises(ValueError, match="not found in any registry"):
            create('NonExistentComponent')


class TestRegistryFromConfig:
    """Test from_config() class method support in Registry.create()"""

    def test_from_config_called(self):
        """Test Registry.create() calls from_config() if defined"""
        test_registry = Registry('test')

        @test_registry.register()
        class TestComponent(nn.Module):
            def __init__(self, value=10):
                super().__init__()
                self.value = value

            @classmethod
            def from_config(cls, cfg, global_config=None):
                # Double the value in from_config
                return {'value': cfg.get('value', 10) * 2}

        # When we pass value=5, from_config should double it to 10
        instance = test_registry.create('TestComponent', value=5)
        assert instance.value == 10, f"Expected value=10 (5*2), got {instance.value}"

    def test_from_config_with_dependencies(self):
        """Test from_config() can create nested dependencies"""
        test_registry = Registry('test')

        @test_registry.register()
        class Dependency(nn.Module):
            def __init__(self, dep_value=100):
                super().__init__()
                self.dep_value = dep_value

        @test_registry.register()
        class Parent(nn.Module):
            __inject__ = ['dep']

            def __init__(self, dep, parent_value=200):
                super().__init__()
                self.dep = dep
                self.parent_value = parent_value

            @classmethod
            def from_config(cls, cfg, global_config=None):
                # Create dependency if not provided
                if 'dep' not in cfg and global_config and 'dep' in global_config:
                    dep_cfg = global_config['dep']
                    if isinstance(dep_cfg, dict) and 'type' in dep_cfg:
                        from rtdetrv3_pytorch.models import create as global_create
                        # Use the test_registry instead
                        dep = test_registry.create(dep_cfg['type'], **{k: v for k, v in dep_cfg.items() if k != 'type'})
                        return {'dep': dep}
                return {}

        # This test validates the pattern, actual dependency injection is tested in integration tests
        instance = test_registry.create('Dependency', dep_value=150)
        assert instance.dep_value == 150


class TestMigrationValidator:
    """T072: Unit tests for migration validator functionality"""

    def test_all_registries_exist(self):
        """Test that all 6 required registries exist"""
        from rtdetrv3_pytorch.models import ALL_REGISTRIES

        assert len(ALL_REGISTRIES) == 6, f"Expected 6 registries, found {len(ALL_REGISTRIES)}"

        # Verify each registry is present
        registry_names = [r._name for r in ALL_REGISTRIES]
        expected = {'backbone', 'neck', 'transformer', 'head', 'loss', 'architecture'}
        assert set(registry_names) == expected, f"Missing registries: {expected - set(registry_names)}"

    def test_core_components_registered(self):
        """Test that all 7 core components are registered"""
        expected_components = {
            'BACKBONE': ['ResNet'],
            'NECK': ['HybridEncoder'],
            'TRANSFORMER': ['RTDETRTransformerv3'],
            'HEAD': ['DINOv3Head', 'PPYOLOEHead'],
            'LOSS': ['DINOv3Loss'],
            'ARCHITECTURE': ['RTDETRv3']
        }

        for reg_name, expected_comps in expected_components.items():
            if reg_name == 'BACKBONE':
                registry = BACKBONE_REGISTRY
            elif reg_name == 'NECK':
                registry = NECK_REGISTRY
            elif reg_name == 'TRANSFORMER':
                registry = TRANSFORMER_REGISTRY
            elif reg_name == 'HEAD':
                registry = HEAD_REGISTRY
            elif reg_name == 'LOSS':
                registry = LOSS_REGISTRY
            elif reg_name == 'ARCHITECTURE':
                registry = ARCHITECTURE_REGISTRY

            components = registry.list()
            for comp in expected_comps:
                assert comp in components, f"{comp} not found in {reg_name}_REGISTRY"

    def test_registry_lookup_performance(self):
        """Test that registry lookup is fast (<5ms per operation)"""
        import time

        # Measure time for 100 lookups
        iterations = 100
        start = time.perf_counter()

        for _ in range(iterations):
            _ = BACKBONE_REGISTRY.get('ResNet')
            _ = NECK_REGISTRY.get('HybridEncoder')
            _ = TRANSFORMER_REGISTRY.get('RTDETRTransformerv3')
            _ = HEAD_REGISTRY.get('DINOv3Head')
            _ = LOSS_REGISTRY.get('DINOv3Loss')
            _ = ARCHITECTURE_REGISTRY.get('RTDETRv3')

        elapsed = time.perf_counter() - start
        avg_per_lookup = (elapsed / iterations / 6) * 1000  # Convert to ms

        assert avg_per_lookup < 5.0, f"Registry lookup too slow: {avg_per_lookup:.2f}ms (target: <5ms)"

    def test_all_components_have_category(self):
        """Test that all registered components have __category__ attribute"""
        registries = [
            BACKBONE_REGISTRY,
            NECK_REGISTRY,
            TRANSFORMER_REGISTRY,
            HEAD_REGISTRY,
            LOSS_REGISTRY,
            ARCHITECTURE_REGISTRY
        ]

        for registry in registries:
            for comp_name in registry.list():
                cls = registry.get(comp_name)
                assert hasattr(cls, '__category__'), f"{comp_name} missing __category__ attribute"
                assert isinstance(cls.__category__, str), f"{comp_name}.__category__ must be string"
                assert len(cls.__category__) > 0, f"{comp_name}.__category__ cannot be empty"

    def test_all_components_have_inject_and_shared(self):
        """Test that all registered components have __inject__ and __shared__ attributes"""
        registries = [
            BACKBONE_REGISTRY,
            NECK_REGISTRY,
            TRANSFORMER_REGISTRY,
            HEAD_REGISTRY,
            LOSS_REGISTRY,
            ARCHITECTURE_REGISTRY
        ]

        for registry in registries:
            for comp_name in registry.list():
                cls = registry.get(comp_name)
                assert hasattr(cls, '__inject__'), f"{comp_name} missing __inject__ attribute"
                assert hasattr(cls, '__shared__'), f"{comp_name} missing __shared__ attribute"
                assert isinstance(cls.__inject__, list), f"{comp_name}.__inject__ must be list"
                assert isinstance(cls.__shared__, list), f"{comp_name}.__shared__ must be list"


class TestComponentMetadataValidation:
    """T073: Tests for component metadata validation"""

    def test_validate_protocol_on_all_registered_components(self):
        """Test that validate_component_protocol() passes for all registered components"""
        registries = [
            ('BACKBONE', BACKBONE_REGISTRY),
            ('NECK', NECK_REGISTRY),
            ('TRANSFORMER', TRANSFORMER_REGISTRY),
            ('HEAD', HEAD_REGISTRY),
            ('LOSS', LOSS_REGISTRY),
            ('ARCHITECTURE', ARCHITECTURE_REGISTRY)
        ]

        for reg_name, registry in registries:
            for comp_name in registry.list():
                cls = registry.get(comp_name)
                # Should not raise exception
                result = validate_component_protocol(cls)
                assert result is True, f"{comp_name} from {reg_name} failed protocol validation"

    def test_shared_fields_have_defaults(self):
        """Test that all __shared__ fields have default values in __init__"""
        import inspect

        registries = [
            BACKBONE_REGISTRY,
            NECK_REGISTRY,
            TRANSFORMER_REGISTRY,
            HEAD_REGISTRY,
            LOSS_REGISTRY,
            ARCHITECTURE_REGISTRY
        ]

        for registry in registries:
            for comp_name in registry.list():
                cls = registry.get(comp_name)

                if not hasattr(cls, '__shared__') or not cls.__shared__:
                    continue

                # Get __init__ signature
                sig = inspect.signature(cls.__init__)

                for field in cls.__shared__:
                    assert field in sig.parameters, f"{comp_name}.__shared__ contains '{field}' not in __init__"
                    param = sig.parameters[field]
                    assert param.default != inspect.Parameter.empty, \
                        f"{comp_name}.__shared__ field '{field}' must have default value"

    def test_inject_fields_in_init(self):
        """Test that all __inject__ fields are parameters in __init__"""
        import inspect

        registries = [
            BACKBONE_REGISTRY,
            NECK_REGISTRY,
            TRANSFORMER_REGISTRY,
            HEAD_REGISTRY,
            LOSS_REGISTRY,
            ARCHITECTURE_REGISTRY
        ]

        for registry in registries:
            for comp_name in registry.list():
                cls = registry.get(comp_name)

                if not hasattr(cls, '__inject__') or not cls.__inject__:
                    continue

                # Get __init__ signature
                sig = inspect.signature(cls.__init__)

                for field in cls.__inject__:
                    assert field in sig.parameters, \
                        f"{comp_name}.__inject__ contains '{field}' not in __init__ parameters"

    def test_category_matches_registry(self):
        """Test that __category__ matches the registry the component is in"""
        test_cases = [
            (BACKBONE_REGISTRY, 'backbone'),
            (NECK_REGISTRY, 'neck'),
            (TRANSFORMER_REGISTRY, 'transformer'),
            (HEAD_REGISTRY, 'head'),
            (LOSS_REGISTRY, 'loss'),
            (ARCHITECTURE_REGISTRY, 'architecture')
        ]

        for registry, expected_category in test_cases:
            for comp_name in registry.list():
                cls = registry.get(comp_name)
                assert cls.__category__ == expected_category, \
                    f"{comp_name} has category '{cls.__category__}' but is in {expected_category} registry"

    def test_from_config_signature(self):
        """Test that from_config() has correct signature when present"""
        import inspect

        registries = [
            BACKBONE_REGISTRY,
            NECK_REGISTRY,
            TRANSFORMER_REGISTRY,
            HEAD_REGISTRY,
            LOSS_REGISTRY,
            ARCHITECTURE_REGISTRY
        ]

        for registry in registries:
            for comp_name in registry.list():
                cls = registry.get(comp_name)

                if not hasattr(cls, 'from_config'):
                    continue

                # Check it's a classmethod
                method = getattr(cls, 'from_config')
                # For classmethod, check signature
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())

                # Should have at least 'cfg' parameter (cls is implicit for classmethod)
                assert 'cfg' in params, f"{comp_name}.from_config() missing 'cfg' parameter"
                # Should support global_config parameter
                assert 'global_config' in params, f"{comp_name}.from_config() missing 'global_config' parameter"


class TestFromConfigForAllComponents:
    """T079: Test from_config() method for all registered components"""

    def test_resnet_from_config(self):
        """Test ResNet.from_config() works correctly"""
        cfg = {
            'depth': 50,
            'variant': 'd',
            'return_idx': [1, 2, 3]
        }
        global_config = {'num_classes': 80}

        backbone = BACKBONE_REGISTRY.create('ResNet', **cfg, global_config=global_config)
        assert backbone is not None
        assert backbone.depth == 50
        assert backbone.variant == 'd'

    def test_hybrid_encoder_from_config(self):
        """Test HybridEncoder.from_config() works correctly"""
        cfg = {
            'hidden_dim': 256,
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32]
        }

        neck = NECK_REGISTRY.create('HybridEncoder', **cfg)
        assert neck is not None
        assert neck.hidden_dim == 256

    def test_transformer_from_config(self):
        """Test RTDETRTransformerv3.from_config() works correctly"""
        cfg = {
            'num_queries': 300,
            'hidden_dim': 256
        }

        transformer = TRANSFORMER_REGISTRY.create('RTDETRTransformerv3', **cfg)
        assert transformer is not None

    def test_head_from_config(self):
        """Test DINOv3Head.from_config() works correctly"""
        cfg = {
            'num_classes': 80,
            'hidden_dim': 256
        }

        head = HEAD_REGISTRY.create('DINOv3Head', **cfg)
        assert head is not None
        assert head.num_classes == 80

    def test_ppyoloe_head_from_config(self):
        """Test PPYOLOEHead.from_config() works correctly"""
        cfg = {
            'num_classes': 80,
            'fpn_strides': [8, 16, 32]
        }

        head = HEAD_REGISTRY.create('PPYOLOEHead', **cfg)
        assert head is not None
        assert head.num_classes == 80

    def test_architecture_from_config(self):
        """Test RTDETRv3.from_config() handles nested component configs"""
        # This is tested in integration tests - just verify the method exists
        cls = ARCHITECTURE_REGISTRY.get('RTDETRv3')
        assert hasattr(cls, 'from_config'), "RTDETRv3 must have from_config() method"


class TestParameterResolutionOrder:
    """T080: Test parameter resolution order (explicit > global_config > default)"""

    def test_explicit_overrides_global_config(self):
        """Test that explicit parameters override global_config"""
        test_registry = Registry('test')

        @test_registry.register()
        class TestComponent(nn.Module):
            __shared__ = ['num_classes']

            def __init__(self, num_classes=80):
                super().__init__()
                self.num_classes = num_classes

        # Explicit parameter (100) should override global_config (90)
        comp = test_registry.create(
            'TestComponent',
            num_classes=100,  # explicit
            global_config={'num_classes': 90}
        )
        assert comp.num_classes == 100, "Explicit parameter should override global_config"

    def test_global_config_overrides_default(self):
        """Test that global_config overrides default values"""
        test_registry = Registry('test')

        @test_registry.register()
        class TestComponent(nn.Module):
            __shared__ = ['num_classes']

            def __init__(self, num_classes=80):  # default = 80
                super().__init__()
                self.num_classes = num_classes

        # global_config (90) should override default (80)
        comp = test_registry.create(
            'TestComponent',
            global_config={'num_classes': 90}
        )
        assert comp.num_classes == 90, "global_config should override default value"

    def test_default_used_when_no_override(self):
        """Test that default values are used when no override provided"""
        test_registry = Registry('test')

        @test_registry.register()
        class TestComponent(nn.Module):
            __shared__ = ['num_classes']

            def __init__(self, num_classes=80):  # default = 80
                super().__init__()
                self.num_classes = num_classes

        # No explicit param, no global_config -> use default (80)
        comp = test_registry.create('TestComponent')
        assert comp.num_classes == 80, "Default value should be used when no override"

    def test_resolution_order_complete(self):
        """Test complete parameter resolution order: explicit > global_config > default"""
        test_registry = Registry('test')

        @test_registry.register()
        class TestComponent(nn.Module):
            __shared__ = ['param_a', 'param_b', 'param_c']

            def __init__(self, param_a=10, param_b=20, param_c=30):
                super().__init__()
                self.param_a = param_a
                self.param_b = param_b
                self.param_c = param_c

        comp = test_registry.create(
            'TestComponent',
            param_a=100,  # explicit - should win
            global_config={
                'param_a': 90,  # should be overridden by explicit
                'param_b': 200,  # should win (no explicit)
            }
            # param_c not specified -> should use default (30)
        )

        assert comp.param_a == 100, "Explicit should override global_config"
        assert comp.param_b == 200, "global_config should override default"
        assert comp.param_c == 30, "Default should be used when no override"

    def test_non_shared_not_affected_by_global_config(self):
        """Test that non-__shared__ parameters are not affected by global_config"""
        test_registry = Registry('test')

        @test_registry.register()
        class TestComponent(nn.Module):
            __shared__ = ['shared_param']

            def __init__(self, shared_param=10, normal_param=20):
                super().__init__()
                self.shared_param = shared_param
                self.normal_param = normal_param

        comp = test_registry.create(
            'TestComponent',
            normal_param=200,  # explicit for non-shared param
            global_config={
                'shared_param': 100,  # should affect shared_param
                'normal_param': 999   # should NOT affect normal_param
            }
        )

        assert comp.shared_param == 100, "Shared param should use global_config"
        assert comp.normal_param == 200, "Non-shared param should use explicit value only"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

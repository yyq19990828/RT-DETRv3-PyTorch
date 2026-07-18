"""
Unit tests for ppdet_pytorch.core.workspace module

Tests the PaddlePaddle-compatible registration system:
- register() decorator
- create() factory function
- __inject__ dependency injection
- __shared__ shared configuration
- merge_config() YAML integration
"""

import pytest
import sys
from pathlib import Path

# Add rtdetrv3_pytorch to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ppdet_pytorch.core.workspace import (
    global_config,
    register,
    create,
    merge_config,
    reset_global_config,
    get_registered_classes,
)


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global_config before each test"""
    reset_global_config()
    yield
    reset_global_config()


class TestRegisterDecorator:
    """Test register() decorator functionality"""

    def test_basic_registration(self):
        """Test basic class registration"""
        @register
        class DummyClass:
            pass

        assert 'DummyClass' in global_config
        assert global_config['DummyClass'] is DummyClass

    def test_custom_name_registration(self):
        """Test registration with custom name"""
        @register(name='CustomName')
        class DummyClass:
            pass

        assert 'CustomName' in global_config
        assert global_config['CustomName'] is DummyClass
        assert 'DummyClass' not in global_config

    def test_metadata_extraction(self):
        """Test __inject__, __shared__, __category__ metadata"""
        @register
        class ComponentWithMeta:
            __inject__ = ['dep1', 'dep2']
            __shared__ = ['num_classes']
            __category__ = 'backbone'

        cls = global_config['ComponentWithMeta']
        assert cls.__inject__ == ['dep1', 'dep2']
        assert cls.__shared__ == ['num_classes']
        assert cls.__category__ == 'backbone'

    def test_default_metadata(self):
        """Test default metadata values"""
        @register
        class ComponentNoMeta:
            pass

        cls = global_config['ComponentNoMeta']
        assert cls.__inject__ == []
        assert cls.__shared__ == []
        assert cls.__category__ == 'component'

    def test_overwrite_warning(self, caplog):
        """Test warning on duplicate registration"""
        @register
        class Duplicate:
            pass

        # Register again with same name
        @register
        class Duplicate:  # noqa: F811
            value = 'new'

        assert 'already registered' in caplog.text.lower()
        assert global_config['Duplicate'].value == 'new'


class TestCreateFunction:
    """Test create() factory function"""

    def test_create_from_string(self):
        """Test creating instance from string name"""
        @register
        class SimpleClass:
            def __init__(self, value=10):
                self.value = value

        instance = create('SimpleClass', value=20)
        assert isinstance(instance, SimpleClass)
        assert instance.value == 20

    def test_create_from_dict(self):
        """Test creating instance from config dict"""
        @register
        class SimpleClass:
            def __init__(self, value=10):
                self.value = value

        cfg = {'type': 'SimpleClass', 'value': 30}
        instance = create(cfg)
        assert isinstance(instance, SimpleClass)
        assert instance.value == 30

    def test_create_missing_type_key(self):
        """Test error when config dict missing 'type' key"""
        with pytest.raises(ValueError, match="must contain 'type' key"):
            create({'value': 10})

    def test_create_unknown_class(self):
        """Test error when class not found"""
        with pytest.raises(KeyError, match="not found in global_config"):
            create('UnknownClass')

    def test_kwargs_precedence(self):
        """Test that kwargs override config values"""
        @register
        class SimpleClass:
            def __init__(self, value=10):
                self.value = value

        cfg = {'type': 'SimpleClass', 'value': 30}
        instance = create(cfg, value=50)  # kwargs override
        assert instance.value == 50


class TestInjectMechanism:
    """Test __inject__ dependency injection"""

    def test_inject_from_config(self):
        """Test injecting dependency from config"""
        @register
        class Dependency:
            def __init__(self, dep_value=100):
                self.dep_value = dep_value

        @register
        class MainClass:
            __inject__ = ['dependency']

            def __init__(self, dependency, main_value=200):
                self.dependency = dependency
                self.main_value = main_value

        cfg = {
            'type': 'MainClass',
            'dependency': {'type': 'Dependency', 'dep_value': 123},
            'main_value': 456
        }
        instance = create(cfg)

        assert isinstance(instance.dependency, Dependency)
        assert instance.dependency.dep_value == 123
        assert instance.main_value == 456

    def test_inject_from_global_config(self):
        """Test injecting dependency from global_config"""
        @register
        class Dependency:
            def __init__(self, value=100):
                self.value = value

        @register
        class MainClass:
            __inject__ = ['dependency']

            def __init__(self, dependency):
                self.dependency = dependency

        # Put dependency config in global_config
        global_config['dependency'] = {'type': 'Dependency', 'value': 999}

        instance = create('MainClass')
        assert isinstance(instance.dependency, Dependency)
        assert instance.dependency.value == 999

    def test_inject_already_provided(self):
        """Test that provided instances are not re-created"""
        @register
        class Dependency:
            def __init__(self, value=100):
                self.value = value

        @register
        class MainClass:
            __inject__ = ['dependency']

            def __init__(self, dependency):
                self.dependency = dependency

        # Provide pre-created instance
        dep = Dependency(value=777)
        instance = create('MainClass', dependency=dep)

        assert instance.dependency is dep
        assert instance.dependency.value == 777

    def test_nested_injection(self):
        """Test nested dependency injection"""
        @register
        class Level0:
            def __init__(self, value=0):
                self.value = value

        @register
        class Level1:
            __inject__ = ['level0']

            def __init__(self, level0, value=1):
                self.level0 = level0
                self.value = value

        @register
        class Level2:
            __inject__ = ['level1']

            def __init__(self, level1, value=2):
                self.level1 = level1
                self.value = value

        cfg = {
            'type': 'Level2',
            'level1': {
                'type': 'Level1',
                'level0': {'type': 'Level0', 'value': 100},
                'value': 200
            },
            'value': 300
        }
        instance = create(cfg)

        assert instance.value == 300
        assert instance.level1.value == 200
        assert instance.level1.level0.value == 100


class TestSharedMechanism:
    """Test __shared__ shared configuration"""

    def test_shared_from_global_config(self):
        """Test shared fields from __shared__ in global_config"""
        @register
        class ComponentA:
            __shared__ = ['num_classes']

            def __init__(self, hidden_dim=256, num_classes=80):
                self.hidden_dim = hidden_dim
                self.num_classes = num_classes

        global_config['__shared__'] = {'num_classes': 91}

        instance = create('ComponentA', hidden_dim=512)
        assert instance.hidden_dim == 512
        assert instance.num_classes == 91  # From __shared__

    def test_shared_override_by_explicit(self):
        """Test that explicit values override shared config"""
        @register
        class ComponentA:
            __shared__ = ['num_classes']

            def __init__(self, num_classes=80):
                self.num_classes = num_classes

        global_config['__shared__'] = {'num_classes': 91}

        instance = create('ComponentA', num_classes=100)  # Explicit override
        assert instance.num_classes == 100

    def test_multiple_shared_fields(self):
        """Test multiple shared fields"""
        @register
        class ComponentA:
            __shared__ = ['num_classes', 'batch_size', 'lr']

            def __init__(self, num_classes=80, batch_size=32, lr=0.001):
                self.num_classes = num_classes
                self.batch_size = batch_size
                self.lr = lr

        global_config['__shared__'] = {
            'num_classes': 91,
            'batch_size': 64,
            'lr': 0.0001
        }

        instance = create('ComponentA')
        assert instance.num_classes == 91
        assert instance.batch_size == 64
        assert instance.lr == 0.0001


class TestMergeConfig:
    """Test merge_config() function"""

    def test_basic_merge(self):
        """Test basic config merging"""
        cfg = {
            'ResNet': {'depth': 50, 'freeze_at': 0},
            'HybridEncoder': {'hidden_dim': 256}
        }
        merge_config(cfg)

        assert 'ResNet' in global_config
        assert global_config['ResNet'] == {'depth': 50, 'freeze_at': 0}
        assert global_config['HybridEncoder'] == {'hidden_dim': 256}

    def test_merge_preserves_registered_classes(self):
        """Test that merge doesn't overwrite registered classes"""
        @register
        class ResNet:
            pass

        original_class = global_config['ResNet']

        cfg = {'ResNet': {'depth': 50}}
        merge_config(cfg)

        # Class should be overwritten by config (expected behavior)
        assert global_config['ResNet'] == {'depth': 50}

    def test_merge_shared_config(self):
        """Test merging __shared__ configuration"""
        cfg = {
            '__shared__': {'num_classes': 80, 'batch_size': 32}
        }
        merge_config(cfg)

        assert '__shared__' in global_config
        assert global_config['__shared__']['num_classes'] == 80
        assert global_config['__shared__']['batch_size'] == 32

    def test_deep_merge(self):
        """Test deep merging of dict values"""
        cfg1 = {'Model': {'param1': 10, 'param2': 20}}
        merge_config(cfg1)

        cfg2 = {'Model': {'param2': 30, 'param3': 40}}
        merge_config(cfg2)

        # Should merge dict values
        assert global_config['Model']['param1'] == 10
        assert global_config['Model']['param2'] == 30  # Overwritten
        assert global_config['Model']['param3'] == 40


class TestHelperFunctions:
    """Test helper utility functions"""

    def test_reset_global_config(self):
        """Test resetting global_config"""
        @register
        class DummyClass:
            pass

        assert 'DummyClass' in global_config

        reset_global_config()
        assert len(global_config) == 0

    def test_get_registered_classes(self):
        """Test getting only registered classes (not config entries)"""
        @register
        class ClassA:
            pass

        @register
        class ClassB:
            pass

        # Add non-class config
        global_config['SomeConfig'] = {'param': 10}
        global_config['__shared__'] = {'num_classes': 80}

        registered = get_registered_classes()

        assert 'ClassA' in registered
        assert 'ClassB' in registered
        assert 'SomeConfig' not in registered  # Not a class
        assert '__shared__' not in registered


class TestIntegration:
    """Integration tests combining multiple features"""

    def test_full_model_creation_flow(self):
        """Test creating a full model with inject + shared"""
        @register
        class Backbone:
            __shared__ = ['num_classes']

            def __init__(self, depth=50, num_classes=80):
                self.depth = depth
                self.num_classes = num_classes

        @register
        class Neck:
            def __init__(self, hidden_dim=256):
                self.hidden_dim = hidden_dim

        @register
        class Model:
            __inject__ = ['backbone', 'neck']
            __shared__ = ['num_classes']

            def __init__(self, backbone, neck, num_classes=80):
                self.backbone = backbone
                self.neck = neck
                self.num_classes = num_classes

        # Set up global config
        global_config['__shared__'] = {'num_classes': 91}

        # Create model from config
        model_cfg = {
            'type': 'Model',
            'backbone': {'type': 'Backbone', 'depth': 101},
            'neck': {'type': 'Neck', 'hidden_dim': 512}
        }

        model = create(model_cfg)

        assert isinstance(model, Model)
        assert model.num_classes == 91  # From __shared__
        assert isinstance(model.backbone, Backbone)
        assert model.backbone.depth == 101
        assert model.backbone.num_classes == 91  # From __shared__
        assert isinstance(model.neck, Neck)
        assert model.neck.hidden_dim == 512

    def test_yaml_style_config_workflow(self):
        """Test workflow mimicking YAML config loading"""
        @register
        class ResNet:
            __shared__ = ['num_classes']

            def __init__(self, depth, num_classes=80):
                self.depth = depth
                self.num_classes = num_classes

        @register
        class RTDETRV3:
            __inject__ = ['backbone']
            __shared__ = ['num_classes']

            def __init__(self, backbone, num_classes=80):
                self.backbone = backbone
                self.num_classes = num_classes

        # Simulate YAML config (avoid overwriting registered classes)
        yaml_cfg = {
            '__shared__': {'num_classes': 91},
        }

        # Merge YAML into global_config
        merge_config(yaml_cfg)

        # Manually setup backbone reference (real config would use create())
        # This simulates how a YAML loader would construct the dependency
        global_config['backbone'] = {'type': 'ResNet', 'depth': 50}

        # Create model - backbone will be injected
        model = create('RTDETRV3')

        assert model.num_classes == 91
        assert isinstance(model.backbone, ResNet)
        assert model.backbone.depth == 50
        assert model.backbone.num_classes == 91


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

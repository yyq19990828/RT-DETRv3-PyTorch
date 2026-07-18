"""
Integration tests for component registration

Test that components are properly registered on module import.
"""

import pytest


def test_component_registration_on_import():
    """
    T016: Test that all components are automatically registered when modules are imported

    This validates that @register decorators execute correctly during import time.
    """
    # Import the models package
    from rtdetrv3_pytorch.models import (
        BACKBONE_REGISTRY,
        NECK_REGISTRY,
        TRANSFORMER_REGISTRY,
        HEAD_REGISTRY,
        LOSS_REGISTRY,
        ARCHITECTURE_REGISTRY
    )

    # Expected components in each registry
    expected_components = {
        'BACKBONE_REGISTRY': ['ResNet'],
        'NECK_REGISTRY': ['HybridEncoder'],
        'TRANSFORMER_REGISTRY': ['RTDETRTransformerv3'],
        'HEAD_REGISTRY': ['DINOv3Head', 'PPYOLOEHead'],
        'LOSS_REGISTRY': ['DINOv3Loss'],
        'ARCHITECTURE_REGISTRY': ['RTDETRv3']
    }

    registries = {
        'BACKBONE_REGISTRY': BACKBONE_REGISTRY,
        'NECK_REGISTRY': NECK_REGISTRY,
        'TRANSFORMER_REGISTRY': TRANSFORMER_REGISTRY,
        'HEAD_REGISTRY': HEAD_REGISTRY,
        'LOSS_REGISTRY': LOSS_REGISTRY,
        'ARCHITECTURE_REGISTRY': ARCHITECTURE_REGISTRY
    }

    # Verify each registry has expected components
    for reg_name, expected_comps in expected_components.items():
        registry = registries[reg_name]
        actual_comps = registry.list()

        for comp_name in expected_comps:
            assert comp_name in actual_comps, (
                f"{comp_name} not found in {reg_name}. "
                f"Expected: {expected_comps}, Got: {actual_comps}"
            )

    # Verify total count (should have at least 7 components)
    total_registered = sum(len(reg.list()) for reg in registries.values())
    assert total_registered >= 7, (
        f"Expected at least 7 components registered, found {total_registered}"
    )

    print(f"✅ All {total_registered} components successfully registered on import")


def test_component_metadata_after_import():
    """
    Verify that all registered components have proper metadata after import
    """
    from rtdetrv3_pytorch.models import (
        BACKBONE_REGISTRY,
        NECK_REGISTRY,
        TRANSFORMER_REGISTRY,
        HEAD_REGISTRY,
        LOSS_REGISTRY,
        ARCHITECTURE_REGISTRY
    )

    # Test each component has required attributes
    test_cases = [
        (BACKBONE_REGISTRY, 'ResNet', 'backbone'),
        (NECK_REGISTRY, 'HybridEncoder', 'neck'),
        (TRANSFORMER_REGISTRY, 'RTDETRTransformerv3', 'transformer'),
        (HEAD_REGISTRY, 'DINOv3Head', 'head'),
        (HEAD_REGISTRY, 'PPYOLOEHead', 'head'),
        (LOSS_REGISTRY, 'DINOv3Loss', 'loss'),
        (ARCHITECTURE_REGISTRY, 'RTDETRv3', 'architecture')
    ]

    for registry, comp_name, expected_category in test_cases:
        cls = registry.get(comp_name)

        # Check __category__
        assert hasattr(cls, '__category__'), (
            f"{comp_name} missing __category__ attribute"
        )
        assert cls.__category__ == expected_category, (
            f"{comp_name}.__category__ = '{cls.__category__}', expected '{expected_category}'"
        )

        # Check __inject__ (should exist, even if empty list)
        assert hasattr(cls, '__inject__'), (
            f"{comp_name} missing __inject__ attribute"
        )
        assert isinstance(cls.__inject__, list), (
            f"{comp_name}.__inject__ should be a list"
        )

        # Check __shared__ (should exist, even if empty list)
        assert hasattr(cls, '__shared__'), (
            f"{comp_name} missing __shared__ attribute"
        )
        assert isinstance(cls.__shared__, list), (
            f"{comp_name}.__shared__ should be a list"
        )

    print("✅ All components have correct metadata")


def test_cross_registry_uniqueness():
    """
    Verify that component names are unique across all registries
    (or intentionally duplicated if needed)
    """
    from rtdetrv3_pytorch.models import ALL_REGISTRIES

    all_names = []
    for registry in ALL_REGISTRIES:
        all_names.extend(registry.list())

    # Check if there are any duplicate names across registries
    unique_names = set(all_names)

    # In RT-DETRv3, components should have unique names across registries
    assert len(all_names) == len(unique_names), (
        f"Found duplicate component names across registries. "
        f"Total: {len(all_names)}, Unique: {len(unique_names)}"
    )

    print(f"✅ All {len(unique_names)} component names are unique across registries")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""Test suite for RT-DETRv3 PyTorch implementation."""

import pytest
import sys
import os

# Add project root to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test configuration
pytest_plugins = []

def run_all_tests():
    """Run all tests in the test suite."""
    test_dir = os.path.dirname(__file__)
    return pytest.main([test_dir, '-v', '--tb=short'])

def run_model_tests():
    """Run model unit tests."""
    test_file = os.path.join(os.path.dirname(__file__), 'test_model.py')
    return pytest.main([test_file, '-v'])

def run_transform_tests():
    """Run transform unit tests."""
    test_file = os.path.join(os.path.dirname(__file__), 'test_transforms.py')
    return pytest.main([test_file, '-v'])

def run_integration_tests():
    """Run integration tests."""
    test_file = os.path.join(os.path.dirname(__file__), 'test_integration.py')
    return pytest.main([test_file, '-v'])

if __name__ == '__main__':
    run_all_tests()
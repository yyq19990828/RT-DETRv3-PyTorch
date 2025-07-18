"""Unit tests for data transforms."""

import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.transforms import (
    Compose, ToTensor, Normalize, Resize, RandomResize, RandomHorizontalFlip,
    RandomVerticalFlip, RandomCrop, ColorJitter, GaussianBlur, RandomPhotometricDistort,
    PadToSize, build_transforms
)


class TestBasicTransforms:
    """Test basic transforms."""
    
    def test_to_tensor(self):
        """Test ToTensor transform."""
        transform = ToTensor()
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {'image': image}
        
        result = transform(sample)
        
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 50, 100)  # C, H, W
        assert result['image'].dtype == torch.float32
    
    def test_normalize(self):
        """Test Normalize transform."""
        transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Create sample tensor
        image = torch.randn(3, 224, 224)
        sample = {'image': image}
        
        result = transform(sample)
        
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 224, 224)
        
        # Check normalization was applied
        assert not torch.allclose(result['image'], image)
    
    def test_resize(self):
        """Test Resize transform."""
        transform = Resize((224, 224))
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'orig_size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30], [50, 20, 80, 40]], dtype=torch.float32)
        }
        
        result = transform(sample)
        
        assert result['image'].size == (224, 224)
        assert result['size'].tolist() == [224, 224]
        
        # Check boxes are scaled correctly
        assert result['boxes'].shape == (2, 4)
        assert torch.all(result['boxes'] >= 0)
    
    def test_resize_with_max_size(self):
        """Test Resize with max_size constraint."""
        transform = Resize(800, max_size=1333)
        
        # Create sample PIL image
        image = Image.new('RGB', (2000, 1000), color='red')
        sample = {
            'image': image,
            'orig_size': torch.tensor([1000, 2000]),
            'boxes': torch.tensor([[100, 100, 400, 300]], dtype=torch.float32)
        }
        
        result = transform(sample)
        
        # Check that max_size constraint is respected
        assert max(result['image'].size) <= 1333
        assert result['boxes'].shape == (1, 4)
    
    def test_random_resize(self):
        """Test RandomResize transform."""
        transform = RandomResize([480, 512, 544, 576, 608, 640])
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'orig_size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30]], dtype=torch.float32)
        }
        
        result = transform(sample)
        
        # Check that one of the target sizes was used
        assert result['image'].size[0] in [480, 512, 544, 576, 608, 640] or \
               result['image'].size[1] in [480, 512, 544, 576, 608, 640]
        assert result['boxes'].shape == (1, 4)


class TestFlipTransforms:
    """Test flip transforms."""
    
    def test_random_horizontal_flip(self):
        """Test RandomHorizontalFlip transform."""
        transform = RandomHorizontalFlip(prob=1.0)  # Always flip
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30]], dtype=torch.float32)
        }
        
        result = transform(sample)
        
        # Check that box coordinates are flipped
        expected_x1 = 100 - 40  # width - x2
        expected_x2 = 100 - 10  # width - x1
        assert torch.allclose(result['boxes'][0, [0, 2]], torch.tensor([expected_x1, expected_x2]))
        assert torch.allclose(result['boxes'][0, [1, 3]], torch.tensor([10, 30]))  # y unchanged
    
    def test_random_vertical_flip(self):
        """Test RandomVerticalFlip transform."""
        transform = RandomVerticalFlip(prob=1.0)  # Always flip
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30]], dtype=torch.float32)
        }
        
        result = transform(sample)
        
        # Check that box coordinates are flipped
        expected_y1 = 50 - 30  # height - y2
        expected_y2 = 50 - 10  # height - y1
        assert torch.allclose(result['boxes'][0, [1, 3]], torch.tensor([expected_y1, expected_y2]))
        assert torch.allclose(result['boxes'][0, [0, 2]], torch.tensor([10, 40]))  # x unchanged
    
    def test_random_horizontal_flip_no_flip(self):
        """Test RandomHorizontalFlip with no flip."""
        transform = RandomHorizontalFlip(prob=0.0)  # Never flip
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30]], dtype=torch.float32)
        }
        
        result = transform(sample)
        
        # Check that boxes are unchanged
        assert torch.allclose(result['boxes'], sample['boxes'])


class TestCropTransforms:
    """Test crop transforms."""
    
    def test_random_crop(self):
        """Test RandomCrop transform."""
        transform = RandomCrop((32, 32))
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30]], dtype=torch.float32)
        }
        
        result = transform(sample)
        
        # Check crop size
        assert result['image'].size == (32, 32)
        assert result['size'].tolist() == [32, 32]
        
        # Check that boxes are clipped to new size
        assert torch.all(result['boxes'][:, [0, 2]] >= 0)
        assert torch.all(result['boxes'][:, [1, 3]] >= 0)
        assert torch.all(result['boxes'][:, [0, 2]] <= 32)
        assert torch.all(result['boxes'][:, [1, 3]] <= 32)
    
    def test_random_crop_with_padding(self):
        """Test RandomCrop with padding."""
        transform = RandomCrop((200, 200), pad_if_needed=True)
        
        # Create sample PIL image smaller than crop size
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30]], dtype=torch.float32)
        }
        
        result = transform(sample)
        
        # Check that image was padded and then cropped
        assert result['image'].size == (200, 200)
        assert result['size'].tolist() == [200, 200]


class TestColorTransforms:
    """Test color transforms."""
    
    def test_color_jitter(self):
        """Test ColorJitter transform."""
        transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {'image': image}
        
        result = transform(sample)
        
        # Check that image is still PIL Image
        assert isinstance(result['image'], Image.Image)
        assert result['image'].size == (100, 50)
    
    def test_gaussian_blur(self):
        """Test GaussianBlur transform."""
        transform = GaussianBlur(radius=1.0, prob=1.0)  # Always apply
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {'image': image}
        
        result = transform(sample)
        
        # Check that image is still PIL Image
        assert isinstance(result['image'], Image.Image)
        assert result['image'].size == (100, 50)
    
    def test_random_photometric_distort(self):
        """Test RandomPhotometricDistort transform."""
        transform = RandomPhotometricDistort(
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18
        )
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {'image': image}
        
        result = transform(sample)
        
        # Check that image is still PIL Image or tensor
        assert isinstance(result['image'], (Image.Image, torch.Tensor))


class TestPadTransforms:
    """Test pad transforms."""
    
    def test_pad_to_size(self):
        """Test PadToSize transform."""
        transform = PadToSize((200, 200))
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {'image': image}
        
        result = transform(sample)
        
        # Check that image is padded to target size
        assert result['image'].size == (200, 200)
        assert result['size'].tolist() == [200, 200]
    
    def test_pad_to_size_no_padding_needed(self):
        """Test PadToSize when no padding is needed."""
        transform = PadToSize((50, 50))
        
        # Create sample PIL image already larger than target
        image = Image.new('RGB', (100, 80), color='red')
        sample = {'image': image}
        
        result = transform(sample)
        
        # Image should remain unchanged
        assert result['image'].size == (100, 80)


class TestCompose:
    """Test Compose transform."""
    
    def test_compose_transforms(self):
        """Test Compose with multiple transforms."""
        transforms = [
            {'type': 'ToTensor'},
            {'type': 'Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            {'type': 'Resize', 'size': [224, 224]}
        ]
        
        compose = Compose(transforms)
        
        # Create sample PIL image
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'orig_size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30]], dtype=torch.float32)
        }
        
        result = compose(sample)
        
        # Check that all transforms were applied
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 224, 224)
        assert result['size'].tolist() == [224, 224]
        assert result['boxes'].shape == (1, 4)
    
    def test_compose_with_invalid_transform(self):
        """Test Compose with invalid transform type."""
        transforms = [
            {'type': 'InvalidTransform'}
        ]
        
        with pytest.raises(ValueError, match="Unknown transform type"):
            Compose(transforms)


class TestBuildTransforms:
    """Test build_transforms function."""
    
    def test_build_transforms_single(self):
        """Test building single transform."""
        transform_cfg = {'type': 'ToTensor'}
        
        compose = build_transforms(transform_cfg)
        
        assert isinstance(compose, Compose)
        assert len(compose.transforms) == 1
        assert isinstance(compose.transforms[0], ToTensor)
    
    def test_build_transforms_list(self):
        """Test building list of transforms."""
        transform_cfg = [
            {'type': 'ToTensor'},
            {'type': 'Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        ]
        
        compose = build_transforms(transform_cfg)
        
        assert isinstance(compose, Compose)
        assert len(compose.transforms) == 2
        assert isinstance(compose.transforms[0], ToTensor)
        assert isinstance(compose.transforms[1], Normalize)
    
    def test_build_transforms_invalid_config(self):
        """Test building transforms with invalid config."""
        transform_cfg = "invalid_config"
        
        with pytest.raises(ValueError, match="Invalid transform configuration"):
            build_transforms(transform_cfg)


class TestTransformRobustness:
    """Test transform robustness and edge cases."""
    
    def test_transforms_with_empty_boxes(self):
        """Test transforms with empty bounding boxes."""
        transform = Resize((224, 224))
        
        # Create sample with empty boxes
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'orig_size': torch.tensor([50, 100]),
            'boxes': torch.empty((0, 4), dtype=torch.float32)
        }
        
        result = transform(sample)
        
        assert result['image'].size == (224, 224)
        assert result['boxes'].shape == (0, 4)
    
    def test_transforms_with_large_boxes(self):
        """Test transforms with boxes outside image bounds."""
        transform = Resize((224, 224))
        
        # Create sample with boxes outside image bounds
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'orig_size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[90, 40, 150, 80]], dtype=torch.float32)  # Outside bounds
        }
        
        result = transform(sample)
        
        assert result['image'].size == (224, 224)
        assert result['boxes'].shape == (1, 4)
        # Boxes should be scaled but coordinates might be outside new image bounds
    
    def test_transforms_consistency(self):
        """Test transform consistency with same random seed."""
        transform = RandomHorizontalFlip(prob=0.5)
        
        # Create sample
        image = Image.new('RGB', (100, 50), color='red')
        sample = {
            'image': image,
            'size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30]], dtype=torch.float32)
        }
        
        # Apply transform with same seed
        import random
        random.seed(42)
        result1 = transform(sample.copy())
        
        random.seed(42)
        result2 = transform(sample.copy())
        
        # Results should be identical
        assert torch.allclose(result1['boxes'], result2['boxes'])
    
    def test_transforms_with_tensor_input(self):
        """Test transforms that can handle tensor input."""
        transform = Resize((224, 224))
        
        # Create sample with tensor image
        image = torch.randn(3, 100, 50)
        sample = {
            'image': image,
            'orig_size': torch.tensor([50, 100]),
            'boxes': torch.tensor([[10, 10, 40, 30]], dtype=torch.float32)
        }
        
        result = transform(sample)
        
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 224, 224)
        assert result['boxes'].shape == (1, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
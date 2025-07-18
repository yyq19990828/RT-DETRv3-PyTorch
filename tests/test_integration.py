"""Integration tests for RT-DETRv3 training and inference pipeline."""

import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tempfile
import shutil
from PIL import Image
import numpy as np
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.config import RTDETRConfig
from core.workspace import create, register
from data.dataset import CocoDetection, build_dataset
from data.transforms import build_transforms
from data.dataloader import build_dataloader, collate_fn
from nn.criterion.rtdetr_criterion import RTDETRCriterion, HungarianMatcher
from zoo.rtdetrv3.rtdetrv3 import RTDETRv3
from solver.trainer import RTDETRTrainer
from solver.evaluator import CocoEvaluator
from optim.optimizer import build_optimizer


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, size=100, image_size=(640, 640), num_classes=80):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create mock image
        image = torch.randn(3, *self.image_size)
        
        # Create mock targets
        num_objects = np.random.randint(1, 6)  # 1-5 objects
        boxes = torch.rand(num_objects, 4)
        boxes[:, 2:] = boxes[:, 2:] * 0.5 + 0.1  # Ensure reasonable box sizes
        labels = torch.randint(1, self.num_classes, (num_objects,))
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(idx),
            'orig_size': torch.tensor(self.image_size),
            'size': torch.tensor(self.image_size)
        }


class TestEndToEndTraining:
    """Test end-to-end training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            'model': {
                'type': 'RTDETRv3',
                'num_classes': 80,
                'backbone': {'type': 'ResNet', 'depth': 50},
                'neck': {'type': 'HybridEncoder', 'hidden_dim': 256},
                'transformer': {'type': 'RTDETRTransformerV3', 'num_queries': 100},
                'head': {'type': 'RTDETRHead', 'num_queries': 100}
            },
            'criterion': {
                'type': 'RTDETRCriterion',
                'num_classes': 80,
                'weight_dict': {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
            },
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.0001,
                'weight_decay': 0.0001
            },
            'train_dataloader': {
                'batch_size': 2,
                'shuffle': True,
                'num_workers': 0,
                'drop_last': True
            },
            'val_dataloader': {
                'batch_size': 2,
                'shuffle': False,
                'num_workers': 0,
                'drop_last': False
            }
        }
    
    def test_model_creation(self, mock_config):
        """Test model creation from configuration."""
        model = RTDETRv3(
            num_classes=mock_config['model']['num_classes'],
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6}
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'neck')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'head')
    
    def test_criterion_creation(self, mock_config):
        """Test criterion creation from configuration."""
        criterion = RTDETRCriterion(
            num_classes=mock_config['criterion']['num_classes'],
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
            weight_dict=mock_config['criterion']['weight_dict'],
            losses=['labels', 'boxes'],
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        assert isinstance(criterion, nn.Module)
        assert hasattr(criterion, 'matcher')
        assert hasattr(criterion, 'weight_dict')
    
    def test_dataloader_creation(self, mock_config):
        """Test dataloader creation."""
        # Create mock dataset
        dataset = MockDataset(size=10, image_size=(640, 640))
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=mock_config['train_dataloader']['batch_size'],
            shuffle=mock_config['train_dataloader']['shuffle'],
            num_workers=mock_config['train_dataloader']['num_workers'],
            collate_fn=collate_fn
        )
        
        # Test batch loading
        batch = next(iter(dataloader))
        
        assert 'images' in batch
        assert 'targets' in batch
        assert batch['images'].shape[0] == mock_config['train_dataloader']['batch_size']
        assert len(batch['targets']) == mock_config['train_dataloader']['batch_size']
    
    def test_forward_pass(self, mock_config):
        """Test forward pass through model."""
        model = RTDETRv3(
            num_classes=mock_config['model']['num_classes'],
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6}
        )
        
        # Create mock input
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 6  # 6 decoder layers
        
        for output in outputs:
            assert 'pred_logits' in output
            assert 'pred_boxes' in output
            assert output['pred_logits'].shape == (batch_size, 100, 80)
            assert output['pred_boxes'].shape == (batch_size, 100, 4)
    
    def test_loss_computation(self, mock_config):
        """Test loss computation."""
        model = RTDETRv3(
            num_classes=mock_config['model']['num_classes'],
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6}
        )
        
        criterion = RTDETRCriterion(
            num_classes=mock_config['criterion']['num_classes'],
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
            weight_dict=mock_config['criterion']['weight_dict'],
            losses=['labels', 'boxes'],
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        # Create mock input and targets
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640)
        targets = [
            {'labels': torch.tensor([1, 2]), 'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])},
            {'labels': torch.tensor([3]), 'boxes': torch.tensor([[0.7, 0.7, 0.3, 0.3]])}
        ]
        
        # Forward pass
        model.train()
        outputs = model(images, targets=targets)
        
        # Compute loss
        losses = criterion(outputs, targets)
        
        assert isinstance(losses, dict)
        assert 'loss_ce' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses
        
        # Check that losses are tensors with gradients
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.requires_grad
    
    def test_training_step(self, mock_config, temp_dir):
        """Test single training step."""
        model = RTDETRv3(
            num_classes=mock_config['model']['num_classes'],
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6}
        )
        
        criterion = RTDETRCriterion(
            num_classes=mock_config['criterion']['num_classes'],
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
            weight_dict=mock_config['criterion']['weight_dict'],
            losses=['labels', 'boxes'],
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=mock_config['optimizer']['lr'],
            weight_decay=mock_config['optimizer']['weight_decay']
        )
        
        # Create mock dataloader
        dataset = MockDataset(size=4, image_size=(640, 640))
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Single training step
        model.train()
        batch = next(iter(dataloader))
        
        images = batch['images']
        targets = batch['targets']
        
        # Forward pass
        outputs = model(images, targets=targets)
        
        # Compute loss
        losses = criterion(outputs, targets)
        total_loss = sum(losses.values())
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        assert torch.isfinite(total_loss)
        
        # Check that gradients were computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"


class TestInferencePipeline:
    """Test inference pipeline."""
    
    def test_inference_mode(self):
        """Test model in inference mode."""
        model = RTDETRv3(
            num_classes=80,
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6}
        )
        
        model.eval()
        
        # Test inference
        batch_size = 1
        images = torch.randn(batch_size, 3, 640, 640)
        
        with torch.no_grad():
            outputs = model(images)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 6
        
        # Check that outputs have correct shape
        for output in outputs:
            assert 'pred_logits' in output
            assert 'pred_boxes' in output
            assert output['pred_logits'].shape == (batch_size, 100, 80)
            assert output['pred_boxes'].shape == (batch_size, 100, 4)
    
    def test_postprocessing(self):
        """Test postprocessing of model outputs."""
        model = RTDETRv3(
            num_classes=80,
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6}
        )
        
        model.eval()
        
        # Test inference
        batch_size = 1
        images = torch.randn(batch_size, 3, 640, 640)
        
        with torch.no_grad():
            outputs = model(images)
        
        # Use last decoder layer output
        pred_logits = outputs[-1]['pred_logits']
        pred_boxes = outputs[-1]['pred_boxes']
        
        # Apply softmax to get probabilities
        probs = torch.softmax(pred_logits, dim=-1)
        scores, labels = probs.max(dim=-1)
        
        # Filter predictions
        score_threshold = 0.5
        valid_mask = scores > score_threshold
        
        # Check that filtering works
        assert valid_mask.shape == (batch_size, 100)
        
        # Apply mask
        filtered_scores = scores[valid_mask]
        filtered_labels = labels[valid_mask]
        filtered_boxes = pred_boxes[valid_mask]
        
        # Check shapes
        assert filtered_scores.shape[0] == filtered_labels.shape[0]
        assert filtered_scores.shape[0] == filtered_boxes.shape[0]
        assert filtered_boxes.shape[1] == 4


class TestModelSaveLoad:
    """Test model save and load functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_save_load(self, temp_dir):
        """Test saving and loading model."""
        model = RTDETRv3(
            num_classes=80,
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6}
        )
        
        # Save model
        model_path = os.path.join(temp_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        
        # Load model
        model_loaded = RTDETRv3(
            num_classes=80,
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6}
        )
        
        model_loaded.load_state_dict(torch.load(model_path))
        
        # Test that loaded model produces same output
        model.eval()
        model_loaded.eval()
        
        images = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            outputs1 = model(images)
            outputs2 = model_loaded(images)
        
        # Check that outputs are identical
        for out1, out2 in zip(outputs1, outputs2):
            assert torch.allclose(out1['pred_logits'], out2['pred_logits'], atol=1e-6)
            assert torch.allclose(out1['pred_boxes'], out2['pred_boxes'], atol=1e-6)
    
    def test_checkpoint_save_load(self, temp_dir):
        """Test saving and loading training checkpoint."""
        model = RTDETRv3(
            num_classes=80,
            backbone_name='ResNet50',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 50, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [512, 1024, 2048], 'hidden_dim': 256},
            transformer_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6},
            head_args={'hidden_dim': 256, 'num_queries': 100, 'num_decoder_layers': 6}
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        
        # Save checkpoint
        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.5
        }
        
        checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        
        # Check checkpoint contents
        assert loaded_checkpoint['epoch'] == 10
        assert 'model_state_dict' in loaded_checkpoint
        assert 'optimizer_state_dict' in loaded_checkpoint
        assert loaded_checkpoint['loss'] == 0.5
        
        # Load states
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])


class TestMemoryUsage:
    """Test memory usage and performance."""
    
    def test_memory_efficient_training(self):
        """Test memory efficient training."""
        model = RTDETRv3(
            num_classes=80,
            backbone_name='ResNet18',  # Use smaller model
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 18, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [128, 256, 512], 'hidden_dim': 128},
            transformer_args={'hidden_dim': 128, 'num_queries': 50, 'num_decoder_layers': 3},
            head_args={'hidden_dim': 128, 'num_queries': 50, 'num_decoder_layers': 3}
        )
        
        # Small batch size
        batch_size = 1
        images = torch.randn(batch_size, 3, 320, 320)  # Smaller input size
        
        # Training step
        model.train()
        outputs = model(images)
        
        # Check that model runs without memory issues
        assert isinstance(outputs, list)
        assert len(outputs) == 3  # 3 decoder layers
        
        # Check output shapes
        for output in outputs:
            assert output['pred_logits'].shape == (batch_size, 50, 80)
            assert output['pred_boxes'].shape == (batch_size, 50, 4)
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        model = RTDETRv3(
            num_classes=80,
            backbone_name='ResNet18',
            neck_name='HybridEncoder',
            transformer_name='RTDETRTransformerV3',
            head_name='RTDETRHead',
            backbone_args={'depth': 18, 'return_idx': [1, 2, 3]},
            neck_args={'in_channels': [128, 256, 512], 'hidden_dim': 128},
            transformer_args={'hidden_dim': 128, 'num_queries': 50, 'num_decoder_layers': 3},
            head_args={'hidden_dim': 128, 'num_queries': 50, 'num_decoder_layers': 3}
        )
        
        criterion = RTDETRCriterion(
            num_classes=80,
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
            weight_dict={'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0},
            losses=['labels', 'boxes'],
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        
        # Simulate gradient accumulation
        model.train()
        optimizer.zero_grad()
        
        accumulation_steps = 2
        total_loss = 0
        
        for step in range(accumulation_steps):
            # Create batch
            images = torch.randn(1, 3, 320, 320)
            targets = [{'labels': torch.tensor([1]), 'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
            
            # Forward pass
            outputs = model(images, targets=targets)
            
            # Compute loss
            losses = criterion(outputs, targets)
            loss = sum(losses.values()) / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item()
        
        # Optimizer step
        optimizer.step()
        
        # Check that gradients were accumulated
        assert total_loss > 0
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
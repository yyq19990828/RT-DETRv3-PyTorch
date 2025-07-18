"""Trainer implementation for RT-DETRv3 PyTorch."""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
from tqdm import tqdm

from ..core.workspace import register
from ..misc.dist_utils import get_world_size, get_rank, is_main_process, synchronize


logger = logging.getLogger(__name__)


@register()
class RTDETRTrainer:
    """RT-DETR Trainer with distributed training support."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device('cuda'),
        output_dir: str = './outputs',
        max_epochs: int = 100,
        print_freq: int = 50,
        checkpoint_freq: int = 10,
        use_amp: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        clip_max_norm: float = 0.1,
        find_unused_parameters: bool = False,
        sync_bn: bool = False,
        **kwargs
    ):
        """Initialize RT-DETR trainer.
        
        Args:
            model: RT-DETR model
            criterion: Loss criterion
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            device: Training device
            output_dir: Output directory for checkpoints and logs
            max_epochs: Maximum number of training epochs
            print_freq: Print frequency for training logs
            checkpoint_freq: Checkpoint save frequency
            use_amp: Whether to use automatic mixed precision
            use_ema: Whether to use exponential moving average
            ema_decay: EMA decay factor
            clip_max_norm: Gradient clipping norm
            find_unused_parameters: Whether to find unused parameters in DDP
            sync_bn: Whether to use synchronized batch normalization
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.print_freq = print_freq
        self.checkpoint_freq = checkpoint_freq
        self.use_amp = use_amp
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.clip_max_norm = clip_max_norm
        self.find_unused_parameters = find_unused_parameters
        self.sync_bn = sync_bn
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup distributed training
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.distributed = self.world_size > 1
        
        # Setup model for distributed training
        self._setup_model()
        
        # Setup mixed precision
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Setup EMA
        if self.use_ema:
            self.ema_model = self._create_ema_model()
        
        # Setup logging
        if is_main_process():
            self.writer = SummaryWriter(os.path.join(output_dir, 'logs'))
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
    
    def _setup_model(self):
        """Setup model for distributed training."""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Synchronized batch normalization
        if self.sync_bn and self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        # Distributed data parallel
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                find_unused_parameters=self.find_unused_parameters
            )
    
    def _create_ema_model(self) -> nn.Module:
        """Create EMA model."""
        ema_model = type(self.model.module if self.distributed else self.model)(
            **self.model.module.__dict__ if self.distributed else self.model.__dict__
        )
        ema_model = ema_model.to(self.device)
        ema_model.eval()
        
        # Initialize EMA weights
        for ema_param, param in zip(ema_model.parameters(), self.model.parameters()):
            ema_param.data.copy_(param.data)
        
        return ema_model
    
    def _update_ema(self):
        """Update EMA model."""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        # Initialize metrics
        epoch_metrics = {
            'loss': 0.0,
            'lr': 0.0,
            'samples': 0,
            'time': 0.0
        }
        
        # Training loop
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            images = batch['images'].to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in batch['targets']]
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    loss = sum(loss_dict.values())
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = sum(loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                
                self.optimizer.step()
            
            # Update EMA
            self._update_ema()
            
            # Update metrics
            batch_size = images.size(0)
            epoch_metrics['loss'] += loss.item() * batch_size
            epoch_metrics['samples'] += batch_size
            
            # Log training step
            if batch_idx % self.print_freq == 0 and is_main_process():
                current_lr = self.optimizer.param_groups[0]['lr']
                
                logger.info(
                    f'Epoch [{self.epoch}/{self.max_epochs}] '
                    f'Step [{batch_idx}/{len(dataloader)}] '
                    f'Loss: {loss.item():.4f} '
                    f'LR: {current_lr:.6f}'
                )
                
                # Tensorboard logging
                self.writer.add_scalar('train/loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
                
                # Log individual losses
                for loss_name, loss_value in loss_dict.items():
                    self.writer.add_scalar(f'train/{loss_name}', loss_value.item(), self.global_step)
            
            self.global_step += 1
        
        # Calculate epoch metrics
        epoch_metrics['time'] = time.time() - start_time
        epoch_metrics['loss'] /= epoch_metrics['samples']
        epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return epoch_metrics
    
    def validate(self, dataloader: DataLoader, evaluator) -> Dict[str, float]:
        """Validate model.
        
        Args:
            dataloader: Validation data loader
            evaluator: Evaluation module
            
        Returns:
            Validation metrics
        """
        model = self.ema_model if self.use_ema else self.model
        model.eval()
        
        # Initialize evaluator
        evaluator.reset()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation', disable=not is_main_process()):
                # Move data to device
                images = batch['images'].to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in batch['targets']]
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                # Update evaluator
                evaluator.update(outputs, targets)
        
        # Synchronize across processes
        if self.distributed:
            synchronize()
        
        # Compute metrics
        val_metrics = evaluator.compute()
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, filename: str = None):
        """Save checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best checkpoint
            filename: Checkpoint filename
        """
        if not is_main_process():
            return
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        # Add scheduler state
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Add EMA model state
        if self.use_ema:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
        
        # Add scaler state
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, 'best.pth')
            torch.save(checkpoint, best_path)
        
        logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            resume_training: Whether to resume training state
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model_state_dict = checkpoint['model_state_dict']
        if self.distributed:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        
        if resume_training:
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            # Load EMA model state
            if self.use_ema and 'ema_model_state_dict' in checkpoint:
                self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            
            # Load scaler state
            if self.use_amp and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load training state
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_metric = checkpoint['best_metric']
            self.train_metrics = checkpoint.get('train_metrics', {})
            self.val_metrics = checkpoint.get('val_metrics', {})
        
        logger.info(f'Checkpoint loaded: {checkpoint_path}')
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None, 
              evaluator=None):
        """Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            evaluator: Evaluation module
        """
        logger.info(f'Starting training for {self.max_epochs} epochs...')
        
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            self.train_metrics = train_metrics
            
            # Log training metrics
            if is_main_process():
                logger.info(
                    f'Epoch [{epoch}/{self.max_epochs}] '
                    f'Loss: {train_metrics["loss"]:.4f} '
                    f'LR: {train_metrics["lr"]:.6f} '
                    f'Time: {train_metrics["time"]:.2f}s'
                )
                
                # Tensorboard logging
                self.writer.add_scalar('train/loss_epoch', train_metrics['loss'], epoch)
                self.writer.add_scalar('train/lr_epoch', train_metrics['lr'], epoch)
            
            # Validation
            if val_dataloader is not None and evaluator is not None:
                val_metrics = self.validate(val_dataloader, evaluator)
                self.val_metrics = val_metrics
                
                # Log validation metrics
                if is_main_process():
                    logger.info(f'Validation metrics: {val_metrics}')
                    
                    # Tensorboard logging
                    for metric_name, metric_value in val_metrics.items():
                        self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)
                
                # Check if best model
                current_metric = val_metrics.get('mAP', 0.0)
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
            else:
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_freq == 0 or epoch == self.max_epochs - 1:
                self.save_checkpoint(epoch, is_best)
        
        logger.info('Training completed!')
        
        if is_main_process():
            self.writer.close()


def build_trainer(trainer_cfg: Dict[str, Any], model: nn.Module, criterion: nn.Module,
                  optimizer: optim.Optimizer, lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> RTDETRTrainer:
    """Build trainer from configuration.
    
    Args:
        trainer_cfg: Trainer configuration
        model: Model to train
        criterion: Loss criterion
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        
    Returns:
        Configured trainer
    """
    return RTDETRTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        **trainer_cfg
    )
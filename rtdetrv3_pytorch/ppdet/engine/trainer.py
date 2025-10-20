# Copyright (c) 2025 RT-DETRv3 PyTorch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from PaddlePaddle RT-DETRv3
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

"""
Trainer for RT-DETRv3 PyTorch.

This module provides training utilities including:
- Distributed Data Parallel (DDP) training
- Mixed precision training with torch.cuda.amp
- Checkpointing and resuming
- Learning rate scheduling
- Logging and metrics tracking
"""

import os
import time
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler, autocast

from ..utils.logger import setup_logger

logger = setup_logger('rtdetrv3.engine')


class Trainer:
    """
    Trainer for RT-DETRv3.

    Handles training loop with support for:
    - Single-GPU and multi-GPU (DDP) training
    - Mixed precision training
    - Checkpointing and resuming
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        cfg: Optional[Dict] = None,
        val_loader: Optional[DataLoader] = None,
        evaluator: Optional[object] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_fn: Loss function (can be None if model returns loss)
            cfg: Configuration dictionary with training parameters
            val_loader: Validation data loader (optional)
            evaluator: Evaluator for validation (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.val_loader = val_loader
        self.evaluator = evaluator

        # Default config
        if cfg is None:
            cfg = {}
        self.cfg = cfg

        # Training parameters
        self.epochs = cfg.get('epochs', 72)
        self.start_epoch = 0
        self.save_dir = cfg.get('save_dir', './output')
        self.save_interval = cfg.get('save_interval', 1)
        self.log_interval = cfg.get('log_interval', 50)
        self.val_interval = cfg.get('val_interval', 1)
        self.grad_clip = cfg.get('grad_clip', 0.1)

        # Mixed precision
        self.use_amp = cfg.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None

        # Distributed training
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.is_main_process = (self.rank == 0)

        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                model,
                device_ids=[self.rank],
                find_unused_parameters=cfg.get('find_unused_parameters', False)
            )

        # Create save directory
        if self.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)

        # Training status
        self.global_step = 0
        self.best_metric = 0.0

    def train(self):
        """
        Main training loop.
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Training on {self.world_size} GPU(s)")

        for epoch in range(self.start_epoch, self.epochs):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            train_stats = self.train_one_epoch(epoch)

            # Validation
            if self.val_loader is not None and (epoch + 1) % self.val_interval == 0:
                val_stats = self.validate(epoch)

                # Save best model
                if self.evaluator is not None and self.is_main_process:
                    metric = val_stats.get('mAP', 0.0)
                    if metric > self.best_metric:
                        self.best_metric = metric
                        self.save_checkpoint(
                            epoch,
                            os.path.join(self.save_dir, 'best_model.pth'),
                            is_best=True
                        )
                        logger.info(f"Saved best model with mAP: {metric:.4f}")

            # Save checkpoint
            if self.is_main_process and (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch, os.path.join(self.save_dir, f'epoch_{epoch + 1}.pth'))

        logger.info("Training completed")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training statistics
        """
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        epoch_start = time.time()
        batch_time = 0.0
        data_time = 0.0

        end = time.time()
        for batch_idx, batch in enumerate(self.train_loader):
            # Measure data loading time
            data_time = time.time() - end

            # Move data to GPU
            if isinstance(batch, dict):
                batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [x.cuda(non_blocking=True) if isinstance(x, torch.Tensor) else x
                        for x in batch]

            # Forward pass
            with autocast(enabled=self.use_amp):
                if self.loss_fn is not None:
                    # Separate forward and loss computation
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs, batch)
                else:
                    # Model returns loss directly
                    outputs = self.model(batch)
                    if isinstance(outputs, dict):
                        loss = outputs['loss']
                    else:
                        loss = outputs

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Statistics
            batch_size = self._get_batch_size(batch)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1

            # Measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            # Logging
            if self.is_main_process and (batch_idx + 1) % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                avg_loss = total_loss / total_samples

                logger.info(
                    f"Epoch [{epoch + 1}/{self.epochs}] "
                    f"Step [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Avg Loss: {avg_loss:.4f} "
                    f"LR: {lr:.6f} "
                    f"Time: {batch_time:.3f}s "
                    f"Data: {data_time:.3f}s"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        if self.is_main_process:
            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}] completed in {epoch_time:.2f}s, "
                f"Average loss: {avg_loss:.4f}"
            )

        return {
            'loss': avg_loss,
            'epoch_time': epoch_time
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation statistics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        if self.evaluator is not None:
            self.evaluator.reset()

        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(self.val_loader):
            # Move data to GPU
            if isinstance(batch, dict):
                batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [x.cuda(non_blocking=True) if isinstance(x, torch.Tensor) else x
                        for x in batch]

            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(batch)

                if self.loss_fn is not None:
                    loss = self.loss_fn(outputs, batch)
                else:
                    if isinstance(outputs, dict):
                        loss = outputs.get('loss', torch.tensor(0.0))
                    else:
                        loss = torch.tensor(0.0)

            # Update evaluator
            if self.evaluator is not None:
                self.evaluator.update(outputs, batch)

            # Statistics
            batch_size = self._get_batch_size(batch)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Compute metrics
        val_stats = {'val_loss': avg_loss}
        if self.evaluator is not None:
            metrics = self.evaluator.compute()
            val_stats.update(metrics)

        if self.is_main_process:
            logger.info(f"Validation at epoch {epoch + 1}: {val_stats}")

        return val_stats

    def save_checkpoint(
        self,
        epoch: int,
        save_path: str,
        is_best: bool = False
    ):
        """
        Save checkpoint.

        Args:
            epoch: Current epoch number
            save_path: Path to save checkpoint
            is_best: Whether this is the best model
        """
        # Get model state dict (unwrap DDP if needed)
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'global_step': self.global_step
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            resume_training: Whether to resume training state (optimizer, scheduler, etc.)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        if resume_training:
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load scaler state
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # Load training state
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_metric = checkpoint.get('best_metric', 0.0)
            self.global_step = checkpoint.get('global_step', 0)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        if resume_training:
            logger.info(f"Resuming from epoch {self.start_epoch}")

    def _get_batch_size(self, batch) -> int:
        """
        Get batch size from batch data.

        Args:
            batch: Batch data (dict, list, tuple, or tensor)

        Returns:
            Batch size
        """
        if isinstance(batch, dict):
            # Get batch size from first tensor in dict
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    return v.size(0)
        elif isinstance(batch, (list, tuple)):
            # Get batch size from first tensor in list/tuple
            for item in batch:
                if isinstance(item, torch.Tensor):
                    return item.size(0)
        elif isinstance(batch, torch.Tensor):
            return batch.size(0)

        return 1

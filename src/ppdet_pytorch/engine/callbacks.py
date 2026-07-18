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
Callbacks for training monitoring and checkpointing.

Migrated from PaddlePaddle to maintain API compatibility.
"""

import os
import datetime
from typing import Dict, List, Optional
import torch
import torch.distributed as dist

from ..utils.logger import setup_logger
from ..utils.checkpoint import save_checkpoint

logger = setup_logger('rtdetrv3.engine.callbacks')

__all__ = [
    'Callback', 'ComposeCallback', 'LogPrinter', 'Checkpointer',
    'LearningRateLogger', 'BestModelSaver'
]


class Callback:
    """
    Base callback class. All callbacks should inherit from this class.

    Compatible with Paddle's callback interface.
    """

    def __init__(self, trainer):
        """
        Args:
            trainer: Trainer instance
        """
        self.trainer = trainer

        # Support log_ranks for distributed training
        log_ranks = getattr(trainer.cfg, 'log_ranks', '0') if hasattr(trainer, 'cfg') else '0'
        if isinstance(log_ranks, str):
            self.log_ranks = [int(i) for i in log_ranks.split(',')]
        elif isinstance(log_ranks, int):
            self.log_ranks = [log_ranks]
        else:
            self.log_ranks = [0]

    def on_step_begin(self, status: Dict):
        """Called at the beginning of each training step"""
        pass

    def on_step_end(self, status: Dict):
        """Called at the end of each training step"""
        pass

    def on_epoch_begin(self, status: Dict):
        """Called at the beginning of each epoch"""
        pass

    def on_epoch_end(self, status: Dict):
        """Called at the end of each epoch"""
        pass

    def on_train_begin(self, status: Dict):
        """Called at the beginning of training"""
        pass

    def on_train_end(self, status: Dict):
        """Called at the end of training"""
        pass

    def _is_log_rank(self):
        """Check if current rank should log"""
        if not dist.is_initialized():
            return True
        return dist.get_rank() in self.log_ranks


class ComposeCallback:
    """
    Compose multiple callbacks.

    Args:
        callbacks: List of callback instances
    """

    def __init__(self, callbacks: List[Callback]):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(c, Callback), \
                f"callback should be subclass of Callback, but got {type(c)}"
        self._callbacks = callbacks

    def on_step_begin(self, status: Dict):
        for c in self._callbacks:
            c.on_step_begin(status)

    def on_step_end(self, status: Dict):
        for c in self._callbacks:
            c.on_step_end(status)

    def on_epoch_begin(self, status: Dict):
        for c in self._callbacks:
            c.on_epoch_begin(status)

    def on_epoch_end(self, status: Dict):
        for c in self._callbacks:
            c.on_epoch_end(status)

    def on_train_begin(self, status: Dict):
        for c in self._callbacks:
            c.on_train_begin(status)

    def on_train_end(self, status: Dict):
        for c in self._callbacks:
            c.on_train_end(status)


class LogPrinter(Callback):
    """
    Callback for printing training logs.

    Compatible with Paddle's LogPrinter output format.
    """

    def __init__(self, trainer):
        super(LogPrinter, self).__init__(trainer)
        self.log_iter = getattr(trainer, 'log_interval', 50)
        self._batch_time_total = 0.0
        self._batch_time_count = 0

    def on_train_begin(self, status: Dict):
        """Reset ETA timing when a new training run starts."""
        self._batch_time_total = 0.0
        self._batch_time_count = 0

    def on_step_end(self, status: Dict):
        """Print training logs at specified intervals"""
        if not self._is_log_rank():
            return

        mode = status.get('mode', 'train')

        if mode == 'train':
            epoch_id = status.get('epoch_id', 0)
            step_id = status.get('step_id', 0)
            steps_per_epoch = status.get('steps_per_epoch', 1)
            batch_time = float(status.get('batch_time', 0))
            if batch_time > 0:
                self._batch_time_total += batch_time
                self._batch_time_count += 1

            if step_id % self.log_iter == 0:
                # Calculate ETA
                total_epochs = getattr(self.trainer, 'end_epoch', 72)
                eta_steps = (total_epochs - epoch_id) * steps_per_epoch - step_id
                average_batch_time = (
                    self._batch_time_total / self._batch_time_count
                    if self._batch_time_count else 0.0)
                eta_sec = eta_steps * average_batch_time
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

                # Get metrics
                loss = status.get('loss', 0)
                learning_rate = status.get('learning_rate', 0)
                data_time = status.get('data_time', 0)

                # IPS (images per second)
                batch_size = status.get('batch_size', 1)
                ips = batch_size / batch_time if batch_time > 0 else 0

                # Memory info
                max_mem_str = ""
                if torch.cuda.is_available():
                    max_mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 2)
                    max_mem_allocated = torch.cuda.max_memory_allocated() // (1024 ** 2)
                    max_mem_str = f", max_mem_reserved: {max_mem_reserved} MB, max_mem_allocated: {max_mem_allocated} MB"

                # Format log message (compatible with Paddle format)
                space_fmt = ':' + str(len(str(steps_per_epoch))) + 'd'
                fmt = (
                    f"Epoch: [{epoch_id}] "
                    f"[{step_id:{space_fmt[1:]}}/{steps_per_epoch}] "
                    f"learning_rate: {learning_rate:.6f}, "
                    f"loss: {loss:.4f}, "
                    f"eta: {eta_str}, "
                    f"batch_cost: {batch_time:.4f}s, "
                    f"data_cost: {data_time:.4f}s, "
                    f"ips: {ips:.4f} images/s"
                    f"{max_mem_str}"
                )
                logger.info(fmt)

        elif mode == 'eval':
            step_id = status.get('step_id', 0)
            if step_id % 100 == 0:
                logger.info(f"Eval iter: {step_id}")

    def on_epoch_end(self, status: Dict):
        """Print epoch summary"""
        if not self._is_log_rank():
            return

        mode = status.get('mode', 'train')

        if mode == 'eval':
            sample_num = status.get('sample_num', 0)
            cost_time = status.get('cost_time', 1)
            fps = sample_num / cost_time if cost_time > 0 else 0
            logger.info(f'Total sample number: {sample_num}, average FPS: {fps:.2f}')


class Checkpointer(Callback):
    """
    Callback for saving checkpoints during training.

    Compatible with Paddle's Checkpointer behavior.
    """

    def __init__(self, trainer, save_interval: int = 1):
        """
        Args:
            trainer: Trainer instance
            save_interval: Save checkpoint every N epochs
        """
        super(Checkpointer, self).__init__(trainer)
        self.save_interval = int(save_interval)
        if self.save_interval < 1:
            raise ValueError("save_interval must be at least 1")
        self.save_dir = getattr(trainer, 'save_dir', './output')
        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, status: Dict):
        """Save checkpoint at epoch end"""
        mode = status.get('mode', 'train')
        if mode != 'train':
            return

        epoch_id = status.get('epoch_id', 0)
        total_epochs = getattr(self.trainer, 'end_epoch', 72)

        # Save at specified intervals or at the last epoch
        if (epoch_id + 1) % self.save_interval == 0 or epoch_id == total_epochs - 1:
            save_name = f"epoch_{epoch_id + 1}" if epoch_id != total_epochs - 1 else "model_final"
            save_path = os.path.join(self.save_dir, f"{save_name}.pth")

            saved = save_checkpoint(
                model=self.trainer.model,
                optimizer=self.trainer.optimizer,
                epoch=epoch_id + 1,
                iteration=getattr(self.trainer, 'global_step', 0),
                save_path=save_path,
                config=self.trainer._convert_cfg_to_dict(self.trainer.cfg),
                scheduler=getattr(self.trainer, 'lr', None),
                scaler=getattr(self.trainer, 'scaler', None),
                ema=(self.trainer.ema
                     if getattr(self.trainer, 'use_ema', False) else None),
                sampler_epoch=epoch_id + 1,
                gather_distributed_rng=True,
                loss=status.get('loss', 0),
            )
            if saved:
                logger.info(f"Saved checkpoint: {save_path}")


class BestModelSaver(Callback):
    """
    Callback for saving the best model based on validation metric.

    Args:
        trainer: Trainer instance
        metric_name: Name of metric to monitor (e.g., 'mAP', 'loss')
        mode: 'max' or 'min' (whether higher or lower is better)
    """

    def __init__(self, trainer, metric_name: str = 'mAP', mode: str = 'max'):
        super(BestModelSaver, self).__init__(trainer)
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = -float('inf') if mode == 'max' else float('inf')
        self.save_dir = getattr(trainer, 'save_dir', './output')
        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, status: Dict):
        """Check and save best model"""
        # Only save on rank 0 in distributed training
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        mode = status.get('mode', 'train')
        if mode != 'eval':
            return

        # Get metric value
        metric_value = status.get(self.metric_name, None)
        if metric_value is None:
            return

        # Check if this is the best model
        is_best = False
        if self.mode == 'max':
            if metric_value > self.best_metric:
                is_best = True
                self.best_metric = metric_value
        else:  # mode == 'min'
            if metric_value < self.best_metric:
                is_best = True
                self.best_metric = metric_value

        if is_best:
            save_path = os.path.join(self.save_dir, "best_model.pth")

            save_checkpoint(
                model=self.trainer.model,
                optimizer=self.trainer.optimizer,
                epoch=status.get('epoch_id', 0) + 1,
                iteration=getattr(self.trainer, 'global_step', 0),
                save_path=save_path,
                config=self.trainer._convert_cfg_to_dict(self.trainer.cfg),
                best_metric=self.best_metric,
                scheduler=getattr(self.trainer, 'lr', None),
                scaler=getattr(self.trainer, 'scaler', None),
                ema=(self.trainer.ema
                     if getattr(self.trainer, 'use_ema', False) else None),
                sampler_epoch=status.get('epoch_id', 0) + 1,
            )
            logger.info(f"Saved best model with {self.metric_name}: {self.best_metric:.4f}")


class LearningRateLogger(Callback):
    """
    Callback for logging learning rate changes.

    Useful for debugging LR schedules.
    """

    def __init__(self, trainer):
        super(LearningRateLogger, self).__init__(trainer)
        self.last_lr = None

    def on_step_end(self, status: Dict):
        """Log learning rate if it changed"""
        if not self._is_log_rank():
            return

        mode = status.get('mode', 'train')
        if mode != 'train':
            return

        current_lr = status.get('learning_rate', 0)

        # Log if LR changed significantly (more than 1e-8)
        if self.last_lr is None or abs(current_lr - self.last_lr) > 1e-8:
            step_id = status.get('step_id', 0)
            epoch_id = status.get('epoch_id', 0)
            logger.debug(f"Epoch {epoch_id}, Step {step_id}: LR = {current_lr:.8f}")
            self.last_lr = current_lr

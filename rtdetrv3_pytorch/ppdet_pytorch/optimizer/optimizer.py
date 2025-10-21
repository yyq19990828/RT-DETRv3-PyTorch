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
Optimizer and Learning Rate Scheduler for RT-DETRv3.

This module provides:
- AdamW optimizer with parameter groups
- MultiStepLR scheduler with linear warmup
- Gradient clipping utilities
"""

from typing import Dict, List, Optional, Union
import re
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR


class LinearWarmupScheduler(LRScheduler):
    """
    Learning rate scheduler with linear warmup.

    This scheduler increases the learning rate linearly from start_factor * base_lr
    to base_lr over warmup_steps iterations.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup iterations
        start_factor: Initial learning rate factor (lr = start_factor * base_lr at step 0)
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        start_factor: float = 0.001,
        last_epoch: int = -1
    ):
        """Initialize warmup scheduler."""
        self.warmup_steps = warmup_steps
        self.start_factor = start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch >= self.warmup_steps:
            # After warmup, return base learning rates
            return [group['lr'] for group in self.optimizer.param_groups]

        # During warmup, linearly interpolate from start_factor to 1.0
        alpha = self.last_epoch / self.warmup_steps
        factor = self.start_factor * (1 - alpha) + alpha

        return [base_lr * factor for base_lr in self.base_lrs]


class MultiStepLRWithWarmup(LRScheduler):
    """
    Multi-step learning rate scheduler with linear warmup.

    This scheduler combines linear warmup with multi-step decay. During warmup,
    the learning rate increases linearly from start_factor * base_lr to base_lr.
    After warmup, it decays by gamma at specified milestones.

    Args:
        optimizer: Wrapped optimizer
        milestones: List of epoch indices when to decay learning rate
        gamma: Multiplicative factor of learning rate decay
        warmup_steps: Number of warmup iterations (default: 1000)
        start_factor: Initial learning rate factor during warmup (default: 0.001)
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_steps: int = 1000,
        start_factor: float = 0.001,
        last_epoch: int = -1
    ):
        """Initialize multi-step LR scheduler with warmup."""
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.start_factor = start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # During warmup phase
            alpha = self.last_epoch / self.warmup_steps
            factor = self.start_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # After warmup, apply multi-step decay
            # Count how many milestones have been passed
            decay_count = sum(1 for milestone in self.milestones if self.last_epoch >= milestone)
            decay_factor = self.gamma ** decay_count
            return [base_lr * decay_factor for base_lr in self.base_lrs]


def build_optimizer(
    model: nn.Module,
    optimizer_cfg: Optional[Dict] = None
) -> Optimizer:
    """
    Build optimizer from configuration.

    Supports parameter groups for different learning rates on different model parts
    (e.g., backbone vs. decoder).

    Args:
        model: Model to optimize
        optimizer_cfg: Optimizer configuration dict with keys:
            - type: Optimizer type (e.g., 'AdamW', 'SGD')
            - lr: Base learning rate
            - weight_decay: Weight decay coefficient
            - param_groups: Optional list of parameter group configs

    Returns:
        Configured optimizer

    Example config:
        {
            'type': 'AdamW',
            'lr': 0.0001,
            'weight_decay': 0.0001,
            'param_groups': [
                {
                    'params': ['backbone'],
                    'lr': 0.00001  # Lower LR for backbone
                }
            ]
        }
    """
    if optimizer_cfg is None:
        optimizer_cfg = {
            'type': 'AdamW',
            'lr': 0.0001,
            'weight_decay': 0.0001
        }

    # Extract optimizer type and parameters
    optim_cfg = optimizer_cfg.copy()
    optim_type = optim_cfg.pop('type', 'AdamW')
    base_lr = optim_cfg.pop('lr', 0.0001)
    weight_decay = optim_cfg.pop('weight_decay', 0.0001)
    param_groups_cfg = optim_cfg.pop('param_groups', None)

    # Build parameter groups
    if param_groups_cfg is not None:
        # Custom parameter groups
        param_groups = []
        visited_params = set()

        for group_cfg in param_groups_cfg:
            group_params = []
            param_patterns = group_cfg.get('params', [])

            # Match parameters by regex pattern
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Check if parameter matches any pattern
                for pattern in param_patterns:
                    if re.search(pattern, name):
                        group_params.append(param)
                        visited_params.add(name)
                        break

            if len(group_params) > 0:
                # Create parameter group with custom settings
                group = {
                    'params': group_params,
                    'lr': group_cfg.get('lr', base_lr),
                    'weight_decay': group_cfg.get('weight_decay', weight_decay)
                }
                param_groups.append(group)

        # Add remaining parameters to default group
        remaining_params = [
            param for name, param in model.named_parameters()
            if param.requires_grad and name not in visited_params
        ]

        if len(remaining_params) > 0:
            param_groups.append({
                'params': remaining_params,
                'lr': base_lr,
                'weight_decay': weight_decay
            })

        params = param_groups
    else:
        # Single parameter group (all parameters)
        params = [
            {'params': [p for p in model.parameters() if p.requires_grad],
             'lr': base_lr,
             'weight_decay': weight_decay}
        ]

    # Create optimizer
    if optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(params, **optim_cfg)
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam(params, **optim_cfg)
    elif optim_type == 'SGD':
        optimizer = torch.optim.SGD(params, **optim_cfg)
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")

    return optimizer


def build_lr_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: Optional[Dict] = None,
    steps_per_epoch: Optional[int] = None
) -> Optional[LRScheduler]:
    """
    Build learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule
        scheduler_cfg: Scheduler configuration dict with keys:
            - type: Scheduler type ('MultiStepLR', 'CosineAnnealingLR', etc.)
            - milestones: Epoch indices for MultiStepLR (in epochs)
            - gamma: Decay factor for MultiStepLR
            - warmup_steps: Number of warmup iterations
            - start_factor: Initial LR factor during warmup
        steps_per_epoch: Number of iterations per epoch (for converting epoch milestones to iterations)

    Returns:
        Configured scheduler or None

    Example config:
        {
            'type': 'MultiStepLR',
            'milestones': [60],  # Decay at epoch 60
            'gamma': 0.1,
            'warmup_steps': 2000,
            'start_factor': 0.001
        }
    """
    if scheduler_cfg is None:
        return None

    sched_cfg = scheduler_cfg.copy()
    sched_type = sched_cfg.pop('type', 'MultiStepLR')
    warmup_steps = sched_cfg.pop('warmup_steps', 0)
    start_factor = sched_cfg.pop('start_factor', 0.001)

    if sched_type == 'MultiStepLR':
        milestones = sched_cfg.get('milestones', [60])
        gamma = sched_cfg.get('gamma', 0.1)

        # Convert epoch milestones to iteration milestones if needed
        if steps_per_epoch is not None:
            milestones = [m * steps_per_epoch for m in milestones]

        if warmup_steps > 0:
            # Use MultiStepLR with warmup
            scheduler = MultiStepLRWithWarmup(
                optimizer,
                milestones=milestones,
                gamma=gamma,
                warmup_steps=warmup_steps,
                start_factor=start_factor
            )
        else:
            # Standard MultiStepLR without warmup
            scheduler = MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma
            )

    elif sched_type == 'CosineAnnealingLR':
        T_max = sched_cfg.get('T_max', 72)
        if steps_per_epoch is not None:
            T_max = T_max * steps_per_epoch

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=sched_cfg.get('eta_min', 0)
        )

        # Wrap with warmup if needed
        if warmup_steps > 0:
            raise NotImplementedError("Warmup for CosineAnnealingLR not yet implemented")

    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

    return scheduler


def clip_gradients(
    model: nn.Module,
    max_norm: Optional[float] = None,
    max_value: Optional[float] = None
) -> Optional[float]:
    """
    Clip gradients by norm or value.

    Args:
        model: Model whose gradients to clip
        max_norm: Maximum norm for gradient clipping (global norm)
        max_value: Maximum absolute value for gradient clipping

    Returns:
        Total norm of gradients (if max_norm is specified), else None
    """
    parameters = [p for p in model.parameters() if p.grad is not None]

    if max_norm is not None:
        # Clip by global norm
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
    elif max_value is not None:
        # Clip by value
        torch.nn.utils.clip_grad_value_(parameters, max_value)
        return None
    else:
        return None

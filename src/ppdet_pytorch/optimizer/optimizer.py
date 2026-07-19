# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

import copy
import math
import re
import sys

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    SequentialLR,
)

from ppdet_pytorch.core.workspace import register, serializable

from .adamw import build_adamwdl

__all__ = ["LearningRate", "OptimizerBuilder"]

from ppdet_pytorch.utils.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# PyTorch-specific schedulers to match PaddlePaddle's behavior
# ============================================================================


class PiecewiseLRScheduler(LRScheduler):
    """
    Piecewise learning rate scheduler using milestones and gamma values.
    Mimics Paddle's PiecewiseDecay(boundary, value).

    Args:
        optimizer: Wrapped optimizer
        milestones: List of step indices where LR changes
        gamma: List of decay factors for each milestone
        last_epoch: The index of last epoch
    """

    def __init__(
        self, optimizer: Optimizer, milestones: list, gamma: list, last_epoch: int = -1
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on current step and milestones."""
        step = self.last_epoch

        # Find which milestone segment we're in
        # Start with base_lrs (the initial learning rates from optimizer)
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr
            for i, milestone in enumerate(self.milestones):
                if step >= milestone:
                    if i < len(self.gamma):
                        lr = base_lr * self.gamma[i]
                else:
                    break
            lrs.append(lr)

        return lrs


# ============================================================================
# end
# ============================================================================


@serializable
class CosineDecay(object):
    """
    Cosine learning rate decay scheduler for PyTorch.

    Args:
        max_epochs (int): Max epochs for the training process.
        use_warmup (bool): Whether to use warmup. Default: True.
        min_lr_ratio (float): Minimum learning rate ratio. Default: 0.
        last_plateau_epochs (int): Use minimum learning rate in
            the last few epochs. Default: 0.

    Note:
        In PyTorch, base_lr is automatically retrieved from optimizer.param_groups[0]['lr'].
        The scheduler binds to the optimizer and modifies its learning rate directly.
    """

    def __init__(
        self, max_epochs=1000, use_warmup=True, min_lr_ratio=0.0, last_plateau_epochs=0
    ):
        self.max_epochs = max_epochs
        self.use_warmup = use_warmup
        self.min_lr_ratio = min_lr_ratio
        self.last_plateau_epochs = last_plateau_epochs

    def build_scheduler(self, optimizer, step_per_epoch):
        """
        Build cosine annealing scheduler.

        Args:
            optimizer: PyTorch optimizer instance
            step_per_epoch: Steps per epoch

        Returns:
            PyTorch LR scheduler or SequentialLR if last_plateau is used
        """
        base_lr = optimizer.param_groups[0]["lr"]
        max_iters = self.max_epochs * int(step_per_epoch)
        min_lr = base_lr * self.min_lr_ratio
        last_plateau_iters = self.last_plateau_epochs * int(step_per_epoch)

        if last_plateau_iters > 0:
            # Need to combine cosine + constant plateau
            from torch.optim.lr_scheduler import ConstantLR

            main_iters = max_iters - last_plateau_iters
            scheds = [
                CosineAnnealingLR(optimizer, T_max=main_iters, eta_min=min_lr),
                ConstantLR(optimizer, factor=1.0, total_iters=last_plateau_iters),
            ]
            return SequentialLR(optimizer, scheds, milestones=[main_iters])
        else:
            # Simple cosine annealing
            return CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=min_lr)


@serializable
class PiecewiseDecay(object):
    """
    Multi-step learning rate decay scheduler for PyTorch.

    Args:
        gamma (float | list): Decay factor(s). If float, will generate a list
            by dividing by powers of 10. If list, directly specifies decay factors.
        milestones (list): Epoch indices at which to decay learning rate.
        values (list|None): If specified, directly use these LR values at each milestone.
        use_warmup (bool): Whether to use warmup. Default: True.

    Note:
        In PyTorch, base_lr is automatically retrieved from optimizer.param_groups[0]['lr'].
        The scheduler binds to the optimizer and modifies its learning rate directly.
    """

    def __init__(
        self, gamma=[0.1, 0.01], milestones=[8, 11], values=None, use_warmup=True
    ):
        super(PiecewiseDecay, self).__init__()
        if type(gamma) is not list:
            self.gamma = []
            for i in range(len(milestones)):
                self.gamma.append(gamma / 10**i)
        else:
            self.gamma = gamma
        self.milestones = milestones
        self.values = values
        self.use_warmup = use_warmup

    def build_scheduler(self, optimizer, step_per_epoch, step_offset=0):
        """
        Build piecewise decay scheduler.

        Args:
            optimizer: PyTorch optimizer instance
            step_per_epoch: Steps per epoch

        Returns:
            PiecewiseLRScheduler instance
        """
        # Paddle milestones are absolute global steps. SequentialLR starts the
        # decay scheduler at zero after warmup, so remove that offset here.
        milestones = [
            max(0, int(step_per_epoch) * i - int(step_offset)) for i in self.milestones
        ]

        # Prepare gamma factors
        if self.values is not None:
            assert len(self.milestones) + 1 == len(self.values)
            # Convert absolute values to relative gamma factors
            gamma_factors = [
                self.values[i + 1] / self.values[0] for i in range(len(self.milestones))
            ]
        else:
            gamma_factors = self.gamma

        return PiecewiseLRScheduler(
            optimizer, milestones=milestones, gamma=gamma_factors
        )


@serializable
class LinearWarmup(object):
    """
    Linear learning rate warmup for PyTorch schedulers.

    Args:
        steps (int): Warmup steps. Default: 500.
        start_factor (float): Initial learning rate factor. Default: 1/3.
        epochs (int|None): Use epochs as warmup steps. If specified,
            this takes priority over `steps`. Default: None.
        epochs_first (bool): Whether to check epochs before steps. Default: True.

    Note:
        Returns torch.optim.lr_scheduler.LinearLR directly.
    """

    def __init__(self, steps=500, start_factor=1.0 / 3, epochs=None, epochs_first=True):
        super(LinearWarmup, self).__init__()
        self.steps = steps
        self.start_factor = start_factor
        self.epochs = epochs
        self.epochs_first = epochs_first

    def build_scheduler(self, optimizer, step_per_epoch):
        """
        Build linear warmup scheduler.

        Args:
            optimizer: PyTorch optimizer instance
            step_per_epoch: Steps per epoch

        Returns:
            torch.optim.lr_scheduler.LinearLR instance
        """
        if self.epochs_first and self.epochs is not None:
            warmup_steps = self.epochs * step_per_epoch
        else:
            warmup_steps = self.steps
        warmup_steps = max(warmup_steps, 1)

        # PyTorch LinearLR requires start_factor > 0
        start_factor = max(self.start_factor, 1e-6)
        return LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps)


@serializable
class ExpWarmup(object):
    """
    Exponential learning rate warmup for PyTorch schedulers.

    Args:
        steps (int): Warmup steps. Default: 1000.
        epochs (int|None): Use epochs as warmup steps. If specified,
            this takes priority over `steps`. Default: None.
        power (int): Exponential coefficient. Default: 2.

    Note:
        Returns torch.optim.lr_scheduler.PolynomialLR directly.
    """

    def __init__(self, steps=1000, epochs=None, power=2):
        super(ExpWarmup, self).__init__()
        self.steps = steps
        self.epochs = epochs
        self.power = power

    def build_scheduler(self, optimizer, step_per_epoch):
        """
        Build exponential warmup scheduler.

        Args:
            optimizer: PyTorch optimizer instance
            step_per_epoch: Steps per epoch

        Returns:
            torch.optim.lr_scheduler.PolynomialLR instance
        """
        warmup_steps = (
            self.epochs * step_per_epoch if self.epochs is not None else self.steps
        )
        warmup_steps = max(warmup_steps, 1)

        return PolynomialLR(optimizer, total_iters=warmup_steps, power=self.power)


@register
class LearningRate(object):
    """
    Learning Rate configuration for PyTorch.

    Args:
        base_lr (float): Base learning rate. This is kept for backward compatibility
            with PaddlePaddle configs, but in PyTorch the actual learning rate comes
            from the optimizer. If provided, it should match the optimizer's LR.
        schedulers (list): Learning rate schedulers (e.g., [CosineDecay(), LinearWarmup()])

    Note:
        In PyTorch, the optimizer holds the actual learning rate via
        optimizer.param_groups[0]['lr']. The base_lr parameter here is optional
        and only used for config compatibility. All schedulers will read the
        learning rate directly from the optimizer.
    """

    __category__ = "optim"

    def __init__(self, base_lr=0.01, schedulers=[PiecewiseDecay(), LinearWarmup()]):
        super(LearningRate, self).__init__()
        self.base_lr = base_lr  # For config compatibility only
        self.schedulers = []

        schedulers = copy.deepcopy(schedulers)
        for sched in schedulers:
            if isinstance(sched, dict):
                # support dict sched instantiate
                module = sys.modules[__name__]
                type = sched.pop("name")
                scheduler = getattr(module, type)(**sched)
                self.schedulers.append(scheduler)
            else:
                self.schedulers.append(sched)

    def __call__(self, step_per_epoch, optimizer=None):
        """
        Create learning rate scheduler by combining warmup and decay schedulers.

        Args:
            step_per_epoch: Steps per epoch
            optimizer: PyTorch optimizer (required for creating scheduler)

        Returns:
            PyTorch LR scheduler instance (SequentialLR if warmup+decay, otherwise single scheduler)

        Note:
            All schedulers read base_lr from optimizer.param_groups[0]['lr'].
            Each scheduler class has a build_scheduler() method that returns PyTorch native schedulers.
        """
        assert len(self.schedulers) >= 1
        assert optimizer is not None, (
            "optimizer is required to create LR scheduler in PyTorch"
        )

        decay_config = self.schedulers[0]

        if not decay_config.use_warmup:
            # No warmup: directly return the decay scheduler
            return decay_config.build_scheduler(optimizer, step_per_epoch)

        # With warmup: combine warmup + decay using SequentialLR
        assert len(self.schedulers) >= 2, (
            "Warmup config is required when use_warmup=True"
        )
        warmup_config = self.schedulers[1]

        # Build warmup scheduler
        warmup_scheduler = warmup_config.build_scheduler(optimizer, step_per_epoch)

        # Calculate warmup steps for milestone
        if hasattr(warmup_config, "epochs") and warmup_config.epochs is not None:
            warmup_steps = warmup_config.epochs * step_per_epoch
        else:
            warmup_steps = getattr(warmup_config, "steps", 500)
        warmup_steps = max(warmup_steps, 1)

        # Build decay scheduler. Piecewise milestones are absolute global
        # steps in Paddle configs, while cosine progress starts after warmup.
        if isinstance(decay_config, PiecewiseDecay):
            decay_scheduler = decay_config.build_scheduler(
                optimizer,
                step_per_epoch,
                step_offset=warmup_steps,
            )
        else:
            decay_scheduler = decay_config.build_scheduler(
                optimizer,
                step_per_epoch,
            )

        # Combine using SequentialLR
        return SequentialLR(
            optimizer, [warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )


@register
class OptimizerBuilder:
    """
    Build optimizer for PyTorch models.

    Args:
        clip_grad_by_norm (float|None): Gradient clipping by global norm
        clip_grad_by_value (float|None): Gradient clipping by value
        regularizer (dict): Regularization config with 'type' and 'factor'
        optimizer (dict): Optimizer config with 'type' and optimizer-specific params

    Note:
        In PyTorch, learning rate schedulers are bound to optimizers, so the
        base learning rate is set in the optimizer and retrieved by schedulers
        via optimizer.param_groups[0]['lr'].
    """

    __category__ = "optim"

    def __init__(
        self,
        clip_grad_by_norm=None,
        clip_grad_by_value=None,
        regularizer={"type": "L2", "factor": 0.0001},
        optimizer={"type": "Momentum", "momentum": 0.9},
    ):
        self.clip_grad_by_norm = clip_grad_by_norm
        self.clip_grad_by_value = clip_grad_by_value
        self.regularizer = regularizer
        self.optimizer = optimizer

    def __call__(self, learning_rate, model=None):
        """
        Build optimizer instance.

        Args:
            learning_rate (float): Base learning rate (will be set in optimizer)
            model (torch.nn.Module): Model to optimize

        Returns:
            torch.optim.Optimizer: PyTorch optimizer with gradient clipping info
        """
        assert model is not None, "model is required to build optimizer"

        # Configure gradient clipping (stored for use during training)
        if self.clip_grad_by_norm is not None:
            grad_clip = ("norm", self.clip_grad_by_norm)
        elif self.clip_grad_by_value is not None:
            var = abs(self.clip_grad_by_value)
            grad_clip = ("value", var)
        else:
            grad_clip = None

        # Configure weight decay from regularizer
        if self.regularizer and self.regularizer != "None":
            reg_type = self.regularizer["type"] + "Decay"
            reg_factor = self.regularizer["factor"]
            weight_decay = reg_factor if reg_type == "L2Decay" else 0.0
        else:
            weight_decay = 0.0

        # Parse optimizer config
        optim_args = self.optimizer.copy()
        optim_type = optim_args.pop("type")

        # Handle custom AdamWDL optimizer
        if optim_type == "AdamWDL":
            optimizer = build_adamwdl(model, lr=learning_rate, **optim_args)
            if grad_clip is not None:
                optimizer._grad_clip = grad_clip
            return optimizer

        # Add weight_decay for non-AdamW optimizers
        # AdamW handles weight decay differently via optimizer args
        if optim_type != "AdamW":
            optim_args["weight_decay"] = weight_decay

        # Explicit config groups take precedence. Otherwise preserve Paddle's
        # per-parameter learning-rate multipliers as PyTorch parameter groups.
        if "param_groups" in optim_args:
            params = self._build_param_groups(model, optim_args.pop("param_groups"))
        else:
            params = self._build_lr_multiplier_groups(model, learning_rate)

        # Create PyTorch optimizer
        optimizer = self._create_optimizer(
            optim_type, params, learning_rate, optim_args
        )

        # Attach gradient clipping info for training loop
        if grad_clip is not None:
            optimizer._grad_clip = grad_clip

        return optimizer

    def _build_lr_multiplier_groups(self, model, learning_rate):
        """Group trainable parameters by their model-declared LR multiplier."""
        grouped_params: dict[float, list[torch.nn.Parameter]] = {}
        for param in model.parameters():
            if not param.requires_grad:
                continue
            multiplier = float(getattr(param, "_optimizer_lr_multiplier", 1.0))
            if not math.isfinite(multiplier) or multiplier < 0:
                raise ValueError(
                    "Learning-rate multiplier must be finite and non-negative, "
                    "but got {}".format(multiplier)
                )
            grouped_params.setdefault(multiplier, []).append(param)

        if not grouped_params:
            raise ValueError("Optimizer received no trainable parameters")
        if set(grouped_params) == {1.0}:
            return grouped_params[1.0]

        # Keep the default-LR group first so trainer logging continues to
        # report the configured base LR when such parameters exist.
        multipliers = sorted(grouped_params)
        if 1.0 in grouped_params:
            multipliers.remove(1.0)
            multipliers.insert(0, 1.0)
        return [
            {
                "params": grouped_params[multiplier],
                "lr": learning_rate * multiplier,
                "lr_multiplier": multiplier,
            }
            for multiplier in multipliers
        ]

    def _build_param_groups(self, model, param_group_configs):
        """Build parameter groups with different hyperparameters."""
        assert isinstance(param_group_configs, list), "param_groups must be a list"

        param_groups = []
        visited: list[str] = []

        for group_config in param_group_configs:
            assert isinstance(group_config, dict) and "params" in group_config, (
                'Each param group must be a dict with "params" key'
            )
            assert isinstance(group_config["params"], list), (
                'group["params"] must be a list of regex patterns'
            )

            # Match parameters by regex patterns
            matched_params = {}
            for param_name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                for pattern in group_config["params"]:
                    if re.search(pattern, param_name):
                        matched_params[param_name] = param
                        break

            # Create parameter group
            group = group_config.copy()
            group["params"] = list(matched_params.values())
            param_groups.append(group)
            visited.extend(matched_params.keys())

        # Add remaining parameters not matched by any group
        remaining_params = [
            p
            for n, p in model.named_parameters()
            if n not in visited and p.requires_grad
        ]

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(remaining_params) < len(trainable_params):
            param_groups.append({"params": remaining_params})
        elif len(remaining_params) > len(trainable_params):
            raise RuntimeError(
                "Parameter group matching error: some params matched multiple times"
            )

        return param_groups

    def _create_optimizer(self, optim_type, params, learning_rate, optim_args):
        """Create PyTorch optimizer instance."""
        optimizer_map = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
            "Momentum": torch.optim.SGD,  # Momentum is SGD with momentum
        }

        if optim_type not in optimizer_map:
            raise ValueError(f"Unsupported optimizer type: {optim_type}")

        return optimizer_map[optim_type](params, lr=learning_rate, **optim_args)

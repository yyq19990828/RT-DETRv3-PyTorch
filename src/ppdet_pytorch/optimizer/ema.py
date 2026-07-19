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
Exponential Moving Average (EMA) for model parameters.

Maintains a moving average of model parameters during training.
The EMA model typically has better generalization performance.
"""

import math
from typing import List, Optional, Set, Union

import torch
import torch.nn as nn

from ..core.workspace import register
from .utils import get_bn_running_state_names

__all__ = ["ModelEMA"]


@register
class ModelEMA:
    """
    Exponential Weighted Average for Deep Neural Networks.

    Compatible with Paddle's ModelEMA API.

    Args:
        model (nn.Module): Model to apply EMA to
        decay (float): The decay used for updating ema parameter.
            Ema's parameter are updated with the formula:
            `ema_param = decay * ema_param + (1 - decay) * cur_param`.
            Defaults to 0.9998.
        ema_decay_type (str): type in ['threshold', 'normal', 'exponential'],
            'threshold' as default.
        cycle_epoch (int): The epoch of interval to reset ema_param and
            step. Defaults to -1, which means not reset. Its function is to
            add a regular effect to ema, which is set according to experience
            and is effective when the total training epoch is large.
        ema_black_list (set|list|tuple, optional): The custom EMA black_list.
            Blacklist of weight names that will not participate in EMA
            calculation. Default: None.
        ema_filter_no_grad (bool): Whether to filter out parameters that
            don't require gradients. Default: False.
        device (str): Device to store EMA parameters. Default: 'cuda'
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9998,
        ema_decay_type: str = "threshold",
        cycle_epoch: int = -1,
        ema_black_list: Optional[Union[Set, List, tuple]] = None,
        ema_filter_no_grad: bool = False,
        device: str = "cuda",
    ):
        self.step = 0
        self.epoch = 0
        self.decay = decay
        self.ema_decay_type = ema_decay_type
        self.cycle_epoch = cycle_epoch
        self.device = device

        # Build EMA black list (parameters that won't participate in EMA)
        model_state_keys = set(model.state_dict().keys())
        self.ema_black_list = self._match_ema_black_list(
            model_state_keys, ema_black_list
        )

        # Get BN running states
        bn_states_names = get_bn_running_state_names(model)

        # Filter out parameters that don't require gradients
        if ema_filter_no_grad:
            for name, param in model.named_parameters():
                if not param.requires_grad and name not in bn_states_names:
                    self.ema_black_list.add(name)

        # Initialize EMA state dict
        self.state_dict = {}
        for k, v in model.state_dict().items():
            if k in self.ema_black_list:
                # For blacklisted parameters, just copy the reference
                self.state_dict[k] = v.clone().to(device)
            else:
                # For EMA parameters, initialize with zeros
                self.state_dict[k] = torch.zeros_like(
                    v, dtype=torch.float32, device=device
                )

        # Store decay for apply() method
        self._decay = decay

    def reset(self):
        """Reset EMA state (used when cycle_epoch is reached)"""
        self.step = 0
        self.epoch = 0
        for k, v in self.state_dict.items():
            if k not in self.ema_black_list:
                self.state_dict[k] = torch.zeros_like(v)

    def resume(self, state_dict: dict, step: int = 0):
        """
        Resume EMA from saved state dict.

        Args:
            state_dict: EMA state dict to restore
            step: Current training step
        """
        for k, v in state_dict.items():
            if k in self.state_dict:
                # Handle dtype mismatch
                if self.state_dict[k].dtype != v.dtype:
                    v = v.to(self.state_dict[k].dtype)
                self.state_dict[k] = v.to(self.device)

        self.step = step

    def update(self, model: nn.Module):
        """
        Update EMA parameters with current model parameters.

        Args:
            model: Current model with updated parameters
        """
        # Calculate decay based on decay type
        if self.ema_decay_type == "threshold":
            decay = min(self.decay, (1 + self.step) / (10 + self.step))
        elif self.ema_decay_type == "exponential":
            decay = self.decay * (1 - math.exp(-(self.step + 1) / 2000))
        else:  # 'normal'
            decay = self.decay

        self._decay = decay

        # Get current model state
        model_dict = model.state_dict()

        # Update EMA parameters
        with torch.no_grad():
            for k, v in self.state_dict.items():
                if k not in self.ema_black_list:
                    # EMA update: ema = decay * ema + (1 - decay) * current
                    model_param = model_dict[k].to(
                        dtype=torch.float32, device=self.device
                    )
                    v.mul_(decay).add_(model_param, alpha=1 - decay)

        self.step += 1

    def apply(self) -> dict:
        """
        Apply bias correction and return the EMA state dict.

        This method applies bias correction (for 'threshold' and 'normal' types)
        and handles cycle_epoch logic.

        Returns:
            EMA state dict with bias correction applied
        """
        if self.step == 0:
            return self.state_dict

        state_dict = {}

        with torch.no_grad():
            for k, v in self.state_dict.items():
                if k in self.ema_black_list:
                    # For blacklisted parameters, return as-is
                    state_dict[k] = v.clone()
                else:
                    # Apply bias correction for non-exponential types
                    if self.ema_decay_type != "exponential":
                        # Bias correction: ema / (1 - decay^step)
                        corrected_v = v / (1 - self._decay**self.step)
                        state_dict[k] = corrected_v
                    else:
                        state_dict[k] = v.clone()

        self.epoch += 1

        # Reset if cycle_epoch is reached
        if self.cycle_epoch > 0 and self.epoch == self.cycle_epoch:
            self.reset()

        return state_dict

    def _match_ema_black_list(
        self,
        weight_names: Set[str],
        ema_black_list: Optional[Union[Set, List, tuple]] = None,
    ) -> Set[str]:
        """
        Match weight names against black list patterns.

        Args:
            weight_names: All weight names in the model
            ema_black_list: Black list patterns (substrings to match)

        Returns:
            Set of matched weight names
        """
        out_list = set()
        if ema_black_list:
            for name in weight_names:
                for key in ema_black_list:
                    if key in name:
                        out_list.add(name)
        return out_list

    def state_dict_for_save(self) -> dict:
        """
        Get state dict for checkpoint saving.

        Returns:
            Dictionary containing EMA state for saving
        """
        return {
            "ema_state_dict": self.state_dict,
            "step": self.step,
            "epoch": self.epoch,
            "decay": self.decay,
            "current_decay": self._decay,
            "ema_decay_type": self.ema_decay_type,
            "ema_black_list": sorted(self.ema_black_list),
        }

    def load_state_dict(self, checkpoint: dict):
        """
        Load EMA state from checkpoint.

        Args:
            checkpoint: Dictionary containing EMA state
        """
        if "ema_state_dict" in checkpoint:
            self.resume(checkpoint["ema_state_dict"], checkpoint.get("step", 0))
            self.epoch = checkpoint.get("epoch", 0)
            self.decay = checkpoint.get("decay", self.decay)
            self._decay = checkpoint.get("current_decay", self.decay)
            self.ema_decay_type = checkpoint.get("ema_decay_type", self.ema_decay_type)

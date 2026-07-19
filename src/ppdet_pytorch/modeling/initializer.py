# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025 PyTorch Migration. All Rights Reserved.
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

"""
Weight initialization utilities - PyTorch Migration from PaddlePaddle

This module is based on PyTorch's initialization methods but follows PaddlePaddle's API.
Reference: ppdet/modeling/initializer.py
"""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "uniform_",
    "normal_",
    "constant_",
    "ones_",
    "zeros_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "linear_init_",
    "conv_init_",
    "bias_init_with_prob",
    "reset_initialized_parameter",
]


def uniform_(tensor, a, b):
    """Modified tensor inplace using uniform distribution"""
    with torch.no_grad():
        tensor.uniform_(a, b)
    return tensor


def normal_(tensor, mean=0.0, std=1.0):
    """Modified tensor inplace using normal distribution"""
    with torch.no_grad():
        tensor.normal_(mean, std)
    return tensor


def constant_(tensor, value=0.0):
    """Modified tensor inplace using constant value"""
    with torch.no_grad():
        tensor.fill_(value)
    return tensor


def ones_(tensor):
    """Modified tensor inplace using ones"""
    with torch.no_grad():
        tensor.fill_(1)
    return tensor


def zeros_(tensor):
    """Modified tensor inplace using zeros"""
    with torch.no_grad():
        tensor.fill_(0)
    return tensor


def vector_(tensor, vector):
    """Set tensor value from vector"""
    with torch.no_grad():
        tensor.copy_(torch.tensor(vector, dtype=tensor.dtype))
    return tensor


def _calculate_fan_in_and_fan_out(tensor, reverse=False):
    """
    Calculate (fan_in, fan_out) for tensor

    Args:
        tensor (Tensor): torch.Tensor
        reverse (bool): tensor data format order, False by default as [fout, fin, ...]
                       e.g. conv.weight [cout, cin, kh, kw] is False;
                            linear.weight [cin, cout] is True

    Return:
        Tuple[fan_in, fan_out]
    """
    if tensor.ndim < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    if reverse:
        num_input_fmaps, num_output_fmaps = tensor.shape[0], tensor.shape[1]
    else:
        num_input_fmaps, num_output_fmaps = tensor.shape[1], tensor.shape[0]

    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = np.prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform_(tensor, gain=1.0, reverse=False):
    """Xavier uniform initialization"""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse=reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    k = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-k, k)
    return tensor


def xavier_normal_(tensor, gain=1.0, reverse=False):
    """Xavier normal initialization"""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse=reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def _calculate_correct_fan(tensor, mode, reverse=False):
    """Calculate fan_in or fan_out"""
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(mode, valid_modes)
        )

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse)

    return fan_in if mode == "fan_in" else fan_out


def _calculate_gain(nonlinearity, param=None):
    """Calculate gain for different nonlinearities"""
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and (
            isinstance(param, int) or isinstance(param, float)
        ):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def kaiming_uniform_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", reverse=False
):
    """Kaiming uniform initialization"""
    fan = _calculate_correct_fan(tensor, mode, reverse)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    k = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-k, k)
    return tensor


def kaiming_normal_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", reverse=False
):
    """Kaiming normal initialization"""
    fan = _calculate_correct_fan(tensor, mode, reverse)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def linear_init_(module):
    """Initialize linear layer"""
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def conv_init_(module):
    """Initialize conv layer"""
    bound = 1 / np.sqrt(np.prod(module.weight.shape[1:]))
    uniform_(module.weight, -bound, bound)
    if module.bias is not None:
        uniform_(module.bias, -bound, bound)


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


@torch.no_grad()
def reset_initialized_parameter(model, include_self=True):
    """
    Reset initialized parameter using following method for [conv, linear, embedding, bn]

    Args:
        model (nn.Module): torch Module
        include_self (bool): Indicate whether including itself
    Return:
        None
    """
    modules = [model] if include_self else []
    modules.extend(list(model.modules()))

    for m in modules:
        if isinstance(m, nn.Conv2d):
            k = float(m.groups) / (m.in_channels * m.kernel_size[0] * m.kernel_size[1])
            k = math.sqrt(k)
            m.weight.uniform_(-k, k)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.uniform_(-k, k)

        elif isinstance(m, nn.Linear):
            k = math.sqrt(1.0 / m.weight.shape[0])
            m.weight.uniform_(-k, k)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.uniform_(-k, k)

        elif isinstance(m, nn.Embedding):
            m.weight.normal_(mean=0.0, std=1.0)

        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            m.weight.fill_(1.0)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.fill_(0)

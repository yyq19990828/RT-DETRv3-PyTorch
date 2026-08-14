# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
Ops Module - PyTorch Migration from PaddlePaddle

Reference: ppdet/modeling/ops.py
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F

__all__ = ["get_act_fn", "get_static_shape"]


def identity(x):
    """Identity activation function"""
    return x


def mish(x):
    """Mish activation function"""
    return x * torch.tanh(F.softplus(x))


def silu(x):
    """SiLU/Swish activation function"""
    return F.silu(x)


def swish(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)


# TRT (TensorRT) compatible activation functions
TRT_ACT_SPEC = {"swish": swish, "silu": swish}

# Custom activation functions
ACT_SPEC = {"mish": mish, "silu": silu}


def get_act_fn(act=None, trt=False):
    """Get activation function by name or return the module directly

    Args:
        act (str|dict|None): Activation function name or config dict
        trt (bool): Whether to use TensorRT compatible version

    Returns:
        callable: Activation function

    Examples:
        >>> act_fn = get_act_fn('relu')
        >>> act_fn = get_act_fn({'name': 'leaky_relu', 'negative_slope': 0.1})
        >>> act_fn = get_act_fn(None)  # Returns identity
    """
    assert act is None or isinstance(act, (str, dict)), (
        "name of activation should be str, dict or None"
    )

    if not act:
        return identity

    if isinstance(act, dict):
        name = act["name"]
        act_copy = act.copy()
        act_copy.pop("name")
        kwargs = act_copy
    else:
        name = act
        kwargs = dict()

    # Get activation function
    if trt and name in TRT_ACT_SPEC:
        fn = TRT_ACT_SPEC[name]
    elif name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        # Try to get from torch.nn.functional
        fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)


def get_static_shape(tensor):
    """Get static shape of tensor

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Shape tensor with gradient stopped
    """
    shape = torch.tensor(tensor.shape, device=tensor.device)
    return shape.detach()

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
CSP-DarkNet Backbone Components - PyTorch Migration from PaddlePaddle

Reference: ppdet/modeling/backbones/csp_darknet.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.workspace import register, serializable
from ..initializer import conv_init_
from ..shape_spec import ShapeSpec

__all__ = ['BaseConv']


class BaseConv(nn.Module):
    """Basic Conv-BN-Act block

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        ksize (int): Kernel size
        stride (int): Stride
        groups (int): Groups for grouped convolution. Default: 1
        bias (bool): Whether to use bias. Default: False
        act (str): Activation function name. Default: 'silu'
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.bn(self.conv(x))
        y = x * torch.sigmoid(x)
        return y

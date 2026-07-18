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
CSPResNet Components - PyTorch Migration from PaddlePaddle

This module contains ConvBNLayer and RepVggBlock used by PPYOLOEHead.
Reference: ppdet/modeling/backbones/cspresnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..batch_norm import ContiguousGradBatchNorm2d

__all__ = ['ConvBNLayer', 'RepVggBlock']


def get_act_fn(act):
    """Get activation function by name or return the module directly"""
    if act is None:
        return nn.Identity()
    elif isinstance(act, (str, dict)):
        if isinstance(act, dict):
            act = act.get('name', 'relu')
        if act == 'relu':
            return nn.ReLU()
        elif act == 'swish' or act == 'silu':
            return nn.SiLU()
        elif act == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif act == 'hardsigmoid':
            return nn.Hardsigmoid()
        else:
            raise ValueError(f"Unsupported activation: {act}")
    else:
        # act is already an nn.Module
        return act


class ConvBNLayer(nn.Module):
    """Convolution + BatchNorm + Activation layer"""

    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = ContiguousGradBatchNorm2d(ch_out)
        self.act = get_act_fn(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RepVggBlock(nn.Module):
    """RepVGG Block with 3x3 and 1x1 branches"""

    def __init__(self, ch_in, ch_out, act='relu', alpha=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out

        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)

        self.act = get_act_fn(act)

        if alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha is not None:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        """Convert to deployment mode by fusing branches"""
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias = nn.Parameter(bias)
        delattr(self, 'conv1')
        delattr(self, 'conv2')

    def get_equivalent_kernel_bias(self):
        """Get equivalent kernel and bias for deployment"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        if self.alpha is not None:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(kernel1x1), \
                   bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), \
                   bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 kernel to 3x3"""
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse conv and bn into one conv"""
        if branch is None:
            return 0, 0

        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return kernel * t, beta - running_mean * gamma / std

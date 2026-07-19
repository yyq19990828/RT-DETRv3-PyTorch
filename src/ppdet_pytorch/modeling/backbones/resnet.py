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
ResNet Backbone Implementation - PyTorch Migration from PaddlePaddle

This module is a strict port of PaddlePaddle's ResNet implementation to PyTorch,
maintaining full compatibility with the original structure and behavior.

Reference: third-party/RT-DETRv3-paddle/ppdet/modeling/backbones/resnet.py
"""

import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from ...core.workspace import register, serializable
from ..batch_norm import ContiguousGradBatchNorm2d
from ..shape_spec import ShapeSpec
from .name_adapter import NameAdapter

__all__ = ["ResNet", "Res5Head", "Blocks", "BasicBlock", "BottleNeck"]

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class ConvNormLayer(nn.Module):
    """
    Convolution + Normalization layer with optional activation

    This is the fundamental building block in PaddleDetection's ResNet,
    combining Conv2D, BatchNorm, and optional activation in one module.
    """

    def __init__(
        self,
        ch_in,
        ch_out,
        filter_size,
        stride,
        groups=1,
        act=None,
        norm_type="bn",
        norm_decay=0.0,
        freeze_norm=True,
        lr=1.0,
        dcn_v2=False,
    ):
        """
        Args:
            ch_in (int): Input channels
            ch_out (int): Output channels
            filter_size (int): Kernel size
            stride (int): Stride
            groups (int): Group convolution cardinality
            act (str): Activation function name ('relu', 'sigmoid', etc.)
            norm_type (str): Normalization type ('bn', 'sync_bn')
            norm_decay (float): Weight decay for normalization layers
            freeze_norm (bool): Freeze normalization layer parameters
            lr (float): Learning rate multiplier for this layer
            dcn_v2 (bool): Use Deformable Convolution V2
        """
        super(ConvNormLayer, self).__init__()
        assert norm_type in ["bn", "sync_bn"]
        self.norm_type = norm_type
        self.act = act
        self.dcn_v2 = dcn_v2

        if not self.dcn_v2:
            self.conv = nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias=False,
            )
        else:
            # use torchvision.ops to implement DCN v2
            self.offset_channel = 2 * filter_size**2
            self.mask_channel = filter_size**2

            self.conv_offset = nn.Conv2d(
                in_channels=ch_in,
                out_channels=3 * filter_size**2,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                bias=True,
            )
            # Initialize offset conv to zero
            nn.init.constant_(self.conv_offset.weight, 0.0)
            nn.init.constant_(self.conv_offset.bias, 0.0)

            # Use standard conv as DCN is not implemented
            self.conv = DeformConv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias=False,
            )

        # Normalization layer
        if norm_type in ["sync_bn", "bn"]:
            self.norm = ContiguousGradBatchNorm2d(
                ch_out,
                momentum=0.1,  # PyTorch default, equivalent to Paddle
                eps=1e-05,
            )

        # Paddle's ParamAttr carries the stage learning-rate multiplier on the
        # parameter itself. Keep the same information for OptimizerBuilder to
        # turn into explicit PyTorch parameter groups.
        for param in self.conv.parameters():
            param._optimizer_lr_multiplier = float(lr)
        for param in self.norm.parameters():
            param._optimizer_lr_multiplier = float(lr)

        # Freeze normalization parameters if required
        self.freeze_norm = freeze_norm
        if freeze_norm:
            for param in self.norm.parameters():
                param.requires_grad = False
            self.norm.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_norm:
            self.norm.eval()
        return self

    def forward(self, inputs):
        """Forward pass"""
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            # DCN v2 forward
            offset_mask = self.conv_offset(inputs)
            offset = offset_mask[:, : self.offset_channel, :, :]
            mask = offset_mask[:, self.offset_channel :, :, :]
            mask = torch.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)

        if self.norm_type in ["bn", "sync_bn"]:
            out = self.norm(out)

        if self.act:
            if self.act == "relu":
                out = F.relu(out)
            elif self.act == "sigmoid":
                out = torch.sigmoid(out)
            else:
                # Support other activations
                out = getattr(F, self.act)(out)

        return out


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer"""

    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        c_ = ch // reduction_ratio

        stdv = 1.0 / math.sqrt(ch)
        self.squeeze = nn.Linear(ch, c_, bias=True)
        # Initialize with uniform distribution
        nn.init.uniform_(self.squeeze.weight, -stdv, stdv)

        stdv = 1.0 / math.sqrt(c_)
        self.extract = nn.Linear(c_, ch, bias=True)
        nn.init.uniform_(self.extract.weight, -stdv, stdv)

    def forward(self, inputs):
        out = self.pool(inputs)
        out = torch.squeeze(out, dim=[2, 3])
        out = self.squeeze(out)
        out = F.relu(out)
        out = self.extract(out)
        out = torch.sigmoid(out)
        out = torch.unsqueeze(torch.unsqueeze(out, 2), 3)
        scale = out * inputs
        return scale


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34

    Structure:
        x -> conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> (+) -> ReLU
        |___________________________________________________|
    """

    expansion = 1

    def __init__(
        self,
        ch_in,
        ch_out,
        stride,
        shortcut,
        variant="b",
        groups=1,
        base_width=64,
        lr=1.0,
        norm_type="bn",
        norm_decay=0.0,
        freeze_norm=True,
        dcn_v2=False,
        std_senet=False,
    ):
        """
        Args:
            ch_in (int): Input channels
            ch_out (int): Output channels (before expansion)
            stride (int): Stride for first conv
            shortcut (bool): Whether this is a shortcut connection (True) or needs downsampling (False)
            variant (str): ResNet variant ('a', 'b', 'c', 'd')
            groups (int): Group convolution (must be 1 for BasicBlock)
            base_width (int): Base width (must be 64 for BasicBlock)
            lr (float): Learning rate multiplier
            norm_type (str): Normalization type
            norm_decay (float): Norm weight decay
            freeze_norm (bool): Freeze norm layers
            dcn_v2 (bool): Use DCN v2
            std_senet (bool): Use SE layer
        """
        super(BasicBlock, self).__init__()
        assert groups == 1 and base_width == 64, (
            "BasicBlock only supports groups=1 and base_width=64"
        )

        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                # ResNet-vd: AvgPool + 1x1 conv
                self.short = nn.Sequential()
                self.short.add_module(
                    "pool",
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                )
                self.short.add_module(
                    "conv",
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr,
                    ),
                )
            else:
                # Standard: 1x1 conv with stride
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr,
                )

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            act="relu",
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
        )

        self.branch2b = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            act=None,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dcn_v2=dcn_v2,
        )

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)

        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = torch.add(out, short)
        out = F.relu(out)

        return out


class BottleNeck(nn.Module):
    """
    Bottleneck residual block for ResNet-50/101/152

    Structure:
        x -> conv1x1 -> BN -> ReLU -> conv3x3 -> BN -> ReLU -> conv1x1 -> BN -> (+) -> ReLU
        |_________________________________________________________________________|
    """

    expansion = 4

    def __init__(
        self,
        ch_in,
        ch_out,
        stride,
        shortcut,
        variant="b",
        groups=1,
        base_width=4,
        lr=1.0,
        norm_type="bn",
        norm_decay=0.0,
        freeze_norm=True,
        dcn_v2=False,
        std_senet=False,
    ):
        """
        Args:
            ch_in (int): Input channels
            ch_out (int): Output channels (before expansion, final is ch_out * 4)
            stride (int): Stride for 3x3 conv
            shortcut (bool): Whether this is a shortcut connection
            variant (str): ResNet variant ('a', 'b', 'c', 'd')
            groups (int): Group convolution cardinality (for ResNeXt)
            base_width (int): Base width for group convolution
            lr (float): Learning rate multiplier
            norm_type (str): Normalization type
            norm_decay (float): Norm weight decay
            freeze_norm (bool): Freeze norm layers
            dcn_v2 (bool): Use DCN v2
            std_senet (bool): Use SE layer
        """
        super(BottleNeck, self).__init__()

        # Variant 'a': stride in first conv, variant 'b'/'c'/'d': stride in second conv
        if variant == "a":
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        # ResNeXt width calculation
        width = int(ch_out * (base_width / 64.0)) * groups

        # 1x1 conv (channel reduction)
        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=width,
            filter_size=1,
            stride=stride1,
            groups=1,
            act="relu",
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
        )

        # 3x3 conv (with optional stride)
        self.branch2b = ConvNormLayer(
            ch_in=width,
            ch_out=width,
            filter_size=3,
            stride=stride2,
            groups=groups,
            act="relu",
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dcn_v2=dcn_v2,
        )

        # 1x1 conv (channel expansion)
        self.branch2c = ConvNormLayer(
            ch_in=width,
            ch_out=ch_out * self.expansion,
            filter_size=1,
            stride=1,
            groups=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
        )

        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                # ResNet-vd: AvgPool + 1x1 conv
                self.short = nn.Sequential()
                self.short.add_module(
                    "pool",
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                )
                self.short.add_module(
                    "conv",
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out * self.expansion,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr,
                    ),
                )
            else:
                # Standard: 1x1 conv with stride
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out * self.expansion,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr,
                )

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out * self.expansion)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = torch.add(out, short)
        out = F.relu(out)

        return out


class Blocks(nn.Module):
    """
    Container for multiple residual blocks forming one stage
    """

    def __init__(
        self,
        block,
        ch_in,
        ch_out,
        count,
        name_adapter,
        stage_num,
        variant="b",
        groups=1,
        base_width=64,
        lr=1.0,
        norm_type="bn",
        norm_decay=0.0,
        freeze_norm=True,
        dcn_v2=False,
        std_senet=False,
    ):
        """
        Args:
            block: Block class (BasicBlock or BottleNeck)
            ch_in (int): Input channels
            ch_out (int): Output channels (before expansion)
            count (int): Number of blocks
            stage_num (int): Stage number (2, 3, 4, 5)
            variant (str): ResNet variant
            groups (int): Group convolution cardinality
            base_width (int): Base width
            lr (float): Learning rate multiplier
            norm_type (str): Normalization type
            norm_decay (float): Norm weight decay
            freeze_norm (bool): Freeze norm layers
            dcn_v2 (bool): Use DCN v2
            std_senet (bool): Use SE layer
        """
        super(Blocks, self).__init__()

        self.blocks = []
        for i in range(count):
            # Use NameAdapter to generate block name (same as Paddle)
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)

            # First block may have stride=2 (except for stage 2)
            # First block always has shortcut=False (needs downsampling or channel matching)
            layer = block(
                ch_in=ch_in,
                ch_out=ch_out,
                stride=2 if i == 0 and stage_num != 2 else 1,
                shortcut=False if i == 0 else True,
                variant=variant,
                groups=groups,
                base_width=base_width,
                lr=lr,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                dcn_v2=dcn_v2,
                std_senet=std_senet,
            )

            # Register with custom name (like Paddle's add_sublayer)
            self.add_module(conv_name, layer)
            self.blocks.append(layer)

            # Update ch_in for next block
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, inputs):
        block_out = inputs
        for block in self.blocks:
            block_out = block(block_out)
        return block_out


@register
@serializable
class ResNet(nn.Module):
    """
    Residual Network

    Reference: https://arxiv.org/abs/1512.03385

    This is a strict PyTorch port of PaddlePaddle's ResNet implementation,
    maintaining full parameter and behavioral compatibility.
    """

    __shared__ = ["norm_type"]

    def __init__(
        self,
        depth=50,
        ch_in=64,
        variant="b",
        lr_mult_list=[1.0, 1.0, 1.0, 1.0],
        groups=1,
        base_width=64,
        norm_type="bn",
        norm_decay=0,
        freeze_norm=True,
        freeze_at=0,
        return_idx=[0, 1, 2, 3],
        dcn_v2_stages=[-1],
        num_stages=4,
        std_senet=False,
        freeze_stem_only=False,
    ):
        """
        Args:
            depth (int): ResNet depth (18, 34, 50, 101, 152)
            ch_in (int): Output channel of first stage (default 64)
            variant (str): ResNet variant ('a', 'b', 'c', 'd')
                - 'd': ResNet-vd with avgpool downsampling and 3x3 stem
            lr_mult_list (list): Learning rate ratio of different resnet stages (2,3,4,5)
            groups (int): Group convolution cardinality (for ResNeXt)
            base_width (int): Base width of each group convolution
            norm_type (str): Normalization type ('bn', 'sync_bn')
            norm_decay (float): Weight decay for normalization layer weights
            freeze_norm (bool): Freeze normalization layers
            freeze_at (int): Freeze the backbone at which stage (0-4)
                Stage 0 is stem, stages 1-4 are residual layers
            return_idx (list): Indices of stages whose features are returned (0-3)
                [0] -> layer1 output (stride 4)
                [1] -> layer2 output (stride 8)
                [2] -> layer3 output (stride 16)
                [3] -> layer4 output (stride 32)
            dcn_v2_stages (list): Indices of stages using deformable conv v2
            num_stages (int): Total number of stages (1-4)
            std_senet (bool): Whether to use SE layer
            freeze_stem_only (bool): Only freeze stem, not residual stages
        """
        super(ResNet, self).__init__()

        self._model_type = "ResNet" if groups == 1 else "ResNeXt"
        assert num_stages >= 1 and num_stages <= 4

        self.depth = depth
        self.variant = variant
        self.groups = groups
        self.base_width = base_width
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at

        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx) < num_stages, (
            "the maximum return index must smaller than num_stages, "
            "but received maximum return index is {} and num_stages "
            "is {}".format(max(return_idx), num_stages)
        )
        self.return_idx = return_idx
        self.num_stages = num_stages

        assert len(lr_mult_list) == 4, (
            "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        )

        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        self.dcn_v2_stages = dcn_v2_stages

        block_nums = ResNet_cfg[depth]

        # Stem layers (conv1)
        if variant in ["c", "d"]:
            # ResNet-vd: Three 3x3 convs
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            # Standard ResNet: One 7x7 conv
            conv_def = [[3, ch_in, 7, 2, "conv1"]]

        self.conv1 = nn.Sequential()
        for i, (c_in, c_out, k, s, _name) in enumerate(conv_def):
            self.conv1.add_module(
                _name,
                ConvNormLayer(
                    ch_in=c_in,
                    ch_out=c_out,
                    filter_size=k,
                    stride=s,
                    groups=1,
                    act="relu",
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=1.0,
                ),
            )

        self.ch_in = ch_in
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        self._out_channels = [block.expansion * v for v in ch_out_list]
        self._out_strides = [4, 8, 16, 32]

        # Create NameAdapter (same as Paddle)
        na = NameAdapter(self)

        # Residual stages
        # Use list to maintain reference, but register with add_module for correct naming
        self.res_layers = []
        for i in range(num_stages):
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            res_name = "res{}".format(
                stage_num
            )  # Generate "res2", "res3", "res4", "res5"

            res_layer = Blocks(
                block,
                self.ch_in,
                ch_out_list[i],
                count=block_nums[i],
                name_adapter=na,
                stage_num=stage_num,
                variant=variant,
                groups=groups,
                base_width=base_width,
                lr=lr_mult,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                dcn_v2=(i in self.dcn_v2_stages),
                std_senet=std_senet,
            )

            # Register sublayer with custom name (like Paddle's add_sublayer)
            self.add_module(res_name, res_layer)
            self.res_layers.append(res_layer)
            self.ch_in = self._out_channels[i]

        # Freeze parameters
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, num_stages)):
                    self._freeze_parameters(self.res_layers[i])

    def _freeze_parameters(self, m):
        """Freeze all parameters in module m"""
        for p in m.parameters():
            p.requires_grad = False

    @property
    def out_shape(self):
        """
        Get output shape specification for each returned stage

        Returns:
            List of ShapeSpec objects
        """
        return [
            ShapeSpec(channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, inputs):
        """
        Forward pass

        Args:
            inputs (dict or Tensor): If dict, expects {'image': tensor}
                                     If Tensor, uses it directly

        Returns:
            List of feature tensors at specified return indices
        """
        # Handle both dict input (PaddlePaddle style) and tensor input
        if isinstance(inputs, dict):
            x = inputs["image"]
        else:
            x = inputs

        # Stem
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)

        # Residual stages
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs


@register
class Res5Head(nn.Module):
    """
    ResNet Stage 5 Head for ROI feature extraction
    """

    def __init__(self, depth=50):
        super(Res5Head, self).__init__()

        feat_in, feat_out = [1024, 512]
        if depth < 50:
            feat_in = 256

        # Create NameAdapter (same as Paddle)
        na = NameAdapter(self)

        block = BottleNeck if depth >= 50 else BasicBlock
        self.res5 = Blocks(
            block, feat_in, feat_out, count=3, name_adapter=na, stage_num=5
        )
        self.feat_out = feat_out if depth < 50 else feat_out * 4

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.feat_out, stride=16)]

    def forward(self, roi_feat, stage=0):
        y = self.res5(roi_feat)
        return y

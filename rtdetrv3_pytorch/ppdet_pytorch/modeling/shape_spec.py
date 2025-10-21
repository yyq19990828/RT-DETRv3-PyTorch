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

# The code is based on:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/shape_spec.py

from collections import namedtuple


class ShapeSpec(
        namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.

    This is used to describe the output shape of backbone features, which helps
    downstream modules to know what to expect.

    Attributes:
        channels (int): Number of channels
        height (int): Height of the feature map (optional)
        width (int): Width of the feature map (optional)
        stride (int): The stride of the feature map relative to input image
    """
    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super(ShapeSpec, cls).__new__(cls, channels, height, width,
                                             stride)

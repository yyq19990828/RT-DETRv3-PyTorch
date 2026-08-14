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
DETR Transformer - PyTorch Migration from PaddlePaddle

Reference: ppdet/modeling/transformers/detr_transformer.py
"""

from __future__ import absolute_import, division, print_function

import torch.nn as nn

from .utils import _get_clones

__all__ = ["TransformerEncoder"]


class TransformerEncoder(nn.Module):
    """Transformer Encoder

    Args:
        encoder_layer (nn.Module): An instance of the encoder layer module
        num_layers (int): Number of encoder layers
        norm (nn.Module|None): Normalization layer to apply after all encoder layers.
                               Default: None
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        """Forward pass of transformer encoder

        Args:
            src (Tensor): Input features, shape (B, L, C) where
                         B is batch size, L is sequence length, C is feature dim
            src_mask (Tensor|None): Attention mask, shape (B, L, L). Default: None
            pos_embed (Tensor|None): Position embeddings, shape (B, L, C). Default: None

        Returns:
            output (Tensor): Encoded features, shape (B, L, C)
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

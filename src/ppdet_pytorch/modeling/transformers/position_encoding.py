# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# PyTorch Conversion: Converted from PaddlePaddle to PyTorch

from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn

from ppdet_pytorch.core.workspace import register, serializable


@register
@serializable
class PositionEmbedding(nn.Module):
    """
    Position Embedding for 2D feature maps

    Supports two types of position embeddings:
    - 'sine': Sinusoidal position embeddings (default)
    - 'learned': Learnable position embeddings

    Args:
        num_pos_feats (int): Half of the embedding dimension (default: 128)
                            Final embedding dim is 2 * num_pos_feats
        temperature (int): Temperature for sinusoidal encoding (default: 10000)
        normalize (bool): Whether to normalize coordinates to [0, 1] (default: True)
        scale (float): Scale factor for normalized coordinates (default: 2*pi)
        embed_type (str): Type of embedding ('sine' or 'learned')
        num_embeddings (int): Number of embeddings for learned type (default: 50)
        offset (float): Offset for position calculation (default: 0.0)
        eps (float): Small epsilon to avoid division by zero (default: 1e-6)
    """

    def __init__(
        self,
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        scale=2 * math.pi,
        embed_type="sine",
        num_embeddings=50,
        offset=0.0,
        eps=1e-6,
    ):
        super(PositionEmbedding, self).__init__()
        assert embed_type in ["sine", "learned"]

        self.embed_type = embed_type
        self.offset = offset
        self.eps = eps
        if self.embed_type == "sine":
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            self.scale = scale
        elif self.embed_type == "learned":
            self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
            self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        else:
            raise ValueError(f"{self.embed_type} is not supported.")

    def forward(self, mask):
        """
        Generate position embeddings for feature map

        Args:
            mask (Tensor): Mask tensor of shape (B, H, W) where True/1 means valid position

        Returns:
            pos (Tensor): Position embeddings of shape (B, H, W, C)
                         where C = 2 * num_pos_feats
        """
        if self.embed_type == "sine":
            # Cumulative sum along height and width
            y_embed = mask.cumsum(1, dtype=torch.float32)
            x_embed = mask.cumsum(2, dtype=torch.float32)

            if self.normalize:
                y_embed = (
                    (y_embed + self.offset)
                    / (y_embed[:, -1:, :] + self.eps)
                    * self.scale
                )
                x_embed = (
                    (x_embed + self.offset)
                    / (x_embed[:, :, -1:] + self.eps)
                    * self.scale
                )

            # Generate sinusoidal embeddings
            dim_t = 2 * torch.div(
                torch.arange(
                    self.num_pos_feats, dtype=torch.float32, device=mask.device
                ),
                2,
                rounding_mode="floor",
            )
            dim_t = self.temperature ** (dim_t / self.num_pos_feats)

            pos_x = x_embed.unsqueeze(-1) / dim_t
            pos_y = y_embed.unsqueeze(-1) / dim_t

            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)

            return torch.cat((pos_y, pos_x), dim=3)

        elif self.embed_type == "learned":
            h, w = mask.shape[-2:]
            i = torch.arange(w, device=mask.device)
            j = torch.arange(h, device=mask.device)
            x_emb = self.col_embed(i)
            y_emb = self.row_embed(j)

            return torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            ).unsqueeze(0)
        else:
            raise ValueError(f"not supported {self.embed_type}")

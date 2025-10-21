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
RT-DETRv3 Architecture - PyTorch Migration from PaddlePaddle

Reference: ppdet/modeling/architectures/rtdetrv3.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from ppdet_pytorch.core.workspace import register, create

__all__ = ['RTDETRV3']


@register
class RTDETRV3(nn.Module):
    """RT-DETRv3 Main Architecture

    Components:
        - backbone: Feature extraction network
        - neck: Feature pyramid network (optional)
        - transformer: Transformer encoder-decoder
        - detr_head: Main detection head
        - aux_o2m_head: Auxiliary one-to-many head (optional, for training)
        - post_process: Post-processing module (for inference)
    """

    __category__ = 'architecture'
    __inject__ = ['post_process', 'post_process_semi']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 aux_o2m_head=None,
                 post_process='DETRPostProcess',
                 post_process_semi=None,
                 with_mask=False,
                 exclude_post_process=False):
        super(RTDETRV3, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.aux_o2m_head = aux_o2m_head
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process
        self.post_process_semi = post_process_semi

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        """Create RTDETRV3 from config following PaddlePaddle's pattern

        Args:
            cfg: Configuration dict with component configs

        Returns:
            Dict of component instances for __init__
        """
        # backbone
        backbone = create(cfg['backbone'])

        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None

        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        transformer = create(cfg['transformer'], **kwargs)

        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        aux_o2m_head = create(cfg['aux_o2m_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck,
            "aux_o2m_head": aux_o2m_head
        }

    def _forward(self, inputs):
        """Forward pass for both training and inference

        Args:
            inputs: Dict containing:
                - image: Input images [B, C, H, W]
                - pad_mask: Padding mask (optional)
                - im_shape: Original image shapes (for inference)
                - scale_factor: Scale factors (for inference)
                - gt_* : Ground truth labels (for training)

        Returns:
            Training: Dict of losses
            Inference: Dict with 'bbox', 'bbox_num', optionally 'mask'
        """
        # Backbone
        body_feats = self.backbone(inputs['image'])

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # Transformer
        pad_mask = inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, body_feats, inputs)
            detr_losses.update({
                'loss': torch.stack(
                    [v for k, v in detr_losses.items() if 'log' not in k]
                ).sum()
            })
            if self.aux_o2m_head is not None:
                aux_o2m_losses = self.aux_o2m_head(body_feats, inputs)
                for k, v in aux_o2m_losses.items():
                    if k == 'loss':
                        detr_losses[k] += v
                    k = k + '_aux_o2m'
                    detr_losses[k] = v
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, inputs['im_shape'], inputs['scale_factor'],
                    inputs['image'].shape[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def forward(self, inputs):
        """Forward pass wrapper

        Args:
            inputs: Dict or Tensor
                - If dict: Must contain 'image' key
                - If Tensor: Converted to {'image': inputs}

        Returns:
            Same as _forward()
        """
        if isinstance(inputs, torch.Tensor):
            inputs = {'image': inputs}
        return self._forward(inputs)

    def get_loss(self, inputs):
        """Get training losses

        Args:
            inputs: Training batch dict

        Returns:
            Dict of losses
        """
        return self.forward(inputs)

    def get_pred(self, inputs):
        """Get predictions for inference

        Args:
            inputs: Inference batch dict

        Returns:
            Dict with detections
        """
        return self.forward(inputs)

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
Naive SyncBatchNorm conversion utility.

Converts BatchNorm layers to SyncBatchNorm for distributed training.
"""

import logging

import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["convert_syncbn"]


def convert_syncbn(model):
    """
    Convert BatchNorm to SyncBatchNorm for distributed training (Paddle compatible).

    Args:
        model: Model with BatchNorm layers to convert

    Returns:
        Model with SyncBatchNorm layers
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        logger.warning("convert_syncbn called but not in distributed mode, skipping")
        return model

    logger.info("Converting BatchNorm to SyncBatchNorm for distributed training")
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info("SyncBatchNorm conversion complete")

    return model

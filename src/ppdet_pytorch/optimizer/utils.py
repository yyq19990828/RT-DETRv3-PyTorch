# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

from typing import Set

import torch.nn as nn


def get_bn_running_state_names(model: nn.Module) -> Set[str]:
    """
    Get names of BatchNorm running states (running_mean, running_var, num_batches_tracked).

    Args:
        model: PyTorch model

    Returns:
        Set of parameter names for BN running states
    """
    bn_states = set()
    for name, module in model.named_modules():
        if isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        ):
            prefix = name + ("." if name else "")
            bn_states.add(prefix + "running_mean")
            bn_states.add(prefix + "running_var")
            bn_states.add(prefix + "num_batches_tracked")
    return bn_states

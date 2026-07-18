# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np

import torch
import torch.distributed as dist

__all__ = ['init_parallel_env', 'set_random_seed', 'init_fleet_env']


def init_fleet_env(find_unused_parameters=False):
    """
    Initialize distributed training environment using PyTorch DDP.

    Args:
        find_unused_parameters (bool): Whether to find unused parameters in DDP.
                                      Set to True if not all parameters are used in forward pass.

    Note:
        This is equivalent to PaddlePaddle's fleet.init() but using PyTorch's DDP.
        Environment variables should be set before calling this function:
        - RANK: Global rank of the process
        - LOCAL_RANK: Local rank on the current node
        - WORLD_SIZE: Total number of processes
        - MASTER_ADDR: Address of rank 0
        - MASTER_PORT: Port of rank 0
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training is not available")

    # Initialize process group if not already initialized
    if not dist.is_initialized():
        # Get backend from environment or use default
        backend = os.environ.get('DIST_BACKEND', 'nccl' if torch.cuda.is_available() else 'gloo')

        # Initialize the process group
        dist.init_process_group(backend=backend)

        # Set device for current process
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)

    # Store find_unused_parameters in environment for later use in DDP wrapper
    os.environ['DDP_FIND_UNUSED_PARAMETERS'] = str(find_unused_parameters)


def init_parallel_env():
    """
    Initialize parallel training environment.

    For PyTorch, this checks for distributed training setup and initializes
    random seeds for reproducibility across processes.

    Note:
        Environment variables checked:
        - RANK or LOCAL_RANK: Process rank
        - WORLD_SIZE: Total number of processes
    """
    env = os.environ

    # Check if we're in distributed mode
    # PyTorch uses RANK/LOCAL_RANK/WORLD_SIZE instead of PADDLE_TRAINER_ID/PADDLE_TRAINERS_NUM
    is_distributed = ('RANK' in env or 'LOCAL_RANK' in env) and 'WORLD_SIZE' in env

    if is_distributed:
        # Get rank (use RANK if available, otherwise LOCAL_RANK)
        rank = int(env.get('RANK', env.get('LOCAL_RANK', 0)))

        # Set process-specific seed for reproducibility
        local_seed = (99 + rank)
        random.seed(local_seed)
        np.random.seed(local_seed)

        # Initialize distributed training if not already done
        if not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)

            # Set device if using CUDA
            if torch.cuda.is_available():
                local_rank = int(env.get('LOCAL_RANK', 0))
                torch.cuda.set_device(local_rank)


def set_random_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed value

    Note:
        This sets seeds for:
        - PyTorch (CPU and CUDA)
        - Python random module
        - NumPy
    """
    # Set PyTorch seed (affects both CPU and CUDA)
    torch.manual_seed(seed)

    # Set CUDA seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Additional settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set Python random seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

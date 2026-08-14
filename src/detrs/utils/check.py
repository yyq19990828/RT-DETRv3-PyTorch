# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys

import torch

from .logger import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "check_gpu",
    "check_version",
    "check_config",
    # 'check_npu', 'check_xpu', 'check_mlu',  # Not supported in PyTorch by default
]


# NOTE: The following device checks are commented out as PyTorch doesn't natively support these devices.
# Uncomment and modify if you have the appropriate extensions installed.

# def check_mlu(use_mlu):
#     """
#     Log error and exit when set use_mlu=true in PyTorch version
#     without MLU support.
#     """
#     err = "Config use_mlu cannot be set as true while you are " \
#           "using PyTorch version without MLU support! \nPlease try: \n" \
#           "\t1. Install PyTorch with MLU support to run model on MLU \n" \
#           "\t2. Set use_mlu as false in config file to run " \
#           "model on CPU/GPU"
#
#     try:
#         if use_mlu:
#             # PyTorch doesn't have native MLU support, check if custom device is available
#             if not hasattr(torch, 'mlu') or not torch.mlu.is_available():
#                 logger.error(err)
#                 sys.exit(1)
#     except Exception as e:
#         pass


# def check_npu(use_npu):
#     """
#     Log error and exit when set use_npu=true in PyTorch version
#     without NPU support.
#     """
#     err = "Config use_npu cannot be set as true while you are " \
#           "using PyTorch version without NPU support! \nPlease try: \n" \
#           "\t1. Install PyTorch with NPU support (torch-npu) to run model on NPU \n" \
#           "\t2. Set use_npu as false in config file to run " \
#           "model on other devices supported."
#
#     try:
#         if use_npu:
#             # Check if torch_npu is available
#             try:
#                 import torch_npu
#                 if not torch.npu.is_available():
#                     logger.error(err)
#                     sys.exit(1)
#             except ImportError:
#                 logger.error(err)
#                 sys.exit(1)
#     except Exception as e:
#         pass


# def check_xpu(use_xpu):
#     """
#     Log error and exit when set use_xpu=true in PyTorch version
#     without XPU support.
#     """
#     err = "Config use_xpu cannot be set as true while you are " \
#           "using PyTorch version without XPU support! \nPlease try: \n" \
#           "\t1. Install PyTorch with XPU support (Intel Extension for PyTorch) to run model on XPU \n" \
#           "\t2. Set use_xpu as false in config file to run " \
#           "model on CPU/GPU"
#
#     try:
#         if use_xpu:
#             # Check if Intel Extension for PyTorch is available
#             if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
#                 logger.error(err)
#                 sys.exit(1)
#     except Exception as e:
#         pass


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in PyTorch
    CPU version or when CUDA is not available.
    """
    err = (
        "Config use_gpu cannot be set as true while CUDA is not available! \n"
        "Please try: \n"
        "\t1. Install PyTorch with CUDA support to run model on GPU \n"
        "\t2. Set use_gpu as false in config file to run model on CPU"
    )

    try:
        if use_gpu and not torch.cuda.is_available():
            logger.error(err)
            sys.exit(1)
    except Exception:
        pass


def check_version(version="2.0"):
    """
    Log error and exit when the installed version of PyTorch is
    not satisfied.

    Args:
        version (str): Minimum required PyTorch version, default is '2.0'
    """
    err = (
        "PyTorch version {} or higher is required. \n"
        "Please make sure the version is good with your code. \n"
        "Current PyTorch version: {}".format(version, torch.__version__)
    )

    try:
        # Get installed PyTorch version
        installed_version = torch.__version__.split("+")[
            0
        ]  # Remove build info like '+cu118'

        # Parse version strings
        required_parts = [int(x) for x in version.split(".")]
        installed_parts = [int(x) for x in installed_version.split(".")]

        # Compare versions
        for i in range(min(len(required_parts), len(installed_parts))):
            if installed_parts[i] > required_parts[i]:
                return  # Installed version is higher
            if installed_parts[i] < required_parts[i]:
                logger.error(err)
                sys.exit(1)

    except Exception as e:
        logger.warning(f"Failed to check PyTorch version: {e}")
        pass


def check_config(cfg):
    """
    Check the correctness of the configuration file. Log error and exit
    when Config is not compliant.

    Args:
        cfg (dict): Configuration dictionary

    Returns:
        dict: Validated configuration with defaults set
    """
    err = "'{}' not specified in config file. Please set it in config file."
    check_list = ["architecture", "num_classes"]
    try:
        for var in check_list:
            if var not in cfg:
                logger.error(err.format(var))
                sys.exit(1)
    except Exception:
        pass

    if "log_iter" not in cfg:
        cfg["log_iter"] = 20

    return cfg

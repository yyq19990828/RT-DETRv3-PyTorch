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
import sys

# add python path of PyTorch Detection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from ppdet_pytorch.core.workspace import load_config, merge_config

from ppdet_pytorch.engine import Trainer, TrainerCot, init_parallel_env, set_random_seed, init_fleet_env
# from ppdet_pytorch.engine.trainer_ssod import Trainer_DenseTeacher, Trainer_ARSL, Trainer_Semi_RTDETR

# from ppdet_pytorch.slim import build_slim_model

from ppdet_pytorch.utils.cli import ArgsParser, merge_args
import ppdet_pytorch.utils.check as check
from ppdet_pytorch.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "-r", "--resume", default=None, help="weights path for resume")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")
    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable auto mixed precision training.")
    parser.add_argument(
        "--ddp",
        action='store_true',
        default=False,
        help="Use DistributedDataParallel for multi-GPU training")
    parser.add_argument(
        "--use_tensorboard",
        type=bool,
        default=False,
        help="whether to record the data to TensorBoard.")
    parser.add_argument(
        '--tensorboard_log_dir',
        type=str,
        default="runs",
        help='TensorBoard logging directory.')
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="whether to record the data to wandb.")
    parser.add_argument(
        '--save_prediction_only',
        action='store_true',
        default=False,
        help='Whether to save the evaluation results only')
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help="The option of profiler, which should be in "
        "format \"key1=value1;key2=value2;key3=value3\"."
        "please see ppdet_pytorch/utils/profiler.py for detail.")
    parser.add_argument(
        '--save_proposals',
        action='store_true',
        default=False,
        help='Whether to save the train proposals')
    parser.add_argument(
        '--proposals_path',
        type=str,
        default="sniper/proposals.json",
        help='Train proposals directory')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training')

    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # init distributed environment if ddp is enabled
    if cfg.get('ddp', False) or FLAGS.ddp:
        init_fleet_env(cfg.get('find_unused_parameters', False))
    else:
        # init parallel environment if nranks > 1
        init_parallel_env()

    if FLAGS.enable_ce:
        set_random_seed(0)

    # build trainer
    # ssod_method = cfg.get('ssod_method', None)
    # if ssod_method is not None:
    #     if ssod_method == 'DenseTeacher':
    #         trainer = Trainer_DenseTeacher(cfg, mode='train')
    #     elif ssod_method == 'ARSL':
    #         trainer = Trainer_ARSL(cfg, mode='train')
    #     elif ssod_method == 'Semi_RTDETR':
    #         trainer = Trainer_Semi_RTDETR(cfg, mode='train')
    #     else:
    #         raise ValueError(
    #             "Semi-Supervised Object Detection only no support this method.")
    # elif cfg.get('use_cot', False):
    #     trainer = TrainerCot(cfg, mode='train')
    # else:
    #     trainer = Trainer(cfg, mode='train')
    trainer = Trainer(cfg, mode='train')

    # load weights
    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_student_weights' in cfg and 'pretrain_teacher_weights' in cfg \
            and cfg.pretrain_teacher_weights and cfg.pretrain_student_weights:
        trainer.load_semi_weights(cfg.pretrain_teacher_weights,
                                  cfg.pretrain_student_weights)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    # training
    trainer.train(FLAGS.eval)


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # Set device configuration
    # Disable special devices by default (NPU, XPU, MLU not supported in PyTorch by default)
    if 'use_npu' not in cfg:
        cfg.use_npu = False
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False
    if 'use_mlu' not in cfg:
        cfg.use_mlu = False

    # Set GPU/CPU device
    if 'use_gpu' not in cfg:
        cfg.use_gpu = torch.cuda.is_available()  # Auto-detect CUDA availability

    # Set PyTorch device
    if cfg.use_gpu:
        device = torch.device('cuda')
        logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU')

    # Store device in config for later use
    cfg.device = device

    # Check for special devices (warn if enabled)
    if cfg.use_npu or cfg.use_xpu or cfg.use_mlu:
        logger.warning(
            "NPU/XPU/MLU devices are not supported in PyTorch by default. "
            "These settings will be ignored. Using GPU/CPU instead."
        )

    # NOTE: slim is not supported temporarily
    # if FLAGS.slim_config:
    #     cfg = build_slim_model(cfg, FLAGS.slim_config)

    # FIXME: Temporarily solve the priority problem of FLAGS.opt
    merge_config(FLAGS.opt)
    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    run(FLAGS, cfg)


if __name__ == "__main__":
    main()

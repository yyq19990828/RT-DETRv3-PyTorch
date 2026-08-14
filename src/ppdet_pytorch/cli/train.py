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

from __future__ import absolute_import, division, print_function

# ignore warning log
import warnings

warnings.filterwarnings("ignore")

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch

import ppdet_pytorch.utils.check as check
from ppdet_pytorch.core.workspace import load_config, merge_config
from ppdet_pytorch.engine import (
    Trainer,
    init_fleet_env,
    init_parallel_env,
    set_random_seed,
)

# from ppdet_pytorch.engine.trainer_ssod import Trainer_DenseTeacher, Trainer_ARSL, Trainer_Semi_RTDETR
# from ppdet_pytorch.slim import build_slim_model
from ppdet_pytorch.utils.cli import ArgsParser, merge_args
from ppdet_pytorch.utils.logger import setup_logger

logger = setup_logger("train")


def create_argument_parser():
    parser = ArgsParser()
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Whether to perform evaluation in train",
    )
    parser.add_argument("-r", "--resume", default=None, help="weights path for resume")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.",
    )
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducible training.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable auto mixed precision training.",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        default=False,
        help="Use DistributedDataParallel for multi-GPU training",
    )
    parser.add_argument(
        "--use_tensorboard",
        type=bool,
        default=False,
        help="whether to record the data to TensorBoard.",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default="runs",
        help="TensorBoard logging directory.",
    )
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="whether to record the data to wandb.",
    )
    parser.add_argument(
        "--save_prediction_only",
        action="store_true",
        default=False,
        help="Whether to save the evaluation results only",
    )
    parser.add_argument(
        "--profiler_options",
        type=str,
        default=None,
        help="The option of profiler, which should be in "
        'format "key1=value1;key2=value2;key3=value3".'
        "please see ppdet_pytorch/utils/profiler.py for detail.",
    )
    parser.add_argument(
        "--save_proposals",
        action="store_true",
        default=False,
        help="Whether to save the train proposals",
    )
    parser.add_argument(
        "--proposals_path",
        type=str,
        default="sniper/proposals.json",
        help="Train proposals directory",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )

    return parser


def parse_args(argv=None):
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    unsupported = {
        "eval": "--eval is not implemented; run rtdetrv3-eval separately",
        "slim_config": "--slim_config is not supported by the PyTorch trainer",
        "use_tensorboard": "--use_tensorboard is not implemented",
        "use_wandb": "--use_wandb is not implemented",
        "save_prediction_only": "--save_prediction_only is not a training option",
        "profiler_options": "--profiler_options is not implemented",
        "save_proposals": "--save_proposals is not implemented",
    }
    for field, message in unsupported.items():
        if getattr(args, field):
            parser.error(message)
    return args


def run(FLAGS, cfg):
    # init distributed environment if ddp is enabled
    if cfg.get("ddp", False) or FLAGS.ddp:
        init_fleet_env(cfg.get("find_unused_parameters", False))
    else:
        # init parallel environment if nranks > 1
        init_parallel_env()

    seed = getattr(FLAGS, "seed", None)
    if seed is None and FLAGS.enable_ce:
        seed = 0
    if seed is not None:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        set_random_seed(seed + rank)
        cfg["seed"] = seed
        logger.info(
            "Using base seed %s (process seed %s on rank %s)", seed, seed + rank, rank
        )

    has_semi_supervised_weights = (
        "pretrain_student_weights" in cfg
        and "pretrain_teacher_weights" in cfg
        and cfg.pretrain_teacher_weights
        and cfg.pretrain_student_weights
    )
    if has_semi_supervised_weights:
        raise NotImplementedError(
            "Semi-supervised teacher/student weights are not supported by "
            "the PyTorch trainer"
        )

    if FLAGS.resume is not None and cfg.get("pretrain_weights"):
        cfg["pretrain_weights"] = None

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
    trainer = Trainer(cfg, mode="train")

    # load weights
    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif "pretrain_weights" in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    # training
    trainer.train()


def main(argv=None):
    FLAGS = parse_args(argv)
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # Set device configuration
    # Disable special devices by default (NPU, XPU, MLU not supported in PyTorch by default)
    if "use_npu" not in cfg:
        cfg.use_npu = False
    if "use_xpu" not in cfg:
        cfg.use_xpu = False
    if "use_mlu" not in cfg:
        cfg.use_mlu = False

    # Set GPU/CPU device
    if "use_gpu" not in cfg:
        cfg.use_gpu = torch.cuda.is_available()  # Auto-detect CUDA availability

    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    # Set PyTorch device
    if cfg.use_gpu:
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Store device in config for later use
    cfg.device = device

    # Check for special devices (warn if enabled)
    if cfg.use_npu or cfg.use_xpu or cfg.use_mlu:
        logger.warning(
            "NPU/XPU/MLU devices are not supported in PyTorch by default. "
            "These settings will be ignored. Using GPU/CPU instead."
        )

    try:
        run(FLAGS, cfg)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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
Training script for RT-DETRv3 PyTorch.

Usage:
    # Single GPU training
    python tools/train.py --config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml

    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=8 tools/train.py --config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml

    # Resume training
    python tools/train.py --config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml --resume output/epoch_10.pth
"""

import argparse
import os
import sys
import warnings

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from models import build_model
from data import build_dataset, build_transform
from engine import Trainer, build_optimizer, build_lr_scheduler, build_coco_evaluator
from models.losses import DINOv3Loss
from utils.config import load_config
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RT-DETRv3')

    # Config
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        required=True,
        help='Path to config file'
    )

    # Training
    parser.add_argument(
        '--resume',
        '-r',
        type=str,
        default=None,
        help='Path to checkpoint for resuming training'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Whether to perform evaluation during training'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable automatic mixed precision training'
    )

    # Override config options
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size per GPU (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )

    # Distributed
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training (set by torchrun)'
    )

    args = parser.parse_args()
    return args


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()

    return rank, world_size, local_rank


def main():
    """Main training function."""
    args = parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    is_main_process = rank == 0

    # Setup logger
    logger = setup_logger('rtdetrv3.train', log_ranks='0')

    if is_main_process:
        logger.info(f"Training RT-DETRv3 on {world_size} GPU(s)")
        logger.info(f"Config: {args.config}")

    # Load config
    cfg = load_config(args.config)

    # Override config with command line arguments
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['optimizer']['lr'] = args.lr
    if args.output_dir is not None:
        cfg['save_dir'] = args.output_dir
    if args.amp:
        cfg['use_amp'] = True

    # Create output directory
    save_dir = cfg.get('save_dir', './output')
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    # Build model
    if is_main_process:
        logger.info("Building model...")
    model = build_model(cfg['model'])
    model = model.cuda()

    # Build dataset and dataloader
    if is_main_process:
        logger.info("Building dataset...")

    train_transform = build_transform(cfg.get('train_transform', None), is_train=True)
    train_dataset = build_dataset(
        cfg['train_dataset'],
        transform=train_transform
    )

    # Use DistributedSampler for DDP
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('batch_size', 2),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=getattr(train_dataset, 'collate_fn', None)
    )

    # Build validation dataset if needed
    val_loader = None
    evaluator = None
    if args.eval and 'val_dataset' in cfg:
        val_transform = build_transform(cfg.get('val_transform', None), is_train=False)
        val_dataset = build_dataset(
            cfg['val_dataset'],
            transform=val_transform
        )

        # Use DistributedSampler for validation too
        if is_distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            val_sampler = None

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get('batch_size', 2),
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.get('num_workers', 4),
            pin_memory=True,
            collate_fn=getattr(val_dataset, 'collate_fn', None)
        )

        # Build evaluator
        evaluator = build_coco_evaluator(
            cfg.get('evaluator', {}),
            dataset=val_dataset
        )

    # Build loss function
    loss_fn = DINOv3Loss(
        num_classes=cfg.get('num_classes', 80),
        loss_coeff=cfg.get('loss_coeff', None),
        use_focal_loss=cfg.get('use_focal_loss', True),
        use_vfl=cfg.get('use_vfl', True)
    )

    # Build optimizer
    if is_main_process:
        logger.info("Building optimizer...")
    optimizer = build_optimizer(model, cfg.get('optimizer', None))

    # Build learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = build_lr_scheduler(
        optimizer,
        cfg.get('lr_scheduler', None),
        steps_per_epoch=steps_per_epoch
    )

    # Build trainer
    trainer_cfg = {
        'epochs': cfg.get('epochs', 72),
        'save_dir': save_dir,
        'save_interval': cfg.get('save_interval', 1),
        'log_interval': cfg.get('log_interval', 50),
        'val_interval': cfg.get('val_interval', 1),
        'grad_clip': cfg.get('grad_clip', 0.1),
        'use_amp': cfg.get('use_amp', False),
        'find_unused_parameters': cfg.get('find_unused_parameters', False)
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        cfg=trainer_cfg,
        val_loader=val_loader,
        evaluator=evaluator
    )

    # Resume from checkpoint if specified
    if args.resume:
        if is_main_process:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume, resume_training=True)

    # Start training
    if is_main_process:
        logger.info("Starting training...")
    trainer.train()

    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()

    if is_main_process:
        logger.info("Training completed!")


if __name__ == '__main__':
    main()

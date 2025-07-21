"""Training script for RT-DETRv3 PyTorch."""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import load_config
from src.core.workspace import create
from src.data import build_dataset, build_transforms, build_dataloader
from src.solver import build_trainer, build_evaluator
from src.misc.dist_utils import init_distributed_mode, cleanup_distributed_mode


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RT-DETRv3 Training Script')
    
    # Required arguments
    parser.add_argument('--config', required=True, help='Path to config file')
    
    # Optional arguments
    parser.add_argument('--resume', help='Path to checkpoint for resuming training')
    parser.add_argument('--pretrained', help='Path to pretrained model')
    parser.add_argument('--output-dir', help='Output directory (override config)')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic training')
    
    # Training parameters
    parser.add_argument('--max-epochs', type=int, help='Maximum epochs (override config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (override config)')
    parser.add_argument('--learning-rate', type=float, help='Learning rate (override config)')
    
    # Distributed training
    parser.add_argument('--world-size', type=int, default=1, help='Number of processes')
    parser.add_argument('--rank', type=int, default=0, help='Process rank')
    parser.add_argument('--dist-url', default='env://', help='URL for distributed training')
    parser.add_argument('--dist-backend', default='nccl', help='Distributed backend')
    parser.add_argument('--local-rank', type=int, default=-1, help='Local rank for distributed training')
    
    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    
    return parser.parse_args()


def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('train.log')
        ]
    )


def set_random_seed(seed, deterministic=False):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def build_model_and_criterion(cfg):
    """Build model and criterion from configuration."""
    # Build model
    model = create(cfg.model)
    
    # Build criterion
    criterion = create(cfg.criterion)
    
    return model, criterion


def build_optimizer_and_scheduler(cfg, model):
    """Build optimizer and scheduler from configuration."""
    # Build optimizer
    optimizer = create(cfg.optimizer, model=model)
    
    # Build scheduler
    scheduler = None
    if hasattr(cfg, 'lr_scheduler') and cfg.lr_scheduler is not None:
        scheduler = create(cfg.lr_scheduler, optimizer=optimizer)
    
    return optimizer, scheduler


def build_dataloaders(cfg, distributed=False):
    """Build train and validation dataloaders."""
    # Build transforms
    train_transforms = build_transforms(cfg.train_dataloader.dataset.transforms)
    val_transforms = build_transforms(cfg.val_dataloader.dataset.transforms)
    
    # Build datasets
    train_dataset = build_dataset({
        **cfg.train_dataloader.dataset,
        'transforms': train_transforms
    })
    
    val_dataset = build_dataset({
        **cfg.val_dataloader.dataset,
        'transforms': val_transforms
    })
    
    # Build dataloaders
    train_dataloader = build_dataloader(
        train_dataset,
        cfg.train_dataloader,
        distributed=distributed,
        training=True
    )
    
    val_dataloader = build_dataloader(
        val_dataset,
        cfg.val_dataloader,
        distributed=distributed,
        training=False
    )
    
    return train_dataloader, val_dataloader


def main_worker(rank, args):
    """Main worker function for distributed training."""
    # Initialize distributed mode
    if args.world_size > 1:
        init_distributed_mode(rank, args.world_size, args.dist_url, args.dist_backend)
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_random_seed(args.seed, args.deterministic)
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_epochs:
        cfg.max_epochs = args.max_epochs
    if args.batch_size:
        cfg.train_dataloader.batch_size = args.batch_size
        cfg.val_dataloader.batch_size = args.batch_size
    if args.learning_rate:
        cfg.optimizer.lr = args.learning_rate
    
    # Set device
    device = torch.device(args.device)
    if args.device == 'cuda':
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    
    logger.info(f'Using device: {device}')
    logger.info(f'Configuration: {cfg}')
    
    # Build model and criterion
    model, criterion = build_model_and_criterion(cfg)
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Build optimizer and scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)
    
    # Build dataloaders
    train_dataloader, val_dataloader = build_dataloaders(cfg, distributed=(args.world_size > 1))
    
    # Build evaluator
    evaluator = build_evaluator(cfg.evaluator)
    
    # Build trainer
    trainer = build_trainer(
        cfg.get('trainer', {}),
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )
    
    # Load checkpoint if specified
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        trainer.load_checkpoint(args.resume, resume_training=True)
    elif args.pretrained:
        logger.info(f'Loading pretrained model: {args.pretrained}')
        trainer.load_checkpoint(args.pretrained, resume_training=False)
    
    # Start training
    logger.info('Starting training...')
    trainer.train(train_dataloader, val_dataloader, evaluator)
    
    # Cleanup distributed mode
    if args.world_size > 1:
        cleanup_distributed_mode()


def main():
    """Main function."""
    args = parse_args()
    
    # Set environment variables for distributed training
    if args.local_rank != -1:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if args.world_size > 1:
        # Multi-process training
        mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)
    else:
        # Single-process training
        main_worker(0, args)


if __name__ == '__main__':
    main()
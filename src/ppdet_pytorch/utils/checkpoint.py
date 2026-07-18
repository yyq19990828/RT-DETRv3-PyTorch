"""
Checkpoint save/load utilities for RT-DETRv3 PyTorch

Handles model checkpoints, optimizer states, and training resumption.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .distributed import get_rank, is_main_process


logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    iteration: int,
    save_path: str,
    config: Optional[Dict] = None,
    best_metric: Optional[float] = None,
    scheduler: Optional[Any] = None,
    **kwargs
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        iteration: Current iteration
        save_path: Path to save checkpoint
        config: Optional config dict
        best_metric: Optional best metric value
        scheduler: Optional LR scheduler
        **kwargs: Additional items to save
    """
    if not is_main_process():
        return  # Only save on main process
    
    # Unwrap DDP model if needed
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'model': model_state,
        'epoch': epoch,
        'iteration': iteration,
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    if best_metric is not None:
        checkpoint['best_metric'] = best_metric
    
    # Add any additional items
    checkpoint.update(kwargs)
    
    # Create directory if needed
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path} (epoch={epoch}, iter={iteration})")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load checkpoint and restore model/optimizer states.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        strict: Whether to strictly enforce state_dict key matching
        map_location: Device to map tensors to
        
    Returns:
        Dictionary with checkpoint metadata (epoch, iteration, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    if map_location is None:
        map_location = f'cuda:{get_rank()}' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load model state
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        # Assume entire checkpoint is model state
        model_state = checkpoint
    
    # Handle DDP wrapped model
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state, strict=strict)
    else:
        model.load_state_dict(model_state, strict=strict)
    
    logger.info("Loaded model weights")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Loaded optimizer state")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded scheduler state")
    
    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'iteration': checkpoint.get('iteration', 0),
        'best_metric': checkpoint.get('best_metric', None),
        'config': checkpoint.get('config', None),
    }
    
    return metadata


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: str,
    strict: bool = False,
    prefix: Optional[str] = None
):
    """
    Load pretrained weights into model.
    
    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained weights file
        strict: Whether to strictly enforce key matching
        prefix: Optional prefix to add/remove from keys
    """
    pretrained_path = Path(pretrained_path)
    
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
    
    logger.info(f"Loading pretrained weights from {pretrained_path}")
    
    # Load weights
    state_dict = torch.load(pretrained_path, map_location='cpu')
    
    # Handle checkpoint format
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Handle prefix
    if prefix is not None:
        state_dict = {
            prefix + k if not k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }
    
    # Load into model
    if hasattr(model, 'module'):
        incompatible = model.module.load_state_dict(state_dict, strict=strict)
    else:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    
    if not strict and incompatible:
        logger.warning(f"Missing keys: {incompatible.missing_keys}")
        logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")
    
    logger.info("Loaded pretrained weights")


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """
    Get path to latest checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pth'))
    
    if not checkpoints:
        # Try 'model_*.pth' pattern
        checkpoints = list(checkpoint_dir.glob('model_*.pth'))
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    return latest


def resume_from_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Resume training from latest checkpoint in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Model to resume
        optimizer: Optional optimizer to resume
        scheduler: Optional scheduler to resume
        strict: Whether to strictly enforce key matching

    Returns:
        Dictionary with checkpoint metadata

    Raises:
        FileNotFoundError if no checkpoint found
    """
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    logger.info(f"Resuming from checkpoint: {latest_checkpoint}")

    return load_checkpoint(
        str(latest_checkpoint),
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        strict=strict
    )


# ============================================================================
# Paddle-compatible API functions
# ============================================================================

def convert_to_dict(obj):
    """
    Convert config object to dict (Paddle compatible).

    Args:
        obj: Config object to convert

    Returns:
        Dictionary representation
    """
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(i) for i in obj]
    elif hasattr(obj, '__dict__'):
        return {k: convert_to_dict(v) for k, v in obj.__dict__.items()
                if not k.startswith('_')}
    else:
        return obj


def load_weight(model, weight, optimizer=None, ema=None, exchange=True):
    """
    Load checkpoint for resuming training (Paddle compatible API).

    This function follows Paddle's load_weight signature and behavior.

    Args:
        model: Model to load weights into
        weight: Path to checkpoint file
        optimizer: Optional optimizer to load state
        ema: Optional EMA model to load state
        exchange: Whether to exchange model and ema weights (for Paddle compatibility)

    Returns:
        last_epoch: Epoch number to resume from
    """
    if not os.path.exists(weight):
        raise ValueError(f"Checkpoint file does not exist: {weight}")

    logger.info(f"Loading checkpoint from: {weight}")

    # Load checkpoint
    checkpoint = torch.load(weight, map_location='cpu')

    # Extract model state
    if 'model' in checkpoint:
        param_state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        param_state_dict = checkpoint['state_dict']
    else:
        param_state_dict = checkpoint

    # Load EMA state if exists
    ema_state_dict = None
    if ema is not None and 'ema' in checkpoint:
        if exchange:
            # Exchange model and ema_model to load (Paddle behavior)
            logger.info('Exchange model and ema_model to load:')
            ema_state_dict = param_state_dict
            param_state_dict = checkpoint['ema']
            logger.info(f'Loading ema_model weights from model checkpoint')
            logger.info(f'Loading model weights from ema checkpoint')
        else:
            ema_state_dict = checkpoint['ema']
            logger.info(f'Loading ema_model weights from ema checkpoint')
            logger.info(f'Loading model weights from model checkpoint')

    # Load model weights
    model_dict = model.state_dict()
    model_weight = {}
    incorrect_keys = 0

    for key in model_dict.keys():
        if key in param_state_dict.keys():
            model_weight[key] = param_state_dict[key]
        else:
            logger.info(f'Unmatched key: {key}')
            incorrect_keys += 1

    if incorrect_keys > 0:
        logger.warning(f"Load weight {weight} incorrectly, "
                      f"{incorrect_keys} keys unmatched")

    logger.info(f'Finish resuming model weights: {weight}')
    model.load_state_dict(model_weight, strict=False)

    # Load optimizer state
    last_epoch = 0
    if optimizer is not None and 'optimizer' in checkpoint:
        optim_state_dict = checkpoint['optimizer']

        # Handle missing keys in optimizer state
        for key in optimizer.state_dict().keys():
            if key not in optim_state_dict.keys():
                optim_state_dict[key] = optimizer.state_dict()[key]

        if 'last_epoch' in optim_state_dict:
            last_epoch = optim_state_dict.pop('last_epoch')

        optimizer.load_state_dict(optim_state_dict)
        logger.info("Loaded optimizer state")

        # Load EMA state
        if ema_state_dict is not None:
            if 'LR_Scheduler' in optim_state_dict and 'last_epoch' in optim_state_dict['LR_Scheduler']:
                ema.resume(ema_state_dict, optim_state_dict['LR_Scheduler']['last_epoch'])
            else:
                ema.resume(ema_state_dict)
            logger.info("Loaded EMA state")
    elif ema_state_dict is not None:
        ema.resume(ema_state_dict)
        logger.info("Loaded EMA state")

    # Get epoch from checkpoint
    if 'epoch' in checkpoint:
        last_epoch = checkpoint['epoch']

    return last_epoch


def load_pretrain_weight(model, pretrain_weight, ARSL_eval=False):
    """
    Load pretrained weights (Paddle compatible API).

    Args:
        model: Model to load weights into
        pretrain_weight: Path to pretrained weights file
        ARSL_eval: ARSL evaluation mode (for compatibility, not used in PyTorch)
    """
    if not os.path.exists(pretrain_weight):
        raise ValueError(f"Pretrained weight file does not exist: {pretrain_weight}")

    logger.info(f"Loading pretrained weights from: {pretrain_weight}")

    # Load checkpoint
    checkpoint = torch.load(pretrain_weight, map_location='cpu')

    # Extract model state
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load weights with non-strict mode
    if hasattr(model, 'module'):
        incompatible = model.module.load_state_dict(state_dict, strict=False)
    else:
        incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        logger.info(f"Missing keys when loading pretrained weights:")
        for key in incompatible.missing_keys[:10]:  # Show first 10
            logger.info(f"  - {key}")
        if len(incompatible.missing_keys) > 10:
            logger.info(f"  ... and {len(incompatible.missing_keys) - 10} more")

    if incompatible.unexpected_keys:
        logger.info(f"Unexpected keys when loading pretrained weights:")
        for key in incompatible.unexpected_keys[:10]:  # Show first 10
            logger.info(f"  - {key}")
        if len(incompatible.unexpected_keys) > 10:
            logger.info(f"  ... and {len(incompatible.unexpected_keys) - 10} more")

    logger.info(f"Loaded pretrained weights from {pretrain_weight}")

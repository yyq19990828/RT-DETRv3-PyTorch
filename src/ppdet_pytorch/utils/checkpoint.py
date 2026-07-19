"""
Checkpoint save/load utilities for RT-DETRv3 PyTorch

Handles model checkpoints, optimizer states, and training resumption.
"""

import logging
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .distributed import get_rank, is_main_process

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1


def capture_rng_state() -> Dict[str, Any]:
    """Capture process RNG state required for deterministic continuation."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    for device_index, cuda_state in enumerate(state.get("torch_cuda", [])):
        if device_index >= torch.cuda.device_count():
            break
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device_index)


def _select_rng_state(checkpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Select the current rank's RNG state with legacy fallback."""
    states_by_rank = checkpoint.get("rng_state_by_rank")
    if states_by_rank is None:
        return checkpoint.get("rng_state")

    rank = get_rank() if dist.is_initialized() else 0
    if rank >= len(states_by_rank) or states_by_rank[rank] is None:
        raise RuntimeError(
            "Checkpoint has no RNG state for rank {} (saved world size {})".format(
                rank, len(states_by_rank)
            )
        )
    return states_by_rank[rank]


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    iteration: int,
    save_path: str,
    config: Optional[Dict] = None,
    best_metric: Optional[float] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema: Optional[Any] = None,
    sampler_epoch: Optional[int] = None,
    gather_distributed_rng: bool = False,
    **kwargs,
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
        scaler: Optional AMP gradient scaler
        ema: Optional ModelEMA instance
        sampler_epoch: Epoch used to seed a distributed sampler
        gather_distributed_rng: Collect every rank's RNG state before rank 0
            writes the checkpoint. All ranks must call this function when it
            is enabled.
        **kwargs: Additional items to save
    """
    local_rng_state = capture_rng_state()
    rng_states_by_rank = None
    if gather_distributed_rng and dist.is_initialized():
        rng_states_by_rank = (
            [None] * dist.get_world_size() if is_main_process() else None
        )
        dist.gather_object(
            local_rng_state,
            object_gather_list=rng_states_by_rank,
            dst=0,
        )

    if not is_main_process():
        return False  # Only rank 0 publishes the shared checkpoint.

    # Unwrap DDP model if needed
    model_state = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model_state,
        "epoch": epoch,
        "iteration": iteration,
        "global_step": iteration,
        "rng_state": local_rng_state,
    }

    if rng_states_by_rank is not None:
        checkpoint["rng_state_by_rank"] = rng_states_by_rank

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if ema is not None:
        checkpoint["ema"] = ema.state_dict_for_save()

    if sampler_epoch is not None:
        checkpoint["sampler_epoch"] = sampler_epoch

    if config is not None:
        checkpoint["config"] = config

    if best_metric is not None:
        checkpoint["best_metric"] = best_metric

    # Add any additional items
    checkpoint.update(kwargs)

    # Create directory if needed
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Publish atomically so an interrupted write does not replace a valid
    # checkpoint with a partial file.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=save_path.parent,
            prefix=".{}.".format(save_path.name),
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(save_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    logger.info(f"Saved checkpoint to {save_path} (epoch={epoch}, iter={iteration})")
    return True


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema: Optional[Any] = None,
    restore_rng: bool = False,
    strict: bool = True,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint and restore model/optimizer states.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        scaler: Optional AMP gradient scaler to restore
        ema: Optional ModelEMA instance to restore
        restore_rng: Restore Python, NumPy and PyTorch RNG state
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
        map_location = f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    # Load model state
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        # Assume entire checkpoint is model state
        model_state = checkpoint

    # Handle DDP wrapped model
    if hasattr(model, "module"):
        model.module.load_state_dict(model_state, strict=strict)
    else:
        model.load_state_dict(model_state, strict=strict)

    logger.info("Loaded model weights")

    # Load optimizer state
    optimizer_state = checkpoint.get(
        "optimizer", checkpoint.get("optimizer_state_dict")
    )
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        logger.info("Loaded optimizer state")

    # Load scheduler state
    scheduler_state = checkpoint.get(
        "scheduler", checkpoint.get("scheduler_state_dict")
    )
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        logger.info("Loaded scheduler state")

    scaler_state = checkpoint.get("scaler", checkpoint.get("scaler_state_dict"))
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
        logger.info("Loaded scaler state")

    if ema is not None and "ema" in checkpoint:
        ema_state = checkpoint["ema"]
        if isinstance(ema_state, dict) and "ema_state_dict" in ema_state:
            ema.load_state_dict(ema_state)
        else:
            ema.resume(ema_state, checkpoint.get("global_step", 0))
        logger.info("Loaded EMA state")

    if restore_rng:
        rng_state = _select_rng_state(checkpoint)
        if rng_state is not None:
            restore_rng_state(rng_state)
            logger.info("Restored RNG state")

    # Extract metadata
    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "iteration": checkpoint.get("iteration", 0),
        "global_step": checkpoint.get("global_step", checkpoint.get("iteration", 0)),
        "sampler_epoch": checkpoint.get("sampler_epoch", checkpoint.get("epoch", 0)),
        "format_version": checkpoint.get("format_version", 0),
        "best_metric": checkpoint.get("best_metric", None),
        "config": checkpoint.get("config", None),
    }

    return metadata


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: str,
    strict: bool = False,
    prefix: Optional[str] = None,
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
    state_dict = torch.load(pretrained_path, map_location="cpu")

    # Handle checkpoint format
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Handle prefix
    if prefix is not None:
        state_dict = {
            prefix + k if not k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }

    # Load into model
    if hasattr(model, "module"):
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
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))

    if not checkpoints:
        # Try 'model_*.pth' pattern
        checkpoints = list(checkpoint_dir.glob("model_*.pth"))

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
    strict: bool = True,
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
        strict=strict,
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
    if isinstance(obj, Mapping):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_dict(i) for i in obj]
    elif isinstance(obj, (Path, torch.device)):
        return str(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif hasattr(obj, "__dict__"):
        return {
            k: convert_to_dict(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
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
    checkpoint = torch.load(weight, map_location="cpu")

    # Extract model state
    if "model" in checkpoint:
        param_state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        param_state_dict = checkpoint["state_dict"]
    else:
        param_state_dict = checkpoint

    # Load EMA state if exists
    ema_state_dict = None
    if ema is not None and "ema" in checkpoint:
        if exchange:
            # Exchange model and ema_model to load (Paddle behavior)
            logger.info("Exchange model and ema_model to load:")
            ema_state_dict = param_state_dict
            param_state_dict = checkpoint["ema"]
            logger.info("Loading ema_model weights from model checkpoint")
            logger.info("Loading model weights from ema checkpoint")
        else:
            ema_state_dict = checkpoint["ema"]
            logger.info("Loading ema_model weights from ema checkpoint")
            logger.info("Loading model weights from model checkpoint")

    # Load model weights
    model_dict = model.state_dict()
    model_weight = {}
    incorrect_keys = 0

    for key in model_dict.keys():
        if key in param_state_dict.keys():
            model_weight[key] = param_state_dict[key]
        else:
            logger.info(f"Unmatched key: {key}")
            incorrect_keys += 1

    if incorrect_keys > 0:
        logger.warning(
            f"Load weight {weight} incorrectly, {incorrect_keys} keys unmatched"
        )

    logger.info(f"Finish resuming model weights: {weight}")
    model.load_state_dict(model_weight, strict=False)

    # Load optimizer state
    last_epoch = 0
    if optimizer is not None and "optimizer" in checkpoint:
        optim_state_dict = checkpoint["optimizer"]

        # Handle missing keys in optimizer state
        for key in optimizer.state_dict().keys():
            if key not in optim_state_dict.keys():
                optim_state_dict[key] = optimizer.state_dict()[key]

        if "last_epoch" in optim_state_dict:
            last_epoch = optim_state_dict.pop("last_epoch")

        optimizer.load_state_dict(optim_state_dict)
        logger.info("Loaded optimizer state")

        # Load EMA state
        if ema_state_dict is not None:
            if (
                "LR_Scheduler" in optim_state_dict
                and "last_epoch" in optim_state_dict["LR_Scheduler"]
            ):
                ema.resume(
                    ema_state_dict, optim_state_dict["LR_Scheduler"]["last_epoch"]
                )
            else:
                ema.resume(ema_state_dict)
            logger.info("Loaded EMA state")
    elif ema_state_dict is not None:
        ema.resume(ema_state_dict)
        logger.info("Loaded EMA state")

    # Get epoch from checkpoint
    if "epoch" in checkpoint:
        last_epoch = checkpoint["epoch"]

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
    checkpoint = torch.load(pretrain_weight, map_location="cpu")

    # Extract model state
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load weights with non-strict mode
    if hasattr(model, "module"):
        incompatible = model.module.load_state_dict(state_dict, strict=False)
    else:
        incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        logger.info("Missing keys when loading pretrained weights:")
        for key in incompatible.missing_keys[:10]:  # Show first 10
            logger.info(f"  - {key}")
        if len(incompatible.missing_keys) > 10:
            logger.info(f"  ... and {len(incompatible.missing_keys) - 10} more")

    if incompatible.unexpected_keys:
        logger.info("Unexpected keys when loading pretrained weights:")
        for key in incompatible.unexpected_keys[:10]:  # Show first 10
            logger.info(f"  - {key}")
        if len(incompatible.unexpected_keys) > 10:
            logger.info(f"  ... and {len(incompatible.unexpected_keys) - 10} more")

    logger.info(f"Loaded pretrained weights from {pretrain_weight}")

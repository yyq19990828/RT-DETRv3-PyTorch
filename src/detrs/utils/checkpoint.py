"""
Checkpoint save/load utilities for RT-DETRv3 PyTorch

Handles model checkpoints, optimizer states, and training resumption.
"""

import hashlib
import logging
import os
import random
import tempfile
from collections.abc import Mapping
from copy import deepcopy
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
    training_state: Optional[Dict[str, Any]] = None,
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
        training_state: Optional family-independent training protocol state
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
    _validate_no_teacher_state(model_state, path="model")

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

    if training_state is not None:
        _validate_no_teacher_state(training_state)
        checkpoint["training_state"] = deepcopy(training_state)

    _validate_no_teacher_state(kwargs, path="checkpoint")
    # Add any additional items
    checkpoint.update(kwargs)

    # Create directory if needed
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Publish atomically so an interrupted write does not replace a valid
    # checkpoint with a partial file.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=save_file.parent,
            prefix=".{}.".format(save_file.name),
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(save_file)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    logger.info(f"Saved checkpoint to {save_file} (epoch={epoch}, iter={iteration})")
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
    protocol: Optional[Any] = None,
    expected_model_identity: Optional[str] = None,
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
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    logger.info(f"Loading checkpoint from {checkpoint_file}")

    # Deserialization and all validation happen on CPU before live state changes.
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    format_version = checkpoint.get("format_version")
    if format_version is not None and format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format_version: {}".format(format_version)
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

    target_model = model.module if hasattr(model, "module") else model
    optimizer_state = checkpoint.get(
        "optimizer", checkpoint.get("optimizer_state_dict")
    )
    scheduler_state = checkpoint.get(
        "scheduler", checkpoint.get("scheduler_state_dict")
    )
    scaler_state = checkpoint.get("scaler", checkpoint.get("scaler_state_dict"))
    ema_state = checkpoint.get("ema")
    training_state = checkpoint.get("training_state")

    _preflight_checkpoint(
        checkpoint_file,
        checkpoint,
        target_model,
        model_state,
        optimizer,
        optimizer_state,
        scheduler,
        scheduler_state,
        scaler,
        scaler_state,
        ema,
        ema_state,
        protocol,
        training_state,
        expected_model_identity,
        strict,
        restore_rng,
    )

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "iteration": checkpoint.get("iteration", 0),
        "global_step": checkpoint.get("global_step", checkpoint.get("iteration", 0)),
        "sampler_epoch": checkpoint.get("sampler_epoch", checkpoint.get("epoch", 0)),
        "format_version": checkpoint.get("format_version", 0),
        "best_metric": checkpoint.get("best_metric", None),
        "config": checkpoint.get("config", None),
    }
    if training_state is not None:
        metadata["training_state"] = training_state

    snapshots = {
        "model": deepcopy(target_model.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict())
        if optimizer is not None
        else None,
        "scheduler": deepcopy(scheduler.state_dict())
        if scheduler is not None
        else None,
        "scaler": deepcopy(scaler.state_dict()) if scaler is not None else None,
        "ema": deepcopy(ema.state_dict_for_save()) if ema is not None else None,
        "protocol": deepcopy(protocol.state_dict()) if protocol is not None else None,
        "rng": capture_rng_state() if restore_rng else None,
    }
    try:
        target_model.load_state_dict(model_state, strict=strict)
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        if ema is not None and ema_state is not None:
            _load_ema_state(ema, ema_state, checkpoint.get("global_step", 0))
        if protocol is not None and training_state is not None:
            protocol.load_state_dict(training_state["protocol_state"])
        if restore_rng:
            rng_state = _select_rng_state(dict(checkpoint))
            if rng_state is not None:
                restore_rng_state(rng_state)
        if protocol is not None and training_state is not None:
            protocol.after_load(training_state, metadata)
    except Exception:
        model_snapshot = snapshots["model"]
        if not isinstance(model_snapshot, Mapping):
            raise TypeError("model snapshot must contain a mapping")
        target_model.load_state_dict(model_snapshot)
        if optimizer is not None:
            optimizer.load_state_dict(snapshots["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(snapshots["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(snapshots["scaler"])
        if ema is not None:
            ema_snapshot = snapshots["ema"]
            if not isinstance(ema_snapshot, Mapping):
                raise TypeError("EMA snapshot must contain a mapping")
            _load_ema_state(ema, ema_snapshot, checkpoint.get("global_step", 0))
        if protocol is not None:
            protocol.load_state_dict(snapshots["protocol"])
        if restore_rng:
            rng_snapshot = snapshots["rng"]
            if not isinstance(rng_snapshot, dict):
                raise TypeError("RNG snapshot must contain a dictionary")
            restore_rng_state(rng_snapshot)
        raise

    return metadata


def _validate_no_teacher_state(value: Any, path: str = "training_state") -> None:
    if isinstance(value, nn.Module):
        raise ValueError(
            "module serialization is forbidden in checkpoints: {}".format(path)
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_path = "{}.{}".format(path, key)
            if "teacher" in str(key).lower():
                raise ValueError(
                    "teacher state is forbidden in checkpoints: {}".format(key_path)
                )
            _validate_no_teacher_state(item, key_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_no_teacher_state(item, "{}[{}]".format(path, index))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ema_state(ema: Any, state: Mapping[str, Any], global_step: int) -> None:
    if isinstance(state, Mapping) and "ema_state_dict" in state:
        ema.load_state_dict(state)
    else:
        ema.resume(state, global_step)


def _validate_component_state(component: Any, state: Any, name: str) -> None:
    if component is None:
        return
    if state is None:
        raise ValueError("checkpoint is missing required {} state".format(name))
    candidate = deepcopy(component)
    candidate.load_state_dict(deepcopy(state))


def _validate_optimizer_scheduler_state(
    optimizer: Any,
    optimizer_state: Any,
    scheduler: Any,
    scheduler_state: Any,
) -> None:
    if optimizer is None and scheduler is None:
        return
    if optimizer is None:
        _validate_component_state(scheduler, scheduler_state, "scheduler")
        return
    if optimizer_state is None:
        raise ValueError("checkpoint is missing required optimizer state")
    if scheduler is None:
        _validate_component_state(optimizer, optimizer_state, "optimizer")
        return
    if scheduler_state is None:
        raise ValueError("checkpoint is missing required scheduler state")

    # Preserve the optimizer reference held by stateful schedulers while
    # rehearsing the same load order used for live components.
    candidate_optimizer, candidate_scheduler = deepcopy((optimizer, scheduler))
    candidate_optimizer.load_state_dict(deepcopy(optimizer_state))
    candidate_scheduler.load_state_dict(deepcopy(scheduler_state))


def _preflight_checkpoint(
    checkpoint_file: Path,
    checkpoint: Mapping[str, Any],
    model: nn.Module,
    model_state: Mapping[str, Any],
    optimizer: Optional[Any],
    optimizer_state: Any,
    scheduler: Optional[Any],
    scheduler_state: Any,
    scaler: Optional[Any],
    scaler_state: Any,
    ema: Optional[Any],
    ema_state: Any,
    protocol: Optional[Any],
    training_state: Any,
    expected_model_identity: Optional[str],
    strict: bool,
    restore_rng: bool,
) -> None:
    deepcopy(model).load_state_dict(deepcopy(model_state), strict=strict)
    protocol_checkpoint = training_state is not None or protocol is not None
    if protocol_checkpoint:
        if training_state is None or not isinstance(training_state, Mapping):
            raise ValueError("checkpoint is missing required training_state")
        _validate_no_teacher_state(training_state)
        required = {
            "model_identity",
            "protocol_identity",
            "protocol_stage",
            "protocol_state",
        }
        missing = sorted(required - set(training_state))
        if missing:
            raise ValueError("training_state is missing: {}".format(", ".join(missing)))
        if protocol is None:
            raise ValueError("checkpoint training_state requires a configured protocol")
        if training_state["protocol_identity"] != protocol.identity:
            raise ValueError("training protocol identity mismatch")
        protocol.validate_checkpoint_stage(training_state["protocol_stage"])
        if (
            expected_model_identity is not None
            and training_state["model_identity"] != expected_model_identity
        ):
            raise ValueError("checkpoint model identity mismatch")
        protocol.validate_state_dict(
            training_state["protocol_state"], str(checkpoint_file)
        )
        candidate_protocol = deepcopy(protocol)
        candidate_protocol.load_state_dict(deepcopy(training_state["protocol_state"]))
        if candidate_protocol.checkpoint_stage != training_state["protocol_stage"]:
            raise ValueError("training protocol stage mismatch")
        protocol_state = training_state["protocol_state"]
        companion_name = protocol_state.get("companion_basename")
        companion_sha = protocol_state.get("companion_sha256")
        if companion_name is not None or companion_sha is not None:
            if not companion_name or not companion_sha:
                raise ValueError("companion basename and SHA-256 must both be present")
            if Path(companion_name).name != companion_name:
                raise ValueError("companion_basename must not contain a path")
            companion_path = checkpoint_file.parent / companion_name
            if not companion_path.is_file():
                raise FileNotFoundError(
                    "checkpoint companion not found: {}".format(companion_path)
                )
            actual_sha = _sha256(companion_path)
            if actual_sha != companion_sha:
                raise ValueError("checkpoint companion SHA-256 mismatch")

    _validate_optimizer_scheduler_state(
        optimizer, optimizer_state, scheduler, scheduler_state
    )
    _validate_component_state(scaler, scaler_state, "scaler")
    if ema is not None:
        if ema_state is None:
            raise ValueError("checkpoint is missing required EMA state")
        candidate_ema = deepcopy(ema)
        _load_ema_state(
            candidate_ema, deepcopy(ema_state), checkpoint.get("global_step", 0)
        )
    if restore_rng:
        rng_state = _select_rng_state(dict(checkpoint))
        if rng_state is None:
            raise ValueError("checkpoint is missing required RNG state")


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
    pretrained_file = Path(pretrained_path)

    if not pretrained_file.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_file}")

    logger.info(f"Loading pretrained weights from {pretrained_file}")

    # Load weights
    state_dict = torch.load(pretrained_file, map_location="cpu")

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
    checkpoint_directory = Path(checkpoint_dir)

    if not checkpoint_directory.exists():
        return None

    # Find all checkpoint files
    checkpoints = list(checkpoint_directory.glob("checkpoint_*.pth"))

    if not checkpoints:
        # Try 'model_*.pth' pattern
        checkpoints = list(checkpoint_directory.glob("model_*.pth"))

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
    checkpoint = torch.load(pretrain_weight, map_location="cpu", weights_only=False)

    # Extract model state
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    target = model.module if hasattr(model, "module") else model
    target_keys = set(target.state_dict())
    state_keys = set(state_dict)
    prefixed_keys = {"backbone." + key for key in state_keys}
    backbone_keys = {
        key.removeprefix("backbone.")
        for key in target_keys
        if key.startswith("backbone.")
    }
    missing_backbone_keys = backbone_keys - state_keys
    if (
        state_keys
        and state_keys <= backbone_keys
        and all(key.endswith(".num_batches_tracked") for key in missing_backbone_keys)
    ):
        state_dict = {"backbone." + key: value for key, value in state_dict.items()}
    elif prefixed_keys & target_keys:
        raise ValueError(
            "Backbone pretrained weight keys only partially match the model"
        )

    # Load weights with non-strict mode
    incompatible = target.load_state_dict(state_dict, strict=False)

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

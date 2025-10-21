"""
Distributed training utilities for RT-DETRv3 PyTorch

Adapted from PaddlePaddle's distributed APIs to PyTorch's torch.distributed.
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def init_distributed_mode(backend: str = 'nccl') -> bool:
    """
    Initialize distributed training process group.
    
    Args:
        backend: Distribution backend ('nccl' for GPU, 'gloo' for CPU)
        
    Returns:
        True if distributed training is enabled, False otherwise
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        logger.info("Not using distributed mode")
        return False
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    dist.barrier()
    
    logger.info(
        f"Initialized distributed mode: rank={rank}, "
        f"world_size={world_size}, local_rank={local_rank}, backend={backend}"
    )
    
    return True


def get_rank() -> int:
    """
    Get current process rank in distributed training.
    
    Returns:
        Process rank (0 if not distributed)
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """
    Get total number of processes in distributed training.
    
    Returns:
        World size (1 if not distributed)
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """
    Get local rank (rank within node) for distributed training.
    
    Returns:
        Local rank from environment variable or 0
    """
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """Check if current process is main process (rank 0)"""
    return get_rank() == 0


def barrier():
    """Synchronize all processes"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Input tensor
        op: Reduction operation (SUM, AVG, MIN, MAX)
        
    Returns:
        Reduced tensor
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    return tensor


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor using average across all processes.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Averaged tensor
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    world_size = get_world_size()
    tensor = reduce_tensor(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / world_size
    return tensor


def gather_tensors(tensor: torch.Tensor) -> Optional[list]:
    """
    Gather tensors from all processes to main process.
    
    Args:
        tensor: Input tensor
        
    Returns:
        List of tensors on main process, None on other processes
    """
    if not dist.is_available() or not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    rank = get_rank()
    
    # Prepare gather list on main process
    if rank == 0:
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None
    
    # Gather
    dist.gather(tensor, gather_list, dst=0)
    
    return gather_list if rank == 0 else None


def all_gather_tensors(tensor: torch.Tensor) -> list:
    """
    Gather tensors from all processes to all processes.
    
    Args:
        tensor: Input tensor
        
    Returns:
        List of tensors from all processes
    """
    if not dist.is_available() or not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    dist.all_gather(gather_list, tensor)
    
    return gather_list


def synchronize():
    """
    Synchronize all processes.
    Helper function that is safe to call even when distributed is not initialized.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    
    world_size = get_world_size()
    if world_size == 1:
        return
    
    dist.barrier()


def setup_for_distributed(is_master: bool):
    """
    Disable printing when not in master process.
    
    Args:
        is_master: Whether current process is master
    """
    import builtins as __builtin__
    
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")

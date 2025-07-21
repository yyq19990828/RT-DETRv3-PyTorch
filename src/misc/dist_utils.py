"""
Distributed training utilities
"""

import os
import torch
import torch.distributed as dist


def init_distributed_mode(rank, world_size, dist_url='env://', dist_backend='nccl'):
    """Initialize distributed mode"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
    else:
        local_rank = 0 # Single-process, non-distributed
    
    if world_size > 1:
        dist.init_process_group(
            backend=dist_backend,
            init_method=dist_url,
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        dist.barrier()


def cleanup_distributed_mode():
    """Cleanup distributed mode"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_world_size():
    """Get the number of processes in distributed training"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    """Get the rank of current process in distributed training"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    """Check if current process is the main process"""
    return get_rank() == 0


def barrier():
    """Synchronize all processes"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor, dst=0, op=dist.ReduceOp.SUM):
    """Reduce tensor across all processes"""
    if dist.is_available() and dist.is_initialized():
        dist.reduce(tensor, dst, op)
    return tensor


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """All-reduce tensor across all processes"""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op)
    return tensor
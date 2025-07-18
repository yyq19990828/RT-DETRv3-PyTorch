"""
Distributed training utilities
"""

import torch
import torch.distributed as dist


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
# NOTE this is a torch-nlp re-implementation of
# import torch
# from torch.utils.data.sampler import BatchSampler
# from torch.utils.data.sampler import Sampler


# class DistributedSampler(Sampler):
#     """ Iterable wrapper that distributes data across multiple workers.

#     Args:
#         iterable (iterable)
#         num_replicas (int, optional): Number of processes participating in distributed training.
#         rank (int, optional): Rank of the current process within ``num_replicas``.

#     Example:
#         >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
#         [0, 2, 4, 6, 8]
#         >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
#         [1, 3, 5, 7, 9]
#     """

#     def __init__(self, iterable, num_replicas=None, rank=None):
#         self.iterable = iterable
#         self.num_replicas = num_replicas
#         self.rank = rank

#         if num_replicas is None or rank is None:  # pragma: no cover
#             if not torch.distributed.is_initialized():
#                 raise RuntimeError('Requires `torch.distributed` to be initialized.')

#             self.num_replicas = (
#                 torch.distributed.get_world_size() if num_replicas is None else num_replicas)
#             self.rank = torch.distributed.get_rank() if rank is None else rank

#         if self.rank >= self.num_replicas:
#             raise IndexError('`rank` must be smaller than the `num_replicas`.')

#     def __iter__(self):
#         return iter(
#             [e for i, e in enumerate(self.iterable) if (i - self.rank) % self.num_replicas == 0])

#     def __len__(self):
#         return len(self.iterable)
    

# class DistributedBatchSampler(BatchSampler):
#     """ `BatchSampler` wrapper that distributes across each batch multiple workers.

#     Args:
#         batch_sampler (torch.utils.data.sampler.BatchSampler)
#         num_replicas (int, optional): Number of processes participating in distributed training.
#         rank (int, optional): Rank of the current process within num_replicas.

#     Example:
#         >>> from torch.utils.data.sampler import BatchSampler
#         >>> from torch.utils.data.sampler import SequentialSampler
#         >>> sampler = SequentialSampler(list(range(12)))
#         >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
#         >>>
#         >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
#         [[0, 2], [4, 6], [8, 10]]
#         >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
#         [[1, 3], [5, 7], [9, 11]]
#     """

#     def __init__(self, batch_sampler, **kwargs):
#         self.batch_sampler = batch_sampler
#         self.kwargs = kwargs

#     def __iter__(self):
#         for batch in self.batch_sampler:
#             yield list(DistributedSampler(batch, **self.kwargs))

#     def __len__(self):
#         return len(self.batch_sampler)


from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler

class DistributedBatchSampler(BatchSampler):
    def __init__(self, batch_sampler, num_replicas=None, rank=None, shuffle=True, seed=0):
        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        for batch in self.batch_sampler:
            # 对每个batch的索引应用分布式采样，使不同进程得到不同的子集
            yield list(DistributedSampler(batch, num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle, seed=self.seed))

    def __len__(self):
        return len(self.batch_sampler)

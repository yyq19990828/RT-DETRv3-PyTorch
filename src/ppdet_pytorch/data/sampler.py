import torch.distributed as dist
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class DistributedBatchSampler(BatchSampler):
    """Build batches for single-process or distributed training.

    PaddleDetection readers configure a dataset, batch size, shuffle flag and
    drop-last flag directly. PyTorch separates sample and batch samplers, so
    this adapter selects the appropriate sample sampler before delegating the
    batching behavior to :class:`torch.utils.data.BatchSampler`.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        drop_last=False,
        num_replicas=None,
        rank=None,
        seed=0,
    ):
        distributed = dist.is_initialized() or num_replicas is not None or rank is not None
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        elif shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        super().__init__(sampler, batch_size=batch_size, drop_last=drop_last)

    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

import random

import numpy as np
import torch

from ppdet_pytorch.data.reader import BaseDataLoader
from ppdet_pytorch.data.sampler import DistributedBatchSampler


class _IndexDataset:
    def __init__(self, size):
        self.size = size
        self.epoch = None
        self.transform = None

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"index": np.asarray(index, dtype=np.int64)}

    def check_or_download_dataset(self):
        pass

    def parse_dataset(self):
        pass

    def set_transform(self, transform):
        self.transform = transform

    def set_kwargs(self, **kwargs):
        pass

    def set_epoch(self, epoch):
        self.epoch = epoch


class _RandomDataset(_IndexDataset):
    def __getitem__(self, index):
        return {
            "index": index,
            "python": random.random(),
            "numpy": np.random.random(),
            "torch": torch.rand(()).item(),
        }


def _flatten_batches(batch_sampler):
    return [index for batch in batch_sampler for index in batch]


def test_single_process_sampler_is_reproducible_per_epoch():
    sampler = DistributedBatchSampler(
        list(range(20)), batch_size=4, shuffle=True, seed=17
    )

    sampler.set_epoch(3)
    epoch_three_first = _flatten_batches(sampler)
    sampler.set_epoch(3)
    epoch_three_second = _flatten_batches(sampler)
    sampler.set_epoch(4)
    epoch_four = _flatten_batches(sampler)

    assert epoch_three_first == epoch_three_second
    assert epoch_three_first != epoch_four


def test_distributed_sampler_uses_same_base_seed_and_disjoint_ranks():
    rank_zero = DistributedBatchSampler(
        list(range(20)),
        batch_size=2,
        shuffle=True,
        num_replicas=2,
        rank=0,
        seed=23,
    )
    rank_one = DistributedBatchSampler(
        list(range(20)),
        batch_size=2,
        shuffle=True,
        num_replicas=2,
        rank=1,
        seed=23,
    )
    rank_zero.set_epoch(5)
    rank_one.set_epoch(5)

    zero_indices = _flatten_batches(rank_zero)
    one_indices = _flatten_batches(rank_one)

    assert set(zero_indices).isdisjoint(one_indices)
    assert sorted(zero_indices + one_indices) == list(range(20))


def test_reader_recreates_epoch_order_and_worker_seed():
    dataset = _IndexDataset(size=20)
    reader = BaseDataLoader(
        batch_size=4,
        shuffle=True,
        collate_batch=True,
        seed=31,
    )(dataset, worker_num=0)

    reader.set_epoch(2)
    first = [batch["index"].tolist() for batch in reader]
    first_worker_seed = reader._worker_generator.initial_seed()
    reader.set_epoch(2)
    second = [batch["index"].tolist() for batch in reader]
    reader.set_epoch(3)
    third = [batch["index"].tolist() for batch in reader]

    assert first == second
    assert first != third
    assert first_worker_seed == 33
    assert dataset.epoch == 3


def test_reader_reseeds_python_numpy_and_torch_workers_per_epoch():
    dataset = _RandomDataset(size=12)
    reader = BaseDataLoader(
        batch_size=3,
        shuffle=False,
        collate_batch=True,
        seed=41,
    )(dataset, worker_num=2)

    reader.set_epoch(2)
    first = list(reader)
    reader.set_epoch(2)
    repeated = list(reader)
    reader.set_epoch(3)
    next_epoch = list(reader)

    for key in ("python", "numpy", "torch"):
        first_values = np.concatenate([batch[key] for batch in first])
        repeated_values = np.concatenate([batch[key] for batch in repeated])
        next_values = np.concatenate([batch[key] for batch in next_epoch])
        np.testing.assert_array_equal(first_values, repeated_values)
        assert not np.array_equal(first_values, next_values)

"""
Batch collation for RT-DETRv3 PyTorch

Handle variable-length annotations and create properly batched tensors.
"""

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import traceback
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ppdet_pytorch.core.workspace import register
from ppdet_pytorch.utils.logger import setup_logger

from . import transform
from .sampler import DistributedBatchSampler
from .utils import default_collate_fn

logger = setup_logger("reader")

MAIN_PID = os.getpid()


class Compose(object):
    def __init__(self, transforms, num_classes=80):
        self.transforms = transforms
        self.transforms_cls = []
        for t in self.transforms:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, "num_classes"):
                    f.num_classes = num_classes

                self.transforms_cls.append(f)

    def _update_transforms_cls(self, data):
        if "transform_schedulers" in data:

            def is_valid(op):
                op_name = op.__class__.__name__
                for t in data["transform_schedulers"]:
                    for k, v in t.items():
                        if op_name == k:
                            # [start_epoch, stop_epoch)
                            start_epoch = v.get("start_epoch", 0)
                            if start_epoch > data["curr_epoch"]:
                                return False
                            stop_epoch = v.get("stop_epoch", float("inf"))
                            if stop_epoch <= data["curr_epoch"]:
                                return False
                return True

            return filter(is_valid, self.transforms_cls)
        else:
            return self.transforms_cls

    def __call__(self, data):
        transforms_cls = self._update_transforms_cls(data)
        for f in transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning(
                    "fail to map sample transform [{}] "
                    "with error: {} and stack:\n{}".format(f, e, str(stack_info))
                )
                raise e

        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        transforms_cls = self._update_transforms_cls(data[0])
        for f in transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning(
                    "fail to map batch transform [{}] "
                    "with error: {} and stack:\n{}".format(f, e, str(stack_info))
                )
                raise e

        # remove keys which is not needed by model
        extra_key = ["h", "w", "flipped", "transform_schedulers"]
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    tmp_data.append(data[i][k])
                if "gt_" not in k and "is_crowd" not in k and "difficult" not in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                if "origin_" in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data


class BaseDataLoader(object):
    """
    Base DataLoader implementation for detection models

    Args:
        sample_transforms (list): a list of transforms to perform
                                  on each sample
        batch_transforms (list): a list of transforms to perform
                                 on batch
        batch_size (int): batch size for batch collating, default 1.
        shuffle (bool): whether to shuffle samples
        drop_last (bool): whether to drop the last incomplete,
                          default False
        num_classes (int): class number of dataset, default 80
        collate_batch (bool): whether to collate batch in dataloader.
            If set to True, the samples will collate into batch according
            to the batch size. Otherwise, the ground-truth will not collate,
            which is used when the number of ground-truch is different in
            samples.
        use_shared_memory (bool): Legacy Paddle-compatible option. In the
                PyTorch loader it enables pinned host memory for faster
                host-to-device transfers. Default False.
    """

    def __init__(
        self,
        sample_transforms=[],
        batch_transforms=[],
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_classes=80,
        collate_batch=True,
        use_shared_memory=False,
        seed=0,
        **kwargs,
    ):
        # sample transform
        self._sample_transforms = Compose(sample_transforms, num_classes=num_classes)

        # batch transfrom
        self._batch_transforms = BatchCompose(
            batch_transforms, num_classes, collate_batch
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = bool(use_shared_memory)
        self.seed = int(seed)
        self.kwargs = kwargs

    def __call__(self, dataset, worker_num, batch_sampler=None, return_list=False):
        # Kept for compatibility with Paddle-style reader configuration.
        # PyTorch always returns the object produced by ``collate_fn``.
        del return_list
        self.dataset = dataset
        self.dataset.check_or_download_dataset()
        self.dataset.parse_dataset()
        # get data
        self.dataset.set_transform(self._sample_transforms)
        # set kwargs
        self.dataset.set_kwargs(**self.kwargs)
        # batch sampler
        if batch_sampler is None:
            self._batch_sampler = DistributedBatchSampler(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                seed=self.seed,
            )
        else:
            self._batch_sampler = batch_sampler

        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._worker_generator = torch.Generator()
        self._seed_worker_generator(epoch=0)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            pin_memory=self.pin_memory,
            generator=self._worker_generator,
        )
        self.loader = None

        return self

    def _seed_worker_generator(self, epoch):
        epoch_seed = self.seed + int(epoch) * self._world_size + self._rank
        self._worker_generator.manual_seed(epoch_seed)

    def set_epoch(self, epoch):
        """Reset sampler and worker RNG deterministically for an epoch."""
        if hasattr(self._batch_sampler, "set_epoch"):
            self._batch_sampler.set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        self._seed_worker_generator(epoch)
        self.loader = None

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        if self.loader is None:
            self.loader = iter(self.dataloader)
        return self

    def __next__(self):
        if self.loader is None:
            self.loader = iter(self.dataloader)
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = None
            raise

    def next(self):
        # python2 compatibility
        return self.__next__()


@register
class TrainReader(BaseDataLoader):
    __shared__ = ["num_classes"]

    def __init__(
        self,
        sample_transforms=[],
        batch_transforms=[],
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_classes=80,
        collate_batch=True,
        **kwargs,
    ):
        super(TrainReader, self).__init__(
            sample_transforms,
            batch_transforms,
            batch_size,
            shuffle,
            drop_last,
            num_classes,
            collate_batch,
            **kwargs,
        )


@register
class EvalReader(BaseDataLoader):
    __shared__ = ["num_classes"]

    def __init__(
        self,
        sample_transforms=[],
        batch_transforms=[],
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_classes=80,
        **kwargs,
    ):
        super(EvalReader, self).__init__(
            sample_transforms,
            batch_transforms,
            batch_size,
            shuffle,
            drop_last,
            num_classes,
            **kwargs,
        )


@register
class TestReader(BaseDataLoader):
    __shared__ = ["num_classes"]

    def __init__(
        self,
        sample_transforms=[],
        batch_transforms=[],
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_classes=80,
        **kwargs,
    ):
        super(TestReader, self).__init__(
            sample_transforms,
            batch_transforms,
            batch_size,
            shuffle,
            drop_last,
            num_classes,
            **kwargs,
        )


@register
class EvalMOTReader(BaseDataLoader):
    __shared__ = ["num_classes"]

    def __init__(
        self,
        sample_transforms=[],
        batch_transforms=[],
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_classes=1,
        **kwargs,
    ):
        super(EvalMOTReader, self).__init__(
            sample_transforms,
            batch_transforms,
            batch_size,
            shuffle,
            drop_last,
            num_classes,
            **kwargs,
        )


@register
class TestMOTReader(BaseDataLoader):
    __shared__ = ["num_classes"]

    def __init__(
        self,
        sample_transforms=[],
        batch_transforms=[],
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_classes=1,
        **kwargs,
    ):
        super(TestMOTReader, self).__init__(
            sample_transforms,
            batch_transforms,
            batch_size,
            shuffle,
            drop_last,
            num_classes,
            **kwargs,
        )


# ===========================================================================================
# For Semi-Supervised Object Detection (SSOD)
# ===========================================================================================
class Compose_SSOD(object):
    def __init__(self, base_transforms, weak_aug, strong_aug, num_classes=80):
        self.base_transforms = base_transforms
        self.base_transforms_cls = []
        for t in self.base_transforms:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, "num_classes"):
                    f.num_classes = num_classes
                self.base_transforms_cls.append(f)

        self.weak_augs = weak_aug
        self.weak_augs_cls = []
        for t in self.weak_augs:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, "num_classes"):
                    f.num_classes = num_classes
                self.weak_augs_cls.append(f)

        self.strong_augs = strong_aug
        self.strong_augs_cls = []
        for t in self.strong_augs:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, "num_classes"):
                    f.num_classes = num_classes
                self.strong_augs_cls.append(f)

    def __call__(self, data):
        for f in self.base_transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning(
                    "fail to map sample transform [{}] "
                    "with error: {} and stack:\n{}".format(f, e, str(stack_info))
                )
                raise e

        weak_data = deepcopy(data)
        strong_data = deepcopy(data)
        for f in self.weak_augs_cls:
            try:
                weak_data = f(weak_data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning(
                    "fail to map weak aug [{}] with error: {} and stack:\n{}".format(
                        f, e, str(stack_info)
                    )
                )
                raise e

        for f in self.strong_augs_cls:
            try:
                strong_data = f(strong_data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning(
                    "fail to map strong aug [{}] with error: {} and stack:\n{}".format(
                        f, e, str(stack_info)
                    )
                )
                raise e

        weak_data["strong_aug"] = strong_data
        return weak_data


class BatchCompose_SSOD(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose_SSOD, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        # split strong_data from data(weak_data)
        strong_data = []
        for sample in data:
            strong_data.append(sample["strong_aug"])
            sample.pop("strong_aug")

        for f in self.transforms_cls:
            try:
                data = f(data)
                if "BatchRandomResizeForSSOD" in f._id:
                    strong_data = f(strong_data, data[1])[0]
                    data = data[0]
                else:
                    strong_data = f(strong_data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning(
                    "fail to map batch transform [{}] "
                    "with error: {} and stack:\n{}".format(f, e, str(stack_info))
                )
                raise e

        # remove keys which is not needed by model
        extra_key = ["h", "w", "flipped"]
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)
            for sample in strong_data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
            strong_batch_data = default_collate_fn(strong_data)
            return batch_data, strong_batch_data
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    tmp_data.append(data[i][k])
                if "gt_" not in k and "is_crowd" not in k and "difficult" not in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data

            strong_batch_data = {}
            for k in strong_data[0].keys():
                tmp_data = []
                for i in range(len(strong_data)):
                    tmp_data.append(strong_data[i][k])
                if "gt_" not in k and "is_crowd" not in k and "difficult" not in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                strong_batch_data[k] = tmp_data

        return batch_data, strong_batch_data


class CombineSSODLoader(object):
    def __init__(self, label_loader, unlabel_loader):
        self.label_loader = label_loader
        self.unlabel_loader = unlabel_loader

    def __iter__(self):
        label_loader_iter = iter(self.label_loader)
        unlabel_loader_iter = iter(self.unlabel_loader)
        while True:
            try:
                label_samples = next(label_loader_iter)
            except StopIteration:
                label_loader_iter = iter(self.label_loader)
                label_samples = next(label_loader_iter)

            try:
                unlabel_samples = next(unlabel_loader_iter)
            except StopIteration:
                unlabel_loader_iter = iter(self.unlabel_loader)
                unlabel_samples = next(unlabel_loader_iter)

            yield (
                label_samples[0],  # sup weak
                label_samples[1],  # sup strong
                unlabel_samples[0],  # unsup weak
                unlabel_samples[1],  # unsup strong
            )

    def __call__(self):
        return self.__iter__()


class BaseSemiDataLoader(object):
    def __init__(
        self,
        sample_transforms=[],
        weak_aug=[],
        strong_aug=[],
        sup_batch_transforms=[],
        unsup_batch_transforms=[],
        sup_batch_size=1,
        unsup_batch_size=1,
        shuffle=True,
        drop_last=True,
        num_classes=80,
        collate_batch=True,
        use_shared_memory=False,
        **kwargs,
    ):
        # sup transforms
        self._sample_transforms_label = Compose_SSOD(
            sample_transforms, weak_aug, strong_aug, num_classes=num_classes
        )
        self._batch_transforms_label = BatchCompose_SSOD(
            sup_batch_transforms, num_classes, collate_batch
        )
        self.batch_size_label = sup_batch_size

        # unsup transforms
        self._sample_transforms_unlabel = Compose_SSOD(
            sample_transforms, weak_aug, strong_aug, num_classes=num_classes
        )
        self._batch_transforms_unlabel = BatchCompose_SSOD(
            unsup_batch_transforms, num_classes, collate_batch
        )
        self.batch_size_unlabel = unsup_batch_size

        # common
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = bool(use_shared_memory)
        self.kwargs = kwargs

    def __call__(
        self,
        dataset_label,
        dataset_unlabel,
        worker_num,
        batch_sampler_label=None,
        batch_sampler_unlabel=None,
        return_list=False,
    ):
        # Kept for compatibility with Paddle-style reader configuration.
        # PyTorch always returns the object produced by ``collate_fn``.
        del return_list
        # sup dataset
        self.dataset_label = dataset_label
        self.dataset_label.check_or_download_dataset()
        self.dataset_label.parse_dataset()
        self.dataset_label.set_transform(self._sample_transforms_label)
        self.dataset_label.set_kwargs(**self.kwargs)
        if batch_sampler_label is None:
            self._batch_sampler_label = DistributedBatchSampler(
                self.dataset_label,
                batch_size=self.batch_size_label,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )
        else:
            self._batch_sampler_label = batch_sampler_label

        # unsup dataset
        self.dataset_unlabel = dataset_unlabel
        self.dataset_unlabel.length = self.dataset_label.__len__()
        self.dataset_unlabel.check_or_download_dataset()
        self.dataset_unlabel.parse_dataset()
        self.dataset_unlabel.set_transform(self._sample_transforms_unlabel)
        self.dataset_unlabel.set_kwargs(**self.kwargs)
        if batch_sampler_unlabel is None:
            self._batch_sampler_unlabel = DistributedBatchSampler(
                self.dataset_unlabel,
                batch_size=self.batch_size_unlabel,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )
        else:
            self._batch_sampler_unlabel = batch_sampler_unlabel

        self.dataloader_label = DataLoader(
            dataset=self.dataset_label,
            batch_sampler=self._batch_sampler_label,
            collate_fn=self._batch_transforms_label,
            num_workers=worker_num,
            pin_memory=self.pin_memory,
        )

        self.dataloader_unlabel = DataLoader(
            dataset=self.dataset_unlabel,
            batch_sampler=self._batch_sampler_unlabel,
            collate_fn=self._batch_transforms_unlabel,
            num_workers=worker_num,
            pin_memory=self.pin_memory,
        )

        self.dataloader = CombineSSODLoader(
            self.dataloader_label, self.dataloader_unlabel
        )
        self.loader = iter(self.dataloader)
        return self

    def __len__(self):
        return len(self._batch_sampler_label)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.loader)

    def next(self):
        # python2 compatibility
        return self.__next__()


@register
class SemiTrainReader(BaseSemiDataLoader):
    __shared__ = ["num_classes"]

    def __init__(
        self,
        sample_transforms=[],
        weak_aug=[],
        strong_aug=[],
        sup_batch_transforms=[],
        unsup_batch_transforms=[],
        sup_batch_size=1,
        unsup_batch_size=1,
        shuffle=True,
        drop_last=True,
        num_classes=80,
        collate_batch=True,
        **kwargs,
    ):
        super(SemiTrainReader, self).__init__(
            sample_transforms,
            weak_aug,
            strong_aug,
            sup_batch_transforms,
            unsup_batch_transforms,
            sup_batch_size,
            unsup_batch_size,
            shuffle,
            drop_last,
            num_classes,
            collate_batch,
            **kwargs,
        )

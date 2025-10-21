"""
Base dataset class for object detection.

Migrated from PaddlePaddle RT-DETRv3 to PyTorch.
Preserves all logic branches from Paddle version for future extensibility.
"""

import os
import copy
import numpy as np
from typing import List, Optional, Any, Dict
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import torch
from torch.utils.data import Dataset

from ppdet_pytorch.core.workspace import register, serializable

import logging
logger = logging.getLogger(__name__)


@serializable
class DetDataset(Dataset):
    """
    Base class for detection datasets.

    Migrated from Paddle ppdet/data/source/dataset.py to maintain compatibility.

    Args:
        dataset_dir (str): Root directory for dataset.
        image_dir (str): Directory for images (relative to dataset_dir).
        anno_path (str): Annotation file path (relative to dataset_dir).
        data_fields (list): Key names of data dictionary, at least have 'image'.
        sample_num (int): Number of samples to load, -1 means all.
        use_default_label (bool): Whether to load default label list (unused, preserved for compatibility).
        repeat (int): Repeat times for dataset, use in benchmark.
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        anno_path: Optional[str] = None,
        data_fields: List[str] = None,
        sample_num: int = -1,
        use_default_label: Optional[bool] = None,
        repeat: int = 1,
        **kwargs
    ):
        super(DetDataset, self).__init__()

        # Core parameters
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.data_fields = data_fields if data_fields is not None else ['image']
        self.sample_num = sample_num
        self.use_default_label = use_default_label  # Preserved for compatibility
        self.repeat = repeat

        # Internal state
        self._epoch = 0
        self._curr_iter = 0

        # Augmentation scheduling (preserved from Paddle for CutMix, Mosaic, etc.)
        self.mixup_epoch = -1
        self.cutmix_epoch = -1
        self.mosaic_epoch = -1
        self.pre_img_epoch = -1  # For CenterTrack (multi-object tracking)
        self.transform_schedulers = None

        # Transform function (set by Trainer/Reader)
        self.transform = None

        # Records storage (populated by parse_dataset)
        self.roidbs = []

    def __len__(self) -> int:
        return len(self.roidbs) * self.repeat

    def __call__(self, *args, **kwargs):
        """Callable interface for compatibility with Paddle."""
        return self

    def __getitem__(self, idx: int) -> Any:
        """
        Get dataset item with optional augmentation scheduling.

        Preserves Paddle logic:
        - Mixup/Cutmix/Mosaic scheduling based on epoch
        - Previous image for CenterTrack
        - Transform scheduler passing

        Args:
            idx: Dataset index

        Returns:
            Transformed sample (dict or list of dicts for multi-image augmentation)
        """
        n = len(self.roidbs)

        # Handle dataset repeat
        if self.repeat > 1:
            idx %= n

        # Deep copy to avoid modifying original records
        roidb = copy.deepcopy(self.roidbs[idx])

        # Augmentation scheduling (epoch-based)
        # Priority: Mixup > Cutmix > Mosaic > PreImg
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            # Mixup: blend current image with another random image
            idx_mix = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx_mix])]
        elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            # Cutmix: similar to Mixup but with cut-and-paste
            idx_cut = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx_cut])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            # Mosaic: combine 4 images in a 2x2 grid
            roidb = [roidb] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)])
                for _ in range(4)
            ]
        elif self.pre_img_epoch == 0 or self._epoch < self.pre_img_epoch:
            # Previous image: for temporal detection (CenterTrack, MOT)
            idx_pre = idx - 1
            if idx_pre < 0:
                idx_pre = idx + 1
            roidb = [roidb, copy.deepcopy(self.roidbs[idx_pre])]

        # Inject iteration and epoch info into records
        if isinstance(roidb, Sequence):
            for r in roidb:
                r['curr_iter'] = self._curr_iter
                r['curr_epoch'] = self._epoch
        else:
            roidb['curr_iter'] = self._curr_iter
            roidb['curr_epoch'] = self._epoch

        self._curr_iter += 1

        # Inject transform schedulers if provided
        if self.transform_schedulers:
            assert isinstance(self.transform_schedulers, list)
            if isinstance(roidb, Sequence):
                for r in roidb:
                    r['transform_schedulers'] = self.transform_schedulers
            else:
                roidb['transform_schedulers'] = self.transform_schedulers

        # Apply transforms
        if self.transform is not None:
            return self.transform(roidb)
        else:
            return roidb

    def check_or_download_dataset(self):
        """
        Check dataset existence or download if needed.

        Note: PyTorch version does not include auto-download,
        preserved method for API compatibility.
        """
        if not os.path.exists(self.dataset_dir):
            logger.warning(
                f"Dataset directory {self.dataset_dir} does not exist. "
                "Please download the dataset manually."
            )

    def set_kwargs(self, **kwargs):
        """Set augmentation scheduling parameters."""
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)
        self.pre_img_epoch = kwargs.get('pre_img_epoch', -1)
        self.transform_schedulers = kwargs.get('transform_schedulers', None)

    def set_transform(self, transform):
        """Set transform function."""
        self.transform = transform

    def set_epoch(self, epoch_id: int):
        """
        Set current epoch for augmentation scheduling.

        Should be called at the beginning of each epoch by Trainer.
        """
        self._epoch = epoch_id

    def parse_dataset(self):
        """
        Parse dataset annotations and populate self.roidbs.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Need to implement parse_dataset method of Dataset"
        )

    def get_anno(self) -> Optional[str]:
        """Get full path to annotation file."""
        if self.anno_path is None:
            return None
        return os.path.join(self.dataset_dir, self.anno_path)


def _is_valid_file(f: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')) -> bool:
    """Check if file has valid image extension."""
    return f.lower().endswith(extensions)


def _make_dataset(dir: str) -> List[str]:
    """
    Make dataset by scanning directory for images.

    Preserved from Paddle for ImageFolder-style datasets (unused in COCO/LVIS but kept for compatibility).
    """
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        raise ValueError(f'{dir} should be a directory')

    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in sorted(fnames):
            if _is_valid_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

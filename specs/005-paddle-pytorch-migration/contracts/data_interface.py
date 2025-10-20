"""
Data Interface Contract: RT-DETRv3 PyTorch Migration

Defines the interface contracts for datasets, transforms, and data loaders.
"""

from typing import Dict, List, Tuple, Callable, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class BaseDataset(Dataset):
    """
    Base interface for all detection datasets.

    All datasets must implement this interface to ensure compatibility
    with the training pipeline.
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Sample dict:
            {
                'image': np.ndarray [H, W, 3] or Tensor [3, H, W],
                'gt_bbox': np.ndarray [N, 4] in (x1, y1, x2, y2) format,
                'gt_class': np.ndarray [N] class IDs (0-indexed),
                'gt_score': np.ndarray [N] confidence scores (default all 1.0),
                'im_shape': np.ndarray [2] original image shape (H, W),
                'scale_factor': np.ndarray [2] resize scale factors,
                'im_id': int image identifier
            }
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Total number of samples."""
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        """Number of object classes (excluding background)."""
        raise NotImplementedError


class COCODatasetInterface(BaseDataset):
    """
    Interface contract for COCO-format datasets.

    Extends BaseDataset with COCO-specific requirements.
    """

    def __init__(
        self,
        dataset_dir: str,
        image_dir: str,
        anno_path: str,
        transforms: Optional[List[Callable]] = None,
        num_classes: int = 80
    ):
        """
        Initialize COCO dataset.

        Args:
            dataset_dir: Root directory of dataset
            image_dir: Image directory (relative to dataset_dir)
            anno_path: Annotation file path (JSON)
            transforms: List of transform functions
            num_classes: Number of classes
        """
        super().__init__()

    def load_annotations(self, anno_path: str) -> List[Dict]:
        """
        Load COCO annotations from JSON file.

        Args:
            anno_path: Path to annotations JSON

        Returns:
            List of annotation dicts
        """
        raise NotImplementedError

    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """
        Get image metadata.

        Args:
            idx: Image index

        Returns:
            Image info dict:
            {
                'id': int,
                'file_name': str,
                'height': int,
                'width': int
            }
        """
        raise NotImplementedError


class TransformInterface:
    """Base interface for data transforms."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transform to a sample.

        Args:
            sample: Input sample dict

        Returns:
            Transformed sample dict (same structure as input)
        """
        raise NotImplementedError


class MosaicInterface(TransformInterface):
    """
    Interface for Mosaic data augmentation.

    Combines 4 images into a 2x2 grid.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        prob: float = 1.0,
        dataset: Optional[BaseDataset] = None
    ):
        """
        Initialize Mosaic transform.

        Args:
            target_size: Output image size (H, W)
            prob: Probability of applying Mosaic
            dataset: Reference to dataset for sampling other images
        """
        pass

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Mosaic augmentation.

        Args:
            sample: Input sample

        Returns:
            Augmented sample with 4 images combined
        """
        raise NotImplementedError


class MixupInterface(TransformInterface):
    """
    Interface for Mixup data augmentation.

    Blends two images together.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        prob: float = 1.0,
        dataset: Optional[BaseDataset] = None
    ):
        """
        Initialize Mixup transform.

        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying Mixup
            dataset: Reference to dataset for sampling another image
        """
        pass

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Mixup augmentation.

        Args:
            sample: Input sample

        Returns:
            Mixed sample
        """
        raise NotImplementedError


class ResizeInterface(TransformInterface):
    """Interface for image resize transform."""

    def __init__(
        self,
        target_size: Tuple[int, int],
        keep_ratio: bool = False,
        interp: int = 1  # cv2.INTER_LINEAR
    ):
        """
        Initialize Resize transform.

        Args:
            target_size: Target size (H, W)
            keep_ratio: Whether to keep aspect ratio
            interp: Interpolation method
        """
        pass


class RandomFlipInterface(TransformInterface):
    """Interface for random horizontal flip."""

    def __init__(self, prob: float = 0.5):
        """
        Initialize RandomFlip.

        Args:
            prob: Flip probability
        """
        pass


class NormalizeInterface(TransformInterface):
    """Interface for image normalization."""

    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        is_scale: bool = True  # Whether to scale to [0, 1] first
    ):
        """
        Initialize Normalize transform.

        Args:
            mean: Channel-wise mean
            std: Channel-wise std
            is_scale: If True, scale to [0, 1] before normalization
        """
        pass


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for detection datasets.

    Handles variable-length bboxes and classes.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched data:
        {
            'image': Tensor [B, 3, H, W],
            'gt_bbox': List of Tensors [N_i, 4],  # Variable length
            'gt_class': List of Tensors [N_i],
            'gt_score': List of Tensors [N_i],
            'im_shape': Tensor [B, 2],
            'scale_factor': Tensor [B, 2],
            'im_id': Tensor [B]
        }
    """
    images = torch.stack([torch.from_numpy(s['image']).permute(2, 0, 1)
                          if isinstance(s['image'], np.ndarray)
                          else s['image']
                          for s in batch], dim=0)

    # Keep variable-length bboxes as list
    gt_bboxes = [torch.from_numpy(s['gt_bbox'])
                 if isinstance(s['gt_bbox'], np.ndarray)
                 else s['gt_bbox']
                 for s in batch]

    gt_classes = [torch.from_numpy(s['gt_class']).long()
                  if isinstance(s['gt_class'], np.ndarray)
                  else s['gt_class']
                  for s in batch]

    gt_scores = [torch.from_numpy(s['gt_score']).float()
                 if isinstance(s['gt_score'], np.ndarray)
                 else s['gt_score']
                 for s in batch]

    im_shapes = torch.stack([torch.from_numpy(s['im_shape'])
                             if isinstance(s['im_shape'], np.ndarray)
                             else s['im_shape']
                             for s in batch], dim=0)

    scale_factors = torch.stack([torch.from_numpy(s['scale_factor'])
                                 if isinstance(s['scale_factor'], np.ndarray)
                                 else s['scale_factor']
                                 for s in batch], dim=0)

    im_ids = torch.tensor([s['im_id'] for s in batch])

    return {
        'image': images,
        'gt_bbox': gt_bboxes,
        'gt_class': gt_classes,
        'gt_score': gt_scores,
        'im_shape': im_shapes,
        'scale_factor': scale_factors,
        'im_id': im_ids
    }


class DataLoaderInterface:
    """
    Interface contract for data loaders.

    Wraps PyTorch DataLoader with detection-specific requirements.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = True
    ):
        """
        Initialize DataLoader.

        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            drop_last: Whether to drop last incomplete batch
            collate_fn: Custom collate function (default: collate_batch)
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        if collate_fn is None:
            collate_fn = collate_batch

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            worker_init_fn=self._worker_init_fn
        )

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """
        Initialize worker with unique random seed.

        Ensures reproducibility in multi-process data loading.
        """
        import numpy as np
        import random
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


# Type aliases for clarity
SampleDict = Dict[str, Any]
BatchDict = Dict[str, Any]
TransformList = List[Callable]

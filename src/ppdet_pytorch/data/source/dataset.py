"""
Base dataset class for object detection.

Migrated from PaddlePaddle RT-DETRv3 to PyTorch.
Preserves all logic branches from Paddle version for future extensibility.
"""

import copy
import os
from typing import Any, List, Optional

import numpy as np

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import logging

from pycocotools.coco import COCO
from torch.utils.data import Dataset

from ppdet_pytorch.core.workspace import register, serializable
from ppdet_pytorch.data import source

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
        data_fields: Optional[List[str]] = None,
        sample_num: int = -1,
        use_default_label: Optional[bool] = None,
        repeat: int = 1,
        **kwargs,
    ):
        super(DetDataset, self).__init__()

        # Core parameters
        self.dataset_dir = dataset_dir if dataset_dir is not None else ""
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ""
        self.data_fields = data_fields if data_fields is not None else ["image"]
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
        self.dense_o2o_policy = None
        self.dense_o2o_seed = 0

        # Transform function (set by Trainer/Reader)
        self.transform = None

        # Records storage (populated by parse_dataset)
        self.roidbs: List[Any] = []

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
                copy.deepcopy(self.roidbs[np.random.randint(n)]) for _ in range(4)
            ]
        elif self.pre_img_epoch == 0 or self._epoch < self.pre_img_epoch:
            # Previous image: for temporal detection (CenterTrack, MOT)
            idx_pre = idx - 1
            if idx_pre < 0:
                idx_pre = idx + 1
            roidb = [roidb, copy.deepcopy(self.roidbs[idx_pre])]

        if self.dense_o2o_policy is not None:
            start, stop, _ = self.dense_o2o_policy["policy_epochs"]
            mosaic_config = self.dense_o2o_policy.get("mosaic", {})
            if start <= self._epoch < stop and not mosaic_config.get(
                "use_cache", False
            ):
                rng = np.random.default_rng(
                    self.dense_o2o_seed + self._epoch * max(n, 1) + idx
                )
                roidb = [roidb] + [
                    copy.deepcopy(self.roidbs[int(rng.integers(n))]) for _ in range(3)
                ]

        # Inject iteration and epoch info into records
        if isinstance(roidb, Sequence):
            for r in roidb:
                r["curr_iter"] = self._curr_iter
                r["curr_epoch"] = self._epoch
        else:
            roidb["curr_iter"] = self._curr_iter
            roidb["curr_epoch"] = self._epoch

        self._curr_iter += 1

        # Inject transform schedulers if provided
        if self.transform_schedulers:
            assert isinstance(self.transform_schedulers, list)
            if isinstance(roidb, Sequence):
                for r in roidb:
                    r["transform_schedulers"] = self.transform_schedulers
            else:
                roidb["transform_schedulers"] = self.transform_schedulers

        # Apply transforms
        if self.transform is not None:
            return self.transform(roidb)
        else:
            return roidb

    # TODO: implement download online dataset method of pytorch if needed
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
        self.mixup_epoch = kwargs.get("mixup_epoch", -1)
        self.cutmix_epoch = kwargs.get("cutmix_epoch", -1)
        self.mosaic_epoch = kwargs.get("mosaic_epoch", -1)
        self.pre_img_epoch = kwargs.get("pre_img_epoch", -1)
        self.transform_schedulers = kwargs.get("transform_schedulers", None)
        self.dense_o2o_policy = kwargs.get("dense_o2o_policy", None)
        self.dense_o2o_seed = int(kwargs.get("dense_o2o_seed", 0))

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
        raise NotImplementedError("Need to implement parse_dataset method of Dataset")

    def get_anno(self) -> Optional[str]:
        """Get full path to annotation file."""
        if self.anno_path is None:
            return None
        return os.path.join(self.dataset_dir, self.anno_path)


def _is_valid_file(
    f: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")
) -> bool:
    """Check if file has valid image extension."""
    return f.lower().endswith(extensions)


def _make_dataset(dir: str) -> List[str]:
    """
    Make dataset by scanning directory for images.

    Preserved from Paddle for ImageFolder-style datasets (unused in COCO/LVIS but kept for compatibility).
    """
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        raise ValueError(f"{dir} should be a directory")

    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in sorted(fnames):
            if _is_valid_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


@register
@serializable
class ImageFolder(DetDataset):
    def __init__(
        self,
        dataset_dir=None,
        image_dir=None,
        anno_path=None,
        sample_num=-1,
        use_default_label=None,
        **kwargs,
    ):
        super(ImageFolder, self).__init__(
            dataset_dir,
            image_dir,
            anno_path,
            sample_num=sample_num,
            use_default_label=use_default_label,
        )
        self._imid2path = {}
        self.roidbs = []
        self.sample_num = sample_num

    def check_or_download_dataset(self):
        return

    def get_anno(self):
        if self.anno_path is None:
            return
        if self.dataset_dir:
            return os.path.join(self.dataset_dir, self.anno_path)
        else:
            return self.anno_path

    def parse_dataset(
        self,
    ):
        if not self.roidbs:
            self.roidbs = self._load_images()

    def _parse(self):
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def get_images(self):
        if self.anno_path is None:
            raise ValueError("anno_path is required to load images from COCO metadata")
        images_path = []
        coco = COCO(os.path.join(self.dataset_dir, self.anno_path))
        imgIds = coco.getImgIds(catIds=[])
        for imgId in imgIds:
            filename = coco.loadImgs(imgId)[0]["file_name"]
            images_path.append(os.path.join(self.dataset_dir, self.image_dir, filename))
        return images_path

    def _load_images(self, do_eval=False):
        images = self._parse()
        ct = 0
        records = []
        anno_file = self.get_anno()
        coco = None
        if do_eval:
            if anno_file is None:
                raise ValueError("anno_path is required when do_eval=True")
            coco = COCO(anno_file)
        for image in images:
            assert image != "" and os.path.isfile(image), "Image {} not found".format(
                image
            )
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            if do_eval:
                assert coco is not None
                image_id = self.get_image_id(image, coco)
                ct = image_id
            rec = {"im_id": np.array([ct]), "im_file": image}
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def get_image_id(self, image, coco):
        image_ids = coco.getImgIds()
        for image_id in image_ids:
            img_info = coco.loadImgs(image_id)[0]
            if img_info["file_name"] in image:
                return image_id
            else:
                continue
        raise ValueError(f"Image {image} is not present in the annotation file")

    def get_imid2path(self):
        return self._imid2path

    def set_images(self, images, do_eval=False):
        self.image_dir = images
        self.roidbs = self._load_images(do_eval=do_eval)

    def set_slice_images(
        self, images, slice_size=[640, 640], overlap_ratio=[0.25, 0.25]
    ):
        self.image_dir = images
        ori_records = self._load_images()
        try:
            import sahi
        except Exception as e:
            logger.error(
                "sahi not found, plaese install sahi. "
                "for example: `pip install sahi`, see https://github.com/obss/sahi."
            )
            raise e

        sub_img_ids = 0
        ct = 0
        ct_sub = 0
        records = []
        for i, ori_rec in enumerate(ori_records):
            im_path = ori_rec["im_file"]
            slice_image_result = sahi.slicing.slice_image(
                image=im_path,
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1],
            )

            sub_img_num = len(slice_image_result)
            for _ind in range(sub_img_num):
                im = slice_image_result.images[_ind]
                rec = (
                    {
                        "image": im,
                        "im_id": np.array([sub_img_ids + _ind]),
                        "h": im.shape[0],
                        "w": im.shape[1],
                        "ori_im_id": np.array([ori_rec["im_id"][0]]),
                        "st_pix": np.array(
                            slice_image_result.starting_pixels[_ind], dtype=np.float32
                        ),
                        "is_last": 1 if _ind == sub_img_num - 1 else 0,
                    }
                    if "image" in self.data_fields
                    else {}
                )
                records.append(rec)
            ct_sub += sub_img_num
            ct += 1
        logger.info("{} samples and slice to {} sub_samples.".format(ct, ct_sub))
        self.roidbs = records

    def get_label_list(self):
        # Only VOC dataset needs label list in ImageFold
        return self.anno_path


@register
class CommonDataset(object):
    def __init__(self, **dataset_args):
        super(CommonDataset, self).__init__()
        dataset_args = copy.deepcopy(dataset_args)
        type = dataset_args.pop("name")
        self.dataset = getattr(source, type)(**dataset_args)

    def __call__(self):
        return self.dataset


@register
class TrainDataset(CommonDataset):
    pass


@register
class EvalMOTDataset(CommonDataset):
    pass


@register
class TestMOTDataset(CommonDataset):
    pass


@register
class EvalDataset(CommonDataset):
    pass


@register
class TestDataset(CommonDataset):
    pass

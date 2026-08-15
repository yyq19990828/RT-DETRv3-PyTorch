"""
YOLO format dataset loader for RT-DETRv3 PyTorch.

Expects the YOLO directory convention: image files under image_dir and one
label .txt per image under label_dir sharing the same stem. Each label line
is `class_id cx cy w h` with box coordinates normalized to [0, 1].

Multiple dataset folders are supported without physically merging them:
image_dir/label_dir accept either a single path or equal-length lists that
are paired by index.
"""

import logging
import os
from typing import Any, List, Optional, Union

import numpy as np
from PIL import Image

from detrs.core.workspace import register, serializable

from .dataset import DetDataset, _make_dataset

logger = logging.getLogger(__name__)

__all__ = ["YOLODataSet"]


def _as_path_list(value: Optional[Union[str, List[str]]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


@register
@serializable
class YOLODataSet(DetDataset):
    """
    Load dataset with YOLO format.

    Args:
        dataset_dir (str): Root directory for dataset.
        image_dir (str | list): Directory(ies) for images (relative to dataset_dir).
        label_dir (str | list): Directory(ies) for label txt files (relative to
            dataset_dir), paired with image_dir by index when lists are given.
        data_fields (list): Key names of data dictionary, at least have 'image'.
        sample_num (int): Number of samples to load, -1 means all.
        label_list (str): Path to a class-name file, one name per line, whose
            line number is the class id. If not provided, placeholder names
            class_0..class_N are derived from the labels.
        allow_empty (bool): Whether to load empty entries (images without a
            label file or with no valid boxes). False as default.
        empty_ratio (float): Ratio of empty records to total records.
                             If out of [0, 1), do not sample and use all empty entries. 1.0 as default.
        repeat (int): Repeat times for dataset, use in benchmark.
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        image_dir: Optional[Any] = None,
        label_dir: Optional[Any] = None,
        data_fields: Optional[List[str]] = None,
        sample_num: int = -1,
        label_list: Optional[str] = None,
        allow_empty: bool = False,
        empty_ratio: float = 1.0,
        repeat: int = 1,
    ):
        if data_fields is None:
            data_fields = ["image"]

        super(YOLODataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir if image_dir is not None else "",
            anno_path=None,
            data_fields=data_fields,
            sample_num=sample_num,
            repeat=repeat,
        )

        self.label_dir = label_dir if label_dir is not None else ""
        self.label_list = label_list
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

        self.cname2cid: dict[str, int] = {}

    def get_anno(self) -> Optional[str]:
        """YOLO labels are per-image txt files; return the class-name list instead."""
        if self.label_list is None:
            return None
        return os.path.join(self.dataset_dir, self.label_list)

    def _load_label_list(self) -> Optional[List[str]]:
        if self.label_list is None:
            return None
        label_path = os.path.join(self.dataset_dir, self.label_list)
        if not os.path.exists(label_path):
            raise ValueError(f"label_list {label_path} does not exist")
        with open(label_path, "r") as fr:
            names = [line.strip() for line in fr.readlines() if line.strip()]
        if len(names) == 0:
            raise ValueError(f"label_list {label_path} is empty")
        return names

    def _parse_label_file(self, label_path: str, im_w: float, im_h: float):
        """
        Parse one YOLO label txt into absolute-pixel xyxy boxes and class ids.

        Returns (gt_bbox (N,4) float32, gt_class (N,1) int32, max_class_id) or
        None when the file has no valid boxes.
        """
        boxes = []
        classes = []
        max_class_id = -1
        with open(label_path, "r") as fr:
            for line in fr:
                parts = line.split()
                if len(parts) < 5:
                    if line.strip():
                        logger.warning(
                            f"Malformed YOLO label line {line!r} in {label_path}, "
                            "will be ignored"
                        )
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x1 = (cx - bw / 2.0) * im_w
                y1 = (cy - bh / 2.0) * im_h
                x2 = (cx + bw / 2.0) * im_w
                y2 = (cy + bh / 2.0) * im_h
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    logger.warning(
                        f"Found an invalid bbox in {label_path}: "
                        f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, will be ignored."
                    )
                    continue
                boxes.append([x1, y1, x2, y2])
                classes.append(cls_id)
                max_class_id = max(max_class_id, cls_id)

        if not boxes:
            return None
        return (
            np.array(boxes, dtype=np.float32),
            np.array(classes, dtype=np.int32).reshape(-1, 1),
            max_class_id,
        )

    def parse_dataset(self):
        """
        Parse YOLO label files and populate self.roidbs.
        """
        image_dirs = _as_path_list(self.image_dir)
        label_dirs = _as_path_list(self.label_dir)
        if not image_dirs:
            raise ValueError("image_dir is required for YOLODataSet")
        if len(image_dirs) != len(label_dirs):
            raise ValueError(
                "image_dir and label_dir must have the same length for "
                f"YOLODataSet, got {len(image_dirs)} and {len(label_dirs)}"
            )

        class_names = self._load_label_list()
        if class_names is not None:
            self.cname2cid = {name: i for i, name in enumerate(class_names)}

        records: List[dict] = []
        empty_records: List[dict] = []
        ct = 0
        im_id = 0
        max_class_id = -1

        for image_rel, label_rel in zip(image_dirs, label_dirs):
            image_root = os.path.join(self.dataset_dir, image_rel)
            label_root = os.path.join(self.dataset_dir, label_rel)
            image_files = _make_dataset(image_root)

            for im_path in image_files:
                if self.sample_num > 0 and ct >= self.sample_num:
                    break

                stem = os.path.splitext(os.path.basename(im_path))[0]
                label_path = os.path.join(label_root, stem + ".txt")

                parsed = None
                im_w = im_h = 0.0
                if os.path.isfile(label_path):
                    with Image.open(im_path) as image:
                        im_w, im_h = image.size
                    parsed = self._parse_label_file(label_path, im_w, im_h)
                elif self.allow_empty:
                    with Image.open(im_path) as image:
                        im_w, im_h = image.size
                else:
                    logger.warning(
                        f"Label file {label_path} not found for image {im_path}, "
                        "will be ignored"
                    )

                if parsed is None:
                    if not self.allow_empty:
                        continue
                    gt_bbox = np.zeros((0, 4), dtype=np.float32)
                    gt_class = np.zeros((0, 1), dtype=np.int32)
                else:
                    gt_bbox, gt_class, file_max_class_id = parsed
                    max_class_id = max(max_class_id, file_max_class_id)
                    if class_names is not None and file_max_class_id >= len(
                        class_names
                    ):
                        raise ValueError(
                            f"Class id {file_max_class_id} in {label_path} exceeds "
                            f"label_list with {len(class_names)} classes"
                        )

                yolo_rec = {
                    "im_file": im_path,
                    "im_id": np.array([im_id]),
                    "h": float(im_h),
                    "w": float(im_w),
                }

                gt_rec = {
                    "gt_bbox": gt_bbox,
                    "gt_class": gt_class,
                    "is_crowd": np.zeros((len(gt_class), 1), dtype=np.int32),
                }
                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        yolo_rec[k] = v

                if parsed is None:
                    empty_records.append(yolo_rec)
                else:
                    records.append(yolo_rec)

                im_id += 1
                ct += 1

            if self.sample_num > 0 and ct >= self.sample_num:
                break

        assert ct > 0, f"Not found any YOLO record in {self.image_dir}"

        if class_names is None:
            self.cname2cid = {f"class_{i}": i for i in range(max_class_id + 1)}
        elif max_class_id >= 0:
            logger.info(
                f"Loaded {len(records)} YOLO samples with {len(class_names)} "
                f"classes from label_list {self.label_list}"
            )

        # Sample and add empty records
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records

        self.roidbs = records
        logger.info(f"YOLO dataset loaded: {len(self.roidbs)} samples")

    def _sample_empty(self, records: List[dict], num: int) -> List[dict]:
        """
        Sample empty records based on empty_ratio.

        Args:
            records: List of empty record dicts
            num: Number of non-empty records

        Returns:
            Sampled empty records
        """
        # If empty_ratio is out of [0, 1), do not sample
        if self.empty_ratio < 0.0 or self.empty_ratio >= 1.0:
            return records

        import random

        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records)
        )
        records = random.sample(records, sample_num)
        return records

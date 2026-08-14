"""
Pascal VOC dataset loader for RT-DETRv3 PyTorch.

Migrated from PaddlePaddle RT-DETRv3 (ppdet/data/source/voc.py).
VOC format dataset with XML annotations.

Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
"""

import logging
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import numpy as np

from detrs.core.workspace import register, serializable

from .dataset import DetDataset

logger = logging.getLogger(__name__)


def _required_text(element: ET.Element, path: str, xml_file: str) -> str:
    value = element.findtext(path)
    if value is None:
        raise ValueError(f"Missing XML field {path!r} in {xml_file}")
    return value


@register
@serializable
class VOCDataSet(DetDataset):
    """
    Load dataset with PascalVOC format.

    Notes:
        `anno_path` must contain a list file where each line has format:
        <image_path> <xml_annotation_path>

    Args:
        dataset_dir (str): Root directory for dataset.
        image_dir (str): Directory for images (relative to dataset_dir).
        anno_path (str): VOC annotation list file path.
        data_fields (list): Key names of data dictionary, at least have 'image'.
        sample_num (int): Number of samples to load, -1 means all.
        label_list (str): Path to label mapping file. If not provided, uses default VOC 20 classes.
        allow_empty (bool): Whether to load empty entries (images without annotations). False as default.
        empty_ratio (float): Ratio of empty records to total records.
                             If out of [0, 1), do not sample and use all empty entries. 1.0 as default.
        repeat (int): Repeat times for dataset, use in benchmark.
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        anno_path: Optional[str] = None,
        data_fields: Optional[List[str]] = None,
        sample_num: int = -1,
        label_list: Optional[str] = None,
        allow_empty: bool = False,
        empty_ratio: float = 1.0,
        repeat: int = 1,
    ):
        if data_fields is None:
            data_fields = ["image"]

        super(VOCDataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            repeat=repeat,
        )

        # VOC-specific parameters
        self.label_list = label_list
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

        # Category mapping (populated in parse_dataset)
        self.cname2cid: Dict[str, int] = {}

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

    def parse_dataset(self):
        """
        Parse VOC dataset annotations.

        Populates self.roidbs and self.cname2cid with dataset records and category mapping.
        Anno_path should be a text file with each line containing:
            <image_path> <xml_annotation_path>
        """
        if self.anno_path is None:
            raise ValueError("anno_path is required for VOCDataSet")
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        # Build category name to class ID mapping
        records = []
        empty_records = []
        ct = 0
        cname2cid = {}

        if self.label_list:
            # Load custom label list
            label_path = os.path.join(self.dataset_dir, self.label_list)
            if not os.path.exists(label_path):
                raise ValueError(f"label_list {label_path} does not exist")

            with open(label_path, "r") as fr:
                label_id = 0
                for line in fr.readlines():
                    cname2cid[line.strip()] = label_id
                    label_id += 1
        else:
            # Use default Pascal VOC 20 classes
            cname2cid = pascalvoc_label()

        logger.info(f"Loading VOC annotations from {anno_path}")

        # Parse annotation list file
        with open(anno_path, "r") as fr:
            while True:
                line = fr.readline()
                if not line:
                    break

                # Parse image and XML paths
                parts = line.strip().split()[:2]
                if len(parts) != 2:
                    logger.warning("Malformed VOC list entry %r, will be ignored", line)
                    continue
                img_file, xml_file = [os.path.join(image_dir, x) for x in parts]

                # Validate image file
                if not os.path.exists(img_file):
                    logger.warning(f"Illegal image file: {img_file}, will be ignored")
                    continue

                # Validate XML file
                if not os.path.isfile(xml_file):
                    logger.warning(f"Illegal xml file: {xml_file}, will be ignored")
                    continue

                # Parse XML annotation
                tree = ET.parse(xml_file)

                # Get image ID (use counter if not in XML)
                image_id_text = tree.findtext("id")
                if image_id_text is None:
                    im_id = np.array([ct])
                else:
                    im_id = np.array([int(image_id_text)])

                # Get image dimensions
                im_w = float(_required_text(tree.getroot(), "size/width", xml_file))
                im_h = float(_required_text(tree.getroot(), "size/height", xml_file))

                # Validate dimensions
                if im_w < 0 or im_h < 0:
                    logger.warning(
                        f"Illegal width: {im_w} or height: {im_h} in annotation, "
                        f"{xml_file} will be ignored"
                    )
                    continue

                # Parse objects
                objs = tree.findall("object")
                num_bbox = len(objs)

                # Initialize annotation arrays
                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_score = np.zeros((num_bbox, 1), dtype=np.float32)
                difficult = np.zeros((num_bbox, 1), dtype=np.int32)

                i = 0
                for obj in objs:
                    cname = _required_text(obj, "name", xml_file)
                    if cname not in cname2cid:
                        raise ValueError(
                            f"Unknown VOC category {cname!r} in {xml_file}"
                        )

                    # Parse difficult flag (user dataset may not contain it)
                    difficult_text = obj.findtext("difficult")
                    difficult_value = int(difficult_text) if difficult_text else 0

                    # Parse bounding box
                    x1 = float(_required_text(obj, "bndbox/xmin", xml_file))
                    y1 = float(_required_text(obj, "bndbox/ymin", xml_file))
                    x2 = float(_required_text(obj, "bndbox/xmax", xml_file))
                    y2 = float(_required_text(obj, "bndbox/ymax", xml_file))

                    # Clip to image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(im_w - 1, x2)
                    y2 = min(im_h - 1, y2)

                    # Validate bbox
                    if x2 > x1 and y2 > y1:
                        gt_bbox[i, :] = [x1, y1, x2, y2]
                        gt_class[i, 0] = cname2cid[cname]
                        gt_score[i, 0] = 1.0
                        difficult[i, 0] = difficult_value
                        i += 1
                    else:
                        logger.warning(
                            f"Found an invalid bbox in annotations: "
                            f"xml_file: {xml_file}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}."
                        )

                # Trim arrays to actual object count
                gt_bbox = gt_bbox[:i, :]
                gt_class = gt_class[:i, :]
                gt_score = gt_score[:i, :]
                difficult = difficult[:i, :]

                # Build record dict
                voc_rec = (
                    {"im_file": img_file, "im_id": im_id, "h": im_h, "w": im_w}
                    if "image" in self.data_fields
                    else {}
                )

                # Add ground truth annotations
                gt_rec = {
                    "gt_class": gt_class,
                    "gt_score": gt_score,
                    "gt_bbox": gt_bbox,
                    "difficult": difficult,
                }

                # Filter by data_fields
                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        voc_rec[k] = v

                # Separate empty and non-empty records
                if i == 0:
                    empty_records.append(voc_rec)
                else:
                    records.append(voc_rec)

                ct += 1

                # Stop if sample_num reached
                if self.sample_num > 0 and ct >= self.sample_num:
                    break

        assert ct > 0, f"Not found any VOC record in {self.anno_path}"

        logger.debug(f"{ct} samples in file {anno_path}")

        # Sample and add empty records
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records

        self.roidbs = records
        self.cname2cid = cname2cid

        logger.info(f"VOC dataset loaded: {len(self.roidbs)} samples")

    def get_label_list(self) -> str:
        """Get full path to label list file."""
        if self.label_list is None:
            raise ValueError(
                "label_list is not configured; Pascal VOC defaults are in use"
            )
        return os.path.join(self.dataset_dir, self.label_list)


def pascalvoc_label() -> Dict[str, int]:
    """
    Get default Pascal VOC 20 class labels.

    Returns:
        Dictionary mapping class names to class IDs (0-19)
    """
    labels_map = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19,
    }
    return labels_map

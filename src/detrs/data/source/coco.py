"""
COCO dataset loader for RT-DETRv3 PyTorch.

Migrated from PaddlePaddle RT-DETRv3 (ppdet/data/source/coco.py).
Compatible with COCO 2017 detection dataset format.

Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
"""

import os
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    pass
except Exception:
    pass

import logging

from detrs.core.workspace import register, serializable
from detrs.utils.stdio import relay_prints

from .dataset import DetDataset

logger = logging.getLogger(__name__)

__all__ = ["COCODataSet"]


@register
@serializable
class COCODataSet(DetDataset):
    """
    Load dataset with COCO format.

    Args:
        dataset_dir (str): Root directory for dataset.
        image_dir (str): Directory for images (relative to dataset_dir).
        anno_path (str | list): COCO annotation file path (JSON format). A list
            merges multiple annotation files without physically merging data:
            each entry is either a path string (uses the global image_dir) or
            a dict {anno_path: ..., image_dir: ...} overriding image_dir.
            All files must share an identical category table. List form is
            only supported for training; evaluation requires a single file.
        data_fields (list): Key names of data dictionary, at least have 'image'.
        sample_num (int): Number of samples to load, -1 means all.
        load_crowd (bool): Whether to load crowded ground-truth. False as default.
        allow_empty (bool): Whether to load empty entries (images without annotations). False as default.
        empty_ratio (float): Ratio of empty records to total records.
                             If out of [0, 1), do not sample and use all empty entries. 1.0 as default.
        repeat (int): Repeat times for dataset, use in benchmark.
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        anno_path: Optional[Any] = None,
        data_fields: Optional[List[str]] = None,
        sample_num: int = -1,
        load_crowd: bool = False,
        allow_empty: bool = False,
        empty_ratio: float = 1.0,
        repeat: int = 1,
    ):
        if data_fields is None:
            data_fields = ["image"]

        super(COCODataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            repeat=repeat,
        )

        # COCO-specific parameters
        self.load_image_only = False
        self.load_semantic = False  # TODO: remove load_semantic (unused but preserved)
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

        # Category mapping (populated in parse_dataset)
        self.catid2clsid: dict[int, int] = {}
        self.cname2cid: dict[str, int] = {}

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

    def _normalize_anno_entries(self) -> List[Tuple[str, str]]:
        """
        Normalize anno_path (str or list of str/dict) into (anno_path, image_dir) pairs.
        """
        if isinstance(self.anno_path, (str, dict)):
            entries: List[Any] = [self.anno_path]
        elif isinstance(self.anno_path, (list, tuple)):
            if len(self.anno_path) == 0:
                raise ValueError("anno_path list must not be empty for COCODataSet")
            entries = list(self.anno_path)
        else:
            raise ValueError(
                f"Invalid anno_path type {type(self.anno_path)} for COCODataSet"
            )

        default_image_dir = self.image_dir if self.image_dir is not None else ""
        normalized = []
        for item in entries:
            if isinstance(item, str):
                normalized.append((item, default_image_dir))
            elif isinstance(item, dict):
                if "anno_path" not in item:
                    raise ValueError(
                        "dict anno_path entries must contain an 'anno_path' key"
                    )
                normalized.append(
                    (item["anno_path"], item.get("image_dir", default_image_dir))
                )
            else:
                raise ValueError(
                    f"Invalid anno_path entry type {type(item)} for COCODataSet"
                )
        return normalized

    def parse_dataset(self):
        """
        Parse COCO dataset annotations (single file or merged list of files).

        Populates self.roidbs with dataset records.
        Each record is a dict containing image path, annotations, etc.
        """
        if self.anno_path is None:
            raise ValueError("anno_path is required for COCODataSet")

        entries = self._normalize_anno_entries()

        records: List[dict] = []
        empty_records: List[dict] = []
        total_ct = 0
        im_id_offset = 0
        category_signature: Optional[List[Tuple[int, str]]] = None
        reference_anno = None

        for anno_rel, image_dir_rel in entries:
            sample_limit = self.sample_num - total_ct if self.sample_num > 0 else -1
            (
                part_records,
                part_empty,
                ct,
                catid2clsid,
                cname2cid,
                max_img_id,
                signature,
            ) = self._parse_single(anno_rel, image_dir_rel, im_id_offset, sample_limit)

            if category_signature is None:
                category_signature = signature
                reference_anno = anno_rel
                self.catid2clsid = catid2clsid
                self.cname2cid = cname2cid
            elif signature != category_signature:
                raise ValueError(
                    "Annotation files merged via anno_path list must share an "
                    f"identical category table: '{anno_rel}' differs from "
                    f"'{reference_anno}'."
                )

            records += part_records
            empty_records += part_empty
            total_ct += ct
            im_id_offset += max_img_id + 1

            if self.sample_num > 0 and total_ct >= self.sample_num:
                break

        assert total_ct > 0, f"Not found any COCO record in {self.anno_path}"

        # Sample and add empty records
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records

        self.roidbs = records
        logger.info(f"COCO dataset loaded: {len(self.roidbs)} samples")

    def _parse_single(
        self,
        anno_rel: str,
        image_dir_rel: str,
        im_id_offset: int,
        sample_limit: int,
    ):
        """
        Parse a single COCO annotation file.

        Returns (records, empty_records, ct, catid2clsid, cname2cid,
        max_img_id, category_signature).
        """
        anno_path = os.path.join(self.dataset_dir, anno_rel)
        image_dir = os.path.join(self.dataset_dir, image_dir_rel)

        assert anno_path.endswith(".json"), f"Invalid COCO annotation file: {anno_path}"

        # Load COCO annotations
        from pycocotools.coco import COCO

        logger.info(f"Loading COCO annotations from {anno_path}")
        with relay_prints(logger):
            coco = COCO(anno_path)

        # Get image and category IDs
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()

        records = []
        empty_records = []
        ct = 0

        # Build category mapping
        catid2clsid = {catid: i for i, catid in enumerate(cat_ids)}
        cname2cid = {
            coco.loadCats(catid)[0]["name"]: clsid
            for catid, clsid in catid2clsid.items()
        }
        signature = [
            (catid, coco.loadCats(catid)[0]["name"]) for catid in sorted(cat_ids)
        ]

        # Check if annotations exist
        load_image_only = "annotations" not in coco.dataset
        if load_image_only:
            self.load_image_only = True
            logger.warning(
                f"Annotation file: {anno_path} does not contain ground truth "
                "and will load image information only."
            )

        # Iterate over images
        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno["file_name"]
            im_w = float(img_anno["width"])
            im_h = float(img_anno["height"])

            im_path = os.path.join(image_dir, im_fname) if image_dir else im_fname
            is_empty = False

            # Validate image file
            if not os.path.exists(im_path):
                logger.warning(f"Illegal image file: {im_path}, will be ignored")
                continue

            # Validate dimensions
            if im_w < 0 or im_h < 0:
                logger.warning(
                    f"Illegal width: {im_w} or height: {im_h} in annotation, "
                    f"im_id: {img_id} will be ignored"
                )
                continue

            # Build record dict
            coco_rec = (
                {
                    "im_file": im_path,
                    "im_id": np.array([img_id + im_id_offset]),
                    "h": im_h,
                    "w": im_w,
                }
                if "image" in self.data_fields
                else {}
            )

            # Parse annotations if not load_image_only
            if not load_image_only:
                # Get annotations (filter crowd if needed)
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False
                )
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                for inst in instances:
                    # Skip ignored annotations
                    if inst.get("ignore", False):
                        continue

                    # Skip annotations without bbox
                    if "bbox" not in inst.keys():
                        continue
                    else:
                        # Skip empty bboxes
                        if not any(np.array(inst["bbox"])):
                            continue

                    # Parse bbox (COCO uses [x, y, w, h] format)
                    x1, y1, box_w, box_h = inst["bbox"]
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5

                    # Validate bbox
                    if inst["area"] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst["clean_bbox"] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            f"Found an invalid bbox in annotations: "
                            f"im_id: {img_id}, area: {float(inst['area'])}, "
                            f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}."
                        )

                num_bbox = len(bboxes)

                # Handle empty annotations
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                # Build annotation arrays
                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly: List[Optional[List[List[float]]]] = [None] * num_bbox
                gt_track_id = -np.ones((num_bbox, 1), dtype=np.int32)

                has_segmentation = False
                has_track_id = False

                for i, box in enumerate(bboxes):
                    catid = box["category_id"]
                    gt_class[i][0] = catid2clsid[catid]
                    gt_bbox[i, :] = box["clean_bbox"]
                    is_crowd[i][0] = box["iscrowd"]

                    # Check RLE format (for instance segmentation)
                    if "segmentation" in box and box["iscrowd"] == 1:
                        # RLE format for crowd annotations
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif "segmentation" in box and box["segmentation"]:
                        # Polygon format
                        if (
                            not np.array(box["segmentation"], dtype=object).size > 0
                            and not self.allow_empty
                        ):
                            # Remove invalid segmentation
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box["segmentation"]
                        has_segmentation = True

                    # Track ID for multi-object tracking (preserved for compatibility)
                    if "track_id" in box:
                        gt_track_id[i][0] = box["track_id"]
                        has_track_id = True

                # Check segmentation validity
                if has_segmentation and not any(gt_poly) and not self.allow_empty:
                    continue

                # Build ground truth record
                gt_rec = {
                    "is_crowd": is_crowd,
                    "gt_class": gt_class,
                    "gt_bbox": gt_bbox,
                    "gt_poly": gt_poly,
                }

                if has_track_id:
                    gt_rec.update({"gt_track_id": gt_track_id})

                # Filter by data_fields
                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # Semantic segmentation (unused, preserved for compatibility)
                # TODO: remove load_semantic
                if self.load_semantic and "semantic" in self.data_fields:
                    seg_path = os.path.join(
                        self.dataset_dir,
                        "stuffthingmaps",
                        "train2017",
                        im_fname[:-3] + "png",
                    )
                    coco_rec.update({"semantic": seg_path})

            logger.debug(
                f"Load file: {im_path}, im_id: {img_id}, h: {im_h}, w: {im_w}."
            )

            # Separate empty and non-empty records
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)

            ct += 1

            # Stop if sample_limit reached
            if sample_limit > 0 and ct >= sample_limit:
                break

        logger.info(
            f"Load [{ct} samples valid, {len(img_ids) - ct} samples invalid] "
            f"in file {anno_path}."
        )

        return (
            records,
            empty_records,
            ct,
            catid2clsid,
            cname2cid,
            max(img_ids) if img_ids else 0,
            signature,
        )

"""
COCO dataset loader for RT-DETRv3 PyTorch.

Migrated from PaddlePaddle RT-DETRv3 (ppdet/data/source/coco.py).
Compatible with COCO 2017 detection dataset format.

Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
"""

import os
import copy
import numpy as np
from typing import List, Optional
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from ppdet_pytorch.core.workspace import register, serializable
from .dataset import DetDataset

import logging
logger = logging.getLogger(__name__)

__all__ = ['COCODataSet']


@register
@serializable
class COCODataSet(DetDataset):
    """
    Load dataset with COCO format.

    Args:
        dataset_dir (str): Root directory for dataset.
        image_dir (str): Directory for images (relative to dataset_dir).
        anno_path (str): COCO annotation file path (JSON format).
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
        anno_path: Optional[str] = None,
        data_fields: List[str] = None,
        sample_num: int = -1,
        load_crowd: bool = False,
        allow_empty: bool = False,
        empty_ratio: float = 1.0,
        repeat: int = 1
    ):
        if data_fields is None:
            data_fields = ['image']

        super(COCODataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            repeat=repeat
        )

        # COCO-specific parameters
        self.load_image_only = False
        self.load_semantic = False  # TODO: remove load_semantic (unused but preserved)
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

        # Category mapping (populated in parse_dataset)
        self.catid2clsid = {}
        self.cname2cid = {}

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
            int(num * self.empty_ratio / (1 - self.empty_ratio)),
            len(records)
        )
        records = random.sample(records, sample_num)
        return records

    def parse_dataset(self):
        """
        Parse COCO dataset annotations.

        Populates self.roidbs with dataset records.
        Each record is a dict containing image path, annotations, etc.
        """
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            f'Invalid COCO annotation file: {anno_path}'

        # Load COCO annotations
        from pycocotools.coco import COCO
        logger.info(f"Loading COCO annotations from {anno_path}")
        coco = COCO(anno_path)

        # Get image and category IDs
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()

        records = []
        empty_records = []
        ct = 0

        # Build category mapping
        self.catid2clsid = {catid: i for i, catid in enumerate(cat_ids)}
        self.cname2cid = {
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        }

        # Check if annotations exist
        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning(
                f'Annotation file: {anno_path} does not contain ground truth '
                'and will load image information only.'
            )

        # Iterate over images
        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir, im_fname) if image_dir else im_fname
            is_empty = False

            # Validate image file
            if not os.path.exists(im_path):
                logger.warning(
                    f'Illegal image file: {im_path}, will be ignored'
                )
                continue

            # Validate dimensions
            if im_w < 0 or im_h < 0:
                logger.warning(
                    f'Illegal width: {im_w} or height: {im_h} in annotation, '
                    f'im_id: {img_id} will be ignored'
                )
                continue

            # Build record dict
            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            # Parse annotations if not load_image_only
            if not self.load_image_only:
                # Get annotations (filter crowd if needed)
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id],
                    iscrowd=None if self.load_crowd else False
                )
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False  # Preserved for rotated bbox support (unused)

                for inst in instances:
                    # Skip ignored annotations
                    if inst.get('ignore', False):
                        continue

                    # Skip annotations without bbox
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        # Skip empty bboxes
                        if not any(np.array(inst['bbox'])):
                            continue

                    # Parse bbox (COCO uses [x, y, w, h] format)
                    x1, y1, box_w, box_h = inst['bbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5

                    # Validate bbox
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            f'Found an invalid bbox in annotations: '
                            f'im_id: {img_id}, area: {float(inst["area"])}, '
                            f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}.'
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
                gt_poly = [None] * num_bbox
                gt_track_id = -np.ones((num_bbox, 1), dtype=np.int32)

                has_segmentation = False
                has_track_id = False

                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']

                    # Check RLE format (for instance segmentation)
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        # RLE format for crowd annotations
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box and box['segmentation']:
                        # Polygon format
                        if not np.array(
                            box['segmentation'],
                            dtype=object
                        ).size > 0 and not self.allow_empty:
                            # Remove invalid segmentation
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                    # Track ID for multi-object tracking (preserved for compatibility)
                    if 'track_id' in box:
                        gt_track_id[i][0] = box['track_id']
                        has_track_id = True

                # Check segmentation validity
                if has_segmentation and not any(gt_poly) and not self.allow_empty:
                    continue

                # Build ground truth record
                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_poly': gt_poly,
                }

                if has_track_id:
                    gt_rec.update({'gt_track_id': gt_track_id})

                # Filter by data_fields
                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # Semantic segmentation (unused, preserved for compatibility)
                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(
                        self.dataset_dir, 'stuffthingmaps', 'train2017',
                        im_fname[:-3] + 'png'
                    )
                    coco_rec.update({'semantic': seg_path})

            logger.debug(
                f'Load file: {im_path}, im_id: {img_id}, h: {im_h}, w: {im_w}.'
            )

            # Separate empty and non-empty records
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)

            ct += 1

            # Stop if sample_num reached
            if self.sample_num > 0 and ct >= self.sample_num:
                break

        assert ct > 0, f'Not found any COCO record in {anno_path}'

        logger.info(
            f'Load [{ct} samples valid, {len(img_ids) - ct} samples invalid] '
            f'in file {anno_path}.'
        )

        # Sample and add empty records
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records

        self.roidbs = records
        logger.info(f"COCO dataset loaded: {len(self.roidbs)} samples")

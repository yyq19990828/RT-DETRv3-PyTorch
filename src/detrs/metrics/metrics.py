# Copyright (c) 2025 RT-DETRv3 PyTorch Authors. All Rights Reserved.
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
#
# Modified from PaddlePaddle RT-DETRv3
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

"""
Metrics for model evaluation.

Migrated from PaddlePaddle, compatible API.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ..data.source.category import get_categories
from ..utils.logger import setup_logger
from .coco_utils import cocoapi_eval, get_infer_results

logger = setup_logger(__name__)

__all__ = ["Metric", "COCOMetric"]

COCO_SIGMAS = (
    np.array(
        [
            0.26,
            0.25,
            0.25,
            0.35,
            0.35,
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
        ]
    )
    / 10.0
)

CROWD_SIGMAS = (
    np.array(
        [
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
            0.79,
            0.79,
        ]
    )
    / 10.0
)


class Metric:
    """
    Base metric class (compatible with Paddle's Metric API).
    """

    def name(self):
        return self.__class__.__name__

    def reset(self):
        """Reset metric state"""
        pass

    def update(self, inputs, outputs):
        """Update metric with new batch results"""
        pass

    def accumulate(self):
        """Accumulate and compute final results"""
        pass

    def log(self):
        """Log metric results"""
        pass

    def get_results(self):
        """Get metric results as dict"""
        pass


class COCOMetric(Metric):
    """
    COCO evaluation metric (compatible with Paddle's COCOMetric API).

    Args:
        anno_file: Path to COCO annotation file
        clsid2catid: Mapping from class id to category id
        classwise: Whether to evaluate per-class metrics
        output_eval: Directory to save evaluation results
        bias: Bias to add to class ids
        save_prediction_only: Only save predictions without evaluation
        IouType: Type of IoU ('bbox', 'segm', 'keypoint')
        save_threshold: Score threshold for saving predictions
    """

    def __init__(self, anno_file, **kwargs):
        self.anno_file = anno_file
        self.clsid2catid = kwargs.get("clsid2catid", None)

        if self.clsid2catid is None:
            self.clsid2catid, _ = get_categories("COCO", anno_file)
            assert self.clsid2catid is not None

        self.classwise = kwargs.get("classwise", False)
        self.output_eval = kwargs.get("output_eval", None)
        self.bias = kwargs.get("bias", 0)
        self.save_prediction_only = kwargs.get("save_prediction_only", False)
        self.iou_type = kwargs.get("IouType", "bbox")

        if not self.save_prediction_only:
            assert os.path.isfile(anno_file), f"anno_file {anno_file} not a file"

        if self.output_eval is not None:
            Path(self.output_eval).mkdir(exist_ok=True, parents=True)

        self.save_threshold = kwargs.get("save_threshold", 0)

        self.reset()

    def reset(self):
        """Reset evaluation results"""
        # Only bbox and mask evaluation support currently
        self.results: Dict[str, list] = {
            "bbox": [],
            "mask": [],
            "segm": [],
            "keypoint": [],
        }
        self.eval_results: Dict[str, object] = {}

    def update(self, inputs: Dict, outputs: Dict):
        """
        Update metric with new batch results.

        Args:
            inputs: Input batch data (contains im_id, im_file, etc.)
            outputs: Model outputs (contains bbox, score, category, etc.)
        """
        outs = {}

        # Convert PyTorch tensors to numpy
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                outs[k] = v.cpu().numpy()
            else:
                outs[k] = v

        # Multi-scale inputs: all inputs have same im_id
        if isinstance(inputs, (list, tuple)):
            im_id = inputs[0]["im_id"]
        else:
            im_id = inputs["im_id"]

        # Convert im_id to numpy
        if isinstance(im_id, torch.Tensor):
            outs["im_id"] = im_id.cpu().numpy()
        else:
            outs["im_id"] = im_id

        # Add image file path if available
        if "im_file" in inputs:
            outs["im_file"] = inputs["im_file"]

        # Get inference results in COCO format
        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias, save_threshold=self.save_threshold
        )

        # Accumulate results by type
        self.results["bbox"] += infer_results.get("bbox", [])
        self.results["mask"] += infer_results.get("mask", [])
        self.results["segm"] += infer_results.get("segm", [])
        self.results["keypoint"] += infer_results.get("keypoint", [])

    def accumulate(self):
        """
        Accumulate results and perform COCO evaluation.
        """
        # Evaluate bounding boxes
        if len(self.results["bbox"]) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)

            with open(output, "w") as f:
                json.dump(self.results["bbox"], f)
                logger.info("The bbox result is saved to bbox.json.")

            if self.save_prediction_only:
                logger.info(
                    f"The bbox result is saved to {output} and do not evaluate the mAP."
                )
            else:
                bbox_stats = cocoapi_eval(
                    output, "bbox", anno_file=self.anno_file, classwise=self.classwise
                )
                self.eval_results["bbox"] = bbox_stats
                sys.stdout.flush()

        # Evaluate masks
        if len(self.results["mask"]) > 0:
            output = "mask.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)

            with open(output, "w") as f:
                json.dump(self.results["mask"], f)
                logger.info("The mask result is saved to mask.json.")

            if self.save_prediction_only:
                logger.info(
                    f"The mask result is saved to {output} and do not evaluate the mAP."
                )
            else:
                seg_stats = cocoapi_eval(
                    output, "segm", anno_file=self.anno_file, classwise=self.classwise
                )
                self.eval_results["mask"] = seg_stats
                sys.stdout.flush()

        # Evaluate segmentation
        if len(self.results["segm"]) > 0:
            output = "segm.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)

            with open(output, "w") as f:
                json.dump(self.results["segm"], f)
                logger.info("The segm result is saved to segm.json.")

            if self.save_prediction_only:
                logger.info(
                    f"The segm result is saved to {output} and do not evaluate the mAP."
                )
            else:
                seg_stats = cocoapi_eval(
                    output, "segm", anno_file=self.anno_file, classwise=self.classwise
                )
                self.eval_results["segm"] = seg_stats
                sys.stdout.flush()

        # Evaluate keypoints
        if len(self.results["keypoint"]) > 0:
            output = "keypoint.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)

            with open(output, "w") as f:
                json.dump(self.results["keypoint"], f)
                logger.info("The keypoint result is saved to keypoint.json.")

            if self.save_prediction_only:
                logger.info(
                    f"The keypoint result is saved to {output} and do not "
                    "evaluate the mAP."
                )
            else:
                style = self.anno_file.split("_")[-1].split(".")[0]
                use_area = True if style == "person" else False
                kpt_stats = cocoapi_eval(
                    output,
                    "keypoints",
                    anno_file=self.anno_file,
                    classwise=self.classwise,
                    sigmas=COCO_SIGMAS,
                    use_area=use_area,
                )
                self.eval_results["keypoint"] = kpt_stats
                sys.stdout.flush()

    def log(self):
        """Log evaluation results"""
        if not self.eval_results:
            logger.warning("No evaluation results available")
            return

        for metric_type, stats in self.eval_results.items():
            logger.info(f"=========== {metric_type} evaluation ===========")
            if isinstance(stats, dict):
                for k, v in stats.items():
                    logger.info(f"{k}: {v}")
            elif isinstance(stats, list):
                for stat in stats:
                    logger.info(stat)

    def get_results(self) -> Dict:
        """
        Get evaluation results.

        Returns:
            Dictionary containing evaluation metrics
        """
        return self.eval_results


# Additional metrics can be added here (VOCMetric, LVISMetric, etc.)
# For now, we focus on COCOMetric which is used by RT-DETRv3

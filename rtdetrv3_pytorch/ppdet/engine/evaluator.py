"""
COCO Evaluator for RT-DETRv3

Evaluates model predictions using COCO metrics (mAP, AP50, AP75, etc.)
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger(__name__)


class COCOEvaluator:
    """
    COCO Evaluator for object detection

    Evaluates model predictions on COCO dataset and computes standard COCO metrics:
    - AP (Average Precision at IoU=0.50:0.95)
    - AP50 (Average Precision at IoU=0.50)
    - AP75 (Average Precision at IoU=0.75)
    - APs (AP for small objects: area < 32^2)
    - APm (AP for medium objects: 32^2 < area < 96^2)
    - APl (AP for large objects: area > 96^2)
    """

    def __init__(
        self,
        anno_file: str,
        iou_types: List[str] = ['bbox'],
        max_dets: List[int] = [1, 10, 100]
    ):
        """
        Initialize COCO evaluator

        Args:
            anno_file: Path to COCO annotation JSON file
            iou_types: List of IoU types to evaluate (e.g., ['bbox', 'segm'])
            max_dets: Maximum detections per image for different settings
        """
        self.anno_file = anno_file
        self.iou_types = iou_types
        self.max_dets = max_dets

        # Load COCO ground truth
        logger.info(f"Loading COCO annotations from {anno_file}")
        self.coco_gt = COCO(anno_file)

        # Storage for predictions
        self.predictions = []
        self.image_ids = []

    def update(
        self,
        predictions: List[Dict],
        image_ids: List[int]
    ):
        """
        Add predictions for a batch of images

        Args:
            predictions: List of predictions for each image, each dict containing:
                - 'boxes': (N, 4) tensor in [x1, y1, x2, y2] format
                - 'scores': (N,) tensor of confidence scores
                - 'labels': (N,) tensor of class labels
            image_ids: List of COCO image IDs corresponding to predictions
        """
        assert len(predictions) == len(image_ids), \
            f"Number of predictions ({len(predictions)}) must match number of image_ids ({len(image_ids)})"

        for pred, img_id in zip(predictions, image_ids):
            boxes = pred['boxes']  # (N, 4) [x1, y1, x2, y2]
            scores = pred['scores']  # (N,)
            labels = pred['labels']  # (N,)

            # Convert to COCO format
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                # COCO format: [x, y, width, height]
                coco_box = [x1, y1, w, h]

                # Convert label (0-indexed) to COCO category_id (1-indexed for COCO)
                # Note: COCO categories are not necessarily continuous, so we need to map
                category_id = int(label) + 1  # Assuming 0-indexed labels

                self.predictions.append({
                    'image_id': int(img_id),
                    'category_id': category_id,
                    'bbox': coco_box,
                    'score': float(score)
                })

            self.image_ids.append(int(img_id))

    def synchronize_between_processes(self):
        """
        Synchronize predictions across multiple processes (for distributed training)

        For single-process evaluation, this is a no-op.
        For multi-process, this would use torch.distributed to gather predictions.
        """
        # TODO: Implement distributed synchronization if needed
        pass

    def accumulate(self):
        """
        Compute and return COCO evaluation results

        Returns:
            Dictionary containing COCO metrics:
                - 'AP': Average Precision at IoU=0.50:0.95
                - 'AP50': Average Precision at IoU=0.50
                - 'AP75': Average Precision at IoU=0.75
                - 'APs': AP for small objects
                - 'APm': AP for medium objects
                - 'APl': AP for large objects
                - 'AR1': Average Recall with 1 detection per image
                - 'AR10': Average Recall with 10 detections per image
                - 'AR100': Average Recall with 100 detections per image
        """
        if len(self.predictions) == 0:
            logger.warning("No predictions to evaluate!")
            return {}

        # Synchronize across processes if needed
        self.synchronize_between_processes()

        # Save predictions to temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.predictions, f)
            pred_file = f.name

        logger.info(f"Saved {len(self.predictions)} predictions for {len(set(self.image_ids))} images")

        # Evaluate using pycocotools
        results = {}
        try:
            for iou_type in self.iou_types:
                coco_dt = self.coco_gt.loadRes(pred_file)
                coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type)

                # Set image IDs to evaluate
                coco_eval.params.imgIds = sorted(set(self.image_ids))

                # Run evaluation
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # Extract metrics
                results[iou_type] = {
                    'AP': coco_eval.stats[0],      # AP @ IoU=0.50:0.95
                    'AP50': coco_eval.stats[1],    # AP @ IoU=0.50
                    'AP75': coco_eval.stats[2],    # AP @ IoU=0.75
                    'APs': coco_eval.stats[3],     # AP for small objects
                    'APm': coco_eval.stats[4],     # AP for medium objects
                    'APl': coco_eval.stats[5],     # AP for large objects
                    'AR1': coco_eval.stats[6],     # AR with 1 detection
                    'AR10': coco_eval.stats[7],    # AR with 10 detections
                    'AR100': coco_eval.stats[8],   # AR with 100 detections
                }

                logger.info(f"\n{iou_type} Results:")
                logger.info(f"  AP      : {results[iou_type]['AP']:.3f}")
                logger.info(f"  AP50    : {results[iou_type]['AP50']:.3f}")
                logger.info(f"  AP75    : {results[iou_type]['AP75']:.3f}")
                logger.info(f"  APs     : {results[iou_type]['APs']:.3f}")
                logger.info(f"  APm     : {results[iou_type]['APm']:.3f}")
                logger.info(f"  APl     : {results[iou_type]['APl']:.3f}")

        except Exception as e:
            logger.error(f"COCO evaluation failed: {e}")
            raise
        finally:
            # Clean up temporary file
            Path(pred_file).unlink(missing_ok=True)

        return results

    def reset(self):
        """Reset evaluator state"""
        self.predictions = []
        self.image_ids = []


def build_coco_evaluator(
    anno_file: str,
    iou_types: Optional[List[str]] = None
) -> COCOEvaluator:
    """
    Build COCO evaluator from annotation file

    Args:
        anno_file: Path to COCO annotation JSON file
        iou_types: List of IoU types to evaluate (defaults to ['bbox'])

    Returns:
        COCOEvaluator instance
    """
    if iou_types is None:
        iou_types = ['bbox']

    return COCOEvaluator(anno_file, iou_types)


__all__ = ['COCOEvaluator', 'build_coco_evaluator']

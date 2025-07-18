"""Evaluator implementation for RT-DETRv3 PyTorch."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import tempfile
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import logging

from ..core.workspace import register
from ..misc.dist_utils import get_world_size, get_rank, is_main_process, gather_tensors


logger = logging.getLogger(__name__)


@register()
class CocoEvaluator:
    """COCO evaluation for object detection."""
    
    def __init__(
        self,
        ann_file: str = None,
        metric_types: List[str] = ['bbox'],
        classwise: bool = False,
        iou_thresholds: Optional[List[float]] = None,
        max_dets: Optional[List[int]] = None,
        **kwargs
    ):
        """Initialize COCO evaluator.
        
        Args:
            ann_file: Path to annotation file
            metric_types: List of metric types to compute
            classwise: Whether to compute classwise metrics
            iou_thresholds: IoU thresholds for evaluation
            max_dets: Maximum number of detections per image
        """
        self.ann_file = ann_file
        self.metric_types = metric_types
        self.classwise = classwise
        self.iou_thresholds = iou_thresholds
        self.max_dets = max_dets or [1, 10, 100]
        
        # Initialize COCO API
        if ann_file is not None:
            self.coco_gt = COCO(ann_file)
        else:
            self.coco_gt = None
        
        # Reset evaluation state
        self.reset()
    
    def reset(self):
        """Reset evaluation state."""
        self.predictions = []
        self.image_ids = []
        self.category_ids = []
        
        if self.coco_gt is not None:
            self.category_ids = self.coco_gt.getCatIds()
        
        self.eval_results = {}
    
    def update(self, outputs: List[Dict[str, torch.Tensor]], targets: List[Dict[str, Any]]):
        """Update evaluation with model outputs and targets.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        """
        # Process each image in the batch
        for output, target in zip(outputs, targets):
            image_id = target['image_id'].item()
            self.image_ids.append(image_id)
            
            # Get predictions from last decoder layer
            if isinstance(output, list):
                pred_logits = output[-1]['pred_logits']
                pred_boxes = output[-1]['pred_boxes']
            else:
                pred_logits = output['pred_logits']
                pred_boxes = output['pred_boxes']
            
            # Convert to CPU and numpy
            pred_logits = pred_logits.cpu().numpy()
            pred_boxes = pred_boxes.cpu().numpy()
            
            # Get image size
            if 'size' in target:
                img_h, img_w = target['size'].cpu().numpy()
            else:
                img_h, img_w = target['orig_size'].cpu().numpy()
            
            # Process predictions
            self._process_predictions(pred_logits, pred_boxes, image_id, img_h, img_w)
    
    def _process_predictions(self, pred_logits: np.ndarray, pred_boxes: np.ndarray,
                           image_id: int, img_h: int, img_w: int):
        """Process predictions for a single image.
        
        Args:
            pred_logits: Predicted logits [num_queries, num_classes]
            pred_boxes: Predicted boxes [num_queries, 4]
            image_id: Image ID
            img_h: Image height
            img_w: Image width
        """
        # Get scores and labels
        scores = pred_logits.max(axis=1)
        labels = pred_logits.argmax(axis=1)
        
        # Convert boxes from [cx, cy, w, h] to [x1, y1, w, h] format
        boxes = pred_boxes.copy()
        boxes[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * img_w
        boxes[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * img_h
        boxes[:, 2] = pred_boxes[:, 2] * img_w
        boxes[:, 3] = pred_boxes[:, 3] * img_h
        
        # Filter valid predictions
        valid_mask = scores > 0.0
        scores = scores[valid_mask]
        labels = labels[valid_mask]
        boxes = boxes[valid_mask]
        
        # Convert to COCO format
        for score, label, box in zip(scores, labels, boxes):
            if self.coco_gt is not None:
                # Map label to category ID
                category_id = self.coco_gt.getCatIds()[label]
            else:
                category_id = label + 1  # COCO categories start from 1
            
            prediction = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': box.tolist(),
                'score': float(score)
            }
            
            self.predictions.append(prediction)
    
    def compute(self) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if len(self.predictions) == 0:
            logger.warning("No predictions to evaluate")
            return {}
        
        # Gather predictions from all processes
        if get_world_size() > 1:
            all_predictions = self._gather_predictions()
        else:
            all_predictions = self.predictions
        
        if not is_main_process():
            return {}
        
        # Compute COCO metrics
        if self.coco_gt is not None:
            metrics = self._compute_coco_metrics(all_predictions)
        else:
            logger.warning("No ground truth available, skipping evaluation")
            metrics = {}
        
        return metrics
    
    def _gather_predictions(self) -> List[Dict[str, Any]]:
        """Gather predictions from all processes."""
        all_predictions = []
        
        # Gather all predictions
        gathered_predictions = gather_tensors(self.predictions)
        
        if is_main_process():
            for predictions in gathered_predictions:
                all_predictions.extend(predictions)
        
        return all_predictions
    
    def _compute_coco_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute COCO metrics.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Dictionary of COCO metrics
        """
        if len(predictions) == 0:
            return {}
        
        # Create temporary file for predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(predictions, f)
            pred_file = f.name
        
        try:
            # Load predictions
            coco_pred = self.coco_gt.loadRes(pred_file)
            
            # Create evaluator
            coco_eval = COCOeval(self.coco_gt, coco_pred, 'bbox')
            
            # Set evaluation parameters
            if self.iou_thresholds is not None:
                coco_eval.params.iouThrs = np.array(self.iou_thresholds)
            
            coco_eval.params.maxDets = self.max_dets
            
            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = self._extract_coco_metrics(coco_eval)
            
            # Compute classwise metrics if requested
            if self.classwise:
                classwise_metrics = self._compute_classwise_metrics(coco_eval)
                metrics.update(classwise_metrics)
            
            return metrics
            
        finally:
            # Clean up temporary file
            if os.path.exists(pred_file):
                os.unlink(pred_file)
    
    def _extract_coco_metrics(self, coco_eval: COCOeval) -> Dict[str, float]:
        """Extract COCO metrics from evaluator.
        
        Args:
            coco_eval: COCO evaluator
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Extract standard COCO metrics
        metrics['mAP'] = coco_eval.stats[0]
        metrics['mAP_50'] = coco_eval.stats[1]
        metrics['mAP_75'] = coco_eval.stats[2]
        metrics['mAP_small'] = coco_eval.stats[3]
        metrics['mAP_medium'] = coco_eval.stats[4]
        metrics['mAP_large'] = coco_eval.stats[5]
        
        metrics['mAR_1'] = coco_eval.stats[6]
        metrics['mAR_10'] = coco_eval.stats[7]
        metrics['mAR_100'] = coco_eval.stats[8]
        metrics['mAR_small'] = coco_eval.stats[9]
        metrics['mAR_medium'] = coco_eval.stats[10]
        metrics['mAR_large'] = coco_eval.stats[11]
        
        return metrics
    
    def _compute_classwise_metrics(self, coco_eval: COCOeval) -> Dict[str, float]:
        """Compute classwise metrics.
        
        Args:
            coco_eval: COCO evaluator
            
        Returns:
            Dictionary of classwise metrics
        """
        classwise_metrics = {}
        
        # Get category information
        categories = self.coco_gt.dataset['categories']
        cat_ids = self.coco_gt.getCatIds()
        
        # Compute AP for each category
        for cat_idx, cat_id in enumerate(cat_ids):
            cat_name = next(cat['name'] for cat in categories if cat['id'] == cat_id)
            
            # Extract AP for this category
            ap = coco_eval.eval['precision'][:, :, cat_idx, 0, 2]
            ap = ap[ap > -1]
            
            if len(ap) > 0:
                classwise_metrics[f'AP_{cat_name}'] = float(ap.mean())
            else:
                classwise_metrics[f'AP_{cat_name}'] = 0.0
        
        return classwise_metrics


@register()
class LVISEvaluator:
    """LVIS evaluation for object detection."""
    
    def __init__(
        self,
        ann_file: str = None,
        metric_types: List[str] = ['bbox'],
        classwise: bool = False,
        **kwargs
    ):
        """Initialize LVIS evaluator.
        
        Args:
            ann_file: Path to annotation file
            metric_types: List of metric types to compute
            classwise: Whether to compute classwise metrics
        """
        try:
            from lvis import LVIS, LVISEval
        except ImportError:
            raise ImportError("Please install lvis-api: pip install lvis")
        
        self.ann_file = ann_file
        self.metric_types = metric_types
        self.classwise = classwise
        
        # Initialize LVIS API
        if ann_file is not None:
            self.lvis_gt = LVIS(ann_file)
        else:
            self.lvis_gt = None
        
        # Reset evaluation state
        self.reset()
    
    def reset(self):
        """Reset evaluation state."""
        self.predictions = []
        self.image_ids = []
        self.category_ids = []
        
        if self.lvis_gt is not None:
            self.category_ids = self.lvis_gt.get_cat_ids()
        
        self.eval_results = {}
    
    def update(self, outputs: List[Dict[str, torch.Tensor]], targets: List[Dict[str, Any]]):
        """Update evaluation with model outputs and targets.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        """
        # Process each image in the batch
        for output, target in zip(outputs, targets):
            image_id = target['image_id'].item()
            self.image_ids.append(image_id)
            
            # Get predictions from last decoder layer
            if isinstance(output, list):
                pred_logits = output[-1]['pred_logits']
                pred_boxes = output[-1]['pred_boxes']
            else:
                pred_logits = output['pred_logits']
                pred_boxes = output['pred_boxes']
            
            # Convert to CPU and numpy
            pred_logits = pred_logits.cpu().numpy()
            pred_boxes = pred_boxes.cpu().numpy()
            
            # Get image size
            if 'size' in target:
                img_h, img_w = target['size'].cpu().numpy()
            else:
                img_h, img_w = target['orig_size'].cpu().numpy()
            
            # Process predictions
            self._process_predictions(pred_logits, pred_boxes, image_id, img_h, img_w)
    
    def _process_predictions(self, pred_logits: np.ndarray, pred_boxes: np.ndarray,
                           image_id: int, img_h: int, img_w: int):
        """Process predictions for a single image."""
        # Get scores and labels
        scores = pred_logits.max(axis=1)
        labels = pred_logits.argmax(axis=1)
        
        # Convert boxes from [cx, cy, w, h] to [x1, y1, w, h] format
        boxes = pred_boxes.copy()
        boxes[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * img_w
        boxes[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * img_h
        boxes[:, 2] = pred_boxes[:, 2] * img_w
        boxes[:, 3] = pred_boxes[:, 3] * img_h
        
        # Filter valid predictions
        valid_mask = scores > 0.0
        scores = scores[valid_mask]
        labels = labels[valid_mask]
        boxes = boxes[valid_mask]
        
        # Convert to LVIS format
        for score, label, box in zip(scores, labels, boxes):
            if self.lvis_gt is not None:
                # Map label to category ID
                category_id = self.lvis_gt.get_cat_ids()[label]
            else:
                category_id = label + 1
            
            prediction = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': box.tolist(),
                'score': float(score)
            }
            
            self.predictions.append(prediction)
    
    def compute(self) -> Dict[str, float]:
        """Compute evaluation metrics."""
        if len(self.predictions) == 0:
            logger.warning("No predictions to evaluate")
            return {}
        
        # Gather predictions from all processes
        if get_world_size() > 1:
            all_predictions = self._gather_predictions()
        else:
            all_predictions = self.predictions
        
        if not is_main_process():
            return {}
        
        # Compute LVIS metrics
        if self.lvis_gt is not None:
            metrics = self._compute_lvis_metrics(all_predictions)
        else:
            logger.warning("No ground truth available, skipping evaluation")
            metrics = {}
        
        return metrics
    
    def _gather_predictions(self) -> List[Dict[str, Any]]:
        """Gather predictions from all processes."""
        all_predictions = []
        
        # Gather all predictions
        gathered_predictions = gather_tensors(self.predictions)
        
        if is_main_process():
            for predictions in gathered_predictions:
                all_predictions.extend(predictions)
        
        return all_predictions
    
    def _compute_lvis_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute LVIS metrics."""
        try:
            from lvis import LVISEval
        except ImportError:
            raise ImportError("Please install lvis-api: pip install lvis")
        
        if len(predictions) == 0:
            return {}
        
        # Create temporary file for predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(predictions, f)
            pred_file = f.name
        
        try:
            # Load predictions
            lvis_pred = self.lvis_gt.load_res(pred_file)
            
            # Create evaluator
            lvis_eval = LVISEval(self.lvis_gt, lvis_pred, 'bbox')
            
            # Run evaluation
            lvis_eval.run()
            lvis_eval.print_results()
            
            # Extract metrics
            metrics = self._extract_lvis_metrics(lvis_eval)
            
            return metrics
            
        finally:
            # Clean up temporary file
            if os.path.exists(pred_file):
                os.unlink(pred_file)
    
    def _extract_lvis_metrics(self, lvis_eval) -> Dict[str, float]:
        """Extract LVIS metrics from evaluator."""
        metrics = {}
        
        # Extract standard LVIS metrics
        metrics['mAP'] = lvis_eval.results['AP']
        metrics['mAP_50'] = lvis_eval.results['AP50']
        metrics['mAP_75'] = lvis_eval.results['AP75']
        metrics['mAP_small'] = lvis_eval.results['APs']
        metrics['mAP_medium'] = lvis_eval.results['APm']
        metrics['mAP_large'] = lvis_eval.results['APl']
        
        metrics['mAR'] = lvis_eval.results['AR']
        metrics['mAR_small'] = lvis_eval.results['ARs']
        metrics['mAR_medium'] = lvis_eval.results['ARm']
        metrics['mAR_large'] = lvis_eval.results['ARl']
        
        return metrics


def build_evaluator(evaluator_cfg: Dict[str, Any]) -> object:
    """Build evaluator from configuration.
    
    Args:
        evaluator_cfg: Evaluator configuration
        
    Returns:
        Configured evaluator
    """
    evaluator_type = evaluator_cfg.get('type', 'CocoEvaluator')
    
    if evaluator_type == 'CocoEvaluator':
        return CocoEvaluator(**evaluator_cfg)
    elif evaluator_type == 'LVISEvaluator':
        return LVISEvaluator(**evaluator_cfg)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
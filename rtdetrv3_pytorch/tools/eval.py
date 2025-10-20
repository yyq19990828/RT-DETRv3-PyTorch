"""
RT-DETRv3 COCO Evaluation Script

Evaluate trained RT-DETRv3 model on COCO val2017 dataset.

Usage:
    # Evaluate with config and checkpoint
    python tools/eval.py -c configs/rtdetrv3_r50vd.yml --checkpoint weights/rtdetrv3_r50vd.pth

    # Evaluate with custom batch size
    python tools/eval.py -c configs/rtdetrv3_r50vd.yml --checkpoint weights/rtdetrv3_r50vd.pth --batch_size 8

    # Evaluate with custom dataset
    python tools/eval.py -c configs/rtdetrv3_r50vd.yml --checkpoint weights/rtdetrv3_r50vd.pth \
        --anno_file data/coco/annotations/instances_val2017.json \
        --image_dir data/coco/val2017
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.utils.data as data
from tqdm import tqdm

# Add parent directory to path
parent_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_path))

from data.coco_dataset import build_coco_dataset
from engine.evaluator import build_coco_evaluator
from models import create
from tools.infer import postprocess
from utils.checkpoint import load_checkpoint
from utils.config import load_config, apply_overrides
from utils.logger import setup_logger

logger = setup_logger('eval')


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='RT-DETRv3 COCO Evaluation')

    # Config and checkpoint
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')

    # Dataset
    parser.add_argument('--anno_file', type=str, default=None,
                       help='Path to COCO annotation JSON file (defaults to config)')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Path to COCO images directory (defaults to config)')

    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--conf_threshold', type=float, default=0.01,
                       help='Confidence threshold (lower for better recall)')
    parser.add_argument('--nms_threshold', type=float, default=0.7,
                       help='NMS IoU threshold')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run evaluation on (cuda or cpu)')

    # Config overrides
    parser.add_argument('-o', '--override', nargs='*', default=[],
                       help='Config overrides (e.g., num_classes=80)')

    args = parser.parse_args()
    return args


def collate_fn(batch):
    """
    Custom collate function for evaluation

    Args:
        batch: List of (image, target) tuples

    Returns:
        Batched images and list of targets
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # Stack images into batch
    images = torch.stack(images, dim=0)

    return images, targets


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: data.DataLoader,
    evaluator,
    device: torch.device,
    conf_threshold: float = 0.01,
    nms_threshold: float = 0.7
):
    """
    Run evaluation on dataset

    Args:
        model: RT-DETRv3 model
        data_loader: COCO data loader
        evaluator: COCO evaluator
        device: Device to run on
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
    """
    model.eval()

    logger.info(f"Starting evaluation on {len(data_loader)} batches...")

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        # Move to device
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        pred_logits = outputs['pred_logits']  # (B, num_queries, num_classes)
        pred_boxes = outputs['pred_boxes']    # (B, num_queries, 4)

        # Get original image sizes for post-processing
        batch_size = images.shape[0]
        orig_sizes = [target['orig_size'] for target in targets]
        image_ids = [target['image_id'] for target in targets]

        # Post-process predictions
        # Note: We need to create meta info for each image
        predictions = []
        for i in range(batch_size):
            # Create meta info
            orig_h, orig_w = orig_sizes[i]
            meta = {
                'orig_size': (orig_h, orig_w),
                'resized_size': (images.shape[2], images.shape[3]),  # Model input size
                'scale': 1.0,  # Assuming images are already resized
                'input_size': images.shape[2]
            }

            # Post-process single image
            result = postprocess(
                pred_logits[i:i+1],
                pred_boxes[i:i+1],
                meta,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )[0]

            predictions.append(result)

        # Update evaluator
        evaluator.update(predictions, image_ids)

    # Compute metrics
    logger.info("Computing COCO metrics...")
    results = evaluator.accumulate()

    return results


def main():
    """Main evaluation function"""
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    # Override dataset paths if specified
    if args.anno_file:
        cfg['anno_file'] = args.anno_file
    if args.image_dir:
        cfg['image_dir'] = args.image_dir

    # Build model
    logger.info("Building model...")
    model = create('RTDETRv3', global_config=cfg, num_classes=cfg.get('num_classes', 80))

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, args.checkpoint, strict=True)

    # Move to device and set to eval mode
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # Build dataset
    logger.info("Building COCO validation dataset...")
    val_dataset = build_coco_dataset(
        anno_file=cfg.get('anno_file', 'data/coco/annotations/instances_val2017.json'),
        image_dir=cfg.get('image_dir', 'data/coco/val2017'),
        input_size=cfg.get('input_size', 640),
        is_train=False
    )
    logger.info(f"Validation dataset: {len(val_dataset)} images")

    # Build data loader
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )

    # Build evaluator
    logger.info("Building COCO evaluator...")
    evaluator = build_coco_evaluator(
        anno_file=cfg.get('anno_file', 'data/coco/annotations/instances_val2017.json'),
        iou_types=['bbox']
    )

    # Run evaluation
    results = evaluate(
        model,
        val_loader,
        evaluator,
        device,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold
    )

    # Print final results
    logger.info("\n" + "="*50)
    logger.info("COCO Evaluation Results")
    logger.info("="*50)
    for iou_type, metrics in results.items():
        logger.info(f"\n{iou_type.upper()} Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name:10s}: {value:.3f}")

    # Compare with PaddlePaddle baseline if available
    if 'bbox' in results:
        ap = results['bbox']['AP']
        ap50 = results['bbox']['AP50']
        logger.info("\n" + "="*50)
        logger.info(f"Final mAP: {ap:.1%} (AP@[.5:.95])")
        logger.info(f"Final AP50: {ap50:.1%} (AP@.5)")
        logger.info("="*50)

        # PaddlePaddle baseline (R50-vd): 53.4% mAP
        if cfg.get('backbone', 'resnet50') == 'resnet50':
            baseline_map = 0.534
            diff = (ap - baseline_map) * 100
            logger.info(f"\nComparison with PaddlePaddle baseline:")
            logger.info(f"  PyTorch: {ap:.1%}")
            logger.info(f"  Paddle : {baseline_map:.1%}")
            logger.info(f"  Diff   : {diff:+.2f} points")


if __name__ == '__main__':
    main()

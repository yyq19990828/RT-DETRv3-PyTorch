"""
RT-DETRv3 Inference Script

Perform object detection inference on images using trained RT-DETRv3 model.

Usage:
    # Single image inference
    python tools/infer.py -c configs/rtdetrv3_r50vd.yml --checkpoint weights/rtdetrv3_r50vd.pth --infer_img demo.jpg

    # Directory inference
    python tools/infer.py -c configs/rtdetrv3_r50vd.yml --checkpoint weights/rtdetrv3_r50vd.pth --infer_dir images/

    # Custom output directory and threshold
    python tools/infer.py -c configs/rtdetrv3_r50vd.yml --checkpoint weights/rtdetrv3_r50vd.pth \
        --infer_dir images/ --output_dir results/ --threshold 0.5
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add parent directory to path
parent_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_path))

from ppdet.core.workspace import create
from ppdet.utils.config import load_config, apply_overrides
from ppdet.utils.checkpoint import load_checkpoint
from ppdet.utils.logger import setup_logger

logger = setup_logger('infer')


# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# Color palette for visualization (COCO colors)
COLORS = np.array([
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
    [0.635, 0.078, 0.184], [0.300, 0.300, 0.300], [0.600, 0.600, 0.600],
    [1.000, 0.000, 0.000], [1.000, 0.500, 0.000], [0.749, 0.749, 0.000],
    [0.000, 1.000, 0.000], [0.000, 0.000, 1.000], [0.667, 0.000, 1.000],
    [0.333, 0.333, 0.000], [0.333, 0.667, 0.000], [0.333, 1.000, 0.000],
    [0.667, 0.333, 0.000], [0.667, 0.667, 0.000], [0.667, 1.000, 0.000],
    [1.000, 0.333, 0.000], [1.000, 0.667, 0.000], [1.000, 1.000, 0.000],
    [0.000, 0.333, 0.500], [0.000, 0.667, 0.500], [0.000, 1.000, 0.500],
    [0.333, 0.000, 0.500], [0.333, 0.333, 0.500], [0.333, 0.667, 0.500],
    [0.333, 1.000, 0.500], [0.667, 0.000, 0.500], [0.667, 0.333, 0.500],
    [0.667, 0.667, 0.500], [0.667, 1.000, 0.500], [1.000, 0.000, 0.500],
    [1.000, 0.333, 0.500], [1.000, 0.667, 0.500], [1.000, 1.000, 0.500],
    [0.000, 0.333, 1.000], [0.000, 0.667, 1.000], [0.000, 1.000, 1.000],
    [0.333, 0.000, 1.000], [0.333, 0.333, 1.000], [0.333, 0.667, 1.000],
    [0.333, 1.000, 1.000], [0.667, 0.000, 1.000], [0.667, 0.333, 1.000],
    [0.667, 0.667, 1.000], [0.667, 1.000, 1.000], [1.000, 0.000, 1.000],
    [1.000, 0.333, 1.000], [1.000, 0.667, 1.000], [0.333, 0.000, 0.000],
    [0.500, 0.000, 0.000], [0.667, 0.000, 0.000], [0.833, 0.000, 0.000],
    [1.000, 0.000, 0.000], [0.000, 0.167, 0.000], [0.000, 0.333, 0.000],
    [0.000, 0.500, 0.000], [0.000, 0.667, 0.000], [0.000, 0.833, 0.000],
    [0.000, 1.000, 0.000], [0.000, 0.000, 0.167], [0.000, 0.000, 0.333],
    [0.000, 0.000, 0.500], [0.000, 0.000, 0.667], [0.000, 0.000, 0.833],
    [0.000, 0.000, 1.000], [0.000, 0.000, 0.000], [0.143, 0.143, 0.143],
    [0.857, 0.857, 0.857], [1.000, 1.000, 1.000]
]) * 255


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='RT-DETRv3 Inference')

    # Config and checkpoint
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')

    # Input images
    parser.add_argument('--infer_img', type=str, default=None,
                       help='Single image path (has higher priority than --infer_dir)')
    parser.add_argument('--infer_dir', type=str, default=None,
                       help='Directory containing images for inference')

    # Output settings
    parser.add_argument('--output_dir', type=str, default='output/infer',
                       help='Directory to save output images')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detection results to JSON file')

    # Inference settings
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Confidence threshold for visualization')
    parser.add_argument('--nms_threshold', type=float, default=0.7,
                       help='NMS IoU threshold')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run inference on (cuda or cpu)')

    # Config overrides
    parser.add_argument('-o', '--override', nargs='*', default=[],
                       help='Config overrides (e.g., num_classes=80)')

    args = parser.parse_args()

    # Validate input arguments
    assert args.infer_img or args.infer_dir, \
        "--infer_img or --infer_dir must be specified"

    return args


def get_image_list(infer_dir: str, infer_img: str) -> List[str]:
    """
    Get list of image paths for inference

    Args:
        infer_dir: Directory containing images
        infer_img: Single image path (has higher priority)

    Returns:
        List of image file paths
    """
    # Single image has higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    # Directory
    assert os.path.isdir(infer_dir), f"{infer_dir} is not a valid directory"

    image_exts = ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']
    images = []
    for ext in image_exts:
        images.extend(glob.glob(os.path.join(infer_dir, f'*.{ext}')))

    images = sorted(images)
    assert len(images) > 0, f"No images found in {infer_dir}"

    logger.info(f"Found {len(images)} images for inference")
    return images


def preprocess_image(
    image: np.ndarray,
    input_size: int,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Tuple[torch.Tensor, Dict]:
    """
    Preprocess image for model inference

    Args:
        image: Input image (H, W, 3) in BGR format
        input_size: Target input size (square)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Preprocessed image tensor (1, 3, H, W) and meta info
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # Resize while maintaining aspect ratio
    scale = input_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    # Resize image
    image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    padded_image = np.ones((input_size, input_size, 3), dtype=np.uint8) * 114
    padded_image[:new_h, :new_w] = image_resized

    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0

    # Normalize: (x - mean) / std
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    image_tensor = (image_tensor - mean_tensor) / std_tensor

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Meta info for post-processing
    meta = {
        'orig_size': (orig_h, orig_w),
        'resized_size': (new_h, new_w),
        'scale': scale,
        'input_size': input_size
    }

    return image_tensor, meta


def postprocess(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    meta: Dict,
    conf_threshold: float = 0.3,
    nms_threshold: float = 0.7
) -> List[Dict]:
    """
    Post-process model predictions

    Args:
        pred_logits: (B, num_queries, num_classes) class logits
        pred_boxes: (B, num_queries, 4) box predictions in [cx, cy, w, h] format, normalized to [0, 1]
        meta: Meta info from preprocessing
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold

    Returns:
        List of detection results, each containing:
            - 'boxes': (N, 4) in [x1, y1, x2, y2] format (original image coordinates)
            - 'scores': (N,) confidence scores
            - 'labels': (N,) class labels
    """
    batch_size = pred_logits.shape[0]
    results = []

    for i in range(batch_size):
        # Get predictions for single image
        logits = pred_logits[i]  # (num_queries, num_classes)
        boxes = pred_boxes[i]    # (num_queries, 4)

        # Get confidence scores and class labels
        scores = logits.sigmoid().max(dim=-1)[0]  # (num_queries,)
        labels = logits.sigmoid().argmax(dim=-1)  # (num_queries,)

        # Filter by confidence threshold
        keep = scores > conf_threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        if len(scores) == 0:
            results.append({'boxes': torch.zeros((0, 4)), 'scores': torch.zeros(0), 'labels': torch.zeros(0, dtype=torch.long)})
            continue

        # Convert boxes from [cx, cy, w, h] (normalized) to [x1, y1, x2, y2] (pixel coordinates)
        orig_h, orig_w = meta['orig_size']
        resized_h, resized_w = meta['resized_size']

        # Scale boxes from normalized [0, 1] to resized image coordinates
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * resized_w  # x1
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * resized_h  # y1
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * resized_w  # x2
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * resized_h  # y2

        # Clip to resized image boundaries
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, resized_w)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, resized_h)

        # Apply NMS per class
        keep_nms = []
        for class_id in labels.unique():
            class_mask = labels == class_id
            class_boxes = boxes_xyxy[class_mask]
            class_scores = scores[class_mask]
            class_indices = torch.where(class_mask)[0]

            # Apply NMS
            nms_keep = torch.ops.torchvision.nms(class_boxes, class_scores, nms_threshold)
            keep_nms.extend(class_indices[nms_keep].tolist())

        keep_nms = torch.tensor(keep_nms, dtype=torch.long, device=boxes.device)
        boxes_xyxy = boxes_xyxy[keep_nms]
        scores = scores[keep_nms]
        labels = labels[keep_nms]

        results.append({
            'boxes': boxes_xyxy.cpu(),
            'scores': scores.cpu(),
            'labels': labels.cpu()
        })

    return results


def visualize_results(
    image: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    threshold: float = 0.3
) -> np.ndarray:
    """
    Visualize detection results on image

    Args:
        image: Input image (H, W, 3) in BGR format
        boxes: (N, 4) boxes in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        labels: (N,) class labels
        class_names: List of class names
        threshold: Confidence threshold for display

    Returns:
        Annotated image
    """
    vis_image = image.copy()

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue

        x1, y1, x2, y2 = box.int().tolist()
        label_id = label.item()

        # Get color
        color = COLORS[label_id % len(COLORS)].astype(int).tolist()

        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = f"{class_names[label_id]}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 5, label_size[1])

        cv2.rectangle(
            vis_image,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0], label_y + 5),
            color,
            -1
        )
        cv2.putText(
            vis_image,
            label_text,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return vis_image


def main():
    """Main inference function"""
    args = parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

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

    # Get image list
    image_list = get_image_list(args.infer_dir, args.infer_img)

    # Run inference
    logger.info(f"Running inference on {len(image_list)} images...")
    for img_path in image_list:
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue

        # Preprocess
        image_tensor, meta = preprocess_image(image, args.imgsz)
        image_tensor = image_tensor.to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            pred_logits = outputs['pred_logits']  # (1, num_queries, num_classes)
            pred_boxes = outputs['pred_boxes']    # (1, num_queries, 4)

        # Post-process
        results = postprocess(
            pred_logits,
            pred_boxes,
            meta,
            conf_threshold=args.threshold,
            nms_threshold=args.nms_threshold
        )

        # Visualize
        result = results[0]
        vis_image = visualize_results(
            image,
            result['boxes'],
            result['scores'],
            result['labels'],
            COCO_CLASSES,
            threshold=args.threshold
        )

        # Save result
        output_path = os.path.join(args.output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, vis_image)

        # Log detections
        num_detections = len(result['boxes'])
        logger.info(f"Processed {img_path}: {num_detections} detections -> {output_path}")

    logger.info(f"Inference complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

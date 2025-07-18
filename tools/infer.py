"""Inference script for RT-DETRv3 PyTorch."""

import os
import sys
import argparse
import logging
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import json
from typing import List, Dict, Any, Tuple
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.config import load_config
from core.workspace import create
from data.transforms import build_transforms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RT-DETRv3 Inference Script')
    
    # Required arguments
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    
    # Input options
    parser.add_argument('--image', help='Path to input image')
    parser.add_argument('--images-dir', help='Directory containing input images')
    parser.add_argument('--video', help='Path to input video')
    parser.add_argument('--camera', type=int, help='Camera device index')
    
    # Output options
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--save-images', action='store_true', help='Save output images')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--save-predictions', action='store_true', help='Save predictions to JSON')
    
    # Inference options
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='Score threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5, help='NMS threshold')
    parser.add_argument('--max-dets', type=int, default=100, help='Maximum detections per image')
    
    # Visualization options
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    parser.add_argument('--show-labels', action='store_true', default=True, help='Show class labels')
    parser.add_argument('--show-scores', action='store_true', default=True, help='Show confidence scores')
    parser.add_argument('--box-thickness', type=int, default=2, help='Bounding box thickness')
    parser.add_argument('--font-size', type=int, default=20, help='Font size for labels')
    
    # Performance options
    parser.add_argument('--warmup-runs', type=int, default=5, help='Number of warmup runs')
    parser.add_argument('--benchmark', action='store_true', help='Enable benchmarking')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    
    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()


def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_model(cfg, checkpoint_path, device):
    """Load model from checkpoint."""
    # Build model
    model = create(cfg.model)
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
    else:
        model_state_dict = checkpoint
    
    # Handle EMA model
    if 'ema_model_state_dict' in checkpoint:
        model_state_dict = checkpoint['ema_model_state_dict']
    
    model.load_state_dict(model_state_dict)
    model.eval()
    
    return model


def build_transforms(cfg):
    """Build inference transforms."""
    # Use validation transforms for inference
    transforms = build_transforms(cfg.val_dataloader.dataset.transforms)
    return transforms


def preprocess_image(image_path: str, transforms) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size  # (width, height)
    
    # Apply transforms
    sample = {
        'image': image,
        'orig_size': torch.tensor([image.height, image.width]),
        'size': torch.tensor([image.height, image.width])
    }
    
    sample = transforms(sample)
    
    # Get preprocessed image
    processed_image = sample['image']
    if len(processed_image.shape) == 3:
        processed_image = processed_image.unsqueeze(0)
    
    return processed_image, orig_size


def postprocess_predictions(outputs, orig_size: Tuple[int, int], score_threshold: float = 0.5,
                          nms_threshold: float = 0.5, max_dets: int = 100) -> List[Dict[str, Any]]:
    """Postprocess model predictions."""
    # Get predictions from last decoder layer
    if isinstance(outputs, list):
        pred_logits = outputs[-1]['pred_logits']
        pred_boxes = outputs[-1]['pred_boxes']
    else:
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
    
    # Convert to CPU
    pred_logits = pred_logits.cpu()
    pred_boxes = pred_boxes.cpu()
    
    # Remove batch dimension
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.squeeze(0)
        pred_boxes = pred_boxes.squeeze(0)
    
    # Get scores and labels
    scores = F.softmax(pred_logits, dim=-1)
    scores, labels = scores.max(dim=-1)
    
    # Filter background class (assume class 0 is background)
    keep = labels > 0
    scores = scores[keep]
    labels = labels[keep]
    pred_boxes = pred_boxes[keep]
    
    # Filter by score threshold
    keep = scores > score_threshold
    scores = scores[keep]
    labels = labels[keep]
    pred_boxes = pred_boxes[keep]
    
    if len(scores) == 0:
        return []
    
    # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
    boxes = pred_boxes.clone()
    boxes[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * orig_size[0]
    boxes[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * orig_size[1]
    boxes[:, 2] = (pred_boxes[:, 0] + pred_boxes[:, 2] / 2) * orig_size[0]
    boxes[:, 3] = (pred_boxes[:, 1] + pred_boxes[:, 3] / 2) * orig_size[1]
    
    # Apply NMS
    if nms_threshold > 0:
        from torchvision.ops import nms
        keep = nms(boxes, scores, nms_threshold)
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]
    
    # Limit number of detections
    if len(scores) > max_dets:
        scores, indices = scores.topk(max_dets)
        labels = labels[indices]
        boxes = boxes[indices]
    
    # Create detection results
    detections = []
    for score, label, box in zip(scores, labels, boxes):
        detection = {
            'bbox': box.tolist(),
            'score': float(score),
            'category_id': int(label),
            'label': int(label) - 1  # Convert to 0-based index
        }
        detections.append(detection)
    
    return detections


def visualize_detections(image_path: str, detections: List[Dict[str, Any]], 
                        class_names: List[str], args) -> Image.Image:
    """Visualize detections on image."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", args.font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw detections
    for detection in detections:
        bbox = detection['bbox']
        score = detection['score']
        label = detection['label']
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline='red', width=args.box_thickness)
        
        # Draw label and score
        if args.show_labels or args.show_scores:
            text_parts = []
            if args.show_labels and label < len(class_names):
                text_parts.append(class_names[label])
            if args.show_scores:
                text_parts.append(f'{score:.2f}')
            
            if text_parts:
                text = ' '.join(text_parts)
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw text background
                draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], 
                             fill='red', outline='red')
                
                # Draw text
                draw.text((x1, y1 - text_height), text, fill='white', font=font)
    
    return image


def infer_image(model, image_path: str, transforms, class_names: List[str], device, args):
    """Inference on a single image."""
    logger = logging.getLogger(__name__)
    
    # Preprocess image
    processed_image, orig_size = preprocess_image(image_path, transforms)
    processed_image = processed_image.to(device)
    
    # Inference
    with torch.no_grad():
        start_time = time.time()
        outputs = model(processed_image)
        inference_time = time.time() - start_time
    
    # Postprocess predictions
    detections = postprocess_predictions(
        outputs, orig_size, args.score_threshold, args.nms_threshold, args.max_dets
    )
    
    logger.info(f'Inference time: {inference_time:.3f}s, Detections: {len(detections)}')
    
    # Visualize results
    if not args.no_visualize:
        vis_image = visualize_detections(image_path, detections, class_names, args)
        
        # Save visualization
        if args.save_images:
            output_path = os.path.join(args.output_dir, 
                                     f'vis_{os.path.basename(image_path)}')
            vis_image.save(output_path)
            logger.info(f'Visualization saved to: {output_path}')
    
    return detections, inference_time


def infer_images_dir(model, images_dir: str, transforms, class_names: List[str], device, args):
    """Inference on directory of images."""
    logger = logging.getLogger(__name__)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for filename in os.listdir(images_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(images_dir, filename))
    
    logger.info(f'Found {len(image_files)} images')
    
    # Process images
    all_detections = {}
    total_time = 0
    
    for image_path in image_files:
        detections, inference_time = infer_image(
            model, image_path, transforms, class_names, device, args
        )
        
        all_detections[image_path] = detections
        total_time += inference_time
    
    logger.info(f'Total inference time: {total_time:.3f}s, Average: {total_time/len(image_files):.3f}s')
    
    return all_detections


def infer_video(model, video_path: str, transforms, class_names: List[str], device, args):
    """Inference on video."""
    logger = logging.getLogger(__name__)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f'Video: {width}x{height} @ {fps} FPS')
    
    # Setup video writer
    if args.save_video:
        output_path = os.path.join(args.output_dir, 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Save temporary image
        temp_path = os.path.join(args.output_dir, 'temp_frame.jpg')
        pil_image.save(temp_path)
        
        # Inference
        detections, inference_time = infer_image(
            model, temp_path, transforms, class_names, device, args
        )
        total_time += inference_time
        
        # Visualize on frame
        if not args.no_visualize:
            vis_image = visualize_detections(temp_path, detections, class_names, args)
            vis_frame = cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR)
            
            if args.save_video:
                out.write(vis_frame)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        if frame_count % 30 == 0:
            logger.info(f'Processed {frame_count} frames')
    
    cap.release()
    if args.save_video:
        out.release()
        logger.info(f'Output video saved to: {output_path}')
    
    logger.info(f'Total frames: {frame_count}, Average FPS: {frame_count/total_time:.1f}')


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f'Using device: {device}')
    
    # Load model
    model = load_model(cfg, args.checkpoint, device)
    
    # Build transforms
    transforms = build_transforms(cfg)
    
    # Load class names (placeholder)
    class_names = [f'class_{i}' for i in range(cfg.model.num_classes)]
    
    # Warmup
    if args.warmup_runs > 0:
        logger.info(f'Warming up model with {args.warmup_runs} runs...')
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        with torch.no_grad():
            for _ in range(args.warmup_runs):
                _ = model(dummy_input)
    
    # Run inference
    all_detections = {}
    
    if args.image:
        logger.info(f'Running inference on image: {args.image}')
        detections, _ = infer_image(model, args.image, transforms, class_names, device, args)
        all_detections[args.image] = detections
        
    elif args.images_dir:
        logger.info(f'Running inference on directory: {args.images_dir}')
        all_detections = infer_images_dir(model, args.images_dir, transforms, class_names, device, args)
        
    elif args.video:
        logger.info(f'Running inference on video: {args.video}')
        infer_video(model, args.video, transforms, class_names, device, args)
        
    elif args.camera is not None:
        logger.info(f'Running inference on camera {args.camera}')
        # Camera inference would be implemented here
        logger.warning('Camera inference not implemented yet')
    
    else:
        logger.error('No input specified. Use --image, --images-dir, --video, or --camera')
        return
    
    # Save predictions
    if args.save_predictions and all_detections:
        predictions_file = os.path.join(args.output_dir, 'predictions.json')
        with open(predictions_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        logger.info(f'Predictions saved to: {predictions_file}')


if __name__ == '__main__':
    main()
"""Evaluation script for RT-DETRv3 PyTorch."""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.config import load_config
from core.workspace import create
from data import build_dataset, build_transforms, build_dataloader
from solver import build_evaluator
from misc.dist_utils import init_distributed_mode, cleanup_distributed_mode, is_main_process


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RT-DETRv3 Evaluation Script')
    
    # Required arguments
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    
    # Optional arguments
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--batch-size', type=int, help='Batch size (override config)')
    parser.add_argument('--num-workers', type=int, help='Number of workers (override config)')
    
    # Evaluation options
    parser.add_argument('--dataset', choices=['val', 'test'], default='val', help='Dataset to evaluate')
    parser.add_argument('--save-predictions', action='store_true', help='Save predictions to file')
    parser.add_argument('--classwise', action='store_true', help='Compute classwise metrics')
    parser.add_argument('--iou-thresholds', nargs='+', type=float, help='IoU thresholds')
    parser.add_argument('--max-dets', nargs='+', type=int, help='Maximum detections per image')
    
    # Distributed evaluation
    parser.add_argument('--world-size', type=int, default=1, help='Number of processes')
    parser.add_argument('--rank', type=int, default=0, help='Process rank')
    parser.add_argument('--dist-url', default='env://', help='URL for distributed training')
    parser.add_argument('--dist-backend', default='nccl', help='Distributed backend')
    parser.add_argument('--local-rank', type=int, default=-1, help='Local rank for distributed training')
    
    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--vis-score-threshold', type=float, default=0.5, help='Score threshold for visualization')
    
    return parser.parse_args()


def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('eval.log')
        ]
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


def build_dataloader(cfg, args, distributed=False):
    """Build evaluation dataloader."""
    # Select dataset config
    if args.dataset == 'val':
        dataset_cfg = cfg.val_dataloader
    else:
        dataset_cfg = cfg.test_dataloader
    
    # Build transforms
    transforms = build_transforms(dataset_cfg.dataset.transforms)
    
    # Build dataset
    dataset = build_dataset({
        **dataset_cfg.dataset,
        'transforms': transforms
    })
    
    # Override config with command line arguments
    if args.batch_size:
        dataset_cfg.batch_size = args.batch_size
    if args.num_workers:
        dataset_cfg.num_workers = args.num_workers
    
    # Build dataloader
    dataloader = build_dataloader(
        dataset,
        dataset_cfg,
        distributed=distributed,
        training=False
    )
    
    return dataloader


def evaluate_model(model, dataloader, evaluator, device, args):
    """Evaluate model on dataset."""
    logger = logging.getLogger(__name__)
    
    # Reset evaluator
    evaluator.reset()
    
    # Evaluation loop
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating', disable=not is_main_process())):
            # Move data to device
            images = batch['images'].to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in batch['targets']]
            
            # Forward pass
            outputs = model(images)
            
            # Update evaluator
            evaluator.update(outputs, targets)
            
            # Save predictions if requested
            if args.save_predictions:
                for output, target in zip(outputs, targets):
                    image_id = target['image_id'].item()
                    
                    # Get predictions from last decoder layer
                    if isinstance(output, list):
                        pred_logits = output[-1]['pred_logits']
                        pred_boxes = output[-1]['pred_boxes']
                    else:
                        pred_logits = output['pred_logits']
                        pred_boxes = output['pred_boxes']
                    
                    # Convert to CPU
                    pred_logits = pred_logits.cpu().numpy()
                    pred_boxes = pred_boxes.cpu().numpy()
                    
                    # Get image size
                    if 'size' in target:
                        img_h, img_w = target['size'].cpu().numpy()
                    else:
                        img_h, img_w = target['orig_size'].cpu().numpy()
                    
                    # Process predictions
                    predictions = process_predictions(pred_logits, pred_boxes, image_id, img_h, img_w)
                    all_predictions.extend(predictions)
            
            # Visualize predictions if requested
            if args.visualize and batch_idx < 10:  # Visualize first 10 batches
                visualize_predictions(images, outputs, targets, args.vis_score_threshold)
    
    # Compute metrics
    metrics = evaluator.compute()
    
    # Save predictions
    if args.save_predictions and all_predictions and is_main_process():
        output_dir = args.output_dir or 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        predictions_file = os.path.join(output_dir, 'predictions.json')
        with open(predictions_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        logger.info(f'Predictions saved to: {predictions_file}')
    
    return metrics


def process_predictions(pred_logits, pred_boxes, image_id, img_h, img_w):
    """Process predictions for a single image."""
    predictions = []
    
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
        prediction = {
            'image_id': image_id,
            'category_id': label + 1,  # COCO categories start from 1
            'bbox': box.tolist(),
            'score': float(score)
        }
        predictions.append(prediction)
    
    return predictions


def visualize_predictions(images, outputs, targets, score_threshold):
    """Visualize predictions (placeholder implementation)."""
    # This is a placeholder for visualization
    # In a real implementation, you would use matplotlib or similar
    pass


def main_worker(rank, args):
    """Main worker function for distributed evaluation."""
    # Initialize distributed mode
    if args.world_size > 1:
        init_distributed_mode(rank, args.world_size, args.dist_url, args.dist_backend)
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    if args.device == 'cuda':
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    
    logger.info(f'Using device: {device}')
    
    # Load model
    model = load_model(cfg, args.checkpoint, device)
    
    # Build dataloader
    dataloader = build_dataloader(cfg, args, distributed=(args.world_size > 1))
    
    # Build evaluator
    evaluator_cfg = cfg.evaluator.copy()
    if args.classwise:
        evaluator_cfg['classwise'] = True
    if args.iou_thresholds:
        evaluator_cfg['iou_thresholds'] = args.iou_thresholds
    if args.max_dets:
        evaluator_cfg['max_dets'] = args.max_dets
    
    evaluator = build_evaluator(evaluator_cfg)
    
    # Evaluate model
    logger.info('Starting evaluation...')
    metrics = evaluate_model(model, dataloader, evaluator, device, args)
    
    # Print results
    if is_main_process():
        logger.info('Evaluation Results:')
        for metric_name, metric_value in metrics.items():
            logger.info(f'  {metric_name}: {metric_value:.4f}')
        
        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            results_file = os.path.join(args.output_dir, 'eval_results.json')
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f'Results saved to: {results_file}')
    
    # Cleanup distributed mode
    if args.world_size > 1:
        cleanup_distributed_mode()


def main():
    """Main function."""
    args = parse_args()
    
    # Set environment variables for distributed training
    if args.local_rank != -1:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if args.world_size > 1:
        # Multi-process evaluation
        mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)
    else:
        # Single-process evaluation
        main_worker(0, args)


if __name__ == '__main__':
    main()
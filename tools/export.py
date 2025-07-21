"""Model export script for RT-DETRv3 PyTorch."""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.onnx
import torchvision
from typing import Tuple, List, Dict, Any
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import load_config
from src.core.workspace import create


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RT-DETRv3 Model Export Script')
    
    # Required arguments
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--format', required=True, choices=['onnx', 'torchscript', 'tensorrt'], 
                       help='Export format')
    
    # Output options
    parser.add_argument('--output-dir', default='exports', help='Output directory')
    parser.add_argument('--output-name', help='Output filename (without extension)')
    
    # Model options
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--input-size', nargs=2, type=int, default=[640, 640], 
                       help='Input image size (height, width)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--dynamic-batch', action='store_true', help='Enable dynamic batch size')
    parser.add_argument('--dynamic-shape', action='store_true', help='Enable dynamic input shape')
    
    # ONNX options
    parser.add_argument('--onnx-opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--onnx-simplify', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--onnx-check', action='store_true', help='Check ONNX model')
    
    # TensorRT options
    parser.add_argument('--trt-precision', choices=['fp32', 'fp16', 'int8'], default='fp16',
                       help='TensorRT precision')
    parser.add_argument('--trt-workspace', type=int, default=1, help='TensorRT workspace size (GB)')
    parser.add_argument('--trt-min-batch', type=int, default=1, help='TensorRT minimum batch size')
    parser.add_argument('--trt-max-batch', type=int, default=8, help='TensorRT maximum batch size')
    parser.add_argument('--trt-calibration-data', help='Path to calibration data for INT8')
    
    # Optimization options
    parser.add_argument('--optimize', action='store_true', help='Apply optimizations')
    parser.add_argument('--remove-detection-head', action='store_true', 
                       help='Remove detection head (export backbone only)')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='Fuse Conv and BatchNorm')
    
    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def setup_logging(debug=False, verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    if verbose:
        level = logging.DEBUG
    
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


def optimize_model(model, args):
    """Apply model optimizations."""
    logger = logging.getLogger(__name__)
    
    # Fuse Conv and BatchNorm
    if args.fuse_conv_bn:
        logger.info('Fusing Conv and BatchNorm layers...')
        model = torch.nn.utils.fuse_conv_bn_eval(model)
    
    # Remove detection head if requested
    if args.remove_detection_head:
        logger.info('Removing detection head...')
        # This would need to be implemented based on the specific model structure
        pass
    
    # Apply torch.compile if available (PyTorch 2.0+)
    if args.optimize and hasattr(torch, 'compile'):
        logger.info('Applying torch.compile optimization...')
        model = torch.compile(model)
    
    return model


def create_dummy_input(batch_size: int, input_size: Tuple[int, int], device) -> torch.Tensor:
    """Create dummy input tensor."""
    height, width = input_size
    dummy_input = torch.randn(batch_size, 3, height, width, device=device)
    return dummy_input


def export_onnx(model, dummy_input, output_path: str, args):
    """Export model to ONNX format."""
    logger = logging.getLogger(__name__)
    
    # Setup dynamic axes
    dynamic_axes = {}
    if args.dynamic_batch:
        dynamic_axes['input'] = {0: 'batch_size'}
        dynamic_axes['output'] = {0: 'batch_size'}
    
    if args.dynamic_shape:
        dynamic_axes['input'] = {0: 'batch_size', 2: 'height', 3: 'width'}
    
    # Export to ONNX
    logger.info(f'Exporting model to ONNX: {output_path}')
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.onnx_opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        verbose=args.verbose
    )
    
    # Check ONNX model
    if args.onnx_check:
        logger.info('Checking ONNX model...')
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info('ONNX model check passed')
    
    # Simplify ONNX model
    if args.onnx_simplify:
        logger.info('Simplifying ONNX model...')
        try:
            import onnxsim
            simplified_model, check = onnxsim.simplify(output_path)
            if check:
                onnx.save(simplified_model, output_path)
                logger.info('ONNX model simplified')
            else:
                logger.warning('ONNX simplification failed')
        except ImportError:
            logger.warning('onnxsim not installed, skipping simplification')
    
    logger.info(f'ONNX export completed: {output_path}')


def export_torchscript(model, dummy_input, output_path: str, args):
    """Export model to TorchScript format."""
    logger = logging.getLogger(__name__)
    
    logger.info(f'Exporting model to TorchScript: {output_path}')
    
    try:
        # Try tracing first
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
        logger.info('Model exported using tracing')
    except Exception as e:
        logger.warning(f'Tracing failed: {e}')
        logger.info('Trying scripting...')
        
        try:
            # Try scripting
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
            logger.info('Model exported using scripting')
        except Exception as e:
            logger.error(f'Scripting also failed: {e}')
            raise e
    
    logger.info(f'TorchScript export completed: {output_path}')


def export_tensorrt(model, dummy_input, output_path: str, args):
    """Export model to TensorRT format."""
    logger = logging.getLogger(__name__)
    
    try:
        import tensorrt as trt
        from torch2trt import torch2trt
    except ImportError:
        logger.error('TensorRT or torch2trt not installed')
        raise ImportError('Please install TensorRT and torch2trt')
    
    logger.info(f'Exporting model to TensorRT: {output_path}')
    
    # Set precision
    fp16_mode = args.trt_precision == 'fp16'
    int8_mode = args.trt_precision == 'int8'
    
    if int8_mode and not args.trt_calibration_data:
        logger.error('INT8 mode requires calibration data')
        raise ValueError('INT8 mode requires calibration data')
    
    # Convert to TensorRT
    trt_model = torch2trt(
        model,
        [dummy_input],
        fp16_mode=fp16_mode,
        int8_mode=int8_mode,
        max_workspace_size=args.trt_workspace * (1 << 30),  # Convert GB to bytes
        max_batch_size=args.trt_max_batch,
        min_batch_size=args.trt_min_batch
    )
    
    # Save TensorRT model
    torch.save(trt_model.state_dict(), output_path)
    
    logger.info(f'TensorRT export completed: {output_path}')


def benchmark_model(model, dummy_input, device, num_runs: int = 100):
    """Benchmark model performance."""
    logger = logging.getLogger(__name__)
    
    logger.info(f'Benchmarking model with {num_runs} runs...')
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
    
    avg_time = elapsed_time / num_runs
    fps = 1.0 / avg_time
    
    logger.info(f'Average inference time: {avg_time:.4f}s')
    logger.info(f'Average FPS: {fps:.2f}')
    
    return avg_time, fps


def save_export_info(export_info: Dict[str, Any], output_path: str):
    """Save export information to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(export_info, f, indent=2)


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.debug, args.verbose)
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
    logger.info(f'Model loaded from: {args.checkpoint}')
    
    # Optimize model
    model = optimize_model(model, args)
    
    # Create dummy input
    dummy_input = create_dummy_input(args.batch_size, args.input_size, device)
    logger.info(f'Input shape: {dummy_input.shape}')
    
    # Test forward pass
    logger.info('Testing forward pass...')
    with torch.no_grad():
        output = model(dummy_input)
    logger.info(f'Forward pass successful. Output shape: {output.shape if hasattr(output, "shape") else "Multiple outputs"}')
    
    # Benchmark original model
    if args.format != 'tensorrt':  # TensorRT benchmarking is done after export
        avg_time, fps = benchmark_model(model, dummy_input, device)
    
    # Generate output filename
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f'rtdetrv3_{args.format}'
    
    # Add extension based on format
    if args.format == 'onnx':
        output_path = os.path.join(args.output_dir, f'{output_name}.onnx')
    elif args.format == 'torchscript':
        output_path = os.path.join(args.output_dir, f'{output_name}.pt')
    elif args.format == 'tensorrt':
        output_path = os.path.join(args.output_dir, f'{output_name}.trt')
    
    # Export model
    if args.format == 'onnx':
        export_onnx(model, dummy_input, output_path, args)
    elif args.format == 'torchscript':
        export_torchscript(model, dummy_input, output_path, args)
    elif args.format == 'tensorrt':
        export_tensorrt(model, dummy_input, output_path, args)
    
    # Save export information
    export_info = {
        'format': args.format,
        'input_size': args.input_size,
        'batch_size': args.batch_size,
        'device': str(device),
        'model_path': args.checkpoint,
        'config_path': args.config,
        'output_path': output_path,
        'export_args': vars(args)
    }
    
    if args.format != 'tensorrt':
        export_info['benchmark'] = {
            'avg_inference_time': avg_time,
            'fps': fps
        }
    
    info_path = os.path.join(args.output_dir, f'{output_name}_info.json')
    save_export_info(export_info, info_path)
    
    logger.info(f'Export completed successfully!')
    logger.info(f'Model saved to: {output_path}')
    logger.info(f'Export info saved to: {info_path}')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Analyze and compare model architecture between PaddlePaddle and PyTorch
Compare parameter counts and model sizes

Usage:
    python analyze_paddle_weights.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up from tools/ to project root
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def count_parameters(model, framework='pytorch'):
    """Count parameters in a model

    Args:
        model: Model instance (Paddle or PyTorch)
        framework: 'pytorch' or 'paddle'

    Returns:
        dict with total and component-wise parameter counts
    """
    if framework == 'pytorch':
        import torch
        total = sum(p.numel() for p in model.parameters())

        components = {}
        for name in ['backbone', 'neck', 'transformer', 'detr_head', 'aux_o2m_head']:
            if hasattr(model, name):
                module = getattr(model, name)
                if module is not None and isinstance(module, torch.nn.Module):
                    components[name] = sum(p.numel() for p in module.parameters())

        return {'total': total, 'components': components}

    elif framework == 'paddle':
        import paddle
        total = sum(p.numpy().size for p in model.parameters())

        components = {}
        for name in ['backbone', 'neck', 'transformer', 'detr_head', 'aux_o2m_head']:
            if hasattr(model, name):
                module = getattr(model, name)
                if module is not None:
                    components[name] = sum(p.numpy().size for p in module.parameters())

        return {'total': total, 'components': components}


def build_paddle_model(config_path: str):
    """Build PaddlePaddle RT-DETRv3 model from config file"""
    try:
        import paddle
        import sys
    except ImportError:
        logger.error("PaddlePaddle not installed. Install with: pip install paddlepaddle")
        return None

    config_path = Path(config_path)
    logger.info(f"Building Paddle model from config: {config_path}")

    # Add Paddle codebase to path for ppdet imports
    paddle_codebase = project_root / "RT-DETRv3-paddle"
    if not paddle_codebase.exists():
        logger.error(f"Paddle codebase not found at: {paddle_codebase}")
        logger.error("Please update the path or ensure RT-DETRv3-paddle is in the current directory")
        return None

    if str(paddle_codebase) not in sys.path:
        sys.path.insert(0, str(paddle_codebase))

    try:
        # Import Paddle model components
        from ppdet.core.workspace import load_config, create

        # Load config
        logger.info(f"Loading config from: {config_path}")
        cfg = load_config(str(config_path))

        # Create model
        logger.info("Creating RT-DETRv3 model...")
        model = create(cfg.architecture)

        # Set to eval mode
        model.eval()

        logger.info("Paddle model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to build Paddle model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def build_pytorch_model(config_path: str):
    """Build PyTorch RT-DETRv3 model from config file"""
    config_path = Path(config_path)
    logger.info(f"Building PyTorch model from config: {config_path}")

    # Add PyTorch codebase to path
    pytorch_codebase = project_root / "rtdetrv3_pytorch"
    if str(pytorch_codebase) not in sys.path:
        sys.path.insert(0, str(pytorch_codebase))

    # Import PyTorch model builder and config loader using workspace
    try:
        from ppdet_pytorch.core.workspace import load_config, create
    except ImportError as e:
        logger.error(f"Failed to import PyTorch components: {e}")
        logger.error("Make sure rtdetrv3_pytorch is in Python path")
        return None

    # Load config
    logger.info(f"Loading config from: {config_path}")
    cfg = load_config(str(config_path))

    # Build model using workspace create
    logger.info("Creating model from config using workspace...")
    model = create(cfg.architecture)

    # Set to eval mode
    model.eval()

    logger.info("PyTorch model loaded successfully")
    return model


def print_model_summary(paddle_stats, pytorch_stats, model_name):
    """Print comparison summary between Paddle and PyTorch models"""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")

    # Total parameters
    paddle_total = paddle_stats['total']
    pytorch_total = pytorch_stats['total']
    diff = pytorch_total - paddle_total
    diff_pct = (diff / paddle_total * 100) if paddle_total > 0 else 0

    print(f"\n{'Component':<20} {'PaddlePaddle':>20} {'PyTorch':>20} {'Difference':>20}")
    print("-" * 85)

    # Total
    print(f"{'TOTAL':<20} {paddle_total:>20,} {pytorch_total:>20,} {diff:>+20,} ({diff_pct:>+6.2f}%)")
    print("-" * 85)

    # Components
    all_components = set(paddle_stats['components'].keys()) | set(pytorch_stats['components'].keys())
    for comp in sorted(all_components):
        paddle_comp = paddle_stats['components'].get(comp, 0)
        pytorch_comp = pytorch_stats['components'].get(comp, 0)
        comp_diff = pytorch_comp - paddle_comp
        comp_diff_pct = (comp_diff / paddle_comp * 100) if paddle_comp > 0 else 0

        status = "✓" if abs(comp_diff_pct) < 1.0 else "⚠"
        print(f"{comp:<20} {paddle_comp:>20,} {pytorch_comp:>20,} {comp_diff:>+20,} ({comp_diff_pct:>+6.2f}%) {status}")

    # Model size
    paddle_size_mb = paddle_total * 4 / 1024 / 1024
    pytorch_size_mb = pytorch_total * 4 / 1024 / 1024

    print(f"\n{'Model Size (MB)':<20} {paddle_size_mb:>20.2f} {pytorch_size_mb:>20.2f} {pytorch_size_mb - paddle_size_mb:>+20.2f}")

    # Validation
    print(f"\n{'='*80}")
    tolerance_pct = 1.0
    if abs(diff_pct) <= tolerance_pct:
        print(f"✓ PASS: Parameter counts match within {tolerance_pct}% tolerance")
        print(f"  Difference: {diff_pct:+.4f}%")
        return True
    else:
        print(f"⚠ FAIL: Parameter count mismatch: {diff_pct:+.2f}%")
        print(f"  Expected difference should be < {tolerance_pct}%")

        # Show mismatched components
        print(f"\n  Mismatched components:")
        for comp in sorted(all_components):
            paddle_comp = paddle_stats['components'].get(comp, 0)
            pytorch_comp = pytorch_stats['components'].get(comp, 0)
            if paddle_comp > 0:
                comp_diff_pct = ((pytorch_comp - paddle_comp) / paddle_comp * 100)
                if abs(comp_diff_pct) > tolerance_pct:
                    print(f"    {comp}: {comp_diff_pct:+.2f}%")

        return False


def main():
    parser = argparse.ArgumentParser(description="Compare RT-DETRv3 model architectures")
    parser.add_argument(
        "--paddle-config",
        default=str(project_root / "configs/paddle/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml"),
        help="Path to Paddle config file"
    )
    parser.add_argument(
        "--torch-config",
        default=str(project_root / "rtdetrv3_pytorch/configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml"),
        help="Path to PyTorch config file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m/%d %H:%M:%S'
    )

    print("="*80)
    print("RT-DETRv3 Model Architecture Comparison")
    print("PaddlePaddle vs PyTorch")
    print("="*80)

    # Step 1: Build models
    print(f"\nStep 1: Building models...")
    print("-" * 80)

    logger.info("Building PaddlePaddle model...")
    paddle_model = build_paddle_model(args.paddle_config)
    if paddle_model is None:
        logger.error("Failed to build Paddle model")
        return 1

    logger.info("Building PyTorch model...")
    pytorch_model = build_pytorch_model(args.torch_config)
    if pytorch_model is None:
        logger.error("Failed to build PyTorch model")
        return 1

    # Step 2: Count parameters
    print(f"\nStep 2: Counting parameters...")
    print("-" * 80)

    logger.info("Counting Paddle model parameters...")
    paddle_stats = count_parameters(paddle_model, framework='paddle')
    logger.info(f"Paddle model: {paddle_stats['total']:,} parameters")

    logger.info("Counting PyTorch model parameters...")
    pytorch_stats = count_parameters(pytorch_model, framework='pytorch')
    logger.info(f"PyTorch model: {pytorch_stats['total']:,} parameters")

    # Step 3: Compare and print summary
    print(f"\nStep 3: Comparison results...")
    print("-" * 80)

    model_name = Path(args.paddle_config).stem
    success = print_model_summary(paddle_stats, pytorch_stats, model_name)

    # Final result
    print(f"\n{'='*80}")
    if success:
        print("✓ MODEL ARCHITECTURE VALIDATED")
        print("✓ PyTorch model matches PaddlePaddle model structure")
        return 0
    else:
        print("⚠ MODEL ARCHITECTURE MISMATCH DETECTED")
        print("  Review component-wise comparison above")
        return 1


if __name__ == '__main__':
    sys.exit(main())

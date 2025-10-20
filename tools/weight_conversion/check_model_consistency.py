"""Model-level consistency check for RT-DETRv3 weight conversion

This script loads both PaddlePaddle and PyTorch models from config files,
runs forward passes with identical inputs, and compares outputs numerically.

Usage:
    python tools/weight_conversion/check_model_consistency.py \
        --paddle-config configs/paddle/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
        --torch-config configs/pytorch/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
        --input-size 640
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.weight_conversion.validation import ModelOutputValidator

logger = logging.getLogger(__name__)


def build_paddle_model(config_path: str):
    """Build PaddlePaddle RT-DETRv3 model from config file

    Args:
        config_path: Path to Paddle config YAML file (e.g., configs/paddle/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml)

    Returns:
        Paddle model in eval mode with loaded weights
    """
    try:
        import paddle
        import sys
    except ImportError:
        raise ImportError("PaddlePaddle not installed. Install with: pip install paddlepaddle")

    config_path = Path(config_path)
    logger.info(f"Building Paddle model from config: {config_path}")

    # Add Paddle codebase to path for ppdet imports
    paddle_codebase = Path("/home/tyjt/文档/Obsidian Vault/Object_Detection/RT-DETRv3/RT-DETRv3-paddle")
    if not paddle_codebase.exists():
        logger.error(f"Paddle codebase not found at: {paddle_codebase}")
        logger.error("Please update the path in build_paddle_model()")
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

        # Load checkpoint if specified in config
        if 'checkpoint' in cfg and cfg['checkpoint']:
            checkpoint_path = Path(cfg['checkpoint'])
            if not checkpoint_path.is_absolute():
                checkpoint_path = project_root / checkpoint_path

            logger.info(f"Loading checkpoint: {checkpoint_path}")
            state_dict = paddle.load(str(checkpoint_path))
            model.set_state_dict(state_dict)
        else:
            logger.warning("No checkpoint path specified in config")

        # Set to eval mode
        model.eval()

        logger.info("Paddle model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to build Paddle model: {e}")
        import traceback
        logger.error("Traceback:")
        logger.error(traceback.format_exc())

        # Provide helpful hints
        if "np.sctypes" in str(e):
            logger.error("\nHint: This error is likely from visualdl. Try:")
            logger.error("  pip install 'numpy<2.0'")
        elif "imgaug" in str(e):
            logger.error("\nHint: Install imgaug: pip install imgaug")

        return None


def build_pytorch_model(config_path: str):
    """Build PyTorch RT-DETRv3 model from config file

    Args:
        config_path: Path to PyTorch config YAML file (e.g., configs/pytorch/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml)

    Returns:
        PyTorch model in eval mode with loaded weights
    """
    config_path = Path(config_path)
    logger.info(f"Building PyTorch model from config: {config_path}")

    # Import PyTorch model builder and config loader
    try:
        from rtdetrv3_pytorch.models.rtdetrv3 import build_rtdetrv3
        from rtdetrv3_pytorch.utils.config import load_config
    except ImportError as e:
        logger.error(f"Failed to import PyTorch components: {e}")
        logger.error("Make sure rtdetrv3_pytorch is in Python path")
        return None

    # Load config
    logger.info(f"Loading config from: {config_path}")
    cfg = load_config(str(config_path))

    # Extract model parameters from config
    resnet_cfg = cfg.get('ResNet', {})
    hybrid_encoder_cfg = cfg.get('HybridEncoder', {})
    transformer_cfg = cfg.get('RTDETRTransformerv3', {})

    # Determine backbone from ResNet depth
    depth = resnet_cfg.get('depth', 50)
    backbone_map = {18: 'resnet18', 34: 'resnet34', 50: 'resnet50', 101: 'resnet101'}
    backbone = backbone_map.get(depth, 'resnet50')

    # IMPORTANT: hidden_dim is a global config parameter (not HybridEncoder.hidden_dim)
    # It controls transformer decoder dimensions
    # HybridEncoder.hidden_dim controls neck (FPN) dimensions (may differ due to expansion)
    global_hidden_dim = cfg.get('hidden_dim', 256)

    # Build model with config parameters
    model = build_rtdetrv3(
        num_classes=cfg.get('num_classes', 80),
        backbone=backbone,
        variant=resnet_cfg.get('variant', 'd'),
        frozen_stages=resnet_cfg.get('freeze_at', -1),
        hidden_dim=global_hidden_dim,  # Use global hidden_dim for transformer
        num_queries=transformer_cfg.get('num_queries', 300),
        num_decoder_layers=transformer_cfg.get('num_decoder_layers', 6),
        num_levels=transformer_cfg.get('num_levels', 3),
        num_points=4,  # Fixed parameter
        eval_idx=transformer_cfg.get('eval_idx', -1),
        o2m=cfg.get('DINOv3Head', {}).get('o2m', 4) if 'DINOv3Head' in cfg else 4,
        o2m_branch=cfg.get('o2m_branch', False),
        num_queries_o2m=cfg.get('num_queries_o2m', 450),
        use_aux_head=False  # No auxiliary head for inference
    )

    # Load checkpoint if specified in config
    if 'checkpoint' in cfg and cfg['checkpoint']:
        checkpoint_path = Path(cfg['checkpoint'])
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)

        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {len(missing_keys)}")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    logger.debug(f"  Missing: {key}")

        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    logger.debug(f"  Unexpected: {key}")
    else:
        logger.warning("No checkpoint path specified in config")

    # Set to eval mode
    model.eval()

    logger.info("PyTorch model loaded successfully")
    return model


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Check RT-DETRv3 model output consistency")
    parser.add_argument(
        "--paddle-config",
        required=True,
        help="Path to Paddle config file (e.g., configs/paddle/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml)"
    )
    parser.add_argument(
        "--torch-config",
        required=True,
        help="Path to PyTorch config file (e.g., configs/pytorch/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml)"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for forward pass (default: 1)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance (default: 1e-4)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance (default: 1e-5)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )

    logger.info("="*80)
    logger.info("RT-DETRv3 Model Output Consistency Check")
    logger.info("="*80)

    # Build models
    logger.info("\nStep 1: Building models...")
    logger.info("-"*80)

    paddle_model = build_paddle_model(args.paddle_config)
    torch_model = build_pytorch_model(args.torch_config)

    if paddle_model is None:
        logger.error("Paddle model not available - cannot perform validation")
        logger.info("\nTo enable validation:")
        logger.info("1. Install PaddlePaddle: pip install paddlepaddle")
        logger.info("2. Configure Paddle RT-DETRv3 codebase path")
        logger.info("3. Implement build_paddle_model() function")
        return 1

    if torch_model is None:
        logger.error("Failed to build PyTorch model")
        return 1

    # Run forward pass comparison
    logger.info("\nStep 2: Model output validation")
    logger.info("-"*80)

    # Generate random input
    input_shape = (args.batch_size, 3, args.input_size, args.input_size)
    logger.info(f"Generating random input: {input_shape}")
    sample_input = np.random.randn(*input_shape).astype(np.float32)

    # Create validator
    validator = ModelOutputValidator(rtol=args.rtol, atol=args.atol)

    # Run validation
    result = validator.validate_forward_pass(
        paddle_model,
        torch_model,
        sample_input
    )

    # Print report
    validator.print_validation_report(result)

    # Return exit code
    if result.passed:
        logger.info("\n" + "="*80)
        logger.info("✅ Model output validation PASSED!")
        logger.info("="*80)
        return 0
    else:
        logger.error("\n" + "="*80)
        logger.error("❌ Model output validation FAILED!")
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

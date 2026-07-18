"""Model-level consistency check for RT-DETRv3 weight conversion

This script loads both PaddlePaddle and PyTorch models from config files,
runs forward passes with identical inputs, and compares outputs numerically.

Usage:
    python tools/dev/check_model_consistency.py \
        --paddle-config third-party/RT-DETRv3-paddle/configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
        --torch-config configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
        --input-size 640
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running this development script without installing the project first.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from ppdet_pytorch.conversion.validation import ModelOutputValidator

logger = logging.getLogger(__name__)


def build_paddle_model(config_path: str):
    """Build PaddlePaddle RT-DETRv3 model from config file

    Args:
        config_path: Path to a Paddle config YAML file in the reference submodule.

    Returns:
        Paddle model in eval mode with loaded weights
    """
    try:
        import paddle
        import sys
    except ImportError:
        raise ImportError(
            "PaddlePaddle is not installed. Run: uv sync --extra dev"
        )

    config_path = Path(config_path)
    logger.info(f"Building Paddle model from config: {config_path}")

    # Add Paddle codebase to path for ppdet imports
    paddle_codebase = project_root / "third-party" / "RT-DETRv3-paddle"
    if not paddle_codebase.exists():
        logger.error(f"Paddle codebase not found at: {paddle_codebase}")
        logger.error("Initialize submodules with: git submodule update --init --recursive")
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

        # Disable post-processing for raw output comparison
        # Post-processing includes NMS which changes output format
        if hasattr(model, 'exclude_post_process'):
            logger.info("Disabling post-processing for raw output comparison")
            model.exclude_post_process = True

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
            logger.error("\nHint: refresh the development environment:")
            logger.error("  uv sync --extra dev")
        elif "imgaug" in str(e):
            logger.error("\nHint: run: uv sync --extra dev")

        return None


def build_pytorch_model(config_path: str):
    """Build PyTorch RT-DETRv3 model from config file

    Args:
        config_path: Path to a PyTorch config YAML file under configs/.

    Returns:
        PyTorch model in eval mode with loaded weights
    """
    config_path = Path(config_path)
    logger.info(f"Building PyTorch model from config: {config_path}")

    # Import PyTorch model builder and config loader using workspace
    try:
        from ppdet_pytorch.modeling.architectures.rtdetrv3 import RTDETRV3
        from ppdet_pytorch.core.workspace import load_config, create
    except ImportError as e:
        logger.error(f"Failed to import PyTorch components: {e}")
        logger.error("Make sure ppdet_pytorch is in Python path")
        return None

    # Load config
    logger.info(f"Loading config from: {config_path}")
    cfg = load_config(str(config_path))

    # Create model using ppdet_pytorch's create() function
    # This uses the unified registration system (PaddlePaddle-compatible)
    logger.info("Creating RT-DETRv3 model...")
    model = create(cfg.architecture)

    # Disable post-processing for raw output comparison
    # Post-processing includes NMS which changes output format
    if hasattr(model, 'exclude_post_process'):
        logger.info("Disabling post-processing for raw output comparison")
        model.exclude_post_process = True

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
                    logger.warning(f"  Missing: {key}")

        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    logger.warning(f"  Unexpected: {key}")
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
        help="Path to Paddle config file in third-party/RT-DETRv3-paddle/configs"
    )
    parser.add_argument(
        "--torch-config",
        required=True,
        help="Path to PyTorch config file under configs"
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
        logger.info("1. Install development dependencies: uv sync --extra dev")
        logger.info(
            "2. Initialize the reference code: "
            "git submodule update --init --recursive"
        )
        logger.info("3. Verify the supplied config and checkpoint paths")
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

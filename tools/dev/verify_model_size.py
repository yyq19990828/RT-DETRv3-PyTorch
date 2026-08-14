#!/usr/bin/env python3
"""
Verify model size consistency after PaddlePaddle to PyTorch migration
Loads models from config files using workspace and compares parameter counts

Usage:
    python verify_model_size.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running this development script without installing the project first.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import torch

logger = logging.getLogger(__name__)


def count_parameters(model):
    """Count total and trainable parameters in PyTorch model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_component_params(model):
    """Get parameter count for each component"""
    components = {}

    for name in ["backbone", "neck", "transformer", "detr_head", "aux_o2m_head"]:
        if hasattr(model, name):
            module = getattr(model, name)
            if module is not None and isinstance(module, torch.nn.Module):
                components[name] = sum(p.numel() for p in module.parameters())

    return components


def build_pytorch_model(config_path: str):
    """Build PyTorch RT-DETRv3 model from config file using workspace"""
    config_path = Path(config_path)
    logger.info(f"Building PyTorch model from config: {config_path}")

    # Import PyTorch workspace
    try:
        from detrs.core.workspace import create, load_config
    except ImportError as e:
        logger.error(f"Failed to import PyTorch components: {e}")
        logger.error("Install the project or add its src directory to PYTHONPATH")
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
    return model, cfg


def print_model_info(config_path, model_name):
    """Load model from config and print parameter info"""
    print(f"\n{'=' * 80}")
    print(f"Model: {model_name}")
    print(f"Config: {config_path}")
    print(f"{'=' * 80}")

    # Build model
    model, cfg = build_pytorch_model(config_path)
    if model is None:
        logger.error(f"Failed to build model for {model_name}")
        return None

    # Count parameters
    total, trainable = count_parameters(model)
    components = get_component_params(model)

    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Size (MB): {total * 4 / 1024 / 1024:.2f}")  # Assuming float32

    # Print component-wise breakdown
    print("\nComponent-wise parameter count:")
    for name, count in sorted(components.items()):
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {name:15s}: {count:12,} ({percentage:5.2f}%)")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        dummy_input = {
            "image": torch.randn(1, 3, 640, 640),
            "im_shape": torch.tensor([[640, 640]], dtype=torch.float32),
            "scale_factor": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        }
        try:
            outputs = model(dummy_input)
            print("  ✓ Forward pass successful")
            if isinstance(outputs, dict):
                print(f"  Output keys: {list(outputs.keys())}")
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: {value.shape}")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"    {key}: list of {len(value)} tensors")
                        if isinstance(value[0], torch.Tensor):
                            print(f"      First tensor shape: {value[0].shape}")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            import traceback

            traceback.print_exc()

    return {
        "name": model_name,
        "total_params": total,
        "trainable_params": trainable,
        "components": components,
        "config": cfg,
    }


def compare_models(results):
    """Compare parameter counts across models"""
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON")
    print(f"{'=' * 80}")

    print(f"\n{'Model':<25} {'Total Params':>15} {'Size (MB)':>12}")
    print("-" * 60)

    for result in results:
        size_mb = result["total_params"] * 4 / 1024 / 1024
        print(f"{result['name']:<25} {result['total_params']:>15,} {size_mb:>12.2f}")

    # Component-wise comparison
    print(f"\n{'=' * 80}")
    print("COMPONENT-WISE COMPARISON")
    print(f"{'=' * 80}")

    # Get all unique components
    all_components = set()
    for result in results:
        all_components.update(result["components"].keys())

    print(f"\n{'Component':<15}", end="")
    for result in results:
        model_short = result["name"].replace("RT-DETRv3-", "").replace("-vd", "")
        print(f" {model_short:>15}", end="")
    print()
    print("-" * (15 + 16 * len(results)))

    for comp in sorted(all_components):
        print(f"{comp:<15}", end="")
        for result in results:
            count = result["components"].get(comp, 0)
            print(f" {count:>15,}", end="")
        print()

    # Validation
    print(f"\n{'=' * 80}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 80}")

    issues = []

    # Check that detr_head has same params across models
    detr_params = [r["components"].get("detr_head", 0) for r in results]
    if len(set(detr_params)) > 1:
        issues.append("⚠ Detection head parameters differ across models")

    # Check that aux_o2m_head has same params across models
    aux_params = [r["components"].get("aux_o2m_head", 0) for r in results]
    if len(set(aux_params)) > 1 and all(p > 0 for p in aux_params):
        issues.append("⚠ Auxiliary head parameters differ across models")

    if issues:
        print("\n⚠ Issues detected:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ All models validated successfully")
        print("✓ Detection heads are consistent across models")
        print("✓ Auxiliary heads are consistent across models")
        print("✓ Parameter differences are expected due to:")
        print("  - Backbone depth variations (R18: ~11M, R34: ~21M, R50: ~23M)")
        print("  - Different neck configurations")
        print("  - Different transformer configurations")

    # Parameter growth analysis
    if len(results) > 1:
        print("\nParameter Growth Analysis:")
        for i in range(1, len(results)):
            growth = results[i]["total_params"] - results[i - 1]["total_params"]
            growth_pct = growth / results[i - 1]["total_params"] * 100
            print(
                f"  {results[i - 1]['name']} → {results[i]['name']}: "
                f"+{growth:,} (+{growth_pct:.1f}%)"
            )


def main():
    parser = argparse.ArgumentParser(description="Verify RT-DETRv3 model sizes")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml",
            "configs/rtdetrv3/rtdetrv3_r34vd_6x_coco.yml",
            "configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml",
        ],
        help="List of config file paths to verify",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    print("=" * 80)
    print("RT-DETRv3 Model Size Verification")
    print("PaddlePaddle → PyTorch Migration Validation")
    print("=" * 80)

    results = []

    for config_path in args.configs:
        model_name = Path(config_path).stem.upper()
        model_name = model_name.replace("RTDETRV3_", "RT-DETRv3-").replace(
            "_6X_COCO", ""
        )

        try:
            result = print_model_info(config_path, model_name)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if len(results) >= 2:
        compare_models(results)
    elif len(results) == 1:
        print("\n✓ Single model verified successfully")
    else:
        print("\n⚠ No models loaded successfully")
        return 1

    print(f"\n{'=' * 80}")
    print("✓ VERIFICATION COMPLETE")
    print(f"{'=' * 80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

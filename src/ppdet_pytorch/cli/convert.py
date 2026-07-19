"""Command-line interface for weight conversion tool

This module provides the CLI interface for converting model weights between
PaddlePaddle and PyTorch formats.
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

from ..conversion import __version__, configure_logging
from ..conversion.converter import WeightConverter
from ..conversion.models import ConversionConfig

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Convert RT-DETRv3 model weights from PaddlePaddle to PyTorch format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Target-aware conversion (recommended)
  python -m ppdet_pytorch.cli.convert \\
      --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \\
      --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \\
      --config configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml

  # With mapping export
  python -m ppdet_pytorch.cli.convert \\
      --input input.pdparams \\
      --output output.pth \\
      --config configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \\
      --save-mapping mapping.json

  # Explicitly skip target model validation
  python -m ppdet_pytorch.cli.convert \\
      --input input.pdparams \\
      --output output.pth \\
      --no-validate

  # Batch conversion for checkpoints sharing one model config
  python -m ppdet_pytorch.cli.convert --batch --input 'checkpoints/*.pdparams' --output converted --config configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --summary converted/summary.json

For more information, see: docs/migrations/weight-conversion.md
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help=(
            "Source PaddlePaddle checkpoint, or a directory/glob when --batch is set"
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output .pth file, or output directory when --batch is set",
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help=(
            "PyTorch model config used to build the target state_dict "
            "(required unless --no-validate is set)"
        ),
    )

    parser.add_argument(
        "--manual-mapping",
        "-m",
        type=str,
        default=None,
        help="Path to JSON file with manual parameter name mapping overrides",
    )

    parser.add_argument(
        "--save-mapping",
        "-s",
        type=str,
        default=None,
        help="Export generated parameter name mapping to JSON file",
    )

    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--strict",
        action="store_true",
        help="Fail on tensor conversion errors and shape mismatches",
    )

    mode_group.add_argument(
        "--permissive",
        action="store_true",
        default=True,
        help="Enable permissive mode (skip mismatched parameters, continue conversion) [default]",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip shape validation against target model",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing output files without confirmation",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Convert every discovered input independently and continue on failures",
    )

    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Write a JSON batch summary (only valid with --batch)",
    )

    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Release source tensors incrementally during conversion",
    )

    parser.add_argument(
        "--parameter-batch-size",
        type=int,
        default=64,
        help="Source tensors released between garbage-collection passes [default: 64]",
    )

    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging verbosity level",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress all output except errors"
    )

    # Version
    parser.add_argument(
        "--version", action="version", version=f"Weight Conversion Tool v{__version__}"
    )

    return parser


def discover_input_paths(input_value: str):
    """Discover batch inputs from a file, directory, or glob pattern."""
    input_path = Path(input_value)
    if input_path.is_dir():
        candidates = input_path.glob("*.pdparams")
    elif glob.has_magic(input_value):
        candidates = (Path(value) for value in glob.glob(input_value, recursive=True))
    else:
        candidates = [input_path]
    return sorted(
        (path for path in candidates if path.is_file() and path.suffix == ".pdparams"),
        key=lambda path: str(path),
    )


def save_batch_summary(summary, output_path: str) -> None:
    """Write a batch summary as UTF-8 JSON."""
    summary_path = Path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments

    Args:
        args: Parsed arguments

    Raises:
        SystemExit: If validation fails
    """
    if args.batch:
        if not discover_input_paths(args.input):
            logger.error(f"No .pdparams inputs found: {args.input}")
            sys.exit(1)
        output_path = Path(args.output)
        if output_path.exists() and not output_path.is_dir():
            logger.error(f"Batch output must be a directory: {args.output}")
            sys.exit(1)
        if args.save_mapping and Path(args.save_mapping).exists():
            if not Path(args.save_mapping).is_dir():
                logger.error(
                    f"Batch mapping output must be a directory: {args.save_mapping}"
                )
                sys.exit(1)
        if args.summary and Path(args.summary).exists() and not args.force:
            logger.error(
                f"Batch summary already exists (use --force to overwrite): {args.summary}"
            )
            sys.exit(1)
    else:
        if not Path(args.input).exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        if not args.input.endswith(".pdparams"):
            logger.warning(
                f"Input file should have .pdparams extension, got: {args.input}"
            )
        if Path(args.output).exists() and not args.force:
            logger.warning(f"Output file already exists: {args.output}")
            logger.error(
                "Refusing to overwrite existing file (use --force to override)"
            )
            sys.exit(1)
        if args.summary:
            logger.error("--summary is only valid with --batch")
            sys.exit(1)

    if args.parameter_batch_size <= 0:
        logger.error("--parameter-batch-size must be positive")
        sys.exit(1)

    # Check manual mapping file exists if specified
    if args.manual_mapping and not Path(args.manual_mapping).exists():
        logger.error(f"Manual mapping file not found: {args.manual_mapping}")
        sys.exit(1)

    if not args.no_validate and not args.config:
        logger.error("--config is required unless --no-validate is set")
        sys.exit(1)

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Target model config not found: {args.config}")
            sys.exit(1)
        if config_path.suffix not in {".yml", ".yaml"}:
            logger.error(f"Target model config must be YAML, got: {args.config}")
            sys.exit(1)


def build_target_state_dict(config_path: str):
    """Build target shape specs and identify weights owned by Linear modules."""
    import torch

    from .. import modeling as _modeling  # noqa: F401
    from ..core.workspace import create, load_config

    cfg = load_config(str(config_path))
    architecture = cfg.architecture
    model = create(architecture)
    transpose_target_keys = {
        f"{name}.weight" if name else "weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    target_shapes = {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    }
    return target_shapes, architecture, transpose_target_keys


def main(argv=None):
    """Main entry point for CLI"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = "ERROR" if args.quiet else args.log_level
    configure_logging(log_level)

    # Check environment variables
    if "PADDLE_CONV_LOG_LEVEL" in os.environ:
        log_level = os.environ["PADDLE_CONV_LOG_LEVEL"]
        configure_logging(log_level)

    logger.info(f"Weight Conversion Tool v{__version__}")

    # Validate arguments
    validate_arguments(args)

    try:
        target_state_dict = None
        target_architecture = None
        transpose_target_keys = None
        if not args.no_validate:
            (
                target_state_dict,
                target_architecture,
                transpose_target_keys,
            ) = build_target_state_dict(args.config)
            logger.info(
                "Built target %s with %d state_dict keys",
                target_architecture,
                len(target_state_dict),
            )

        output_metadata = {
            "target_validation": not args.no_validate,
            "batch_conversion": args.batch,
        }
        if args.config:
            output_metadata["target_config"] = str(args.config)
        if target_architecture:
            output_metadata["target_architecture"] = target_architecture

        config = ConversionConfig(
            strict_mode=args.strict,
            manual_mapping_file=args.manual_mapping,
            export_mapping=args.save_mapping is not None,
            export_mapping_path=args.save_mapping,
            memory_efficient_mode=args.memory_efficient,
            batch_size=(args.parameter_batch_size if args.memory_efficient else None),
            log_level=log_level,
            output_metadata=output_metadata,
        )
        converter = WeightConverter(config)

        if args.batch:
            input_paths = discover_input_paths(args.input)
            summary = converter.convert_batch(
                input_paths=[str(path) for path in input_paths],
                output_directory=args.output,
                target_model_state_dict=target_state_dict,
                transpose_target_keys=transpose_target_keys,
                mapping_directory=args.save_mapping,
                overwrite=args.force,
            )
            if args.summary:
                save_batch_summary(summary, args.summary)
            logger.info("Batch inputs: %d", summary.total_count)
            logger.info("Succeeded: %d", summary.succeeded_count)
            logger.info("Failed: %d", summary.failed_count)
            logger.info("Duration: %.2f seconds", summary.duration_seconds)
            sys.exit(1 if summary.failed_count else 0)

        session = converter.convert(
            input_path=args.input,
            output_path=args.output,
            target_model_state_dict=target_state_dict,
            transpose_target_keys=transpose_target_keys,
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Source: {args.input}")
        logger.info(f"Target: {args.output}")
        logger.info(f"Session ID: {session.session_id}")
        logger.info(f"Duration: {session.duration_seconds:.2f} seconds")
        logger.info(f"Total parameters: {session.statistics.total_parameters}")
        logger.info(f"Converted: {session.statistics.converted_count}")
        logger.info(f"Skipped: {session.statistics.skipped_count}")

        if session.statistics.unmapped_source_keys:
            logger.warning(
                f"Unmapped source parameters: {len(session.statistics.unmapped_source_keys)}"
            )
        if session.statistics.unmapped_target_keys:
            logger.warning(
                f"Unmapped target parameters: {len(session.statistics.unmapped_target_keys)}"
            )

        if session.warnings:
            logger.warning(f"Warnings: {len(session.warnings)}")

        logger.info("=" * 60)
        logger.info("Conversion completed successfully!")

        sys.exit(0)

    except KeyboardInterrupt:
        logger.error("Conversion interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

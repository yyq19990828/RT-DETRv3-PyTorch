"""Command-line interface for weight conversion tool

This module provides the CLI interface for converting model weights between
PaddlePaddle and PyTorch formats.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__, configure_logging
from .converter import WeightConverter
from .models import ConversionConfig

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
  # Basic conversion
  python -m tools.weight_conversion.cli \\
      --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \\
      --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth

  # With mapping export
  python -m tools.weight_conversion.cli \\
      --input input.pdparams \\
      --output output.pth \\
      --save-mapping mapping.json

  # Strict mode (fail on any mismatch)
  python -m tools.weight_conversion.cli \\
      --input input.pdparams \\
      --output output.pth \\
      --strict

For more information, see: specs/003-paddle-pytorch-conversion/quickstart.md
        """
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to source PaddlePaddle checkpoint file (.pdparams)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path for output PyTorch checkpoint file (.pth)"
    )

    # Optional arguments
    parser.add_argument(
        "--manual-mapping", "-m",
        type=str,
        default=None,
        help="Path to JSON file with manual parameter name mapping overrides"
    )

    parser.add_argument(
        "--save-mapping", "-s",
        type=str,
        default=None,
        help="Export generated parameter name mapping to JSON file"
    )

    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode (fail on any shape mismatch or unmapped parameter)"
    )

    mode_group.add_argument(
        "--permissive",
        action="store_true",
        default=True,
        help="Enable permissive mode (skip mismatched parameters, continue conversion) [default]"
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip shape validation against target model"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing output files without confirmation"
    )

    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging verbosity level"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"Weight Conversion Tool v{__version__}"
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments

    Args:
        args: Parsed arguments

    Raises:
        SystemExit: If validation fails
    """
    # Check input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Check input file extension
    if not args.input.endswith(".pdparams"):
        logger.warning(f"Input file should have .pdparams extension, got: {args.input}")

    # Check output file doesn't exist (unless --force)
    if Path(args.output).exists() and not args.force:
        logger.warning(f"Output file already exists: {args.output}")
        logger.error("Refusing to overwrite existing file (use --force to override)")
        sys.exit(1)

    # Check manual mapping file exists if specified
    if args.manual_mapping and not Path(args.manual_mapping).exists():
        logger.error(f"Manual mapping file not found: {args.manual_mapping}")
        sys.exit(1)


def main():
    """Main entry point for CLI"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

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

    # Create conversion configuration
    config = ConversionConfig(
        strict_mode=args.strict,
        manual_mapping_file=args.manual_mapping,
        export_mapping=args.save_mapping is not None,
        export_mapping_path=args.save_mapping,
        log_level=log_level,
    )

    # Create converter and run conversion
    converter = WeightConverter(config)

    try:
        session = converter.convert(
            input_path=args.input,
            output_path=args.output,
            target_model_state_dict=None  # No validation mode for MVP
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
            logger.warning(f"Unmapped source parameters: {len(session.statistics.unmapped_source_keys)}")
        if session.statistics.unmapped_target_keys:
            logger.warning(f"Unmapped target parameters: {len(session.statistics.unmapped_target_keys)}")

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

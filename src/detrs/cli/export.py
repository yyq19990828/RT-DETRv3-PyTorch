"""Export supported detection checkpoints to ONNX and TorchScript."""

import argparse
from pathlib import Path

import torch

from detrs.cli.infer import build_model
from detrs.core.workspace import load_config
from detrs.deploy import (
    DetectionExportAdapter,
    export_onnx,
    export_torchscript,
    make_example_inputs,
    run_onnx,
    run_torchscript,
    validate_detection_outputs,
)
from detrs.utils.config import apply_overrides
from detrs.utils.logger import setup_logger

logger = setup_logger("export")


def create_argument_parser():
    parser = argparse.ArgumentParser(description="Export a detector for deployment")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--format",
        choices=("onnx", "torchscript", "both"),
        default="both",
    )
    parser.add_argument("--output-dir", default="output/export")
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Fixed spatial size; defaults to TestReader.inputs_def.image_shape.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument(
        "--fixed-batch",
        action="store_true",
        help="Do not mark the ONNX batch axes as dynamic.",
    )
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX Runtime/TorchScript output comparison.",
    )
    parser.add_argument("-o", "--override", nargs="*", default=[])
    return parser


def parse_args(argv=None):
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.input_size and min(args.input_size) < 1:
        parser.error("--input-size values must be positive")
    if args.opset_version != 17:
        parser.error("--opset-version must be 17")
    return args


def _input_size(cfg, override=None):
    if override:
        return tuple(override)
    try:
        image_shape = cfg.TestReader["inputs_def"]["image_shape"]
    except (AttributeError, KeyError, TypeError):
        raise ValueError(
            "Config must define TestReader.inputs_def.image_shape or --input-size"
        ) from None
    if not isinstance(image_shape, (list, tuple)) or len(image_shape) != 3:
        raise ValueError("TestReader image_shape must be [channels, height, width]")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in image_shape
    ):
        raise ValueError("TestReader image_shape values must be integers")
    channels, height, width = image_shape
    if channels != 3:
        raise ValueError("TestReader image_shape channels must be 3")
    if height < 1 or width < 1:
        raise ValueError("TestReader image_shape height and width must be positive")
    return height, width


def _output_paths(cfg, output_directory, export_format):
    prefix = cfg.get("filename", "rtdetrv3")
    output_directory = Path(output_directory)
    paths = {}
    if export_format in ("onnx", "both"):
        paths["onnx"] = output_directory / "{}.onnx".format(prefix)
    if export_format in ("torchscript", "both"):
        paths["torchscript"] = output_directory / "{}.torchscript.pt".format(prefix)
    return paths


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    height, width = _input_size(cfg, args.input_size)
    cfg.eval_size = [height, width]
    cfg.eval_spatial_size = [height, width]
    paths = _output_paths(cfg, args.output_dir, args.format)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Export output already exists: {} (use --force)".format(existing[0])
        )

    model = build_model(
        cfg,
        args.checkpoint,
        torch.device("cpu"),
        use_ema=args.use_ema,
    )
    if hasattr(model, "deploy"):
        model.deploy()
    adapter = DetectionExportAdapter(model).eval()
    inputs = make_example_inputs(args.batch_size, height, width)
    with torch.inference_mode():
        reference = adapter(*inputs)

    if "onnx" in paths:
        validate = None
        if not args.no_verify:

            def validate(path):
                logger.info(
                    "ONNX verification: %s",
                    validate_detection_outputs(reference, run_onnx(path, inputs)),
                )

        export_onnx(
            adapter,
            inputs,
            paths["onnx"],
            opset_version=args.opset_version,
            dynamic_batch=not args.fixed_batch,
            validate=validate,
        )
        logger.info("Exported ONNX model to %s", paths["onnx"])

    if "torchscript" in paths:
        validate_torchscript = None
        if not args.no_verify:

            def validate_torchscript(path):
                logger.info(
                    "TorchScript verification: %s",
                    validate_detection_outputs(
                        reference, run_torchscript(path, inputs)
                    ),
                )

        export_torchscript(
            adapter, inputs, paths["torchscript"], validate=validate_torchscript
        )
        logger.info("Exported TorchScript model to %s", paths["torchscript"])
    return 0


if __name__ == "__main__":
    main()

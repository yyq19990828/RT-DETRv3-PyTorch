"""Unified ``detrs`` command dispatch (``detrs <command> ...``)."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Optional, Sequence

from detrs.utils.cli import DetrsHelpFormatter

COMMANDS = {
    "train": ("detrs.cli.train", "Train a detector from a YAML config."),
    "eval": ("detrs.cli.eval", "Evaluate a checkpoint on COCO-style data."),
    "infer": ("detrs.cli.infer", "Run inference on an image or a directory."),
    "export": (
        "detrs.cli.export",
        "Export a detector for deployment (ONNX/TorchScript).",
    ),
    "convert": ("detrs.cli.convert", "Convert PaddlePaddle weights to PyTorch."),
    "models": ("detrs.cli.models", "List, verify, or download released checkpoints."),
}


def create_argument_parser(command: Optional[str] = None) -> argparse.ArgumentParser:
    """Build the top-level ``detrs`` parser.

    Only ``command`` gets its real argument parser attached (via ``parents=``)
    so lightweight subcommands do not pay unrelated modules' import cost; the
    remaining names are registered as placeholders so they still show up in
    help output and error messages. ``command=None`` attaches every real
    parser and is used for top-level ``--help``.
    """
    parser = argparse.ArgumentParser(
        prog="detrs",
        formatter_class=DetrsHelpFormatter,
        description=(
            "DETR-series PyTorch toolbox: train, evaluate, run inference on, "
            "export, and convert detectors, and manage released checkpoints."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="detrs_command",
        required=True,
        metavar="{%s}" % ",".join(COMMANDS),
    )
    for name, (_, help_text) in COMMANDS.items():
        if command is None or command == name:
            module = importlib.import_module(COMMANDS[name][0])
            module_parser = module.create_argument_parser()
            subparsers.add_parser(
                name,
                parents=[module_parser],
                formatter_class=DetrsHelpFormatter,
                help=help_text,
                description=module_parser.description,
                add_help=False,
            )
        else:
            subparsers.add_parser(
                name,
                formatter_class=DetrsHelpFormatter,
                help=help_text,
                add_help=False,
            )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        create_argument_parser().print_help()
        return 0
    parser = create_argument_parser(args[0])
    parsed = parser.parse_args(args)
    module = importlib.import_module(COMMANDS[parsed.detrs_command][0])
    return module.main(args[1:])


if __name__ == "__main__":
    main()

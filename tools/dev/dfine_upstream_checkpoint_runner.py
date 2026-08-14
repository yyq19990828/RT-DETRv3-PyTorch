#!/usr/bin/env python3
"""Run a pinned D-FINE model without importing its optional data stack."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch


def _package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load {} from {}".format(name, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream(root: Path):
    source = root / "src"
    tensorboard = types.ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = object
    sys.modules[tensorboard.__name__] = tensorboard

    _package("src", source)
    core = _package("src.core", source / "core")
    for name in ("workspace", "yaml_utils", "_config", "yaml_config"):
        _load("src.core." + name, source / "core" / (name + ".py"))
    workspace = sys.modules["src.core.workspace"]
    core.register = workspace.register
    core.create = workspace.create
    core.GLOBAL_CONFIG = workspace.GLOBAL_CONFIG
    core.YAMLConfig = sys.modules["src.core.yaml_config"].YAMLConfig

    _package("src.nn", source / "nn")
    backbone = _package("src.nn.backbone", source / "nn/backbone")
    common = _load("src.nn.backbone.common", source / "nn/backbone/common.py")
    backbone.FrozenBatchNorm2d = common.FrozenBatchNorm2d
    _load("src.nn.backbone.hgnetv2", source / "nn/backbone/hgnetv2.py")

    _package("src.zoo", source / "zoo")
    _package("src.zoo.dfine", source / "zoo/dfine")
    for name in (
        "box_ops",
        "utils",
        "dfine_utils",
        "denoising",
        "hybrid_encoder",
        "dfine_decoder",
        "dfine",
    ):
        _load("src.zoo.dfine." + name, source / "zoo/dfine" / (name + ".py"))
    return core.YAMLConfig


def _flatten(value, name, output):
    if torch.is_tensor(value):
        output[name] = value.detach().cpu()
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, "{}.{}".format(name, key), output)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten(item, "{}[{}]".format(name, index), output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--outputs-only", action="store_true")
    args = parser.parse_args()

    yaml_config = _load_upstream(args.upstream_root)
    config_path = (
        args.upstream_root
        / "configs/dfine"
        / "dfine_hgnetv2_{}_coco.yml".format(args.variant)
    )
    config = yaml_config(
        str(config_path), HGNetv2={"pretrained": False, "local_model_dir": None}
    )
    model = config.model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    captures = {}
    handles = []
    if not args.outputs_only:
        for name, module in (
            ("activation.backbone_stem", model.backbone.stem),
            ("activation.backbone", model.backbone),
            ("activation.encoder", model.encoder),
        ):
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, value, name=name: _flatten(
                        value, name, captures
                    )
                )
            )
    try:
        image = torch.load(args.input, map_location="cpu", weights_only=True)
        with torch.inference_mode():
            output = model(image)
        _flatten(output, "output", captures)
        torch.save(captures, args.output)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()

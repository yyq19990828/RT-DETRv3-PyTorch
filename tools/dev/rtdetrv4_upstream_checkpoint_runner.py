#!/usr/bin/env python3
"""Run the pinned RT-DETRv4 student without constructing its teacher."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import torch

PINNED_SHA = "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"


def _package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify_checkout(root: Path) -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_SHA:
        raise ValueError(f"RT-DETRv4 upstream revision mismatch: {revision}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("RT-DETRv4 upstream checkout has modifications")


def _load_upstream(root: Path):
    engine = root / "engine"
    _package("engine", engine)
    core = _package("engine.core", engine / "core")
    for name in ("workspace", "yaml_utils"):
        _load(f"engine.core.{name}", engine / "core" / f"{name}.py")
    workspace = sys.modules["engine.core.workspace"]
    core.register = workspace.register
    core.create = workspace.create
    core.GLOBAL_CONFIG = workspace.GLOBAL_CONFIG

    _package("engine.backbone", engine / "backbone")
    _load("engine.backbone.common", engine / "backbone/common.py")
    _load("engine.backbone.hgnetv2", engine / "backbone/hgnetv2.py")

    _package("engine.rtv4", engine / "rtv4")
    for name in (
        "utils",
        "dfine_utils",
        "denoising",
        "hybrid_encoder",
        "dfine_decoder",
        "rtv4",
    ):
        _load(f"engine.rtv4.{name}", engine / "rtv4" / f"{name}.py")
    return workspace, sys.modules["engine.core.yaml_utils"]


def _flatten(value, name, output):
    if torch.is_tensor(value):
        output[name] = value.detach().cpu()
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, f"{name}.{key}", output)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten(item, f"{name}[{index}]", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=tuple("smlx"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    _verify_checkout(args.upstream_root)
    workspace, yaml_utils = _load_upstream(args.upstream_root)
    config = yaml_utils.load_config(
        str(args.upstream_root / f"configs/rtv4/rtv4_hgnetv2_{args.variant}_coco.yml")
    )
    config["HGNetv2"].update(pretrained=False, local_model_dir=None)
    config = yaml_utils.merge_config(config, inplace=False, overwrite=False)
    model = workspace.create(config["model"], config).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["ema"]["module"], strict=True)

    captures = {}
    handles = []
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

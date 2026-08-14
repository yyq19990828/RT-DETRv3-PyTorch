#!/usr/bin/env python3
"""Run a pinned DEIM-RT-DETRv2 model without importing its data stack."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import torch

PINNED_SHA = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
CONFIGS = {
    "s": "deim_r18vd_120e_coco.yml",
    "m": "deim_r34vd_120e_coco.yml",
    "m-star": "deim_r50vd_m_60e_coco.yml",
    "l": "deim_r50vd_60e_coco.yml",
    "x": "deim_r101vd_60e_coco.yml",
}


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify_checkout(root):
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_SHA:
        raise ValueError(f"DEIM upstream revision mismatch: {revision}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("DEIM upstream checkout has modifications")


def _load_upstream(root):
    engine = root / "engine"
    tensorboard = types.ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = object
    sys.modules[tensorboard.__name__] = tensorboard

    _package("engine", engine)
    core = _package("engine.core", engine / "core")
    for name in ("workspace", "yaml_utils", "_config", "yaml_config"):
        _load(f"engine.core.{name}", engine / "core" / f"{name}.py")
    workspace = sys.modules["engine.core.workspace"]
    core.register = workspace.register
    core.create = workspace.create
    core.GLOBAL_CONFIG = workspace.GLOBAL_CONFIG

    _package("engine.backbone", engine / "backbone")
    _load("engine.backbone.common", engine / "backbone/common.py")
    _load("engine.backbone.presnet", engine / "backbone/presnet.py")

    _package("engine.deim", engine / "deim")
    for name in (
        "utils",
        "denoising",
        "hybrid_encoder",
        "rtdetrv2_decoder",
        "deim",
    ):
        _load(f"engine.deim.{name}", engine / "deim" / f"{name}.py")
    return sys.modules["engine.core.yaml_config"].YAMLConfig


def _flatten(value, name, output):
    if torch.is_tensor(value):
        output[name] = value.detach().cpu()
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, f"{name}.{key}", output)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten(item, f"{name}[{index}]", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=CONFIGS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    _verify_checkout(args.upstream_root)
    yaml_config = _load_upstream(args.upstream_root)
    config = yaml_config(
        str(args.upstream_root / "configs/deim_rtdetrv2" / CONFIGS[args.variant]),
        PResNet={"pretrained": False, "local_model_dir": None},
    )
    model = config.model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)

    captures = {}
    handles = []
    for name, module in (
        ("activation.backbone_stem", model.backbone.conv1),
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

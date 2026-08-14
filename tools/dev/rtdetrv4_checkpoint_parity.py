"""Manifest-bound RT-DETRv4 official checkpoint parity helpers."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from dfine_checkpoint_parity import _compare, _flatten, sha256_file

from detrs.cli.eval import load_evaluation_weights
from detrs.cli.infer import create_preprocessors, prepare_image_batch
from detrs.core.workspace import create, load_config

PINNED_SHA = "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/checkpoints/rtdetrv4_coco.yml"


def verify_upstream_checkout(upstream_root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_SHA:
        raise ValueError(f"RT-DETRv4 upstream revision mismatch: {revision}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("RT-DETRv4 upstream checkout has modifications")
    return revision


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported RT-DETRv4 checkpoint manifest schema")
    if manifest.get("source_repository", {}).get("revision") != PINNED_SHA:
        raise ValueError("RT-DETRv4 manifest does not pin the required revision")
    if set(manifest.get("models", {})) != set("smlx"):
        raise ValueError("RT-DETRv4 manifest must contain S/M/L/X")
    return manifest


def preflight_artifact(checkpoint_root, variant, manifest):
    entry = manifest["models"][variant]
    checkpoint = Path(checkpoint_root) / entry["filename"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.stat().st_size != entry["source_size_bytes"]:
        raise ValueError(f"RT-DETRv4 {variant} checkpoint size mismatch")
    if sha256_file(checkpoint) != entry["source_sha256"]:
        raise ValueError(f"RT-DETRv4 {variant} checkpoint checksum mismatch")
    container = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(container, dict) or not isinstance(container.get("ema"), dict):
        raise ValueError("RT-DETRv4 checkpoint must contain EMA state")
    state = container["ema"].get("module")
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        )
    ):
        raise ValueError("RT-DETRv4 ema.module must be a tensor state dict")
    if len(state) != entry["state_tensor_count"]:
        raise ValueError(f"RT-DETRv4 {variant} state tensor count mismatch")
    return checkpoint, state


def build_local_model(variant, state):
    config = load_config(ROOT / f"configs/rtdetrv4/rtdetrv4_hgnetv2_{variant}_coco.yml")
    model = create(config.architecture)
    model.load_state_dict(state, strict=True)
    return model


def _run_upstream(upstream_root, variant, checkpoint, image, directory, suffix):
    input_path = directory / f"input-{suffix}.pt"
    output_path = directory / f"upstream-{suffix}.pt"
    torch.save(image, input_path)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/dev/rtdetrv4_upstream_checkpoint_runner.py"),
            "--upstream-root",
            str(upstream_root),
            "--variant",
            variant,
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=ROOT,
    )
    return torch.load(output_path, map_location="cpu", weights_only=True)


def validate_variant(
    variant, checkpoint_root, upstream_root, manifest_path=DEFAULT_MANIFEST
):
    torch.set_num_threads(1)
    manifest = load_manifest(manifest_path)
    verify_upstream_checkout(upstream_root)
    checkpoint, state = preflight_artifact(checkpoint_root, variant, manifest)
    model = build_local_model(variant, state).eval()
    model.exclude_post_process = True
    image = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(0))
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
        with tempfile.TemporaryDirectory(prefix="rtdetrv4-parity-") as path:
            reference = _run_upstream(
                upstream_root, variant, checkpoint, image, Path(path), "fixed"
            )
        with torch.inference_mode():
            output = model({"image": image})
        _flatten(output, "output", captures)
    finally:
        for handle in handles:
            handle.remove()
    tolerance = manifest["numerical_tolerance"]
    checks = _compare(reference, captures, tolerance["rtol"], tolerance["atol"])
    return {
        "artifact": {
            "filename": checkpoint.name,
            "sha256": manifest["models"][variant]["source_sha256"],
            "size_bytes": checkpoint.stat().st_size,
        },
        "checks": checks,
        "key_mapping": "identity",
        "state_tensor_count": len(state),
        "status": "APPROVE",
        "variant": variant,
    }


def validate_real_images(
    variant, checkpoint_root, upstream_root, image_paths, manifest_path=DEFAULT_MANIFEST
):
    torch.set_num_threads(1)
    if len(image_paths) != 4 or not all(path.is_file() for path in image_paths):
        raise ValueError("RT-DETRv4 real-image parity requires exactly four images")
    manifest = load_manifest(manifest_path)
    revision = verify_upstream_checkout(upstream_root)
    checkpoint, state = preflight_artifact(checkpoint_root, variant, manifest)
    config = load_config(ROOT / manifest["models"][variant]["config"])
    sample_transform, batch_transform = create_preprocessors(config)
    batch = prepare_image_batch(
        image_paths,
        range(4),
        sample_transform,
        batch_transform,
        torch.device("cpu"),
    )
    checks = []
    tolerance = manifest["numerical_tolerance"]
    with tempfile.TemporaryDirectory(prefix="rtdetrv4-images-") as path:
        directory = Path(path)
        for index, image_path in enumerate(image_paths):
            model = build_local_model(variant, state).eval()
            load_evaluation_weights(model, checkpoint, use_ema=True)
            model.exclude_post_process = True
            image = batch["image"][index : index + 1]
            reference = _run_upstream(
                upstream_root, variant, checkpoint, image, directory, str(index)
            )
            with torch.inference_mode():
                output = model({"image": image})
            captures = {}
            _flatten(output, "output", captures)
            for check in _compare(
                {
                    key: value
                    for key, value in reference.items()
                    if key.startswith("output")
                },
                captures,
                tolerance["rtol"],
                tolerance["atol"],
            ):
                check["image"] = image_path.name
                checks.append(check)
    return {
        "checks": checks,
        "images": [
            {"filename": path.name, "sha256": sha256_file(path)} for path in image_paths
        ],
        "input": {
            "execution_batch_size": 1,
            "shape": list(batch["image"].shape),
            "torch_num_threads": 1,
        },
        "status": "APPROVE",
        "upstream_revision": revision,
        "variant": variant,
    }

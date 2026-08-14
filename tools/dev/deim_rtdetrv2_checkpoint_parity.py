"""Manifest-bound DEIM-RT-DETRv2 official checkpoint parity helpers."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from dfine_checkpoint_parity import _compare, _flatten, sha256_file

from ppdet_pytorch.cli.eval import load_evaluation_weights
from ppdet_pytorch.cli.infer import create_preprocessors, prepare_image_batch
from ppdet_pytorch.core.workspace import create, load_config

PINNED_SHA = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/checkpoints/deim_rtdetrv2_coco.yml"
CONFIGS = {
    "s": "deim_r18vd_120e_coco.yml",
    "m": "deim_r34vd_120e_coco.yml",
    "m-star": "deim_r50vd_m_60e_coco.yml",
    "l": "deim_r50vd_60e_coco.yml",
    "x": "deim_r101vd_60e_coco.yml",
}


def verify_upstream_checkout(upstream_root):
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_SHA:
        raise ValueError(f"DEIM upstream revision mismatch: {revision}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("DEIM upstream checkout has modifications")
    return revision


def load_manifest(path=DEFAULT_MANIFEST):
    manifest = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported DEIM-RT-DETRv2 checkpoint manifest schema")
    if manifest.get("source_repository", {}).get("revision") != PINNED_SHA:
        raise ValueError("DEIM-RT-DETRv2 manifest does not pin the required revision")
    if set(manifest.get("models", {})) != set(CONFIGS):
        raise ValueError("DEIM-RT-DETRv2 manifest must contain exactly S/M/M*/L/X")
    return manifest


def preflight_artifact(checkpoint_root, variant, manifest):
    entry = manifest["models"][variant]
    path = checkpoint_root / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"missing DEIM-RT-DETRv2 {variant} checkpoint: {path}")
    if path.stat().st_size != entry["source_size_bytes"]:
        raise ValueError(f"DEIM-RT-DETRv2 {variant} checkpoint size mismatch")
    if sha256_file(path) != entry["source_sha256"]:
        raise ValueError(f"DEIM-RT-DETRv2 {variant} checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if sorted(checkpoint) != sorted(entry["container_keys"]):
        raise ValueError(f"DEIM-RT-DETRv2 {variant} container keys mismatch")
    state = checkpoint["model"]
    if len(state) != entry["state_tensor_count"] or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError(f"DEIM-RT-DETRv2 {variant} model state is invalid")
    return path, state


def build_local_model(variant, state):
    config = load_config(ROOT / "configs/deim/rtdetrv2" / CONFIGS[variant])
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
            str(ROOT / "tools/dev/deim_rtdetrv2_upstream_checkpoint_runner.py"),
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
        with tempfile.TemporaryDirectory(prefix="deim-rtdetrv2-parity-") as path:
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
    return {
        "artifact": {
            "filename": checkpoint.name,
            "sha256": manifest["models"][variant]["source_sha256"],
            "size_bytes": checkpoint.stat().st_size,
        },
        "checks": _compare(reference, captures, tolerance["rtol"], tolerance["atol"]),
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
        raise ValueError("DEIM-RT-DETRv2 parity requires exactly four images")
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
    with tempfile.TemporaryDirectory(prefix="deim-rtdetrv2-images-") as path:
        directory = Path(path)
        for index, image_path in enumerate(image_paths):
            model = build_local_model(variant, state).eval()
            load_evaluation_weights(model, checkpoint)
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

"""Manifest-bound DEIM-D-FINE official checkpoint parity helpers."""

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

PINNED_SHA = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/checkpoints/deim_dfine_coco.yml"


def verify_upstream_checkout(upstream_root: Path) -> str:
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


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported DEIM-D-FINE checkpoint manifest schema")
    if manifest.get("source_repository", {}).get("revision") != PINNED_SHA:
        raise ValueError("DEIM-D-FINE manifest does not pin the required revision")
    if set(manifest.get("models", {})) != set("nsmlx"):
        raise ValueError("DEIM-D-FINE manifest must contain exactly N/S/M/L/X")
    return manifest


def preflight_artifact(checkpoint_root: Path, variant: str, manifest: dict):
    entry = manifest["models"][variant]
    path = checkpoint_root / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"missing DEIM-D-FINE {variant} checkpoint: {path}")
    if path.stat().st_size != entry["source_size_bytes"]:
        raise ValueError(f"DEIM-D-FINE {variant} checkpoint size mismatch")
    if sha256_file(path) != entry["source_sha256"]:
        raise ValueError(f"DEIM-D-FINE {variant} checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if sorted(checkpoint) != sorted(entry["container_keys"]):
        raise ValueError(f"DEIM-D-FINE {variant} checkpoint container keys mismatch")
    state = checkpoint["model"]
    if len(state) != entry["state_tensor_count"] or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError(f"DEIM-D-FINE {variant} model state is invalid")
    return path, state


def build_local_model(variant: str, state: dict):
    config = load_config(ROOT / f"configs/deim/dfine/deim_hgnetv2_{variant}_coco.yml")
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
            str(ROOT / "tools/dev/deim_upstream_checkpoint_runner.py"),
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
        with tempfile.TemporaryDirectory(prefix="deim-dfine-parity-") as path:
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
        raise ValueError("DEIM-D-FINE real-image parity requires exactly four images")
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
    with tempfile.TemporaryDirectory(prefix="deim-dfine-images-") as path:
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

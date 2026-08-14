"""Manifest-bound D-FINE official checkpoint parity helpers."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from detrs import modeling as _modeling  # noqa: F401
from detrs.cli.eval import load_evaluation_weights
from detrs.cli.infer import create_preprocessors, prepare_image_batch
from detrs.core.workspace import create, load_config

PINNED_SHA = "267a6da6d04c8ad52e54120692896515b9e55981"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/checkpoints/dfine_coco.yml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream_checkout(upstream_root: Path) -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if sha != PINNED_SHA:
        raise ValueError("D-FINE upstream revision mismatch: {}".format(sha))
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("D-FINE upstream checkout has modifications")
    return sha


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported D-FINE checkpoint manifest schema")
    if manifest.get("source_repository", {}).get("revision") != PINNED_SHA:
        raise ValueError("D-FINE manifest does not pin the required revision")
    if set(manifest.get("models", {})) != set("nsmlx"):
        raise ValueError("D-FINE manifest must contain exactly N/S/M/L/X")
    return manifest


def preflight_artifact(checkpoint_root: Path, variant: str, manifest: dict):
    entry = manifest["models"][variant]
    path = checkpoint_root / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(
            "missing D-FINE {} checkpoint: {}".format(variant, path)
        )
    if path.stat().st_size != entry["source_size_bytes"]:
        raise ValueError("D-FINE {} checkpoint size mismatch".format(variant))
    if sha256_file(path) != entry["source_sha256"]:
        raise ValueError("D-FINE {} checkpoint SHA-256 mismatch".format(variant))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if sorted(checkpoint) != sorted(entry["container_keys"]):
        raise ValueError(
            "D-FINE {} checkpoint container keys mismatch: {}".format(
                variant, sorted(checkpoint)
            )
        )
    state = checkpoint["model"]
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in state.items()
        )
    ):
        raise ValueError(
            "D-FINE {} model state is not a tensor mapping".format(variant)
        )
    return path, state


def build_local_model(variant: str, state: dict):
    config = load_config(
        ROOT / "configs/dfine/dfine_hgnetv2_{}_coco.yml".format(variant)
    )
    model = create(config.architecture)
    target = model.state_dict()
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    if missing or unexpected:
        raise ValueError(
            "D-FINE {} identity mapping mismatch: missing={}, unexpected={}".format(
                variant, missing, unexpected
            )
        )
    for key in sorted(state):
        if state[key].shape != target[key].shape:
            raise ValueError(
                "D-FINE {} layout mismatch at {}: checkpoint={}, target={}".format(
                    variant, key, tuple(state[key].shape), tuple(target[key].shape)
                )
            )
        if state[key].dtype != target[key].dtype:
            raise ValueError(
                "D-FINE {} dtype mismatch at {}: checkpoint={}, target={}".format(
                    variant, key, state[key].dtype, target[key].dtype
                )
            )
    return model


def _flatten(value, name, output):
    if torch.is_tensor(value):
        output[name] = value.detach().cpu()
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, "{}.{}".format(name, key), output)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten(item, "{}[{}]".format(name, index), output)


def _compare(reference: dict, candidate: dict, rtol: float, atol: float):
    if set(reference) != set(candidate):
        raise AssertionError(
            "capture names differ: missing={}, unexpected={}".format(
                sorted(set(reference) - set(candidate)),
                sorted(set(candidate) - set(reference)),
            )
        )
    checks = []
    for name in sorted(reference):
        try:
            torch.testing.assert_close(
                candidate[name], reference[name], rtol=rtol, atol=atol, msg=name
            )
        except AssertionError as error:
            difference = (candidate[name].double() - reference[name].double()).abs()
            raise AssertionError(
                "first divergent activation/output {}: max_abs_error={}, rtol={}, "
                "atol={}; {}".format(
                    name,
                    float(difference.max()) if difference.numel() else 0.0,
                    rtol,
                    atol,
                    error,
                )
            ) from error
        difference = (candidate[name].double() - reference[name].double()).abs()
        checks.append(
            {
                "name": name,
                "shape": list(reference[name].shape),
                "max_abs_error": float(difference.max()) if difference.numel() else 0.0,
                "status": "APPROVE",
            }
        )
    return checks


def validate_variant(
    variant: str,
    checkpoint_root: Path,
    upstream_root: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict:
    manifest = load_manifest(manifest_path)
    verify_upstream_checkout(upstream_root)
    checkpoint_path, state = preflight_artifact(checkpoint_root, variant, manifest)
    model = build_local_model(variant, state).eval()
    load_evaluation_weights(model, checkpoint_path)
    loaded = model.state_dict()
    if not all(torch.equal(loaded[key], value) for key, value in state.items()):
        raise AssertionError(
            "D-FINE {} common evaluation load changed state values".format(variant)
        )

    generator = torch.Generator().manual_seed(0)
    image = torch.rand(1, 3, 640, 640, generator=generator)
    with tempfile.TemporaryDirectory(prefix="dfine-parity-") as directory:
        directory = Path(directory)
        input_path = directory / "input.pt"
        output_path = directory / "upstream.pt"
        torch.save(image, input_path)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/dev/dfine_upstream_checkpoint_runner.py"),
                "--upstream-root",
                str(upstream_root),
                "--variant",
                variant,
                "--checkpoint",
                str(checkpoint_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            check=True,
            cwd=upstream_root,
        )
        reference = torch.load(output_path, map_location="cpu", weights_only=True)

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
        model.exclude_post_process = True
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
            "filename": checkpoint_path.name,
            "sha256": manifest["models"][variant]["source_sha256"],
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "checks": checks,
        "container_keys": ["model"],
        "key_mapping": "identity",
        "state_tensor_count": len(state),
        "status": "APPROVE",
        "tensor_layout": "pytorch-native",
        "variant": variant,
    }


def validate_real_images(
    variant: str,
    checkpoint_root: Path,
    upstream_root: Path,
    image_paths: list[Path],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict:
    if len(image_paths) != 4 or not all(path.is_file() for path in image_paths):
        raise ValueError("D-FINE real-image parity requires exactly four images")

    manifest = load_manifest(manifest_path)
    sha = verify_upstream_checkout(upstream_root)

    checkpoint_path, state = preflight_artifact(checkpoint_root, variant, manifest)
    config = load_config(
        ROOT / "configs/dfine/dfine_hgnetv2_{}_coco.yml".format(variant)
    )
    sample_transform, batch_transform = create_preprocessors(config)
    batch = prepare_image_batch(
        image_paths,
        range(len(image_paths)),
        sample_transform,
        batch_transform,
        torch.device("cpu"),
    )
    image = batch["image"]
    if image.shape != (4, 3, 640, 640) or image.dtype != torch.float32:
        raise ValueError(
            "unexpected D-FINE parity input: shape={}, dtype={}".format(
                tuple(image.shape), image.dtype
            )
        )

    tolerance = manifest["numerical_tolerance"]
    checks = []
    with tempfile.TemporaryDirectory(prefix="dfine-real-image-parity-") as directory:
        directory = Path(directory)
        for index, image_path in enumerate(image_paths):
            model = build_local_model(variant, state).eval()
            load_evaluation_weights(model, checkpoint_path)
            model.exclude_post_process = True
            input_path = directory / "input-{}.pt".format(index)
            output_path = directory / "upstream-{}.pt".format(index)
            image_input = image[index : index + 1]
            torch.save(image_input, input_path)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/dev/dfine_upstream_checkpoint_runner.py"),
                    "--upstream-root",
                    str(upstream_root),
                    "--variant",
                    variant,
                    "--checkpoint",
                    str(checkpoint_path),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--outputs-only",
                ],
                check=True,
                cwd=upstream_root,
            )
            reference = torch.load(output_path, map_location="cpu", weights_only=True)
            with torch.inference_mode():
                output = model({"image": image_input})
            captures = {}
            _flatten(output, "output", captures)
            image_checks = _compare(
                reference, captures, tolerance["rtol"], tolerance["atol"]
            )
            for check in image_checks:
                check["image"] = image_path.name
            checks.extend(image_checks)
    return {
        "checks": checks,
        "images": [
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in image_paths
        ],
        "input": {
            "execution_batch_size": 1,
            "device": "cpu",
            "dtype": str(image.dtype),
            "shape": list(image.shape),
        },
        "status": "APPROVE",
        "upstream_revision": sha,
        "variant": variant,
    }

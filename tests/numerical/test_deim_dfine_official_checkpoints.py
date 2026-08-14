"""Opt-in parity checks for official DEIM-D-FINE N/S/M/L/X checkpoints."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from ppdet_pytorch.core.workspace import create, load_config

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/checkpoints/deim_dfine_coco.yml"
PINNED_SHA = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"

pytestmark = [pytest.mark.numerical, pytest.mark.slow]


def _assets():
    checkpoint_value = os.environ.get("DEIM_DFINE_CHECKPOINT_ROOT")
    upstream_value = os.environ.get("DEIM_UPSTREAM_ROOT")
    if not checkpoint_value or not upstream_value:
        pytest.skip("set DEIM_DFINE_CHECKPOINT_ROOT and DEIM_UPSTREAM_ROOT")
    return Path(checkpoint_value), Path(upstream_value)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(value, name, output):
    if torch.is_tensor(value):
        output[name] = value.detach().cpu()
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, f"{name}.{key}", output)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten(item, f"{name}[{index}]", output)


@pytest.mark.parametrize("variant", "nsmlx")
def test_official_deim_dfine_checkpoint_matches_pinned_upstream(variant):
    torch.set_num_threads(1)
    checkpoint_root, upstream_root = _assets()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == PINNED_SHA
    assert not dirty

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    entry = manifest["models"][variant]
    checkpoint_path = checkpoint_root / entry["filename"]
    assert checkpoint_path.stat().st_size == entry["source_size_bytes"]
    assert _sha256(checkpoint_path) == entry["source_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = load_config(ROOT / entry["config"])
    model = create(config.architecture).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.exclude_post_process = True

    generator = torch.Generator().manual_seed(0)
    image = torch.rand(1, 3, 640, 640, generator=generator)
    with tempfile.TemporaryDirectory(prefix="deim-dfine-parity-") as directory:
        directory = Path(directory)
        input_path = directory / "input.pt"
        output_path = directory / "upstream.pt"
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
                str(checkpoint_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            check=True,
            cwd=ROOT,
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
        with torch.inference_mode():
            output = model({"image": image})
        _flatten(output, "output", captures)
    finally:
        for handle in handles:
            handle.remove()

    assert captures.keys() == reference.keys()
    tolerance = manifest["numerical_tolerance"]
    for name in captures:
        torch.testing.assert_close(
            captures[name],
            reference[name],
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            msg=name,
        )


def test_rejects_swapped_deim_dfine_variant_before_prediction():
    checkpoint_root, _ = _assets()
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    state = torch.load(
        checkpoint_root / manifest["models"]["n"]["filename"],
        map_location="cpu",
        weights_only=False,
    )["model"]
    config = load_config(ROOT / manifest["models"]["s"]["config"])
    model = create(config.architecture)

    with pytest.raises(RuntimeError, match="Missing key|size mismatch"):
        model.load_state_dict(state, strict=True)

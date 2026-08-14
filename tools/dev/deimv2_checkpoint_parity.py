"""Manifest-bound DEIMv2 official checkpoint parity helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch
import yaml
from dfine_checkpoint_parity import sha256_file

from detrs.core.workspace import create, load_config

PINNED_SHA = "add5bcdb499bf7b8a366bfeac1a47d3dc278de27"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/checkpoints/deimv2_coco.yml"

VARIANT_CONFIGS = {
    "x": "configs/deimv2/deimv2_dinov3_x_coco.yml",
    "l": "configs/deimv2/deimv2_dinov3_l_coco.yml",
    "m": "configs/deimv2/deimv2_dinov3_m_coco.yml",
    "s": "configs/deimv2/deimv2_dinov3_s_coco.yml",
    "n": "configs/deimv2/deimv2_hgnetv2_n_coco.yml",
    "pico": "configs/deimv2/deimv2_hgnetv2_pico_coco.yml",
    "femto": "configs/deimv2/deimv2_hgnetv2_femto_coco.yml",
    "atto": "configs/deimv2/deimv2_hgnetv2_atto_coco.yml",
}


def verify_upstream_checkout(upstream_root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_SHA:
        raise ValueError(f"DEIMv2 upstream revision mismatch: {revision}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("DEIMv2 upstream checkout has modifications")
    return revision


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported DEIMv2 checkpoint manifest schema")
    if manifest.get("source_repository", {}).get("revision") != PINNED_SHA:
        raise ValueError("DEIMv2 manifest does not pin the required revision")
    if set(manifest.get("models", {})) != set(VARIANT_CONFIGS):
        raise ValueError(
            "DEIMv2 manifest must contain exactly X/L/M/S/N/Pico/Femto/Atto"
        )
    return manifest


def preflight_artifact(checkpoint_root: Path, variant: str, manifest: dict):
    entry = manifest["models"][variant]
    path = checkpoint_root / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"missing DEIMv2 {variant} checkpoint: {path}")
    if path.stat().st_size != entry["source_size_bytes"]:
        raise ValueError(f"DEIMv2 {variant} checkpoint size mismatch")
    if sha256_file(path) != entry["source_sha256"]:
        raise ValueError(f"DEIMv2 {variant} checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if sorted(checkpoint) != sorted(entry["container_keys"]):
        raise ValueError(f"DEIMv2 {variant} checkpoint container keys mismatch")
    state = checkpoint["model"]
    if len(state) != entry["state_tensor_count"] or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError(f"DEIMv2 {variant} model state is invalid")
    return path, state


def build_local_model(variant: str, state: dict):
    config = load_config(ROOT / VARIANT_CONFIGS[variant])
    model = create(config.architecture)
    model.load_state_dict(state, strict=True)
    return model

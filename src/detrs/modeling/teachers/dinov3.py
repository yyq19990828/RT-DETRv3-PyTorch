"""Strict training-only adapter for the external DINOv3 teacher."""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from detrs.core.workspace import register

__all__ = ["DINOv3TeacherModel"]

PINNED_DINOV3_REVISION = "346f38fee679c56a6888f91c51670fae61d364e0"
DINOV3_MODEL_TYPE = "dinov3_vitb16"
DINOV3_EMBED_DIM = 768
DINOV3_PATCH_SIZE = 16


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repo: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise ValueError("invalid DINOv3 git checkout: {}".format(detail)) from error
    return completed.stdout.strip()


def _square_patch_size(model: nn.Module) -> int:
    patch_size = getattr(model, "patch_size", None)
    if patch_size is None and hasattr(model, "patch_embed"):
        patch_size = getattr(model.patch_embed, "patch_size", None)
    sizes: tuple[int, int]
    if isinstance(patch_size, int):
        sizes = (patch_size, patch_size)
    elif isinstance(patch_size, Sequence) and len(patch_size) == 2:
        sizes = (int(patch_size[0]), int(patch_size[1]))
    else:
        raise ValueError("DINOv3 teacher does not expose a valid patch geometry")
    if sizes != (DINOV3_PATCH_SIZE, DINOV3_PATCH_SIZE):
        raise ValueError(
            "DINOv3 teacher patch geometry must be 16x16, got {}x{}".format(*sizes)
        )
    return sizes[0]


@register
class DINOv3TeacherModel(nn.Module):
    """Load the pinned external ViT-B/16 teacher and return spatial patch features."""

    def __init__(
        self,
        dinov3_repo_path: str,
        dinov3_weights_path: str,
        weights_filename: str,
        weights_size_bytes: int,
        weights_sha256: str,
        dinov3_model_type: str = DINOV3_MODEL_TYPE,
        patch_size: int = DINOV3_PATCH_SIZE,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        if dinov3_model_type != DINOV3_MODEL_TYPE:
            raise ValueError("DINOv3 teacher model type must be dinov3_vitb16")
        if patch_size != DINOV3_PATCH_SIZE:
            raise ValueError("DINOv3 teacher patch_size must be 16")

        repo = Path(dinov3_repo_path).expanduser().resolve()
        weights = Path(dinov3_weights_path).expanduser().resolve()
        if not repo.is_dir() or not (repo / ".git").exists():
            raise FileNotFoundError(
                "DINOv3 repository checkout is missing: {}".format(repo)
            )
        if not weights.is_file():
            raise FileNotFoundError("DINOv3 weights are missing: {}".format(weights))
        if weights.suffix.lower() == ".safetensors":
            raise ValueError(
                "DINOv3 safetensors cannot replace the authorized .pth file"
            )
        if weights.suffix.lower() != ".pth":
            raise ValueError("DINOv3 teacher weights must be the authorized .pth file")
        if not weights_filename or Path(weights_filename).name != weights_filename:
            raise ValueError("DINOv3 weights_filename must be a basename")
        if weights.name != weights_filename:
            raise ValueError("DINOv3 weights filename mismatch")
        if not isinstance(weights_size_bytes, int) or weights_size_bytes <= 0:
            raise ValueError("DINOv3 weights_size_bytes must be positive")
        if weights.stat().st_size != weights_size_bytes:
            raise ValueError("DINOv3 weights size mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", weights_sha256):
            raise ValueError("invalid DINOv3 weights SHA-256")
        if _sha256_file(weights) != weights_sha256:
            raise ValueError("DINOv3 weights SHA-256 mismatch")

        revision = _git_output(repo, "rev-parse", "HEAD")
        if revision != PINNED_DINOV3_REVISION:
            raise ValueError(
                "DINOv3 repository revision mismatch: expected {}, got {}".format(
                    PINNED_DINOV3_REVISION, revision
                )
            )
        if _git_output(repo, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("DINOv3 repository has local modifications")
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError("DINOv3 hubconf.py is missing")
        if sys.version_info < (3, 11):
            raise RuntimeError("DINOv3 teacher training requires Python 3.11+")

        model = torch.hub.load(
            str(repo),
            DINOV3_MODEL_TYPE,
            source="local",
            weights=str(weights),
        )
        if not isinstance(model, nn.Module):
            raise TypeError("DINOv3 hub entry must return a torch.nn.Module")
        if getattr(model, "embed_dim", None) != DINOV3_EMBED_DIM:
            raise ValueError("DINOv3 teacher embed_dim must be 768")
        self.patch_size: int = _square_patch_size(model)
        self.model: nn.Module = model
        self.dinov3_repo_path = str(repo)
        self.dinov3_weights_path = str(weights)
        self.weights_filename = weights_filename
        self.weights_size_bytes = weights_size_bytes
        self.weights_sha256 = weights_sha256
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)
        )
        if self.mean.numel() != 3 or self.std.numel() != 3 or (self.std <= 0).any():
            raise ValueError(
                "DINOv3 normalization must contain three positive channels"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        self.model.train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise TypeError("DINOv3 teacher images must be a BCHW tensor")
        if images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("DINOv3 teacher images must be floating-point RGB")
        if images.shape[-2] != images.shape[-1]:
            raise ValueError("DINOv3 teacher requires square training images")
        normalized = (images - self.mean.to(images)) / self.std.to(images)
        processed = F.avg_pool2d(normalized, kernel_size=2, stride=2)
        if any(size % self.patch_size for size in processed.shape[-2:]):
            raise ValueError("DINOv3 teacher input is not divisible by patch geometry")

        with torch.no_grad():
            outputs = self.model(processed, is_training=True, masks=None)
        if not isinstance(outputs, Mapping) or "x_norm_patchtokens" not in outputs:
            raise ValueError("DINOv3 teacher output is missing x_norm_patchtokens")
        tokens = outputs["x_norm_patchtokens"]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError(
                "DINOv3 x_norm_patchtokens must be a three-dimensional tensor"
            )
        batch, patch_count, channels = tokens.shape
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError("DINOv3 patch token count must form a square feature map")
        expected_side = processed.shape[-1] // self.patch_size
        if batch != images.shape[0] or channels != DINOV3_EMBED_DIM:
            raise ValueError("DINOv3 patch token batch or channel dimension mismatch")
        if side != expected_side:
            raise ValueError("DINOv3 patch token count does not match input geometry")
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("DINOv3 patch features must be finite")
        return (
            tokens.permute(0, 2, 1)
            .reshape(batch, channels, side, side)
            .contiguous()
            .detach()
        )

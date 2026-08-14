"""Fixtures for deterministic upstream/local PyTorch parity tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def tensor_document(path: Path, values: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(values, sort_keys=True), encoding="utf-8")
    return path


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_driver(
    root: Path, script: str, arguments: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "tools" / "dev" / script), *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

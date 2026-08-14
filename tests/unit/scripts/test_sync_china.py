import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/sync_china.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("sync_china", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_locked_linux_wheel_uses_official_lock_hash():
    script = _load_script()
    lock_text = (ROOT / "uv.lock").read_text(encoding="utf-8")

    torch_filename, torch_sha = script.locked_linux_wheel(lock_text, "torch", "cp312")
    vision_filename, vision_sha = script.locked_linux_wheel(
        lock_text, "torchvision", "cp312"
    )

    assert torch_filename == "torch-2.5.1+cu121-cp312-cp312-linux_x86_64.whl"
    assert (
        torch_sha == "222be02548c2e74a21a8fbc8e5b8d2eef9f9faee865d70385d2eb1b9aabcbc76"
    )
    assert vision_filename == "torchvision-0.20.1+cu121-cp312-cp312-linux_x86_64.whl"
    assert (
        vision_sha == "48cf3a716f70370ed5dcb656e7497415ef37860b07e67ea4b1ef8598efe28445"
    )


def test_locked_linux_wheel_rejects_unknown_abi():
    script = _load_script()
    with pytest.raises(ValueError, match="cp38"):
        script.locked_linux_wheel(
            (ROOT / "uv.lock").read_text(encoding="utf-8"), "torch", "cp38"
        )


def test_download_verified_reuses_valid_cache_and_rejects_bad_content(
    tmp_path, monkeypatch
):
    script = _load_script()
    destination = tmp_path / "wheel.whl"
    destination.write_bytes(b"locked wheel")
    expected = hashlib.sha256(b"locked wheel").hexdigest()

    monkeypatch.setattr(
        script.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("valid cache must not download"),
    )
    script.download_verified("https://mirror.invalid/wheel.whl", destination, expected)

    destination.write_bytes(b"stale")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            if hasattr(self, "sent"):
                return b""
            self.sent = True
            return b"tampered"

    monkeypatch.setattr(
        script.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        script.download_verified(
            "https://mirror.invalid/wheel.whl", destination, expected
        )
    assert destination.read_bytes() == b"stale"
    assert not list(tmp_path.glob("*.part"))

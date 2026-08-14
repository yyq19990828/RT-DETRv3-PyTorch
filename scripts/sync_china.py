#!/usr/bin/env python3
"""Preload locked PyTorch wheels from a Chinese mirror, then run uv sync."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "uv.lock"
MIRRORS = {
    "aliyun": "https://mirrors.aliyun.com/pytorch-wheels/cu121",
    "sjtug": "https://mirror.sjtu.edu.cn/pytorch-wheels/cu121",
}
PACKAGES = ("torch", "torchvision")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", choices=sorted(MIRRORS), default="aliyun")
    parser.add_argument(
        "--extra",
        choices=("dev", "export", "export-gpu", "quality", "test"),
        action="append",
        default=[],
        help="Extra passed to the final locked sync; repeat for multiple extras.",
    )
    parser.add_argument(
        "--python",
        default="3.12",
        help="Python version used to create .venv when it does not exist.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Persistent wheel cache; defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-wheels",
        action="store_true",
        help="Keep a temporary wheel directory after successful installation.",
    )
    return parser


def _python_tag(python: Path) -> str:
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def locked_linux_wheel(
    lock_text: str, package: str, python_tag: str
) -> tuple[str, str]:
    """Return the official filename and SHA for a locked Linux x86_64 wheel."""
    sections = re.split(r"(?m)^\[\[package\]\]\s*$", lock_text)
    section = next(
        (
            item
            for item in sections
            if re.search(r'(?m)^name = "{}"$'.format(re.escape(package)), item)
        ),
        None,
    )
    if section is None:
        raise ValueError("{} is absent from uv.lock".format(package))
    pattern = re.compile(
        r'url = "[^"]*/([^"/]*-{}-{}-linux_x86_64\.whl)", '
        r'hash = "sha256:([0-9a-f]{{64}})"'.format(python_tag, python_tag)
    )
    match = pattern.search(section)
    if match is None:
        raise ValueError(
            "uv.lock has no {} wheel for {} Linux x86_64".format(package, python_tag)
        )
    return urllib.parse.unquote(match.group(1)), match.group(2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    """Download atomically and reject mirror content differing from the lock."""
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        print("verified cached {}".format(destination.name), flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=".{}-".format(destination.name),
        suffix=".part",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "rtdetrv3-pytorch-mirror-sync/0.1"}
        )
        print("downloading {}".format(url), flush=True)
        with (
            urllib.request.urlopen(request, timeout=600) as response,
            temporary_path.open("wb") as output,
        ):
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        actual_sha256 = sha256_file(temporary_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "SHA-256 mismatch for {}: expected {}, got {}".format(
                    destination.name, expected_sha256, actual_sha256
                )
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run(command: list[str]) -> None:
    print("+ {}".format(" ".join(command)), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("the CUDA 12.1 mirror helper supports Linux x86_64 only")

    venv_python = REPO_ROOT / ".venv/bin/python"
    if not venv_python.is_file():
        _run(["uv", "venv", "--python", args.python, ".venv"])
    python_tag = _python_tag(venv_python)
    lock_text = LOCK_PATH.read_text(encoding="utf-8")

    temporary_directory = None
    if args.download_dir is None:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="rtdetrv3-pytorch-wheels-"
        )
        download_dir = Path(temporary_directory.name)
    else:
        download_dir = args.download_dir.expanduser().resolve()

    wheels = []
    try:
        for package in PACKAGES:
            filename, expected_sha256 = locked_linux_wheel(
                lock_text, package, python_tag
            )
            destination = download_dir / filename
            mirror_url = "{}/{}".format(
                MIRRORS[args.mirror], urllib.parse.quote(filename, safe="-_.")
            )
            download_verified(mirror_url, destination, expected_sha256)
            wheels.append(destination)

        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "--no-deps",
                *[str(path) for path in wheels],
            ]
        )
        sync_command = ["uv", "sync", "--locked"]
        for extra in args.extra:
            sync_command.extend(("--extra", extra))
        _run(sync_command)
    finally:
        if temporary_directory is not None:
            if args.keep_wheels:
                temporary_directory._finalizer.detach()
                print("kept wheels in {}".format(download_dir), flush=True)
            else:
                temporary_directory.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

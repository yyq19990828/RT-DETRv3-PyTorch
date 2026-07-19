#!/usr/bin/env python3
"""Run one reproducible COCO stability experiment for community collaboration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = {
    "r18": {
        "config": "configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml",
        "pretrain": "pretrained_models/pytorch/ResNet18_vd_pretrained.pth",
    },
    "r34": {
        "config": "configs/rtdetrv3/rtdetrv3_r34vd_6x_coco.yml",
        "pretrain": "pretrained_models/pytorch/ResNet34_vd_pretrained.pth",
    },
    "r50": {
        "config": "configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml",
        "pretrain": ("pretrained_models/pytorch/ResNet50_vd_ssld_v2_pretrained.pth"),
    },
}
COCO_FILES = (
    "train2017",
    "val2017",
    "annotations/instances_train2017.json",
    "annotations/instances_val2017.json",
)
METRIC_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run one model/seed shard of the RT-DETRv3 COCO stability protocol."
        )
    )
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated physical GPU IDs (protocol default: 0,1).",
    )
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size per rank.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--snapshot-epoch", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--pretrain",
        type=Path,
        help="Converted official ImageNet backbone checkpoint.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a prior shard checkpoint; takes precedence over pretraining.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/stability"),
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without creating artifacts.",
    )
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def parse_gpu_ids(value: str) -> list[str]:
    raw_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_ids or any(not item.isascii() or not item.isdigit() for item in raw_ids):
        raise ValueError("--gpus must be a comma-separated list of non-negative IDs")
    gpu_ids = [str(int(item)) for item in raw_ids]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus contains duplicate IDs")
    return gpu_ids


def validate_inputs(args, config: Path, pretrain: Optional[Path]) -> None:
    for name in (
        "--epochs",
        "--batch-size",
        "--num-workers",
        "--snapshot-epoch",
        "--eval-batch-size",
    ):
        positive_int(name, getattr(args, name[2:].replace("-", "_")))

    if args.seed < 0:
        raise ValueError(f"--seed must be non-negative, got {args.seed}")
    if not config.is_file():
        raise FileNotFoundError(f"model config not found: {config}")

    missing_coco = [
        str(args.coco_root / relative)
        for relative in COCO_FILES
        if not (args.coco_root / relative).exists()
    ]
    if missing_coco:
        raise FileNotFoundError(
            "COCO root is incomplete; missing:\n  " + "\n  ".join(missing_coco)
        )

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
    elif pretrain is None or not pretrain.is_file():
        raise FileNotFoundError(
            "converted backbone pretraining checkpoint not found: "
            f"{pretrain}. Prepare it as documented in "
            "docs/migrations/weight-conversion.md or pass --pretrain."
        )

    for executable in ("torchrun", "rtdetrv3-train", "rtdetrv3-eval"):
        path = REPO_ROOT / ".venv" / "bin" / executable
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; create the development environment with "
                "`uv sync --extra dev`."
            )


def yaml_override(key: str, value) -> str:
    return f"{key}={json.dumps(value, ensure_ascii=False)}"


def build_train_command(args, config: Path, pretrain: Optional[Path], run_dir: Path):
    gpu_ids = parse_gpu_ids(args.gpus)
    command = [
        str(REPO_ROOT / ".venv/bin/torchrun"),
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        str(REPO_ROOT / ".venv/bin/rtdetrv3-train"),
        "--ddp",
        "--amp",
        "--seed",
        str(args.seed),
        "-c",
        str(config),
    ]
    if args.resume is not None:
        command.extend(("--resume", str(args.resume)))

    command.append("-o")
    command.extend(
        (
            yaml_override("TrainDataset.dataset_dir", str(args.coco_root)),
            yaml_override("TrainReader.batch_size", args.batch_size),
            yaml_override("worker_num", args.num_workers),
            yaml_override("epoch", args.epochs),
            yaml_override("log_iter", 50),
            yaml_override("use_ema", True),
            yaml_override("accumulate_steps", 1),
            yaml_override("snapshot_epoch", args.snapshot_epoch),
            yaml_override("save_dir", str(run_dir)),
        )
    )
    if args.resume is None:
        command.append(yaml_override("pretrain_weights", str(pretrain)))
    return command


def build_eval_command(args, config: Path, run_dir: Path):
    return [
        str(REPO_ROOT / ".venv/bin/rtdetrv3-eval"),
        "-c",
        str(config),
        "--checkpoint",
        str(run_dir / "model_final.pth"),
        "--anno_file",
        str(args.coco_root / "annotations/instances_val2017.json"),
        "--image_dir",
        str(args.coco_root / "val2017"),
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--output-dir",
        str(run_dir / "eval"),
        "--use-ema",
        "--device",
        "cuda",
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_tensor_sha256(path: Path) -> str:
    """Hash ordered model tensors while ignoring conversion metadata."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint does not contain a state dictionary: {path}")

    digest = hashlib.sha256()
    tensor_count = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            continue
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        tensor_count += 1
    if tensor_count == 0:
        raise ValueError(f"checkpoint contains no model tensors: {path}")
    return digest.hexdigest()


def command_output(command: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_environment(gpu_ids: list[str]) -> dict:
    torch_probe = (
        "import json, torch; "
        "print(json.dumps({"
        "'torch': torch.__version__, "
        "'cuda_runtime': torch.version.cuda, "
        "'cudnn': torch.backends.cudnn.version(), "
        "'cuda_available': torch.cuda.is_available(), "
        "'visible_gpu_count': torch.cuda.device_count(), "
        "'visible_gpu_names': [torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]}))"
    )
    probe_env = os.environ.copy()
    probe_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    try:
        completed = subprocess.run(
            [str(REPO_ROOT / ".venv/bin/python"), "-c", torch_probe],
            cwd=REPO_ROOT,
            env=probe_env,
            check=True,
            capture_output=True,
            text=True,
        )
        torch_info = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RuntimeError(f"failed to inspect the PyTorch/CUDA environment: {error}")

    if not torch_info["cuda_available"]:
        raise RuntimeError("CUDA is unavailable in the uv-managed environment")
    if torch_info["visible_gpu_count"] != len(gpu_ids):
        raise RuntimeError(
            "requested GPU count does not match visible CUDA devices: "
            f"requested={len(gpu_ids)}, visible={torch_info['visible_gpu_count']}"
        )

    nvidia_smi = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_info,
        "nvidia_smi": nvidia_smi.splitlines() if nvidia_smi else None,
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_initial_run(run_dir: Path) -> Optional[dict]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("initial_run"):
        return metadata["initial_run"]
    return {
        key: metadata.get(key)
        for key in ("created_at", "git", "environment", "inputs", "commands")
    }


def run_logged(command: list[str], env: dict, log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            return process.wait()


def parse_eval_metrics(log_path: Path) -> dict:
    metrics = {}
    pattern = re.compile(rf"\b({'|'.join(METRIC_NAMES)})\s*:\s*(-?\d+(?:\.\d+)?)\s*$")
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def main(argv=None) -> int:
    args = parse_args(argv)
    args.coco_root = resolve_path(args.coco_root)
    args.output_root = resolve_path(args.output_root)
    if args.resume is not None:
        args.resume = resolve_path(args.resume)
    spec = MODEL_SPECS[args.model]
    config = resolve_path(Path(spec["config"]))
    pretrain = (
        resolve_path(args.pretrain)
        if args.pretrain is not None
        else resolve_path(Path(spec["pretrain"]))
    )
    gpu_ids = parse_gpu_ids(args.gpus)
    run_dir = args.output_root / f"{args.model}-seed-{args.seed}"

    validate_inputs(args, config, pretrain)
    train_command = build_train_command(args, config, pretrain, run_dir)
    eval_command = build_eval_command(args, config, run_dir)

    print(f"Run directory: {run_dir}")
    print(f"Train command: {shlex.join(train_command)}")
    if not args.skip_eval:
        print(f"Eval command:  {shlex.join(eval_command)}")
    if args.dry_run:
        print("Dry run completed; no files were created.")
        return 0

    if run_dir.exists() and args.resume is None:
        raise FileExistsError(
            f"run directory already exists: {run_dir}; pass --resume to continue it"
        )

    initial_run = load_initial_run(run_dir) if args.resume is not None else None
    initial_inputs = initial_run.get("inputs", {}) if initial_run else {}
    environment = collect_environment(gpu_ids)
    git_commit = command_output(["git", "rev-parse", "HEAD"])
    git_status = command_output(["git", "status", "--short"])
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "protocol": {
            "epochs": args.epochs,
            "gpu_ids": gpu_ids,
            "world_size": len(gpu_ids),
            "batch_size_per_rank": args.batch_size,
            "global_batch_size": args.batch_size * len(gpu_ids),
            "num_workers": args.num_workers,
            "snapshot_epoch": args.snapshot_epoch,
            "eval_batch_size": args.eval_batch_size,
            "accumulate_steps": 1,
            "log_iter": 50,
            "amp": True,
            "ema": True,
            "ddp": True,
            "train_dtype": "amp",
            "eval_dtype": "float32",
        },
        "git": {"commit": git_commit, "dirty": bool(git_status), "status": git_status},
        "environment": environment,
        "initial_run": initial_run,
        "inputs": {
            "config": str(config),
            "config_sha256": sha256(config),
            "coco_root": str(args.coco_root),
            "train_annotations_sha256": sha256(
                args.coco_root / "annotations/instances_train2017.json"
            ),
            "val_annotations_sha256": sha256(
                args.coco_root / "annotations/instances_val2017.json"
            ),
            "pretrain": (
                initial_inputs.get("pretrain")
                if args.resume is not None
                else str(pretrain)
            ),
            "pretrain_file_sha256": (
                initial_inputs.get("pretrain_file_sha256")
                if args.resume is not None
                else sha256(pretrain)
            ),
            "pretrain_tensor_sha256": (
                initial_inputs.get("pretrain_tensor_sha256")
                if args.resume is not None
                else checkpoint_tensor_sha256(pretrain)
            ),
            "resume": None if args.resume is None else str(args.resume),
            "resume_sha256": (None if args.resume is None else sha256(args.resume)),
        },
        "commands": {
            "train": train_command,
            "eval": None if args.skip_eval else eval_command,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "metadata.json", metadata)

    train_env = os.environ.copy()
    train_env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = {
        "schema_version": 1,
        "model": args.model,
        "seed": args.seed,
        "train_exit_code": None,
        "eval_exit_code": None,
        "final_checkpoint_sha256": None,
        "coco_metrics": None,
    }

    result["train_exit_code"] = run_logged(
        train_command, train_env, run_dir / "train.log"
    )
    if result["train_exit_code"] != 0:
        write_json(run_dir / "result.json", result)
        return int(result["train_exit_code"])

    final_checkpoint = run_dir / "model_final.pth"
    if not final_checkpoint.is_file():
        result["train_exit_code"] = 1
        result["error"] = f"final checkpoint not found: {final_checkpoint}"
        write_json(run_dir / "result.json", result)
        return 1
    result["final_checkpoint_sha256"] = sha256(final_checkpoint)

    if not args.skip_eval:
        eval_env = train_env.copy()
        eval_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        eval_log = run_dir / "eval.log"
        result["eval_exit_code"] = run_logged(eval_command, eval_env, eval_log)
        if result["eval_exit_code"] == 0:
            result["coco_metrics"] = parse_eval_metrics(eval_log)

    write_json(run_dir / "result.json", result)
    return int(result["eval_exit_code"] or 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

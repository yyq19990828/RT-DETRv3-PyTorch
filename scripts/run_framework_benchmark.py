#!/usr/bin/env python3
"""Benchmark Paddle and PyTorch RT-DETRv3 in isolated worker processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PADDLE_SOURCE = REPO_ROOT / "third-party/RT-DETRv3-paddle"
DEFAULT_MODEL = "rtdetrv3_r18vd_6x_coco"
DEFAULT_TORCH_CONFIG = "configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml"
DEFAULT_PADDLE_CONFIG = (
    "third-party/RT-DETRv3-paddle/configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml"
)
DEFAULT_TORCH_CHECKPOINT = "pretrained_models/pytorch/rtdetrv3_r18vd_6x_coco.pth"
DEFAULT_PADDLE_CHECKPOINT = "pretrained_models/paddle/rtdetrv3_r18vd_6x_coco.pdparams"
_DERIVED_TORCH_BUFFERS = {
    "aux_o2m_head.anchor_points",
    "aux_o2m_head.stride_tensor",
}


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark RT-DETRv3 model-only inference or a synthetic training "
            "step in isolated Paddle/PyTorch processes."
        )
    )
    parser.add_argument(
        "--framework",
        choices=("both", "paddle", "pytorch"),
        default="both",
    )
    parser.add_argument(
        "--workload",
        choices=("inference", "train-step"),
        default="inference",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--torch-config", default=DEFAULT_TORCH_CONFIG)
    parser.add_argument("--paddle-config", default=DEFAULT_PADDLE_CONFIG)
    parser.add_argument("--torch-checkpoint", default=DEFAULT_TORCH_CHECKPOINT)
    parser.add_argument("--paddle-checkpoint", default=DEFAULT_PADDLE_CHECKPOINT)
    parser.add_argument(
        "--output",
        help="Write the aggregate JSON report to this path; otherwise print it.",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-result", help=argparse.SUPPRESS)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.input_size < 32:
        parser.error("--input-size must be at least 32")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args._worker and args.framework == "both":
        parser.error("an internal worker must select exactly one framework")
    if args._worker and not args._worker_result:
        parser.error("an internal worker requires --_worker-result")
    return args


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def summarize_durations(
    durations_seconds: Sequence[float], batch_size: int
) -> dict[str, Any]:
    if not durations_seconds:
        raise ValueError("at least one measured duration is required")
    values = sorted(float(value) for value in durations_seconds)
    mean_seconds = sum(values) / len(values)
    variance = sum((value - mean_seconds) ** 2 for value in values) / len(values)
    return {
        "samples": len(values),
        "durations_seconds": list(durations_seconds),
        "mean_batch_latency_ms": mean_seconds * 1000.0,
        "median_batch_latency_ms": _percentile(values, 0.5) * 1000.0,
        "p90_batch_latency_ms": _percentile(values, 0.9) * 1000.0,
        "p95_batch_latency_ms": _percentile(values, 0.95) * 1000.0,
        "min_batch_latency_ms": values[0] * 1000.0,
        "max_batch_latency_ms": values[-1] * 1000.0,
        "std_batch_latency_ms": variance**0.5 * 1000.0,
        "throughput_images_per_second": batch_size / mean_seconds,
    }


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _current_rss_bytes() -> Optional[int]:
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def measure_iterations(
    step: Callable[[], Any],
    synchronize: Callable[[], None],
    reset_device_peak: Callable[[], None],
    device_memory: Callable[[], dict[str, Optional[int]]],
    *,
    warmup: int,
    samples: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Optional[int]], Any]:
    last_output: Any = None
    for _ in range(warmup):
        synchronize()
        last_output = step()
        synchronize()

    baseline_peak_rss = _peak_rss_bytes()
    baseline_rss = _current_rss_bytes()
    reset_device_peak()
    durations = []
    for _ in range(samples):
        synchronize()
        started_at = time.perf_counter()
        last_output = step()
        synchronize()
        durations.append(time.perf_counter() - started_at)

    process_peak_rss = _peak_rss_bytes()
    memory = {
        "baseline_rss_bytes": baseline_rss,
        "process_peak_rss_bytes": process_peak_rss,
        "measurement_peak_rss_increase_bytes": max(
            0, process_peak_rss - baseline_peak_rss
        ),
        **device_memory(),
    }
    return summarize_durations(durations, batch_size), memory, last_output


def _synthetic_arrays(batch_size: int, input_size: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    image = rng.standard_normal(
        (batch_size, 3, input_size, input_size), dtype=np.float32
    )
    gt_bbox = np.array(
        [[0.30, 0.40, 0.20, 0.15], [0.72, 0.65, 0.12, 0.25]],
        dtype=np.float32,
    )
    gt_class = np.array([[2], [17]], dtype=np.int64)
    origin_gt_bbox = np.empty_like(gt_bbox)
    origin_gt_bbox[:, :2] = (gt_bbox[:, :2] - gt_bbox[:, 2:] / 2) * input_size
    origin_gt_bbox[:, 2:] = (gt_bbox[:, :2] + gt_bbox[:, 2:] / 2) * input_size
    return {
        "image": image,
        "im_shape": np.full((batch_size, 2), input_size, dtype=np.float32),
        "scale_factor": np.ones((batch_size, 2), dtype=np.float32),
        "gt_bbox": [gt_bbox.copy() for _ in range(batch_size)],
        "gt_class": [gt_class.copy() for _ in range(batch_size)],
        "origin_gt_bbox": np.repeat(origin_gt_bbox[None], batch_size, axis=0),
        "origin_gt_class": np.repeat(gt_class[None], batch_size, axis=0),
        "pad_origin_gt_mask": np.ones((batch_size, 2, 1), dtype=np.float32),
    }


def _load_torch_model(config_path: Path, checkpoint_path: Path, device: str) -> Any:
    import torch

    from ppdet_pytorch import modeling as _modeling  # noqa: F401
    from ppdet_pytorch.core.workspace import create, load_config

    cfg = load_config(str(config_path))
    model = create(cfg.architecture)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    unknown_missing = set(incompatible.missing_keys) - _DERIVED_TORCH_BUFFERS
    if unknown_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "PyTorch checkpoint mismatch: missing={}, unexpected={}".format(
                sorted(unknown_missing), sorted(incompatible.unexpected_keys)
            )
        )
    return model.to(device)


def _torch_inputs(arrays: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {
        "image": torch.from_numpy(arrays["image"]).to(device),
        "im_shape": torch.from_numpy(arrays["im_shape"]).to(device),
        "scale_factor": torch.from_numpy(arrays["scale_factor"]).to(device),
        "gt_bbox": [torch.from_numpy(value).to(device) for value in arrays["gt_bbox"]],
        "gt_class": [
            torch.from_numpy(value).to(device) for value in arrays["gt_class"]
        ],
        "origin_gt_bbox": torch.from_numpy(arrays["origin_gt_bbox"]).to(device),
        "origin_gt_class": torch.from_numpy(arrays["origin_gt_class"]).to(device),
        "pad_origin_gt_mask": torch.from_numpy(arrays["pad_origin_gt_mask"]).to(device),
        "epoch_id": 0,
    }


def run_pytorch(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    config_path = _resolve_path(args.torch_config)
    checkpoint_path = _resolve_path(args.torch_checkpoint)
    model = _load_torch_model(config_path, checkpoint_path, args.device)
    inputs = _torch_inputs(
        _synthetic_arrays(args.batch_size, args.input_size, args.seed), args.device
    )

    if args.workload == "inference":
        model.eval()

        def step() -> Any:
            with torch.inference_mode():
                return model(inputs)

    else:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)

        def step() -> Any:
            optimizer.zero_grad(set_to_none=True)
            losses = model(inputs)
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()
            return total_loss

    def synchronize() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    def reset_device_peak() -> None:
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def device_memory() -> dict[str, Optional[int]]:
        if args.device != "cuda":
            return {
                "device_peak_allocated_bytes": None,
                "device_peak_reserved_bytes": None,
            }
        return {
            "device_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    timing, memory, last_output = measure_iterations(
        step,
        synchronize,
        reset_device_peak,
        device_memory,
        warmup=args.warmup,
        samples=args.samples,
        batch_size=args.batch_size,
    )
    output_summary: dict[str, Any]
    if args.workload == "inference":
        output_summary = {
            key: list(value.shape)
            for key, value in last_output.items()
            if hasattr(value, "shape")
        }
    else:
        output_summary = {"last_total_loss": float(last_output.detach().cpu())}

    return {
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "framework_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
        ),
        "config": _display_path(config_path),
        "checkpoint": _display_path(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "timing": timing,
        "memory": memory,
        "output_summary": output_summary,
    }


def _load_paddle_model(config_path: Path, checkpoint_path: Path) -> Any:
    if str(PADDLE_SOURCE) not in sys.path:
        sys.path.insert(0, str(PADDLE_SOURCE))
    import paddle
    from ppdet import modeling as _modeling  # noqa: F401
    from ppdet.core.workspace import create, load_config

    cfg = load_config(str(config_path))
    model = create(cfg.architecture)
    model.set_state_dict(paddle.load(str(checkpoint_path)))
    return model


def _paddle_inputs(arrays: dict[str, Any]) -> dict[str, Any]:
    import paddle

    return {
        "image": paddle.to_tensor(arrays["image"]),
        "im_shape": paddle.to_tensor(arrays["im_shape"]),
        "scale_factor": paddle.to_tensor(arrays["scale_factor"]),
        "gt_bbox": [paddle.to_tensor(value) for value in arrays["gt_bbox"]],
        "gt_class": [paddle.to_tensor(value) for value in arrays["gt_class"]],
        "origin_gt_bbox": paddle.to_tensor(arrays["origin_gt_bbox"]),
        "origin_gt_class": paddle.to_tensor(arrays["origin_gt_class"]),
        "pad_origin_gt_mask": paddle.to_tensor(arrays["pad_origin_gt_mask"]),
        "epoch_id": 0,
    }


def run_paddle(args: argparse.Namespace) -> dict[str, Any]:
    import paddle

    if args.device == "cuda" and not paddle.is_compiled_with_cuda():
        raise RuntimeError("Paddle CUDA was requested but this build is CPU-only")
    paddle.set_device("gpu" if args.device == "cuda" else "cpu")
    paddle.seed(args.seed)

    config_path = _resolve_path(args.paddle_config)
    checkpoint_path = _resolve_path(args.paddle_checkpoint)
    model = _load_paddle_model(config_path, checkpoint_path)
    inputs = _paddle_inputs(
        _synthetic_arrays(args.batch_size, args.input_size, args.seed)
    )

    if args.workload == "inference":
        model.eval()

        def step() -> Any:
            with paddle.no_grad():
                return model(inputs)

    else:
        model.train()
        optimizer = paddle.optimizer.AdamW(
            learning_rate=0.0,
            parameters=model.parameters(),
            weight_decay=0.0,
        )

        def step() -> Any:
            optimizer.clear_grad()
            losses = model(inputs)
            total_loss = paddle.add_n(list(losses.values()))
            total_loss.backward()
            optimizer.step()
            return total_loss

    def synchronize() -> None:
        if args.device == "cuda":
            paddle.device.synchronize()

    def reset_device_peak() -> None:
        if args.device == "cuda":
            paddle.device.cuda.reset_max_memory_allocated()
            paddle.device.cuda.reset_max_memory_reserved()

    def device_memory() -> dict[str, Optional[int]]:
        if args.device != "cuda":
            return {
                "device_peak_allocated_bytes": None,
                "device_peak_reserved_bytes": None,
            }
        return {
            "device_peak_allocated_bytes": (paddle.device.cuda.max_memory_allocated()),
            "device_peak_reserved_bytes": paddle.device.cuda.max_memory_reserved(),
        }

    timing, memory, last_output = measure_iterations(
        step,
        synchronize,
        reset_device_peak,
        device_memory,
        warmup=args.warmup,
        samples=args.samples,
        batch_size=args.batch_size,
    )
    output_summary: dict[str, Any]
    if args.workload == "inference":
        output_summary = {
            key: list(value.shape)
            for key, value in last_output.items()
            if hasattr(value, "shape")
        }
    else:
        output_summary = {"last_total_loss": float(last_output)}

    cudnn_version = None
    if args.device == "cuda":
        cudnn_version = paddle.version.cudnn()
    return {
        "framework": "paddle",
        "framework_version": paddle.__version__,
        "framework_cuda_version": (
            paddle.version.cuda() if paddle.is_compiled_with_cuda() else None
        ),
        "cudnn_version": cudnn_version,
        "device": args.device,
        "device_name": (
            paddle.device.cuda.get_device_name() if args.device == "cuda" else "cpu"
        ),
        "config": _display_path(config_path),
        "checkpoint": _display_path(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "timing": timing,
        "memory": memory,
        "output_summary": output_summary,
    }


def build_comparison(results: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    if set(results) != {"paddle", "pytorch"}:
        return None
    paddle_result = results["paddle"]
    torch_result = results["pytorch"]
    paddle_throughput = paddle_result["timing"]["throughput_images_per_second"]
    torch_throughput = torch_result["timing"]["throughput_images_per_second"]
    paddle_latency = paddle_result["timing"]["mean_batch_latency_ms"]
    torch_latency = torch_result["timing"]["mean_batch_latency_ms"]
    paddle_rss = paddle_result["memory"]["process_peak_rss_bytes"]
    torch_rss = torch_result["memory"]["process_peak_rss_bytes"]
    return {
        "pytorch_over_paddle_throughput": torch_throughput / paddle_throughput,
        "pytorch_over_paddle_mean_latency": torch_latency / paddle_latency,
        "pytorch_over_paddle_process_peak_rss": torch_rss / paddle_rss,
        "interpretation": (
            "Observed ratios only; performance thresholds are not correctness gates."
        ),
    }


def _run_command(command: Sequence[str]) -> Optional[str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_host_metadata() -> dict[str, Any]:
    cpu_model = None
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", maxsplit=1)[1].strip()
                break
    memory_total = None
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_total = int(line.split()[1]) * 1024
                break
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    gpu_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    commit = _run_command(["git", "rev-parse", "HEAD"])
    dirty_output = _run_command(["git", "status", "--porcelain"])
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "memory_total_bytes": memory_total,
        "nvidia_smi": gpu_query.splitlines() if gpu_query else [],
        "git_commit": commit,
        "git_dirty": bool(dirty_output),
    }


def build_worker_command(
    args: argparse.Namespace, framework: str, result_path: Path
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--framework",
        framework,
        "--workload",
        args.workload,
        "--model",
        args.model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--input-size",
        str(args.input_size),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--threads",
        str(args.threads),
        "--seed",
        str(args.seed),
        "--torch-config",
        args.torch_config,
        "--paddle-config",
        args.paddle_config,
        "--torch-checkpoint",
        args.torch_checkpoint,
        "--paddle-checkpoint",
        args.paddle_checkpoint,
        "--_worker",
        "--_worker-result",
        str(result_path),
    ]


def run_isolated_worker(
    args: argparse.Namespace, framework: str, result_path: Path
) -> dict[str, Any]:
    environment = os.environ.copy()
    thread_count = str(args.threads)
    environment.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
            "FLAGS_paddle_num_threads": thread_count,
            "PYTHONHASHSEED": str(args.seed),
        }
    )
    completed = subprocess.run(
        build_worker_command(args, framework, result_path),
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{framework} benchmark worker exited with {completed.returncode}"
        )
    with result_path.open(encoding="utf-8") as result_file:
        result = json.load(result_file)
    if not isinstance(result, dict):
        raise ValueError(f"{framework} worker result must be a JSON object")
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    training_step = None
    if args.workload == "train-step":
        training_step = (
            "clear gradients + forward + loss aggregation + backward + "
            "AdamW step (lr=0, weight_decay=0)"
        )
    return {
        "model": args.model,
        "scope": "model-only synthetic preprocessed input",
        "workload": args.workload,
        "training_step": training_step,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "input_size": [args.input_size, args.input_size],
        "warmup_iterations": args.warmup,
        "measured_iterations": args.samples,
        "cpu_threads": args.threads,
        "seed": args.seed,
        "synchronization": "before and after every measured iteration",
        "includes_data_loader": False,
        "includes_preprocessing": False,
        "includes_scheduler_ema_amp_ddp": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args._worker:
        result = run_paddle(args) if args.framework == "paddle" else run_pytorch(args)
        _write_json(Path(args._worker_result), result)
        return 0

    frameworks = ["paddle", "pytorch"] if args.framework == "both" else [args.framework]
    host = collect_host_metadata()
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="rtdetrv3-benchmark-") as temp_directory:
        temp_path = Path(temp_directory)
        for framework in frameworks:
            results[framework] = run_isolated_worker(
                args, framework, temp_path / f"{framework}.json"
            )

    payload = {
        "schema_version": 1,
        "host": host,
        "protocol": _protocol(args),
        "results": results,
        "comparison": build_comparison(results),
    }
    if args.output:
        _write_json(_resolve_path(args.output), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

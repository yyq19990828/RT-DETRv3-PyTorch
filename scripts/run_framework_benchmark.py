#!/usr/bin/env python3
"""Benchmark Paddle and PyTorch RT-DETRv3 in isolated worker processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
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
        choices=("inference", "train-step", "e2e-inference"),
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
    parser.add_argument(
        "--dataset-root",
        help="COCO root containing val2017 and annotations for e2e-inference",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--profile-top-k",
        type=int,
        default=10,
        help="Number of model-forward operator rows to retain; 0 disables profiling",
    )
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
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.profile_top_k < 0:
        parser.error("--profile-top-k cannot be negative")
    if args.workload == "e2e-inference" and not args.dataset_root:
        parser.error("--dataset-root is required for e2e-inference")
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


def summarize_e2e_durations(
    input_pipeline_seconds: Sequence[float],
    model_seconds: Sequence[float],
    batch_size: int,
) -> dict[str, Any]:
    """Summarize visible input-pipeline stalls separately from model execution."""
    if len(input_pipeline_seconds) != len(model_seconds):
        raise ValueError("input-pipeline and model duration counts must match")
    end_to_end_seconds = [
        input_seconds + forward_seconds
        for input_seconds, forward_seconds in zip(input_pipeline_seconds, model_seconds)
    ]
    total_seconds = sum(end_to_end_seconds)
    return {
        "end_to_end": summarize_durations(end_to_end_seconds, batch_size),
        "input_pipeline": summarize_durations(input_pipeline_seconds, batch_size),
        "model": summarize_durations(model_seconds, batch_size),
        "input_pipeline_fraction": sum(input_pipeline_seconds) / total_seconds,
    }


def _batch_image_ids(batch: dict[str, Any]) -> list[int]:
    value = batch.get("im_id")
    if value is None:
        raise ValueError("COCO batch does not contain im_id")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return [int(item) for item in np.asarray(value).reshape(-1)]


def measure_e2e_iterations(
    loader: Any,
    prepare_batch: Callable[[dict[str, Any]], dict[str, Any]],
    forward: Callable[[dict[str, Any]], Any],
    synchronize: Callable[[], None],
    reset_device_peak: Callable[[], None],
    device_memory: Callable[[], dict[str, Optional[int]]],
    *,
    warmup: int,
    samples: int,
    batch_size: int,
) -> tuple[
    dict[str, Any],
    dict[str, Optional[int]],
    Any,
    dict[str, Any],
    list[list[int]],
]:
    """Measure batches exactly as consumed by evaluation, excluding profiler tax."""
    iterator = iter(loader)
    last_batch: dict[str, Any]
    last_output: Any = None
    for _ in range(warmup):
        try:
            raw_batch = next(iterator)
        except StopIteration as error:
            raise ValueError("DataLoader ended during benchmark warmup") from error
        last_batch = prepare_batch(raw_batch)
        synchronize()
        last_output = forward(last_batch)
        synchronize()

    baseline_peak_rss = _peak_rss_bytes()
    baseline_rss = _current_rss_bytes()
    reset_device_peak()
    input_durations: list[float] = []
    model_durations: list[float] = []
    measured_image_ids: list[list[int]] = []
    for _ in range(samples):
        input_started_at = time.perf_counter()
        try:
            raw_batch = next(iterator)
        except StopIteration as error:
            raise ValueError("DataLoader ended during measured iterations") from error
        measured_image_ids.append(_batch_image_ids(raw_batch))
        last_batch = prepare_batch(raw_batch)
        synchronize()
        model_started_at = time.perf_counter()
        last_output = forward(last_batch)
        synchronize()
        finished_at = time.perf_counter()
        input_durations.append(model_started_at - input_started_at)
        model_durations.append(finished_at - model_started_at)

    process_peak_rss = _peak_rss_bytes()
    memory = {
        "baseline_rss_bytes": baseline_rss,
        "process_peak_rss_bytes": process_peak_rss,
        "measurement_peak_rss_increase_bytes": max(
            0, process_peak_rss - baseline_peak_rss
        ),
        **device_memory(),
    }
    breakdown = summarize_e2e_durations(
        input_durations,
        model_durations,
        batch_size,
    )
    return (
        breakdown["end_to_end"],
        memory,
        last_output,
        {
            "input_pipeline": breakdown["input_pipeline"],
            "model": breakdown["model"],
            "input_pipeline_fraction": breakdown["input_pipeline_fraction"],
            "definition": (
                "input pipeline is next(DataLoader) plus conversion/transfer until "
                "the batch is synchronized and ready for model forward"
            ),
        },
        measured_image_ids,
    )


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


def _configure_eval_data(
    cfg: Any,
    dataset_root: Path,
    batch_size: int,
    input_size: int,
) -> tuple[Path, Path]:
    dataset_config = cfg.EvalDataset
    reader_config = cfg.EvalReader
    dataset_config["dataset_dir"] = str(dataset_root)
    reader_config["batch_size"] = batch_size

    resize_found = False
    for transform in reader_config["sample_transforms"]:
        if "Resize" in transform:
            transform["Resize"]["target_size"] = [input_size, input_size]
            resize_found = True
            break
    if not resize_found:
        raise ValueError("EvalReader must define a Resize transform")

    annotation_path = dataset_root / dataset_config["anno_path"]
    image_directory = dataset_root / dataset_config["image_dir"]
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
    if not image_directory.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {image_directory}")
    return annotation_path, image_directory


def _dataset_summary(
    annotation_path: Path,
    image_directory: Path,
    dataset_size: int,
    measured_image_ids: list[list[int]],
) -> dict[str, Any]:
    return {
        "name": "COCO 2017 val2017",
        "annotation_file": annotation_path.name,
        "annotation_size_bytes": annotation_path.stat().st_size,
        "annotation_sha256": _sha256(annotation_path),
        "image_directory": image_directory.name,
        "dataset_size": dataset_size,
        "measured_image_ids": measured_image_ids,
    }


def _summarize_paddle_trace(trace_path: Path, top_k: int) -> dict[str, Any]:
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    events = document.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Paddle profiler trace has no traceEvents list")

    category_counts: dict[str, int] = {}
    operator_totals: dict[str, list[float]] = {}
    kernel_totals: dict[str, list[float]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        category = event.get("cat")
        name = event.get("name")
        duration = event.get("dur")
        if not isinstance(category, str) or not isinstance(name, str):
            continue
        if not isinstance(duration, (int, float)):
            continue
        name = re.sub(r"\[[0-9.]+ (?:ns|us|ms|s)\]$", "", name)
        if category == "Operator" and name.endswith(" dygraph"):
            name = name[: -len(" dygraph")]
        category_counts[category] = category_counts.get(category, 0) + 1
        destination: Optional[dict[str, list[float]]] = None
        if category == "Operator":
            destination = operator_totals
        elif category == "Kernel":
            destination = kernel_totals
        if destination is None:
            continue
        count_and_duration = destination.setdefault(name, [0.0, 0.0])
        count_and_duration[0] += 1
        count_and_duration[1] += float(duration)

    def top_rows(values: dict[str, list[float]]) -> list[dict[str, Any]]:
        ordered = sorted(values.items(), key=lambda item: item[1][1], reverse=True)
        return [
            {
                "name": name,
                "count": int(count_and_duration[0]),
                "total_duration_us": count_and_duration[1],
            }
            for name, count_and_duration in ordered[:top_k]
        ]

    return {
        "scope": "one model forward after timing",
        "timing_effect": "excluded from benchmark durations",
        "operator_duration_kind": "inclusive host trace duration",
        "categories": dict(sorted(category_counts.items())),
        "operators": top_rows(operator_totals),
        "device_kernels": top_rows(kernel_totals),
    }


def _load_torch_model(config_path: Path, checkpoint_path: Path, device: str) -> Any:
    import torch

    from detrs import modeling as _modeling  # noqa: F401
    from detrs.core.workspace import create, load_config

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


def _prepare_torch_eval_batch(batch: Any, device: str) -> Any:
    import torch

    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=device == "cuda")
    if isinstance(batch, (np.ndarray, np.generic)):
        return torch.as_tensor(batch).to(device, non_blocking=device == "cuda")
    if isinstance(batch, dict):
        return {
            key: _prepare_torch_eval_batch(value, device)
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(_prepare_torch_eval_batch(value, device) for value in batch)
    if isinstance(batch, list):
        return [_prepare_torch_eval_batch(value, device) for value in batch]
    return batch


def _profile_torch_forward(
    model: Any,
    batch: dict[str, Any],
    device: str,
    top_k: int,
) -> Optional[dict[str, Any]]:
    if top_k == 0:
        return None
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        with torch.inference_mode():
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()

    use_device_time = device == "cuda"
    rows = []
    for event in profile.key_averages():
        if not event.key.startswith("aten::"):
            continue
        sort_duration = (
            event.self_device_time_total
            if use_device_time
            else event.self_cpu_time_total
        )
        if sort_duration <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "count": event.count,
                "self_cpu_time_us": event.self_cpu_time_total,
                "cpu_time_total_us": event.cpu_time_total,
                "self_device_time_us": event.self_device_time_total,
                "device_time_total_us": event.device_time_total,
            }
        )
    sort_key = "self_device_time_us" if use_device_time else "self_cpu_time_us"
    rows.sort(key=lambda row: row[sort_key], reverse=True)
    return {
        "scope": "one model forward after timing",
        "timing_effect": "excluded from benchmark durations",
        "sort_key": sort_key,
        "operators": rows[:top_k],
    }


def _run_pytorch_e2e(
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import BatchSampler, SequentialSampler

    from detrs import data as _data  # noqa: F401
    from detrs.core.workspace import create, load_config

    dataset_root = _resolve_path(args.dataset_root)
    model = _load_torch_model(config_path, checkpoint_path, args.device)
    cfg = load_config(str(config_path))
    annotation_path, image_directory = _configure_eval_data(
        cfg,
        dataset_root,
        args.batch_size,
        args.input_size,
    )
    model.load_meanstd(cfg.TestReader["sample_transforms"])
    model.eval()

    dataset = create(cfg.EvalDataset)
    sampler = BatchSampler(
        SequentialSampler(dataset),
        batch_size=args.batch_size,
        drop_last=False,
    )
    loader = create(cfg.EvalReader)(dataset, args.num_workers, sampler)

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

    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return _prepare_torch_eval_batch(batch, args.device)

    def forward(batch: dict[str, Any]) -> Any:
        with torch.inference_mode():
            return model(batch)

    timing, memory, last_output, pipeline, measured_image_ids = measure_e2e_iterations(
        loader,
        prepare_batch,
        forward,
        synchronize,
        reset_device_peak,
        device_memory,
        warmup=args.warmup,
        samples=args.samples,
        batch_size=args.batch_size,
    )
    profile_batch = prepare_batch(next(iter(loader)))
    synchronize()
    operator_profile = _profile_torch_forward(
        model,
        profile_batch,
        args.device,
        args.profile_top_k,
    )
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
        "pipeline": pipeline,
        "memory": memory,
        "operator_profile": operator_profile,
        "dataset": _dataset_summary(
            annotation_path,
            image_directory,
            len(dataset),
            measured_image_ids,
        ),
        "output_summary": {
            key: list(value.shape)
            for key, value in last_output.items()
            if hasattr(value, "shape")
        },
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
    if args.workload == "e2e-inference":
        return _run_pytorch_e2e(args, config_path, checkpoint_path)
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


def _prepare_paddle_eval_batch(batch: Any) -> Any:
    import paddle

    if isinstance(batch, paddle.Tensor):
        return batch
    if isinstance(batch, np.ndarray):
        return paddle.to_tensor(batch)
    if isinstance(batch, np.generic):
        return paddle.to_tensor(batch.item())
    if isinstance(batch, dict):
        return {key: _prepare_paddle_eval_batch(value) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_prepare_paddle_eval_batch(value) for value in batch)
    if isinstance(batch, list):
        return [_prepare_paddle_eval_batch(value) for value in batch]
    return batch


def _profile_paddle_forward(
    model: Any,
    batch: dict[str, Any],
    device: str,
    top_k: int,
) -> Optional[dict[str, Any]]:
    if top_k == 0:
        return None
    import paddle

    targets = [paddle.profiler.ProfilerTarget.CPU]
    if device == "cuda":
        targets.append(paddle.profiler.ProfilerTarget.GPU)
    with tempfile.TemporaryDirectory(prefix="detrs-paddle-profile-") as directory:
        trace_path = Path(directory) / "trace.json"

        def export_trace(profiler: Any) -> None:
            profiler.export(str(trace_path), "json")

        with paddle.profiler.Profiler(
            targets=targets,
            scheduler=(0, 1),
            on_trace_ready=export_trace,
        ) as profiler:
            with paddle.no_grad():
                model(batch)
            if device == "cuda":
                paddle.device.synchronize()
            profiler.step()
        if not trace_path.is_file():
            raise RuntimeError("Paddle profiler did not produce a trace")
        return _summarize_paddle_trace(trace_path, top_k)


def _run_paddle_e2e(
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if str(PADDLE_SOURCE) not in sys.path:
        sys.path.insert(0, str(PADDLE_SOURCE))
    import paddle
    from ppdet.core.workspace import create, load_config

    dataset_root = _resolve_path(args.dataset_root)
    model = _load_paddle_model(config_path, checkpoint_path)
    cfg = load_config(str(config_path))
    annotation_path, image_directory = _configure_eval_data(
        cfg,
        dataset_root,
        args.batch_size,
        args.input_size,
    )
    model.load_meanstd(cfg.TestReader["sample_transforms"])
    model.eval()

    dataset = create("EvalDataset")()
    sampler = paddle.io.BatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    loader = create("EvalReader")(dataset, args.num_workers, sampler)

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
            "device_peak_allocated_bytes": paddle.device.cuda.max_memory_allocated(),
            "device_peak_reserved_bytes": paddle.device.cuda.max_memory_reserved(),
        }

    def forward(batch: dict[str, Any]) -> Any:
        with paddle.no_grad():
            return model(batch)

    timing, memory, last_output, pipeline, measured_image_ids = measure_e2e_iterations(
        loader,
        _prepare_paddle_eval_batch,
        forward,
        synchronize,
        reset_device_peak,
        device_memory,
        warmup=args.warmup,
        samples=args.samples,
        batch_size=args.batch_size,
    )
    profile_batch = _prepare_paddle_eval_batch(next(iter(loader)))
    synchronize()
    operator_profile = _profile_paddle_forward(
        model,
        profile_batch,
        args.device,
        args.profile_top_k,
    )
    return {
        "framework": "paddle",
        "framework_version": paddle.__version__,
        "framework_cuda_version": (
            paddle.version.cuda() if paddle.is_compiled_with_cuda() else None
        ),
        "cudnn_version": (paddle.version.cudnn() if args.device == "cuda" else None),
        "device": args.device,
        "device_name": (
            paddle.device.cuda.get_device_name() if args.device == "cuda" else "cpu"
        ),
        "config": _display_path(config_path),
        "checkpoint": _display_path(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "timing": timing,
        "pipeline": pipeline,
        "memory": memory,
        "operator_profile": operator_profile,
        "dataset": _dataset_summary(
            annotation_path,
            image_directory,
            len(dataset),
            measured_image_ids,
        ),
        "output_summary": {
            key: list(value.shape)
            for key, value in last_output.items()
            if hasattr(value, "shape")
        },
    }


def run_paddle(args: argparse.Namespace) -> dict[str, Any]:
    import paddle

    if args.device == "cuda" and not paddle.is_compiled_with_cuda():
        raise RuntimeError("Paddle CUDA was requested but this build is CPU-only")
    paddle.set_device("gpu" if args.device == "cuda" else "cpu")
    paddle.seed(args.seed)

    config_path = _resolve_path(args.paddle_config)
    checkpoint_path = _resolve_path(args.paddle_checkpoint)
    if args.workload == "e2e-inference":
        return _run_paddle_e2e(args, config_path, checkpoint_path)
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
    if "dataset" in paddle_result or "dataset" in torch_result:
        paddle_dataset = paddle_result.get("dataset")
        torch_dataset = torch_result.get("dataset")
        if not isinstance(paddle_dataset, dict) or not isinstance(torch_dataset, dict):
            raise ValueError("both frameworks must report dataset identity")
        identity_keys = (
            "annotation_sha256",
            "dataset_size",
            "measured_image_ids",
        )
        if any(
            paddle_dataset.get(key) != torch_dataset.get(key) for key in identity_keys
        ):
            raise ValueError(
                "Paddle and PyTorch did not benchmark the same COCO samples"
            )
    paddle_throughput = paddle_result["timing"]["throughput_images_per_second"]
    torch_throughput = torch_result["timing"]["throughput_images_per_second"]
    paddle_latency = paddle_result["timing"]["mean_batch_latency_ms"]
    torch_latency = torch_result["timing"]["mean_batch_latency_ms"]
    paddle_rss = paddle_result["memory"]["process_peak_rss_bytes"]
    torch_rss = torch_result["memory"]["process_peak_rss_bytes"]
    comparison = {
        "pytorch_over_paddle_throughput": torch_throughput / paddle_throughput,
        "pytorch_over_paddle_mean_latency": torch_latency / paddle_latency,
        "pytorch_over_paddle_process_peak_rss": torch_rss / paddle_rss,
        "interpretation": (
            "Observed ratios only; performance thresholds are not correctness gates."
        ),
    }
    if "pipeline" in paddle_result and "pipeline" in torch_result:
        paddle_pipeline = paddle_result["pipeline"]
        torch_pipeline = torch_result["pipeline"]
        comparison.update(
            {
                "pytorch_over_paddle_input_pipeline_latency": (
                    torch_pipeline["input_pipeline"]["mean_batch_latency_ms"]
                    / paddle_pipeline["input_pipeline"]["mean_batch_latency_ms"]
                ),
                "pytorch_over_paddle_model_latency": (
                    torch_pipeline["model"]["mean_batch_latency_ms"]
                    / paddle_pipeline["model"]["mean_batch_latency_ms"]
                ),
                "paddle_input_pipeline_fraction": paddle_pipeline[
                    "input_pipeline_fraction"
                ],
                "pytorch_input_pipeline_fraction": torch_pipeline[
                    "input_pipeline_fraction"
                ],
            }
        )
    return comparison


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
    command = [
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
        "--num-workers",
        str(args.num_workers),
        "--profile-top-k",
        str(args.profile_top_k),
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
    if args.dataset_root:
        command.extend(["--dataset-root", args.dataset_root])
    return command


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
    e2e_inference = args.workload == "e2e-inference"
    return {
        "model": args.model,
        "scope": (
            "COCO 2017 val2017 end-to-end inference"
            if e2e_inference
            else "model-only synthetic preprocessed input"
        ),
        "workload": args.workload,
        "training_step": training_step,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "input_size": [args.input_size, args.input_size],
        "warmup_iterations": args.warmup,
        "measured_iterations": args.samples,
        "cpu_threads": args.threads,
        "data_loader_workers": args.num_workers if e2e_inference else None,
        "operator_profile_top_k": args.profile_top_k if e2e_inference else None,
        "seed": args.seed,
        "synchronization": "before and after every measured iteration",
        "includes_data_loader": e2e_inference,
        "includes_preprocessing": e2e_inference,
        "includes_host_to_device_transfer": e2e_inference,
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
    with tempfile.TemporaryDirectory(prefix="detrs-benchmark-") as temp_directory:
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

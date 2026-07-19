import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts/run_framework_benchmark.py"


def load_script(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "run_framework_benchmark", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_summarize_durations_reports_latency_percentiles_and_throughput(monkeypatch):
    script = load_script(monkeypatch)

    summary = script.summarize_durations([0.1, 0.2, 0.3], batch_size=2)

    assert summary["mean_batch_latency_ms"] == pytest.approx(200.0)
    assert summary["median_batch_latency_ms"] == pytest.approx(200.0)
    assert summary["p90_batch_latency_ms"] == pytest.approx(280.0)
    assert summary["throughput_images_per_second"] == pytest.approx(10.0)


def test_summarize_e2e_durations_reports_visible_pipeline_fraction(monkeypatch):
    script = load_script(monkeypatch)

    summary = script.summarize_e2e_durations([0.02, 0.04], [0.08, 0.06], 1)

    assert summary["end_to_end"]["mean_batch_latency_ms"] == pytest.approx(100.0)
    assert summary["input_pipeline"]["mean_batch_latency_ms"] == pytest.approx(30.0)
    assert summary["model"]["mean_batch_latency_ms"] == pytest.approx(70.0)
    assert summary["input_pipeline_fraction"] == pytest.approx(0.3)


def test_summarize_e2e_durations_rejects_mismatched_samples(monkeypatch):
    script = load_script(monkeypatch)

    with pytest.raises(ValueError, match="counts must match"):
        script.summarize_e2e_durations([0.02], [0.08, 0.09], 1)


def test_build_comparison_keeps_performance_ratios_observational(monkeypatch):
    script = load_script(monkeypatch)
    results = {
        "paddle": {
            "timing": {
                "throughput_images_per_second": 5.0,
                "mean_batch_latency_ms": 200.0,
            },
            "memory": {"process_peak_rss_bytes": 1000},
        },
        "pytorch": {
            "timing": {
                "throughput_images_per_second": 8.0,
                "mean_batch_latency_ms": 125.0,
            },
            "memory": {"process_peak_rss_bytes": 1100},
        },
    }

    comparison = script.build_comparison(results)

    assert comparison["pytorch_over_paddle_throughput"] == pytest.approx(1.6)
    assert comparison["pytorch_over_paddle_mean_latency"] == pytest.approx(0.625)
    assert comparison["pytorch_over_paddle_process_peak_rss"] == pytest.approx(1.1)
    assert "not correctness gates" in comparison["interpretation"]


def test_build_comparison_requires_identical_e2e_samples(monkeypatch):
    script = load_script(monkeypatch)

    def result(framework, image_ids):
        return {
            "framework": framework,
            "timing": {
                "throughput_images_per_second": 10.0,
                "mean_batch_latency_ms": 100.0,
            },
            "pipeline": {
                "input_pipeline": {"mean_batch_latency_ms": 20.0},
                "model": {"mean_batch_latency_ms": 80.0},
                "input_pipeline_fraction": 0.2,
            },
            "memory": {"process_peak_rss_bytes": 1000},
            "dataset": {
                "annotation_sha256": "a" * 64,
                "dataset_size": 5000,
                "measured_image_ids": image_ids,
            },
        }

    results = {
        "paddle": result("paddle", [[139], [285]]),
        "pytorch": result("pytorch", [[139], [285]]),
    }
    comparison = script.build_comparison(results)
    assert comparison["pytorch_over_paddle_input_pipeline_latency"] == 1.0
    assert comparison["pytorch_over_paddle_model_latency"] == 1.0

    results["pytorch"]["dataset"]["measured_image_ids"] = [[139], [632]]
    with pytest.raises(ValueError, match="same COCO samples"):
        script.build_comparison(results)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--batch-size", "0"],
        ["--input-size", "16"],
        ["--warmup", "-1"],
        ["--samples", "0"],
        ["--threads", "0"],
        ["--num-workers", "-1"],
        ["--profile-top-k", "-1"],
    ],
)
def test_parse_args_rejects_invalid_protocol_values(monkeypatch, arguments):
    script = load_script(monkeypatch)

    with pytest.raises(SystemExit):
        script.parse_args(arguments)


def test_main_runs_frameworks_in_isolated_workers_and_writes_report(
    monkeypatch, tmp_path
):
    script = load_script(monkeypatch)
    observed = []
    output_path = tmp_path / "benchmark.json"

    monkeypatch.setattr(script, "collect_host_metadata", lambda: {"git_dirty": False})

    def fake_worker(args, framework, result_path):
        observed.append((framework, result_path.parent))
        return {
            "framework": framework,
            "timing": {
                "throughput_images_per_second": 2.0,
                "mean_batch_latency_ms": 500.0,
            },
            "memory": {"process_peak_rss_bytes": 1000},
        }

    monkeypatch.setattr(script, "run_isolated_worker", fake_worker)

    assert script.main(["--output", str(output_path), "--samples", "2"]) == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item[0] for item in observed] == ["paddle", "pytorch"]
    assert observed[0][1] == observed[1][1]
    assert report["protocol"]["measured_iterations"] == 2
    assert set(report["results"]) == {"paddle", "pytorch"}
    assert report["comparison"]["pytorch_over_paddle_throughput"] == 1.0


def test_e2e_workload_requires_dataset_root(monkeypatch):
    script = load_script(monkeypatch)

    with pytest.raises(SystemExit):
        script.parse_args(["--workload", "e2e-inference"])


def test_worker_command_carries_exact_protocol(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    args = script.parse_args(
        [
            "--framework",
            "pytorch",
            "--workload",
            "train-step",
            "--batch-size",
            "2",
            "--warmup",
            "1",
            "--samples",
            "3",
            "--threads",
            "4",
        ]
    )

    command = script.build_worker_command(args, "pytorch", tmp_path / "result.json")

    assert command[command.index("--workload") + 1] == "train-step"
    assert command[command.index("--batch-size") + 1] == "2"
    assert command[command.index("--warmup") + 1] == "1"
    assert command[command.index("--samples") + 1] == "3"
    assert command[command.index("--threads") + 1] == "4"
    assert "--_worker" in command


def test_worker_command_carries_e2e_data_and_profile_protocol(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    args = script.parse_args(
        [
            "--framework",
            "pytorch",
            "--workload",
            "e2e-inference",
            "--dataset-root",
            "/datasets/coco2017",
            "--num-workers",
            "3",
            "--profile-top-k",
            "7",
        ]
    )

    command = script.build_worker_command(args, "pytorch", tmp_path / "result.json")

    assert command[command.index("--dataset-root") + 1] == "/datasets/coco2017"
    assert command[command.index("--num-workers") + 1] == "3"
    assert command[command.index("--profile-top-k") + 1] == "7"
    protocol = script._protocol(args)
    assert protocol["includes_data_loader"] is True
    assert protocol["includes_preprocessing"] is True


def test_paddle_trace_summary_keeps_top_operators_and_kernels(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "Operator",
                        "name": "conv2d dygraph[5.000 us]",
                        "dur": 5,
                    },
                    {"ph": "X", "cat": "Operator", "name": "conv2d", "dur": 7},
                    {"ph": "X", "cat": "Operator", "name": "reshape", "dur": 3},
                    {"ph": "X", "cat": "Kernel", "name": "kernel_a", "dur": 9},
                    {"ph": "M", "cat": "Operator", "name": "ignored", "dur": 99},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = script._summarize_paddle_trace(trace_path, top_k=1)

    assert summary["operators"] == [
        {"name": "conv2d", "count": 2, "total_duration_us": 12.0}
    ]
    assert summary["device_kernels"] == [
        {"name": "kernel_a", "count": 1, "total_duration_us": 9.0}
    ]

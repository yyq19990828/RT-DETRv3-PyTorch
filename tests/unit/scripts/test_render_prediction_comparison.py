import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts/render_prediction_comparison.py"


def load_script(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "render_prediction_comparison", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_render_and_summarize_predictions(tmp_path, monkeypatch):
    script = load_script(monkeypatch)
    image_path = tmp_path / "000000000139.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(
        json.dumps({"categories": [{"id": 1, "name": "person"}]}),
        encoding="utf-8",
    )
    paddle_results = tmp_path / "paddle.json"
    pytorch_results = tmp_path / "pytorch.json"
    paddle_results.write_text(
        json.dumps(
            [
                {"category_id": 1, "bbox": [1, 2, 10, 12], "score": 0.9},
                {"category_id": 1, "bbox": [3, 4, 5, 6], "score": 0.2},
            ]
        ),
        encoding="utf-8",
    )
    pytorch_results.write_text(
        json.dumps([{"category_id": 1, "bbox": [1.1, 2, 10, 12], "score": 0.90001}]),
        encoding="utf-8",
    )
    paddle_checkpoint = tmp_path / "model.pdparams"
    pytorch_checkpoint = tmp_path / "model.pth"
    paddle_checkpoint.write_bytes(b"paddle")
    pytorch_checkpoint.write_bytes(b"pytorch")
    output_image = tmp_path / "comparison.png"
    output_json = tmp_path / "comparison.json"

    assert (
        script.main(
            [
                "--image",
                str(image_path),
                "--annotations",
                str(annotations_path),
                "--paddle-results",
                str(paddle_results),
                "--pytorch-results",
                str(pytorch_results),
                "--paddle-checkpoint",
                str(paddle_checkpoint),
                "--pytorch-checkpoint",
                str(pytorch_checkpoint),
                "--output-image",
                str(output_image),
                "--output-json",
                str(output_json),
                "--model",
                "RT-DETRv3-Test",
            ]
        )
        == 0
    )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["protocol"]["model"] == "RT-DETRv3-Test"
    assert summary["comparison"]["matched_count"] == 1
    assert summary["comparison"]["unmatched_paddle_indices"] == []
    assert summary["comparison"]["unmatched_pytorch_indices"] == []
    assert summary["comparison"]["max_box_linf_px"] == pytest.approx(0.1)
    assert summary["comparison_image"]["size_bytes"] > 0
    assert len(summary["comparison_image"]["sha256"]) == 64
    rendered = cv2.imread(str(output_image))
    assert rendered.shape == (106, 60, 3)

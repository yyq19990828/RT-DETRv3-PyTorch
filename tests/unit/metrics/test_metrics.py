import json

import numpy as np
import pytest
import torch

from detrs.metrics import map_utils
from detrs.metrics.map_utils import DetectionMAP, compute_ap, prune_zero_padding
from detrs.metrics.metrics import COCOMetric


def test_compute_ap_integrates_precision_envelope():
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 1.0])

    assert compute_ap(recall, precision) == pytest.approx(1.0)


def test_prune_zero_padding_keeps_aligned_ground_truth_fields():
    boxes = np.array([[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 7, 8]])
    labels = np.array([4, 5, 6])
    difficult = np.array([0, 1, 0])

    valid_boxes, valid_labels, valid_difficult = prune_zero_padding(
        boxes, labels, difficult
    )

    np.testing.assert_array_equal(valid_boxes, boxes[:1])
    np.testing.assert_array_equal(valid_labels, labels[:1])
    np.testing.assert_array_equal(valid_difficult, difficult[:1])


def test_detection_map_accumulates_and_formats_classwise_result(monkeypatch):
    plotted = []
    monkeypatch.setattr(
        map_utils,
        "draw_pr_curve",
        lambda precision, recall, **kwargs: plotted.append((precision, recall, kwargs)),
    )
    metric = DetectionMAP(
        class_num=1,
        map_type="integral",
        catid2name={0: "object"},
        classwise=True,
    )
    box = np.array([[0.0, 0.0, 10.0, 10.0]])

    metric.update(
        bbox=box,
        score=np.array([0.9]),
        label=np.array([0]),
        gt_box=box.copy(),
        gt_label=np.array([0]),
    )
    metric.accumulate()

    assert metric.get_map() == pytest.approx(1.0)
    assert metric.class_score_poss == [[[pytest.approx(0.9), 1.0]]]
    assert plotted[0][2]["file_name"] == "object_precision_recall_curve.jpg"


def test_coco_metric_saves_bbox_predictions_without_evaluation(tmp_path):
    metric = COCOMetric(
        "unused.json",
        clsid2catid={0: 7},
        output_eval=str(tmp_path),
        save_prediction_only=True,
    )
    inputs = {"im_id": torch.tensor([[42]])}
    outputs = {
        "bbox": torch.tensor([[0.0, 0.9, 1.0, 2.0, 5.0, 8.0]]),
        "bbox_num": torch.tensor([1]),
    }

    metric.update(inputs, outputs)
    metric.accumulate()

    assert metric.results["bbox"] == [
        {
            "image_id": 42,
            "category_id": 7,
            "bbox": [1.0, 2.0, 4.0, 6.0],
            "score": pytest.approx(0.9),
        }
    ]
    saved = json.loads((tmp_path / "bbox.json").read_text(encoding="utf-8"))
    assert saved == metric.results["bbox"]
    assert metric.get_results() == {}

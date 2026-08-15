import numpy as np
import pytest
from PIL import Image

import detrs.data.source  # noqa: F401
from detrs.core.workspace import create
from detrs.data.source.category import get_categories
from detrs.metrics import YOLOMetric

_DATA_FIELDS = ["image", "gt_bbox", "gt_class", "is_crowd"]


@pytest.fixture
def parsed_yolo_dataset(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (80, 48), color=(10, 20, 30)).save(images / "one.jpg")
    Image.new("RGB", (64, 64), color=(40, 50, 60)).save(images / "two.jpg")
    (labels / "one.txt").write_text("0 0.5 0.25 0.25 0.5\n1 0.25 0.75 0.5 0.5\n")
    (labels / "two.txt").write_text("0 0.5 0.5 0.25 0.25\n")

    dataset = create(
        {
            "name": "YOLODataSet",
            "dataset_dir": str(tmp_path),
            "image_dir": "images",
            "label_dir": "labels",
            "label_list": "names.txt",
            "data_fields": _DATA_FIELDS,
        }
    )
    (tmp_path / "names.txt").write_text("person\ncar\n")
    dataset.parse_dataset()
    return dataset


def _perfect_outputs(dataset):
    """Model outputs [num_id, score, xmin, ymin, xmax, ymax] mirroring the GT."""
    rows = []
    bbox_num = []
    for rec in dataset.roidbs:
        for bbox, cls in zip(rec["gt_bbox"], rec["gt_class"]):
            rows.append([int(cls[0]), 0.9, bbox[0], bbox[1], bbox[2], bbox[3]])
        bbox_num.append(len(rec["gt_bbox"]))
    return {
        "bbox": np.array(rows, dtype=np.float32),
        "bbox_num": bbox_num,
    }


def test_yolo_metric_perfect_predictions_score_full_ap(tmp_path, parsed_yolo_dataset):
    dataset = parsed_yolo_dataset
    metric = YOLOMetric(dataset, output_eval=str(tmp_path / "eval"))

    im_ids = np.stack([rec["im_id"] for rec in dataset.roidbs])
    metric.update({"im_id": im_ids}, _perfect_outputs(dataset))
    metric.accumulate()

    stats = metric.get_results()["bbox"]
    assert len(stats) == 12
    assert stats[0] == pytest.approx(1.0, abs=1e-6)  # AP50-95
    assert stats[1] == pytest.approx(1.0, abs=1e-6)  # AP50


def test_yolo_metric_without_predictions_does_not_crash(tmp_path, parsed_yolo_dataset):
    metric = YOLOMetric(parsed_yolo_dataset, output_eval=str(tmp_path / "eval"))

    metric.accumulate()

    assert metric.get_results() == {}


def test_yolo_metric_requires_gt_fields(tmp_path):
    class _BareDataset:
        roidbs = [{"im_file": "x.jpg", "im_id": np.array([0])}]
        cname2cid = {"class_0": 0}

    metric = YOLOMetric(_BareDataset(), output_eval=str(tmp_path / "eval"))
    metric.results["bbox"] = [
        {"image_id": 0, "category_id": 0, "bbox": [1, 1, 2, 2], "score": 0.9}
    ]

    with pytest.raises(ValueError, match="gt_bbox"):
        metric.accumulate()


def test_yolo_categories_from_label_list(tmp_path):
    names = tmp_path / "names.txt"
    names.write_text("person\ncar\n")

    clsid2catid, catid2name = get_categories("YOLO", anno_file=str(names))
    assert clsid2catid == {0: 0, 1: 1}
    assert catid2name == {0: "person", 1: "car"}


def test_yolo_categories_require_label_list():
    with pytest.raises(ValueError, match="class-name list"):
        get_categories("YOLO", anno_file=None)

import numpy as np
import pytest
from PIL import Image

import detrs.data.source  # noqa: F401
from detrs.core.workspace import create

_DATA_FIELDS = ["image", "gt_bbox", "gt_class", "is_crowd"]


def _build_yolo_tree(tmp_path, folder="data1"):
    """Create images/ and labels/ with one 80x48 image and two normalized boxes."""
    images = tmp_path / folder / "images"
    labels = tmp_path / folder / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (80, 48), color=(10, 20, 30)).save(images / "one.jpg")
    (labels / "one.txt").write_text("0 0.5 0.25 0.25 0.5\n1 0.25 0.75 0.5 0.5\n")
    return images, labels


def _make_config(tmp_path, **overrides):
    config = {
        "name": "YOLODataSet",
        "dataset_dir": str(tmp_path),
        "image_dir": "data1/images",
        "label_dir": "data1/labels",
        "data_fields": _DATA_FIELDS,
    }
    config.update(overrides)
    return config


def test_yolo_dataset_converts_normalized_boxes(tmp_path):
    _build_yolo_tree(tmp_path)
    dataset = create(_make_config(tmp_path))
    dataset.parse_dataset()

    assert len(dataset) == 1
    record = dataset.roidbs[0]
    # cx,cy,w,h normalized on an 80x48 image -> absolute pixel xyxy
    np.testing.assert_allclose(
        record["gt_bbox"], [[30.0, 0.0, 50.0, 24.0], [0.0, 24.0, 40.0, 48.0]]
    )
    np.testing.assert_array_equal(record["gt_class"], [[0], [1]])
    np.testing.assert_array_equal(record["is_crowd"], [[0], [0]])
    assert record["h"] == 48.0
    assert record["w"] == 80.0
    assert dataset.cname2cid == {"class_0": 0, "class_1": 1}


def test_yolo_dataset_merges_multiple_folders(tmp_path):
    _build_yolo_tree(tmp_path, "data1")
    _build_yolo_tree(tmp_path, "data2")
    dataset = create(
        _make_config(
            tmp_path,
            image_dir=["data1/images", "data2/images"],
            label_dir=["data1/labels", "data2/labels"],
        )
    )
    dataset.parse_dataset()

    assert len(dataset) == 2
    folders = [
        "data1" if "data1" in rec["im_file"] else "data2" for rec in dataset.roidbs
    ]
    assert folders == ["data1", "data2"]
    im_ids = [int(rec["im_id"][0]) for rec in dataset.roidbs]
    assert im_ids == [0, 1]


def test_yolo_dataset_mismatched_dir_lengths_raise(tmp_path):
    _build_yolo_tree(tmp_path, "data1")
    _build_yolo_tree(tmp_path, "data2")
    dataset = create(
        _make_config(
            tmp_path,
            image_dir=["data1/images", "data2/images"],
            label_dir=["data1/labels"],
        )
    )
    with pytest.raises(ValueError, match="same length"):
        dataset.parse_dataset()


def test_yolo_dataset_filters_invalid_boxes(tmp_path):
    images, labels = _build_yolo_tree(tmp_path)
    (labels / "one.txt").write_text("0 0.5 0.25 0.25 0.5\n0 0.5 0.5 0.0 0.5\n")
    dataset = create(_make_config(tmp_path))
    dataset.parse_dataset()

    assert len(dataset) == 1
    assert dataset.roidbs[0]["gt_bbox"].shape == (1, 4)


def test_yolo_dataset_missing_label_skipped_by_default(tmp_path):
    images, _ = _build_yolo_tree(tmp_path)
    Image.new("RGB", (64, 64), color=(40, 50, 60)).save(images / "two.jpg")
    dataset = create(_make_config(tmp_path))
    dataset.parse_dataset()

    assert len(dataset) == 1
    assert dataset.roidbs[0]["im_file"].endswith("one.jpg")


def test_yolo_dataset_missing_label_empty_when_allowed(tmp_path):
    images, _ = _build_yolo_tree(tmp_path)
    Image.new("RGB", (64, 64), color=(40, 50, 60)).save(images / "two.jpg")
    dataset = create(_make_config(tmp_path, allow_empty=True))
    dataset.parse_dataset()

    assert len(dataset) == 2
    empty_record = next(
        rec for rec in dataset.roidbs if rec["im_file"].endswith("two.jpg")
    )
    assert empty_record["gt_bbox"].shape == (0, 4)
    assert empty_record["gt_class"].shape == (0, 1)
    assert empty_record["h"] == 64.0
    assert empty_record["w"] == 64.0


def test_yolo_dataset_label_list_names_and_bounds(tmp_path):
    images, labels = _build_yolo_tree(tmp_path)
    (tmp_path / "names.txt").write_text("person\ncar\n")

    dataset = create(_make_config(tmp_path, label_list="names.txt"))
    dataset.parse_dataset()
    assert dataset.cname2cid == {"person": 0, "car": 1}

    (labels / "one.txt").write_text("2 0.5 0.25 0.25 0.5\n")
    dataset = create(_make_config(tmp_path, label_list="names.txt"))
    with pytest.raises(ValueError, match="exceeds"):
        dataset.parse_dataset()


def test_yolo_dataset_sample_num_truncates(tmp_path):
    images, labels = _build_yolo_tree(tmp_path)
    Image.new("RGB", (64, 64), color=(40, 50, 60)).save(images / "two.jpg")
    (labels / "two.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    dataset = create(_make_config(tmp_path, sample_num=1))
    dataset.parse_dataset()

    assert len(dataset) == 1

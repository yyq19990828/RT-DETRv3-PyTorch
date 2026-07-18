import json
from pathlib import Path

import numpy as np

from ppdet_pytorch.core.workspace import create
import ppdet_pytorch.data.reader  # noqa: F401
import ppdet_pytorch.data.source  # noqa: F401


def test_coco_dataset_parses_xyxy_boxes_and_contiguous_classes(
    minimal_coco_config,
):
    dataset = create(minimal_coco_config)
    dataset.parse_dataset()

    assert len(dataset) == 2
    np.testing.assert_array_equal(dataset.roidbs[0]["gt_bbox"], [[8, 6, 32, 24]])
    np.testing.assert_array_equal(dataset.roidbs[0]["gt_class"], [[0]])
    np.testing.assert_array_equal(
        dataset.roidbs[1]["gt_class"],
        [[1], [0]],
    )


def test_train_reader_builds_deterministic_detection_batch(
    minimal_coco_config,
    deterministic_train_reader_config,
):
    dataset = create(minimal_coco_config)
    reader = create(deterministic_train_reader_config)

    batch = next(iter(reader(dataset, worker_num=0)))

    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["image"].dtype == np.float32
    assert batch["im_shape"].shape == (2, 2)
    assert batch["scale_factor"].shape == (2, 2)
    assert [boxes.shape for boxes in batch["gt_bbox"]] == [(1, 4), (2, 4)]
    assert [classes.shape for classes in batch["gt_class"]] == [(1, 1), (2, 1)]
    assert batch["origin_gt_bbox"].shape == (2, 2, 4)
    assert batch["origin_gt_class"].shape == (2, 2, 1)
    assert batch["pad_origin_gt_mask"].shape == (2, 2, 1)
    np.testing.assert_array_equal(batch["pad_origin_gt_mask"].sum(axis=1), [[1], [2]])

    for boxes in batch["gt_bbox"]:
        assert np.isfinite(boxes).all()
        assert (boxes >= 0).all()
        assert (boxes <= 1).all()


def test_coco_dataset_filters_invalid_bbox(minimal_coco_config):
    annotation_path = (
        Path(minimal_coco_config["dataset_dir"]) / minimal_coco_config["anno_path"]
    )
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations["annotations"].append(
        {
            "id": 4,
            "image_id": 1,
            "category_id": 3,
            "bbox": [10, 10, 0, 5],
            "area": 0,
            "iscrowd": 0,
        }
    )
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

    dataset = create(minimal_coco_config)
    dataset.parse_dataset()

    assert [len(record["gt_bbox"]) for record in dataset.roidbs] == [1, 2]

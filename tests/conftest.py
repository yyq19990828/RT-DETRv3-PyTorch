import json
from copy import deepcopy

import pytest
from PIL import Image

from ppdet_pytorch.core.config.schema import SchemaDict
from ppdet_pytorch.core.workspace import global_config


def _clone_workspace_entry(value):
    if not isinstance(value, SchemaDict):
        return deepcopy(value)

    cloned = value.copy()
    cloned.schema = value.schema.copy()
    for key, item in value.items():
        cloned[key] = deepcopy(item)
    return cloned


@pytest.fixture
def isolated_workspace():
    snapshot = {
        key: _clone_workspace_entry(value) for key, value in global_config.items()
    }
    yield global_config
    global_config.clear()
    global_config.update(snapshot)


@pytest.fixture
def minimal_coco_config(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    Image.new("RGB", (80, 48), color=(10, 20, 30)).save(image_dir / "one.jpg")
    Image.new("RGB", (64, 64), color=(40, 50, 60)).save(image_dir / "two.jpg")

    annotations = {
        "images": [
            {"id": 1, "file_name": "one.jpg", "width": 80, "height": 48},
            {"id": 2, "file_name": "two.jpg", "width": 64, "height": 64},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "bbox": [8, 6, 24, 18],
                "area": 432,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 7,
                "bbox": [4, 8, 20, 24],
                "area": 480,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 3,
                "bbox": [32, 16, 16, 32],
                "area": 512,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 3, "name": "first"},
            {"id": 7, "name": "second"},
        ],
    }
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

    return {
        "name": "COCODataSet",
        "dataset_dir": str(tmp_path),
        "image_dir": "images",
        "anno_path": "instances.json",
        "data_fields": ["image", "gt_bbox", "gt_class", "is_crowd"],
    }


@pytest.fixture
def deterministic_train_reader_config():
    return {
        "name": "TrainReader",
        "sample_transforms": [{"Decode": {}}],
        "batch_transforms": [
            {
                "BatchRandomResize": {
                    "target_size": [64, 64],
                    "random_size": False,
                    "random_interp": False,
                    "keep_ratio": False,
                }
            },
            {
                "NormalizeImage": {
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                    "norm_type": "none",
                }
            },
            {"NormalizeBox": {"retain_origin_box": True}},
            {"BboxXYXY2XYWH": {}},
            {"Permute": {}},
            {"PadGT": {"only_origin_box": True}},
        ],
        "batch_size": 2,
        "shuffle": False,
        "drop_last": False,
        "collate_batch": False,
    }

import json

import pytest
from PIL import Image

import detrs.data.source  # noqa: F401
from detrs.core.workspace import create

_DATA_FIELDS = ["image", "gt_bbox", "gt_class", "is_crowd"]


def _build_two_folder_coco(tmp_path, subpath_filenames):
    """
    Build datasetA/ and datasetB/ folders with identical category tables and
    colliding image ids (both start at 1) to exercise merged loading.

    subpath_filenames=True writes file_name as "<folder>/images/<name>.jpg"
    (pairs with an empty global image_dir); False writes bare file names
    (pairs with per-entry image_dir overrides).
    """
    for prefix, sizes in (
        ("datasetA", [(80, 48), (64, 64)]),
        ("datasetB", [(96, 32), (48, 72)]),
    ):
        images_dir = tmp_path / prefix / "images"
        images_dir.mkdir(parents=True)
        anno_dir = tmp_path / prefix / "annotations"
        anno_dir.mkdir()

        images, annotations = [], []
        ann_id = 1 if prefix == "datasetA" else 11
        for img_id, size in enumerate(sizes, start=1):
            fname = f"{prefix[-1].lower()}{img_id}.jpg"
            Image.new("RGB", size, color=(10, 20, 30)).save(images_dir / fname)
            file_name = f"{prefix}/images/{fname}" if subpath_filenames else fname
            images.append(
                {
                    "id": img_id,
                    "file_name": file_name,
                    "width": size[0],
                    "height": size[1],
                }
            )
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 3,
                    "bbox": [4, 6, 20, 12],
                    "area": 240,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        (anno_dir / "train.json").write_text(
            json.dumps(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": [
                        {"id": 3, "name": "first"},
                        {"id": 7, "name": "second"},
                    ],
                }
            ),
            encoding="utf-8",
        )


def test_string_list_merges_two_datasets(tmp_path):
    _build_two_folder_coco(tmp_path, subpath_filenames=True)
    dataset = create(
        {
            "name": "COCODataSet",
            "dataset_dir": str(tmp_path),
            "image_dir": "",
            "anno_path": [
                "datasetA/annotations/train.json",
                "datasetB/annotations/train.json",
            ],
            "data_fields": _DATA_FIELDS,
        }
    )
    dataset.parse_dataset()

    assert len(dataset) == 4
    im_ids = [int(rec["im_id"][0]) for rec in dataset.roidbs]
    # datasetB ids are shifted by max(datasetA ids) + 1 to avoid collisions
    assert im_ids == [1, 2, 4, 5]
    assert "datasetA" in dataset.roidbs[0]["im_file"]
    assert "datasetB" in dataset.roidbs[2]["im_file"]
    assert dataset.catid2clsid == {3: 0, 7: 1}


def test_dict_entry_overrides_image_dir(tmp_path):
    _build_two_folder_coco(tmp_path, subpath_filenames=False)
    dataset = create(
        {
            "name": "COCODataSet",
            "dataset_dir": str(tmp_path),
            "image_dir": "datasetA/images",
            "anno_path": [
                "datasetA/annotations/train.json",
                {
                    "anno_path": "datasetB/annotations/train.json",
                    "image_dir": "datasetB/images",
                },
            ],
            "data_fields": _DATA_FIELDS,
        }
    )
    dataset.parse_dataset()

    assert len(dataset) == 4
    assert "datasetA" in dataset.roidbs[0]["im_file"]
    assert "datasetB" in dataset.roidbs[2]["im_file"]


def test_category_mismatch_raises(tmp_path):
    _build_two_folder_coco(tmp_path, subpath_filenames=True)
    b_json = tmp_path / "datasetB" / "annotations" / "train.json"
    annotations = json.loads(b_json.read_text(encoding="utf-8"))
    annotations["categories"] = [
        {"id": 3, "name": "first"},
        {"id": 9, "name": "other"},
    ]
    b_json.write_text(json.dumps(annotations), encoding="utf-8")

    dataset = create(
        {
            "name": "COCODataSet",
            "dataset_dir": str(tmp_path),
            "image_dir": "",
            "anno_path": [
                "datasetA/annotations/train.json",
                "datasetB/annotations/train.json",
            ],
            "data_fields": _DATA_FIELDS,
        }
    )
    with pytest.raises(ValueError, match="identical category table"):
        dataset.parse_dataset()


def test_sample_num_truncates_across_files(tmp_path):
    _build_two_folder_coco(tmp_path, subpath_filenames=True)
    dataset = create(
        {
            "name": "COCODataSet",
            "dataset_dir": str(tmp_path),
            "image_dir": "",
            "anno_path": [
                "datasetA/annotations/train.json",
                "datasetB/annotations/train.json",
            ],
            "data_fields": _DATA_FIELDS,
            "sample_num": 3,
        }
    )
    dataset.parse_dataset()

    assert len(dataset) == 3
    prefixes = [
        "datasetA" if "datasetA" in rec["im_file"] else "datasetB"
        for rec in dataset.roidbs
    ]
    assert prefixes == ["datasetA", "datasetA", "datasetB"]


def test_single_dict_anno_path(tmp_path):
    _build_two_folder_coco(tmp_path, subpath_filenames=False)
    dataset = create(
        {
            "name": "COCODataSet",
            "dataset_dir": str(tmp_path),
            "anno_path": {
                "anno_path": "datasetA/annotations/train.json",
                "image_dir": "datasetA/images",
            },
            "data_fields": _DATA_FIELDS,
        }
    )
    dataset.parse_dataset()

    assert len(dataset) == 2
    assert all("datasetA" in rec["im_file"] for rec in dataset.roidbs)


def test_lvis_string_list_merges_two_datasets(tmp_path):
    pytest.importorskip("lvis")
    _build_two_folder_coco(tmp_path, subpath_filenames=True)
    # LVIS derives file names from coco_url and has no iscrowd handling
    for prefix in ("datasetA", "datasetB"):
        json_path = tmp_path / prefix / "annotations" / "train.json"
        annotations = json.loads(json_path.read_text(encoding="utf-8"))
        for image in annotations["images"]:
            image["coco_url"] = f"http://images.cocodataset.org/{image['file_name']}"
        json_path.write_text(json.dumps(annotations), encoding="utf-8")

    dataset = create(
        {
            "name": "LVISDataSet",
            "dataset_dir": str(tmp_path),
            "image_dir": "",
            "anno_path": [
                "datasetA/annotations/train.json",
                "datasetB/annotations/train.json",
            ],
            "data_fields": _DATA_FIELDS,
        }
    )
    dataset.parse_dataset()

    assert len(dataset) == 4
    im_ids = [int(rec["im_id"][0]) for rec in dataset.roidbs]
    assert im_ids == [1, 2, 4, 5]

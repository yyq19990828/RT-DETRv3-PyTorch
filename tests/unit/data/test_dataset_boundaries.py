import numpy as np
import pytest

from ppdet_pytorch.data.reader import CombineSSODLoader
from ppdet_pytorch.data.source.dataset import ImageFolder
from ppdet_pytorch.data.source.voc import VOCDataSet
from ppdet_pytorch.data.transform.autoaugment_utils import (
    shear_x,
    shear_y,
    translate_x,
    translate_y,
)
from ppdet_pytorch.data.transform.batch_operators import (
    BatchRandomResizeForSSOD,
    Gt2GFLTarget,
)
from ppdet_pytorch.data.transform.gridmask_utils import Gridmask
from ppdet_pytorch.data.transform.operators import RandomResize


def test_image_folder_loads_plain_images_without_annotation(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"image-placeholder")
    dataset = ImageFolder(dataset_dir=str(tmp_path))

    dataset.set_images([str(image_path)])

    assert len(dataset.roidbs) == 1
    assert dataset.roidbs[0]["im_file"] == str(image_path)
    np.testing.assert_array_equal(dataset.roidbs[0]["im_id"], [0])


def test_voc_dataset_reports_missing_required_xml_field(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"image-placeholder")
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        "<annotation><size><height>20</height></size></annotation>",
        encoding="utf-8",
    )
    list_path = tmp_path / "train.txt"
    list_path.write_text(
        f"{image_path.name} {xml_path.name}\n",
        encoding="utf-8",
    )
    dataset = VOCDataSet(
        dataset_dir=str(tmp_path),
        image_dir="",
        anno_path=list_path.name,
    )

    with pytest.raises(ValueError, match="size/width"):
        dataset.parse_dataset()


def test_combine_ssod_loader_initializes_and_cycles_iterators():
    label_loader = [("sup-weak", "sup-strong")]
    unlabel_loader = [
        ("unsup-weak-1", "unsup-strong-1"),
        ("unsup-weak-2", "unsup-strong-2"),
    ]
    combined = iter(CombineSSODLoader(label_loader, unlabel_loader))

    assert next(combined) == (
        "sup-weak",
        "sup-strong",
        "unsup-weak-1",
        "unsup-strong-1",
    )
    assert next(combined) == (
        "sup-weak",
        "sup-strong",
        "unsup-weak-2",
        "unsup-strong-2",
    )


def test_autoaugment_affine_operations_support_current_pillow_api():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    replace = [128, 128, 128]

    outputs = [
        translate_x(image, 1, replace),
        translate_y(image, 1, replace),
        shear_x(image, 0.1, replace),
        shear_y(image, 0.1, replace),
    ]

    assert all(output.shape == image.shape for output in outputs)
    assert all(output.dtype == np.uint8 for output in outputs)


def test_gridmask_executes_rotation_branch_without_changing_contract():
    np.random.seed(2026)
    image = np.ones((8, 8, 3), dtype=np.float32)
    gridmask = Gridmask(prob=1.0, upper_iter=1, rotate=1, mode=0)

    output = gridmask(image, curr_iter=1)

    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert not np.array_equal(output, image)


def test_random_resize_accepts_fixed_scalar_target():
    sample = {
        "image": np.ones((4, 6, 3), dtype=np.uint8),
        "scale_factor": np.ones(2, dtype=np.float32),
    }
    resize = RandomResize(target_size=8, keep_ratio=False, random_size=False)

    result = resize(sample)

    assert result["image"].shape == (8, 8, 3)
    np.testing.assert_array_equal(result["im_shape"], [8, 8])


def test_ssod_fixed_batch_resize_returns_reusable_empty_selection():
    resize = BatchRandomResizeForSSOD(
        target_size=[8, 8],
        keep_ratio=False,
        random_size=False,
    )
    weak_sample = {
        "image": np.ones((4, 6, 3), dtype=np.uint8),
        "scale_factor": np.ones(2, dtype=np.float32),
    }
    strong_sample = {
        "image": np.ones((4, 6, 3), dtype=np.uint8),
        "scale_factor": np.ones(2, dtype=np.float32),
    }

    weak_batch, selection = resize([weak_sample])
    strong_batch, strong_selection = resize([strong_sample], selection)

    assert selection is None
    assert strong_selection is None
    assert weak_batch[0]["image"].shape == (8, 8, 3)
    assert strong_batch[0]["image"].shape == (8, 8, 3)


def test_gfl_target_reuses_concatenated_grid_shape_across_batch():
    samples = [
        {
            "image": np.zeros((3, 32, 48), dtype=np.float32),
            "gt_bbox": np.empty((0, 4), dtype=np.float32),
            "gt_class": np.empty((0, 1), dtype=np.int32),
        }
        for _ in range(2)
    ]
    target = Gt2GFLTarget(downsample_ratios=[8, 16])

    result = target(samples)

    assert len(result) == 2
    assert all(sample["grid_cells"].shape == (30, 4) for sample in result)
    assert all(sample["bbox_targets"].shape == (30, 4) for sample in result)
    assert all(sample["labels"].shape == (30,) for sample in result)

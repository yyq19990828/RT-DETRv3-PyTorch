from __future__ import annotations

import importlib.util
import os
import random
import subprocess
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import tv_tensors

from ppdet_pytorch.data.transform.operators import DEIMDenseO2OMosaic

UPSTREAM_SHA = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"


def _sample(value):
    return {
        "image": np.full((6 + value % 3, 8 + value % 2, 3), value, dtype=np.uint8),
        "gt_bbox": np.array([[1, 1, 5, 5]], dtype=np.float32),
        "gt_class": np.array([[value % 3]], dtype=np.int64),
        "is_crowd": np.array([[value % 2]], dtype=np.int64),
        "curr_epoch": 0,
    }


def _target(sample):
    height, width = sample["image"].shape[:2]
    return {
        "boxes": tv_tensors.BoundingBoxes(
            torch.from_numpy(sample["gt_bbox"]),
            format="XYXY",
            canvas_size=(height, width),
        ),
        "gt_class": torch.from_numpy(sample["gt_class"]),
        "is_crowd": torch.from_numpy(sample["is_crowd"]),
    }


@pytest.fixture(scope="module")
def upstream_mosaic():
    root_value = os.environ.get("DEIM_UPSTREAM_ROOT")
    if not root_value:
        pytest.skip("set DEIM_UPSTREAM_ROOT to the pinned DEIM checkout")
    root = Path(root_value).expanduser().resolve()
    source = root / "engine/data/transforms/mosaic.py"
    if not source.is_file():
        pytest.skip("pinned DEIM Mosaic source is unavailable")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == UPSTREAM_SHA

    for name in ("_deim_data", "_deim_data.data", "_deim_data.data.transforms"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    core = types.ModuleType("_deim_data.core")
    core.register = lambda: lambda cls: cls
    sys.modules[core.__name__] = core
    misc = types.ModuleType("_deim_data.data._misc")

    def convert_to_tv_tensor(value, key, box_format=None, spatial_size=None):
        if key == "boxes":
            return tv_tensors.BoundingBoxes(
                value, format=box_format.upper(), canvas_size=spatial_size
            )
        return value

    misc.convert_to_tv_tensor = convert_to_tv_tensor
    sys.modules[misc.__name__] = misc
    spec = importlib.util.spec_from_file_location(
        "_deim_data.data.transforms.mosaic", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned DEIM Mosaic")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Mosaic


def test_dense_o2o_mosaic_matches_pinned_upstream_pixels_and_targets(
    upstream_mosaic,
):
    samples = [_sample(value) for value in (11, 22, 33, 44)]
    choice_seed = 19
    local_seed = 37
    random.seed(choice_seed)
    selected_indices = random.choices(range(3), k=3)
    selected = [samples[0]] + [samples[index + 1] for index in selected_indices]

    class Dataset:
        def __len__(self):
            return 3

        def load_item(self, index):
            sample = samples[index + 1]
            return Image.fromarray(sample["image"]), _target(sample)

    reference = upstream_mosaic(
        output_size=8,
        rotation_range=5,
        translation_range=(0.1, 0.1),
        scaling_range=(0.8, 1.2),
        probability=1.0,
        fill_value=114,
        use_cache=False,
    )
    random.seed(choice_seed)
    torch.manual_seed(local_seed)
    expected_image, expected_target, _ = reference(
        Image.fromarray(samples[0]["image"]), _target(samples[0]), Dataset()
    )

    actual = DEIMDenseO2OMosaic(
        output_size=8,
        rotation_range=5,
        translation_range=(0.1, 0.1),
        scaling_range=(0.8, 1.2),
        probability=1.0,
        fill_value=114,
        use_cache=False,
        seed=local_seed,
    )([deepcopy(sample) for sample in selected])

    np.testing.assert_array_equal(actual["image"], np.asarray(expected_image))
    np.testing.assert_allclose(
        actual["gt_bbox"], expected_target["boxes"].as_subclass(torch.Tensor).numpy()
    )
    np.testing.assert_array_equal(
        actual["gt_class"], expected_target["gt_class"].numpy()
    )
    np.testing.assert_array_equal(
        actual["is_crowd"], expected_target["is_crowd"].numpy()
    )

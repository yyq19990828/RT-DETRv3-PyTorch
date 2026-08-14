from copy import deepcopy

import numpy as np
import pytest

from ppdet_pytorch.data.reader import BatchCompose, Compose, TrainReader


class _Recorder:
    def __init__(self, name):
        self.name = name

    def __call__(self, data):
        data.setdefault("called", []).append(self.name)
        return data


@pytest.mark.parametrize(
    ("stop_epoch", "case_id"),
    [(148, "dfine_stop_148"), (120, "dfine_stop_120"), (72, "dfine_stop_72")],
    ids=lambda value: str(value),
)
def test_dfine_ordinary_stop_policy(stop_epoch, case_id):
    del case_id
    reader = TrainReader(
        ordinary_transform_policy={"stop_epoch": stop_epoch, "ops": ["RandomFlip"]}
    )
    assert not any(
        "Mosaic" in type(op).__name__ or "MixUp" in type(op).__name__
        for op in (
            reader._sample_transforms.transforms_cls
            + reader._batch_transforms.transforms_cls
        )
    )
    op = _Recorder("flip")
    op.__class__.__name__ = "RandomFlip"
    reader._sample_transforms.transforms_cls = [op]
    assert reader._sample_transforms({"curr_epoch": stop_epoch - 1})["called"] == [
        "flip"
    ]
    assert "called" not in reader._sample_transforms({"curr_epoch": stop_epoch})
    assert "called" not in reader._sample_transforms({"curr_epoch": stop_epoch + 1})


def test_dfine_multiscale_stop_switches_to_fixed_base_size():
    reader = TrainReader(
        batch_transforms=[
            {
                "BatchRandomResize": {
                    "target_size": [32, 64, 96],
                    "keep_ratio": False,
                    "random_size": True,
                }
            }
        ],
        ordinary_transform_policy={"stop_epoch": 72, "ops": [], "base_size": 64},
    )
    before = list(reader._batch_transforms._update_transforms_cls({"curr_epoch": 71}))
    after = list(reader._batch_transforms._update_transforms_cls({"curr_epoch": 72}))

    assert before[0].target_size == [32, 64, 96]
    assert before[0].random_size is True
    assert after[0].target_size == 64
    assert after[0].random_size is False
    assert reader._batch_transforms.transforms_cls[0].random_size is True


SCHEDULES = [
    ([4, 78, 148], [4, 78], "deim_dfine_n_4_78_148"),
    ([4, 64, 120], [4, 64], "deim_dfine_s_4_64_120"),
    ([4, 49, 90], [4, 49], "deim_dfine_m_4_49_90"),
    ([4, 29, 50], [4, 29], "deim_dfine_lx_4_29_50"),
    ([4, 64, 117], [4, 64], "rtv2_4_64_117"),
    ([4, 34, 58], [4, 34], "rtv2_4_34_58"),
    ([4, 64, 120], [4, 64], "rtv4_s_4_64_120"),
    ([4, 49, 90], [4, 49], "rtv4_m_4_49_90"),
    ([4, 29, 50], [4, 29], "rtv4_lx_4_29_50"),
]


@pytest.mark.parametrize(
    "policy_epochs,mixup_epochs,case_id", SCHEDULES, ids=lambda x: str(x)
)
def test_dense_o2o_schedule_boundaries(policy_epochs, mixup_epochs, case_id):
    del case_id
    reader = TrainReader(
        dense_o2o_policy={
            "policy_epochs": policy_epochs,
            "mixup_epochs": mixup_epochs,
            "multiscale_stop_epoch": policy_epochs[-1],
            "mosaic_prob": 1,
            "mixup_prob": 1,
            "multiscale_sizes": [8, 12],
            "mosaic": {"output_size": 8},
        }
    )
    mosaic = reader._sample_transforms.transforms_cls[-1]
    collate = reader._batch_transforms.transforms_cls[-1]

    boundaries = sorted(set(policy_epochs + mixup_epochs))
    for boundary in boundaries:
        for epoch in (boundary - 1, boundary, boundary + 1):
            expected_mosaic = policy_epochs[0] <= epoch < policy_epochs[1]
            expected_mixup = mixup_epochs[0] <= epoch < mixup_epochs[1]
            expected_multiscale = epoch < policy_epochs[-1]
            assert mosaic.is_active(epoch) is expected_mosaic
            assert collate.mixup_active(epoch) is expected_mixup
            assert collate.multiscale_active(epoch) is expected_multiscale


def test_v3_compose_and_batch_compose_are_byte_equivalent_when_policy_omitted():
    sample = {"image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3), "curr_epoch": 9}
    before = deepcopy(sample)
    result = Compose([])(sample)
    np.testing.assert_array_equal(result["image"], before["image"])

    batch = BatchCompose([], collate_batch=True)([deepcopy(result)])
    np.testing.assert_array_equal(batch["image"], before["image"][None])

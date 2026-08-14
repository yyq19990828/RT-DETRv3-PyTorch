from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ppdet_pytorch.data.reader import TrainReader
from ppdet_pytorch.data.transform.operators import DEIMDenseO2OMosaic


def _sample(value=32):
    return {
        "image": np.full((8, 8, 3), value, dtype=np.uint8),
        "gt_bbox": np.array([[1, 1, 7, 7]], dtype=np.float32),
        "gt_class": np.array([[1]], dtype=np.int32),
        "is_crowd": np.array([[0]], dtype=np.int32),
        "curr_epoch": 5,
    }


def _policy(**updates):
    policy = {
        "policy_epochs": [4, 29, 50],
        "mixup_epochs": [4, 29],
        "multiscale_stop_epoch": 50,
        "mosaic_prob": 1.0,
        "mixup_prob": 1.0,
        "multiscale_sizes": [8, 12],
        "mosaic": {"output_size": 8, "use_cache": False},
    }
    policy.update(updates)
    return policy


def test_dense_o2o_builds_uniquely_named_components():
    reader = TrainReader(
        sample_transforms=[{"Decode": {}}, {"RandomFlip": {}}],
        batch_transforms=[{"NormalizeImage": {}}],
        dense_o2o_policy=_policy(),
    )

    assert [type(op).__name__ for op in reader._sample_transforms.transforms_cls] == [
        "Decode",
        "DEIMDenseO2OMosaic",
        "RandomFlip",
    ]
    assert [type(op).__name__ for op in reader._batch_transforms.transforms_cls] == [
        "DEIMDenseO2OCollate",
        "NormalizeImage",
    ]


def test_dense_o2o_mosaic_clips_boxes_and_preserves_upstream_alignment():
    samples = [_sample(value) for value in (16, 32, 48, 64)]
    samples[0]["gt_bbox"] = np.array([[-2, -2, 4, 4], [7, 7, 7, 8]])
    samples[0]["gt_class"] = np.array([[1], [2]])
    samples[0]["is_crowd"] = np.array([[0], [0]])
    op = DEIMDenseO2OMosaic(output_size=8, probability=1, use_cache=False)

    result = op(samples)

    assert result["image"].shape == (16, 16, 3)
    assert len(result["gt_bbox"]) == len(result["gt_class"]) == 5
    assert np.isfinite(result["gt_bbox"]).all()
    assert (result["gt_bbox"] >= 0).all()
    assert (result["gt_bbox"] <= 16).all()
    assert (result["gt_bbox"][:, 2:] >= result["gt_bbox"][:, :2]).all()
    assert (result["gt_bbox"][:, 2:] == result["gt_bbox"][:, :2]).any()


def test_dense_o2o_cache_is_bounded_and_cloned():
    op = DEIMDenseO2OMosaic(
        output_size=8,
        probability=1,
        use_cache=True,
        max_cached_images=2,
        random_pop=False,
        seed=9,
    )
    first = _sample(1)
    op(first)
    first["image"].fill(255)
    op(_sample(2))
    op(_sample(3))

    assert len(op.cache) == 2
    assert all(item["image"].max() < 255 for item in op.cache)


def test_dense_o2o_seed_is_deterministic():
    samples = [_sample(value) for value in (1, 2, 3, 4)]
    left = DEIMDenseO2OMosaic(output_size=8, probability=0.5, seed=17)
    right = DEIMDenseO2OMosaic(output_size=8, probability=0.5, seed=17)

    left_result = left(deepcopy(samples))
    right_result = right(deepcopy(samples))

    np.testing.assert_array_equal(left_result["image"], right_result["image"])
    np.testing.assert_array_equal(left_result["gt_bbox"], right_result["gt_bbox"])


def test_dense_o2o_set_epoch_propagates_rank_and_epoch():
    reader = TrainReader(dense_o2o_policy=_policy(), seed=23)
    reader._rank = 2
    reader._world_size = 4
    reader._worker_generator = torch.Generator()
    reader._batch_sampler = object()

    class Dataset:
        def set_epoch(self, epoch):
            self.epoch = epoch

    reader.dataset = Dataset()
    reader.set_epoch(11)

    mosaic = reader._sample_transforms.transforms_cls[-1]
    collate = reader._batch_transforms.transforms_cls[-1]
    assert reader.dataset.epoch == 11
    assert (mosaic.epoch, mosaic.rank) == (11, 2)
    assert (collate.epoch, collate.rank) == (11, 2)


def test_rejects_unsorted_epoch_policy():
    with pytest.raises(ValueError, match="policy_epochs"):
        TrainReader(dense_o2o_policy=_policy(policy_epochs=[29, 4, 50]))


def test_rejects_duplicate_epoch_policy():
    with pytest.raises(ValueError, match="policy_epochs"):
        TrainReader(dense_o2o_policy=_policy(policy_epochs=[4, 4, 50]))


@pytest.mark.parametrize("mixup_epochs", ([3, 29], [4, 50]))
def test_rejects_mixup_outside_policy(mixup_epochs):
    with pytest.raises(ValueError, match="mixup_epochs"):
        TrainReader(dense_o2o_policy=_policy(mixup_epochs=mixup_epochs))


def test_rejects_multiscale_stop_outside_policy():
    with pytest.raises(ValueError, match="multiscale_stop_epoch"):
        TrainReader(dense_o2o_policy=_policy(multiscale_stop_epoch=49))


def test_dense_o2o_cache_does_not_run_outside_policy_window():
    op = DEIMDenseO2OMosaic(
        output_size=8,
        probability=1,
        use_cache=True,
        policy_epochs=[4, 29, 50],
    )
    sample = _sample()
    sample["curr_epoch"] = 50

    assert op(sample) is sample
    assert op.cache == []


def test_rejects_dense_o2o_for_dfine():
    with pytest.raises(ValueError, match="ordinary.*Dense O2O"):
        TrainReader(
            ordinary_transform_policy={"stop_epoch": 120, "ops": ["RandomFlip"]},
            dense_o2o_policy=_policy(),
        )


def test_rejects_mosaic_transform_for_dfine():
    with pytest.raises(ValueError, match="ordinary.*Mosaic/MixUp"):
        TrainReader(
            sample_transforms=[{"Mosaic": {}}],
            ordinary_transform_policy={"stop_epoch": 120, "ops": ["RandomFlip"]},
        )


def test_rejects_impossible_box():
    samples = [_sample() for _ in range(4)]
    samples[0]["gt_bbox"][0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        DEIMDenseO2OMosaic(output_size=8, probability=1)(samples)


def test_rejects_malformed_cache():
    op = DEIMDenseO2OMosaic(output_size=8, use_cache=True)
    op.cache.append({"image": np.zeros((8, 8, 3), dtype=np.uint8)})

    with pytest.raises(ValueError, match="cache"):
        op(_sample())


class _WorkerMosaicDataset(Dataset):
    def __init__(self, seed):
        self.op = DEIMDenseO2OMosaic(output_size=8, probability=1, seed=seed)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        samples = [_sample(index * 4 + offset) for offset in range(4)]
        result = self.op(samples)
        return {
            "image": result["image"],
            "gt_bbox": result["gt_bbox"],
        }


def _run_worker_mosaic(seed):
    loader = DataLoader(_WorkerMosaicDataset(seed), batch_size=2, num_workers=2)
    return [(batch["image"].clone(), batch["gt_bbox"].clone()) for batch in loader]


def test_dense_o2o_real_multi_worker_run_is_deterministic():
    first = _run_worker_mosaic(seed=29)
    repeated = _run_worker_mosaic(seed=29)

    assert len(first) == len(repeated)
    for (first_image, first_boxes), (second_image, second_boxes) in zip(
        first, repeated
    ):
        torch.testing.assert_close(first_image, second_image, rtol=0, atol=0)
        torch.testing.assert_close(first_boxes, second_boxes, rtol=0, atol=0)

"""Reduced D-FINE runtime and deterministic checkpoint contracts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from PIL import Image

from detrs.core.workspace import load_config
from detrs.engine.trainer import Trainer
from detrs.utils.checkpoint import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def _fixture(root):
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(20, 30, 40)).save(images / "one.jpg")
    annotations = {
        "images": [{"id": 1, "file_name": "one.jpg", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 8, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    (root / "instances.json").write_text(json.dumps(annotations), encoding="utf-8")
    return {
        "name": "COCODataSet",
        "dataset_dir": str(root),
        "image_dir": "images",
        "anno_path": "instances.json",
        "data_fields": ["image", "gt_bbox", "gt_class", "is_crowd"],
    }


def _trainer(variant, root, seed=19, *, protocol=False):
    torch.manual_seed(seed)
    cfg = load_config(
        ROOT / "configs/dfine/dfine_hgnetv2_{}_coco.yml".format(variant)
    ).copy()
    dataset = _fixture(root)
    cfg["TrainDataset"] = dataset
    if protocol:
        cfg["EvalDataset"] = deepcopy(dataset)
    cfg["TrainReader"] = {
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
            {"NormalizeBox": {}},
            {"BboxXYXY2XYWH": {}},
            {"Permute": {}},
        ],
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
        "collate_batch": False,
    }
    cfg["worker_num"] = 0
    cfg["save_dir"] = str(root / "output")
    cfg["device"] = torch.device("cpu")
    cfg["amp"] = False
    cfg["validate"] = protocol
    if not protocol:
        cfg.pop("TrainingProtocol", None)
    cfg["DFINETransformer"]["num_queries"] = 4
    cfg["DFINETransformer"]["num_denoising"] = 0
    cfg["DFINETransformer"]["eval_spatial_size"] = None
    trainer = Trainer(cfg, mode="train")
    trainer._compose_callback = None
    return trainer


def _state(trainer):
    return {
        "model": deepcopy(trainer.model.state_dict()),
        "optimizer": deepcopy(trainer.optimizer.state_dict()),
        "scheduler": deepcopy(trainer.lr.state_dict()),
        "ema": deepcopy(trainer.ema.state_dict_for_save()),
        "global_step": trainer.global_step,
    }


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.integration
@pytest.mark.parametrize("variant", "nsmlx")
def test_dfine_reduced_optimizer_ema_update_is_finite(variant, tmp_path):
    trainer = _trainer(variant, tmp_path / variant)
    trainer._train_epoch(0)

    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(trainer.status["loss"]))
    assert torch.isfinite(torch.tensor(trainer.status["gradient_norm"]))
    assert trainer.ema.step == 1


@pytest.mark.integration
@pytest.mark.parametrize("variant", "nsmlx")
def test_dfine_epoch_resume_next_update_matches_uninterrupted(variant, tmp_path):
    uninterrupted = _trainer(variant, tmp_path / "run")
    uninterrupted._train_epoch(0)
    checkpoint = tmp_path / "run" / "epoch.pth"
    save_checkpoint(
        uninterrupted.model,
        uninterrupted.optimizer,
        epoch=1,
        iteration=uninterrupted.global_step,
        save_path=str(checkpoint),
        scheduler=uninterrupted.lr,
        ema=uninterrupted.ema,
        sampler_epoch=1,
    )
    uninterrupted.loader.set_epoch(1)
    uninterrupted._train_epoch(1)
    expected = _state(uninterrupted)

    resumed = _trainer(variant, tmp_path / "run", seed=99)
    resumed.resume_weights(str(checkpoint))
    checkpoint.unlink()
    resumed.loader.set_epoch(1)
    resumed._train_epoch(1)

    _assert_nested_equal(_state(resumed), expected)


@pytest.mark.integration
@pytest.mark.parametrize("variant", "nsmlx")
def test_dfine_stage1_transition_reloads_real_components_and_restarts_ema(
    variant, tmp_path
):
    trainer = _trainer(variant, tmp_path / "run", protocol=True)
    trainer._train_epoch(0)
    trainer.notify_validation({"bbox": 0.4})
    stage1 = _state(trainer)
    protocol = trainer.training_protocol

    trainer._train_epoch(1)
    protocol.before_epoch(protocol.stop_epoch, {})
    trainer._execute_protocol_actions()

    assert protocol.stage == 2
    assert protocol.companion_basename == "best_stg1.pth"
    assert len(protocol.companion_sha256) == 64
    assert trainer.ema.decay == pytest.approx(protocol.ema_restart_decay)
    restored = _state(trainer)
    for section in ("model", "optimizer", "scheduler"):
        _assert_nested_equal(restored[section], stage1[section])
    for key in ("ema_state_dict", "step", "epoch"):
        _assert_nested_equal(restored["ema"][key], stage1["ema"][key])
    assert restored["global_step"] == stage1["global_step"]

    trainer.notify_validation({"bbox": 0.3})
    trainer.notify_validation({"bbox": 0.41})
    stage2_checkpoint = tmp_path / "run" / "output" / "best_stg2.pth"
    assert stage2_checkpoint.is_file()
    stage2_checkpoint.unlink()
    trainer.notify_validation({"bbox": 0.4})
    trainer.notify_validation({"bbox": 0.39})

    assert protocol.restart_count == 1
    assert protocol.top_metric == pytest.approx(0.41)
    assert trainer.ema.decay == pytest.approx(
        protocol.ema_restart_decay - protocol.decay_decrement
    )
    (tmp_path / "run" / "output" / "best_stg1.pth").unlink()


def _corrupt_checkpoint(source, target, field):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint.pop(field)
    torch.save(checkpoint, target)


@pytest.mark.parametrize(
    ("test_id", "field", "message"),
    [
        ("rejects_missing_rng", "rng_state", "RNG state"),
        ("rejects_missing_scheduler", "scheduler", "scheduler state"),
    ],
)
def test_rejects_incomplete_resume_without_state_mutation(
    test_id, field, message, tmp_path
):
    del test_id
    trainer = _trainer("n", tmp_path / "run")
    trainer._train_epoch(0)
    valid = tmp_path / "valid.pth"
    save_checkpoint(
        trainer.model,
        trainer.optimizer,
        epoch=1,
        iteration=trainer.global_step,
        save_path=str(valid),
        scheduler=trainer.lr,
        ema=trainer.ema,
    )
    corrupt = tmp_path / "corrupt.pth"
    _corrupt_checkpoint(valid, corrupt, field)
    before = _state(trainer)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(
            str(corrupt),
            trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.lr,
            ema=trainer.ema,
            restore_rng=True,
        )

    _assert_nested_equal(_state(trainer), before)


def test_rejects_missing_stage1_without_state_mutation(tmp_path):
    trainer = _trainer("n", tmp_path / "missing", protocol=True)
    protocol = trainer.training_protocol
    before = _state(trainer)

    with pytest.raises(FileNotFoundError, match="companion is missing"):
        protocol.before_epoch(protocol.stop_epoch, {})

    _assert_nested_equal(_state(trainer), before)


def test_rejects_stage1_sha_without_state_mutation(tmp_path):
    trainer = _trainer("n", tmp_path / "tampered", protocol=True)
    trainer.notify_validation({"bbox": 0.4})
    companion = tmp_path / "tampered" / "output" / "best_stg1.pth"
    companion.write_bytes(companion.read_bytes() + b"tampered")
    trainer.training_protocol.before_epoch(trainer.training_protocol.stop_epoch, {})
    before = _state(trainer)

    with pytest.raises(ValueError, match="companion SHA-256 mismatch"):
        trainer._execute_protocol_actions()

    _assert_nested_equal(_state(trainer), before)


def test_rejects_nan_without_publish(tmp_path):
    trainer = _trainer("n", tmp_path / "run")
    batch = trainer._prepare_batch(next(iter(trainer.loader)))
    batch["image"].fill_(float("nan"))
    trainer.loader = [batch]
    before = _state(trainer)
    files_before = set((tmp_path / "run" / "output").iterdir())

    with pytest.raises(FloatingPointError):
        trainer._train_epoch(0)

    _assert_nested_equal(_state(trainer), before)
    assert set((tmp_path / "run" / "output").iterdir()) == files_before

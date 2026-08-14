"""Reduced DEIM-D-FINE training, resume, and protocol contracts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from PIL import Image

from ppdet_pytorch.core.workspace import create, load_config
from ppdet_pytorch.engine.trainer import Trainer
from ppdet_pytorch.utils.checkpoint import load_checkpoint, save_checkpoint

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
        ROOT / f"configs/deim/dfine/deim_hgnetv2_{variant}_coco.yml"
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
        "dense_o2o_policy": {
            "policy_epochs": [4, 29, 50],
            "mixup_epochs": [4, 29],
            "multiscale_stop_epoch": 50,
            "mosaic_prob": 0.5,
            "mixup_prob": 0.5,
            "multiscale_sizes": None,
            "mosaic": {"output_size": 32, "use_cache": False},
        },
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
        "collate_batch": False,
    }
    cfg["LearningRate"] = {
        "base_lr": cfg["base_lr"],
        "schedulers": [
            {
                "name": "FlatCosineLRScheduler",
                "total_epochs": 2,
                "warmup_iter": 0,
                "flat_epochs": 1,
                "no_aug_epochs": 0,
                "lr_gamma": 0.5,
            }
        ],
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
def test_deim_dfine_reduced_optimizer_ema_update_is_finite(variant, tmp_path):
    trainer = _trainer(variant, tmp_path / variant)
    trainer._train_epoch(0)

    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(trainer.status["loss"]))
    assert torch.isfinite(torch.tensor(trainer.status["gradient_norm"]))
    assert trainer.ema.step == 1


@pytest.mark.integration
@pytest.mark.parametrize("variant", "nsmlx")
def test_deim_dfine_epoch_resume_next_update_matches_uninterrupted(variant, tmp_path):
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
def test_deim_dfine_stage1_transition_reloads_components_and_restarts_ema(tmp_path):
    trainer = _trainer("n", tmp_path / "run", protocol=True)
    trainer._train_epoch(0)
    trainer.notify_validation({"bbox": 0.4})
    stage1 = _state(trainer)
    protocol = trainer.training_protocol

    trainer._train_epoch(1)
    protocol.before_epoch(protocol.stop_epoch, {})
    trainer._execute_protocol_actions()

    assert protocol.stage == 2
    assert trainer.ema.decay == pytest.approx(protocol.ema_restart_decay)
    restored = _state(trainer)
    for section in ("model", "optimizer", "scheduler"):
        _assert_nested_equal(restored[section], stage1[section])
    (tmp_path / "run" / "output" / "best_stg1.pth").unlink()


def test_rejects_vfl_substitution(isolated_workspace):
    config = load_config(ROOT / "configs/deim/dfine/deim_hgnetv2_n_coco.yml")
    config.DEIMCriterion["losses"] = ["vfl", "boxes", "local"]

    with pytest.raises(ValueError, match="unsupported DEIM losses"):
        create(config.architecture)


def test_rejects_augmentation_after_stop(isolated_workspace):
    config = load_config(ROOT / "configs/deim/dfine/deim_hgnetv2_s_coco.yml")
    config.TrainReader["dense_o2o_policy"]["multiscale_stop_epoch"] = 119

    with pytest.raises(ValueError, match="multiscale_stop_epoch"):
        create("TrainReader")


def test_rejects_wrong_stage_companion(tmp_path):
    trainer = _trainer("n", tmp_path / "run", protocol=True)
    checkpoint = tmp_path / "wrong-stage.pth"
    training_state = trainer._protocol_training_state()
    training_state["protocol_stage"] = 2
    save_checkpoint(
        trainer.model,
        trainer.optimizer,
        epoch=0,
        iteration=0,
        save_path=str(checkpoint),
        scheduler=trainer.lr,
        ema=trainer.ema,
        training_state=training_state,
    )
    before = _state(trainer)

    with pytest.raises(ValueError, match="training protocol stage mismatch"):
        load_checkpoint(
            str(checkpoint),
            trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.lr,
            ema=trainer.ema,
            protocol=trainer.training_protocol,
            expected_model_identity=str(trainer.cfg.get("architecture")),
        )

    _assert_nested_equal(_state(trainer), before)

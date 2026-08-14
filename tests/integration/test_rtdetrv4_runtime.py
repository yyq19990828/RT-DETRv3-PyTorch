"""Reduced RT-DETRv4 student, DSI, GAM, and resume contracts."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ppdet_pytorch.core.workspace import load_config
from ppdet_pytorch.engine.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]


class _Teacher(nn.Module):
    def forward(self, images):
        pooled = F.adaptive_avg_pool2d(images.mean(1, keepdim=True), (2, 2))
        return pooled.expand(-1, 768, -1, -1).detach()


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


def _trainer(root, seed=19, *, variant="s", real_teacher=None):
    torch.manual_seed(seed)
    cfg = load_config(
        ROOT / f"configs/rtdetrv4/rtdetrv4_hgnetv2_{variant}_coco.yml"
    ).copy()
    dataset = _fixture(root)
    cfg["TrainDataset"] = dataset
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
    if real_teacher is None:
        cfg.pop("teacher_model")
    else:
        cfg["teacher_model"].update(real_teacher)
    cfg["DFINETransformer"]["num_queries"] = 4
    cfg["DFINETransformer"]["num_denoising"] = 0
    cfg["DFINETransformer"]["eval_spatial_size"] = None
    cfg["eval_spatial_size"] = None
    trainer = Trainer(cfg, mode="train")
    if real_teacher is None:
        trainer.teacher_model = _Teacher()
    trainer._compose_callback = None
    return trainer


def _state(trainer):
    return {
        "model": deepcopy(trainer.model.state_dict()),
        "optimizer": deepcopy(trainer.optimizer.state_dict()),
        "scheduler": deepcopy(trainer.lr.state_dict()),
        "ema": deepcopy(trainer.ema.state_dict_for_save()),
        "protocol": deepcopy(trainer.training_protocol.state_dict()),
        "criterion_weight": trainer.model.criterion.weight_dict["loss_distill"],
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
def test_reduced_teacher_dsi_gam_update_is_finite(tmp_path):
    trainer = _trainer(tmp_path / "run")
    projector_before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if name.startswith("encoder.feature_projector")
    }

    trainer._train_epoch(0)

    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(trainer.status["loss"]))
    assert trainer.ema.step == 1
    assert trainer.training_protocol._gam_observation_count == 1
    assert any(
        not torch.equal(parameter, projector_before[name])
        for name, parameter in trainer.model.named_parameters()
        if name.startswith("encoder.feature_projector")
    )
    assert not any("teacher" in key for key in trainer.model.state_dict())


@pytest.mark.integration
def test_epoch_boundary_resume_matches_next_dsi_gam_update(tmp_path):
    uninterrupted = _trainer(tmp_path / "run")
    uninterrupted._train_epoch(0)
    uninterrupted.training_protocol.after_epoch(0, {})
    uninterrupted._execute_protocol_actions()
    checkpoint = tmp_path / "run" / "epoch.pth"
    uninterrupted._save_protocol_checkpoint(str(checkpoint))
    uninterrupted.loader.set_epoch(1)
    uninterrupted._train_epoch(1)
    expected = _state(uninterrupted)

    resumed = _trainer(tmp_path / "run", seed=99)
    resumed.resume_weights(str(checkpoint))
    checkpoint.unlink()
    resumed.loader.set_epoch(1)
    resumed._train_epoch(1)

    _assert_nested_equal(_state(resumed), expected)


def test_student_eval_has_no_teacher_or_distillation_output(tmp_path):
    trainer = _trainer(tmp_path / "run")
    trainer.model.eval()
    trainer.model.decoder.eval_spatial_size = None
    batch = trainer._prepare_batch(next(iter(trainer.loader)))

    with torch.inference_mode():
        outputs = trainer.model(batch)

    assert set(outputs) == {"bbox", "bbox_num"}


def test_rejects_missing_teacher_preflight(tmp_path):
    with pytest.raises(FileNotFoundError, match="repository checkout is missing"):
        _trainer(
            tmp_path / "missing",
            real_teacher={
                "dinov3_repo_path": str(tmp_path / "missing-repo"),
                "dinov3_weights_path": str(tmp_path / "missing-weights.pth"),
            },
        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("variant", tuple("smlx"))
def test_real_dinov3_teacher_reduced_update(tmp_path, variant):
    repo = os.environ.get("DINOV3_REPO")
    weights = os.environ.get("DINOV3_WEIGHTS")
    sha256 = os.environ.get("DINOV3_WEIGHTS_SHA256")
    if not repo or not weights or not sha256:
        pytest.skip("set DINOV3_REPO, DINOV3_WEIGHTS and DINOV3_WEIGHTS_SHA256")
    trainer = _trainer(
        tmp_path / variant,
        variant=variant,
        real_teacher={
            "dinov3_repo_path": repo,
            "dinov3_weights_path": weights,
            "weights_sha256": sha256,
        },
    )

    trainer._train_epoch(0)

    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(trainer.status["loss"]))
    assert trainer.training_protocol._gam_observation_count == 1
    assert not any("teacher" in key for key in trainer.model.state_dict())

"""Reduced DEIM-RT-DETRv2 training, resume, and rejection contracts."""

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ppdet_pytorch.core.workspace import create, load_config
from ppdet_pytorch.engine.trainer import Trainer
from ppdet_pytorch.utils.checkpoint import save_checkpoint

from .test_deim_dfine_runtime import _assert_nested_equal, _fixture, _state

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("r18vd_120e", "r34vd_120e", "r50vd_m_60e", "r50vd_60e", "r101vd_60e")


def _trainer(variant, root, seed=19, *, protocol=False, pretrain_weights=None):
    torch.manual_seed(seed)
    cfg = load_config(ROOT / f"configs/deim/rtdetrv2/deim_{variant}_coco.yml").copy()
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
    if pretrain_weights is None:
        cfg.pop("pretrain_weights", None)
    else:
        cfg["pretrain_weights"] = str(pretrain_weights)
    cfg["save_dir"] = str(root / "output")
    cfg["device"] = torch.device("cpu")
    cfg["amp"] = False
    cfg["validate"] = protocol
    if not protocol:
        cfg.pop("TrainingProtocol", None)
    cfg["RTDETRTransformerv2"]["num_queries"] = 4
    cfg["RTDETRTransformerv2"]["num_denoising"] = 0
    cfg["RTDETRTransformerv2"]["eval_spatial_size"] = None
    trainer = Trainer(cfg, mode="train")
    trainer._compose_callback = None
    return trainer


@pytest.mark.integration
@pytest.mark.parametrize("variant", VARIANTS)
def test_reduced_optimizer_ema_update_is_finite(variant, tmp_path):
    trainer = _trainer(variant, tmp_path / variant)
    trainer._train_epoch(0)

    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(trainer.status["loss"]))
    assert torch.isfinite(torch.tensor(trainer.status["gradient_norm"]))
    assert trainer.ema.step == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    "variant,filename",
    [
        ("r18vd_120e", "ResNet18_vd_pretrained_from_paddle.pth"),
        ("r34vd_120e", "ResNet34_vd_pretrained_from_paddle.pth"),
        ("r50vd_60e", "ResNet50_vd_ssld_v2_pretrained_from_paddle.pth"),
        ("r101vd_60e", "ResNet101_vd_ssld_pretrained_from_paddle.pth"),
    ],
)
def test_official_pretrain_initializes_model_and_ema(variant, filename, tmp_path):
    root_value = os.environ.get("DEIM_RTDETRV2_PRETRAINED_ROOT")
    if not root_value:
        pytest.skip("set DEIM_RTDETRV2_PRETRAINED_ROOT")
    path = Path(root_value) / filename
    state = torch.load(path, map_location="cpu", weights_only=True)
    trainer = _trainer(variant, tmp_path / "run", pretrain_weights=path)

    assert trainer.is_loaded_weights
    assert trainer.ema.step == 0
    for key, value in state.items():
        torch.testing.assert_close(
            trainer.model.backbone.state_dict()[key], value, rtol=0, atol=0
        )
        torch.testing.assert_close(
            trainer.ema.state_dict[f"backbone.{key}"], value, rtol=0, atol=0
        )


@pytest.mark.integration
@pytest.mark.parametrize("variant", VARIANTS)
def test_epoch_resume_next_update_matches_uninterrupted(variant, tmp_path):
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
def test_stage1_transition_reloads_components_and_restarts_ema(tmp_path):
    trainer = _trainer("r18vd_120e", tmp_path / "run", protocol=True)
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


def test_rejects_local_loss(isolated_workspace):
    config = load_config(ROOT / "configs/deim/rtdetrv2/deim_r18vd_120e_coco.yml")
    config.DEIMCriterion["losses"].append("local")

    with pytest.raises(ValueError, match="does not support D-FINE local loss"):
        create(config.architecture)


def test_rejects_r50_variant_swap(isolated_workspace):
    medium = create(
        load_config(
            ROOT / "configs/deim/rtdetrv2/deim_r50vd_m_60e_coco.yml"
        ).architecture
    )
    large = create(
        load_config(ROOT / "configs/deim/rtdetrv2/deim_r50vd_60e_coco.yml").architecture
    )

    with pytest.raises(RuntimeError, match="Missing key"):
        large.load_state_dict(medium.state_dict(), strict=True)


def test_rejects_v3_checkpoint(isolated_workspace):
    v2 = create(
        load_config(
            ROOT / "configs/deim/rtdetrv2/deim_r18vd_120e_coco.yml"
        ).architecture
    )
    v3 = create(
        load_config(ROOT / "configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml").architecture
    )

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key|size mismatch"):
        v2.load_state_dict(v3.state_dict(), strict=True)

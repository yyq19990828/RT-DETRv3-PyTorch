from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ppdet_pytorch.core.workspace import create, load_config, merge_config
from ppdet_pytorch.engine.trainer import Trainer
from ppdet_pytorch.modeling.architectures.rtdetrv3 import RTDETRV3
from ppdet_pytorch.modeling.backbones.resnet import ResNet
from ppdet_pytorch.modeling.heads.detr_head import DINOv3Head
from ppdet_pytorch.modeling.heads.ppyoloe_head import PPYOLOEHead
from ppdet_pytorch.modeling.post_process import DETRPostProcess
from ppdet_pytorch.modeling.transformers.hybrid_encoder import HybridEncoder
from ppdet_pytorch.modeling.transformers.rtdetr_transformerv3 import (
    RTDETRTransformerv3,
)

ROOT = Path(__file__).resolve().parents[2]
R18_CONFIG = ROOT / "configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml"


class _StepRecorder:
    def __init__(self):
        self.steps = []

    def on_step_begin(self, status):
        pass

    def on_step_end(self, status):
        self.steps.append(dict(status))


def _build_reduced_r18_trainer(
    workspace,
    minimal_coco_config,
    deterministic_train_reader_config,
    output_dir,
    repeat=1,
):
    torch.manual_seed(2026)
    load_config(str(R18_CONFIG))
    merge_config(
        {
            "eval_size": [96, 96],
            "num_queries_o2m": 20,
            "RTDETRTransformerv3": {
                "num_queries": 20,
                "num_decoder_layers": 2,
                "num_denoising": 8,
                "num_noises": 0,
                "num_noise_queries": [],
                "num_noise_denoising": 8,
            },
        }
    )
    cfg = workspace.copy()
    dataset_config = deepcopy(minimal_coco_config)
    dataset_config["repeat"] = repeat
    reader_config = deepcopy(deterministic_train_reader_config)
    reader_config["batch_transforms"][0]["BatchRandomResize"]["target_size"] = [
        96,
        96,
    ]
    cfg["TrainDataset"] = dataset_config
    cfg["TrainReader"] = reader_config
    cfg["worker_num"] = 0
    cfg["save_dir"] = str(output_dir)
    cfg["use_ema"] = False
    cfg["device"] = torch.device("cpu")
    return Trainer(cfg, mode="train")


@pytest.mark.integration
def test_r18_config_builds_complete_model(isolated_workspace):
    cfg = load_config(str(R18_CONFIG)).copy()

    model = create(cfg.architecture)

    assert isinstance(model, RTDETRV3)
    assert isinstance(model.backbone, ResNet)
    assert isinstance(model.neck, HybridEncoder)
    assert isinstance(model.transformer, RTDETRTransformerv3)
    assert isinstance(model.detr_head, DINOv3Head)
    assert isinstance(model.aux_o2m_head, PPYOLOEHead)
    assert isinstance(model.post_process, DETRPostProcess)
    assert sum(parameter.numel() for parameter in model.parameters()) == 22_942_893
    assert cfg["TrainReader"]["name"] == "TrainReader"
    assert cfg["EvalReader"]["name"] == "EvalReader"
    assert cfg["TestReader"]["name"] == "TestReader"


@pytest.mark.integration
def test_trainer_builds_r18_training_components(
    isolated_workspace,
    minimal_coco_config,
    deterministic_train_reader_config,
    tmp_path,
):
    cfg = load_config(str(R18_CONFIG)).copy()
    cfg["TrainDataset"] = minimal_coco_config
    cfg["TrainReader"] = deterministic_train_reader_config
    cfg["worker_num"] = 0
    cfg["save_dir"] = str(tmp_path / "output")
    cfg["use_ema"] = False
    cfg["device"] = torch.device("cpu")

    trainer = Trainer(cfg, mode="train")

    assert len(trainer.loader) == 1
    assert isinstance(trainer.model, RTDETRV3)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.lr is not None

    batch = trainer._prepare_batch(next(iter(trainer.loader)))
    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["image"].dtype == torch.float32
    assert batch["image"].device.type == "cpu"
    assert all(isinstance(boxes, torch.Tensor) for boxes in batch["gt_bbox"])
    assert [boxes.shape for boxes in batch["gt_bbox"]] == [(1, 4), (2, 4)]
    assert all(classes.dtype == torch.int64 for classes in batch["gt_class"])


@pytest.mark.integration
def test_r18_training_forward_produces_finite_losses(
    isolated_workspace,
    minimal_coco_config,
    deterministic_train_reader_config,
    tmp_path,
):
    trainer = _build_reduced_r18_trainer(
        isolated_workspace,
        minimal_coco_config,
        deterministic_train_reader_config,
        tmp_path / "output",
    )
    batch = trainer._prepare_batch(next(iter(trainer.loader)))
    batch["epoch_id"] = 0

    losses = trainer.model(batch)

    assert isinstance(losses, dict)
    assert "loss" in losses
    loss_shapes = {name: tuple(value.shape) for name, value in losses.items()}
    assert losses["loss"].ndim == 0, loss_shapes
    assert all(torch.isfinite(value).all() for value in losses.values())

    tracked_parameter = next(
        parameter
        for parameter in trainer.model.backbone.parameters()
        if parameter.requires_grad
    )
    parameter_before_step = tracked_parameter.detach().clone()
    frozen_projection = trainer.model.aux_o2m_head.proj_conv.weight
    frozen_projection_before_step = frozen_projection.detach().clone()

    trainer.optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()

    modules_with_gradients = {
        "backbone": trainer.model.backbone,
        "neck": trainer.model.neck,
        "transformer": trainer.model.transformer,
        "aux_o2m_head": trainer.model.aux_o2m_head,
    }
    for name, module in modules_with_gradients.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients, f"{name} did not receive gradients"
        assert all(torch.isfinite(gradient).all() for gradient in gradients)

    gradient_norm = trainer._clip_gradients()
    assert torch.isfinite(torch.as_tensor(gradient_norm))
    trainer.optimizer.step()
    lr_before_step = trainer.optimizer.param_groups[0]["lr"]
    trainer.lr.step()
    lr_after_step = trainer.optimizer.param_groups[0]["lr"]

    assert not torch.equal(tracked_parameter, parameter_before_step)
    assert frozen_projection.requires_grad is False
    assert torch.equal(frozen_projection, frozen_projection_before_step)
    assert lr_before_step != lr_after_step


@pytest.mark.integration
@pytest.mark.slow
def test_r18_short_training_runs_five_iterations(
    isolated_workspace,
    minimal_coco_config,
    deterministic_train_reader_config,
    tmp_path,
):
    trainer = _build_reduced_r18_trainer(
        isolated_workspace,
        minimal_coco_config,
        deterministic_train_reader_config,
        tmp_path / "output",
        repeat=5,
    )
    recorder = _StepRecorder()
    trainer._compose_callback = recorder

    trainer._train_epoch(epoch_id=0)

    assert len(recorder.steps) == 5
    for step in recorder.steps:
        assert torch.isfinite(torch.tensor(step["loss"]))
        assert torch.isfinite(torch.tensor(step["gradient_norm"]))
        assert torch.isfinite(torch.tensor(step["learning_rate"]))
    assert trainer.status["step_id"] == 4
    assert torch.isfinite(torch.tensor(trainer.status["loss"]))
    assert torch.isfinite(torch.tensor(trainer.status["gradient_norm"]))
    assert torch.isfinite(torch.tensor(trainer.status["learning_rate"]))

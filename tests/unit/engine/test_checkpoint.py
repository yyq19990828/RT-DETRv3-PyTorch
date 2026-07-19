import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ppdet_pytorch.engine.callbacks import Checkpointer
from ppdet_pytorch.engine.trainer import Trainer
from ppdet_pytorch.optimizer.ema import ModelEMA
from ppdet_pytorch.optimizer.optimizer import (
    LearningRate,
    LinearWarmup,
    OptimizerBuilder,
    PiecewiseDecay,
)
from ppdet_pytorch.utils import checkpoint as checkpoint_utils
from ppdet_pytorch.utils.checkpoint import (
    capture_rng_state,
    convert_to_dict,
    load_checkpoint,
    save_checkpoint,
)


class _ResumeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.stage = nn.Linear(2, 1)
        for parameter in self.stage.parameters():
            parameter._optimizer_lr_multiplier = 0.1

    def forward(self, inputs):
        return self.stage(torch.tanh(self.base(inputs)))


def _build_training_state(seed):
    torch.manual_seed(seed)
    model = _ResumeModel()
    optimizer = OptimizerBuilder(
        regularizer=False,
        optimizer={"type": "AdamW", "weight_decay": 0.0001},
    )(0.0004, model)
    scheduler = LearningRate(
        base_lr=0.0004,
        schedulers=[
            PiecewiseDecay(gamma=[0.1], milestones=[2], use_warmup=True),
            LinearWarmup(steps=2, start_factor=0.5),
        ],
    )(step_per_epoch=2, optimizer=optimizer)
    ema = ModelEMA(model, decay=0.9, ema_decay_type="normal", device="cpu")
    return model, optimizer, scheduler, ema


def _training_step(model, optimizer, scheduler, ema):
    inputs = torch.tensor([[0.2, -0.4], [0.7, 0.3]])
    targets = torch.tensor([[0.5], [-0.1]])
    optimizer.zero_grad(set_to_none=True)
    loss = torch.square(model(inputs) - targets).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    ema.update(model)
    return loss.detach().clone()


@pytest.mark.filterwarnings(
    "ignore:The epoch parameter in `scheduler.step\\(\\)` was not necessary"
)
def test_checkpoint_restores_next_step_and_rng_state(tmp_path):
    model, optimizer, scheduler, ema = _build_training_state(seed=11)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    for _ in range(3):
        _training_step(model, optimizer, scheduler, ema)

    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    checkpoint_path = tmp_path / "resume.pth"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        iteration=3,
        save_path=str(checkpoint_path),
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        sampler_epoch=1,
    )
    expected_random = (
        random.random(),
        np.random.random(),
        torch.rand(1),
    )

    lr_before_continuation = [group["lr"] for group in optimizer.param_groups]
    expected_loss = _training_step(model, optimizer, scheduler, ema)
    expected_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    restored_model, restored_optimizer, restored_scheduler, restored_ema = (
        _build_training_state(seed=99)
    )
    restored_scaler = torch.amp.GradScaler("cpu", init_scale=32.0)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    metadata = load_checkpoint(
        str(checkpoint_path),
        restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        ema=restored_ema,
        restore_rng=True,
    )

    assert metadata == {
        "epoch": 1,
        "iteration": 3,
        "global_step": 3,
        "sampler_epoch": 1,
        "format_version": 1,
        "best_metric": None,
        "config": None,
    }
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    assert torch.equal(torch.rand(1), expected_random[2])
    assert [group["lr"] for group in restored_optimizer.param_groups] == (
        pytest.approx(lr_before_continuation)
    )
    assert restored_scheduler.last_epoch == scheduler.last_epoch - 1
    assert restored_ema.step == ema.step - 1
    assert restored_ema._decay == ema._decay
    assert restored_scaler.state_dict() == scaler.state_dict()

    restored_loss = _training_step(
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_ema,
    )

    assert restored_loss == pytest.approx(expected_loss)
    for name, parameter in restored_model.named_parameters():
        assert torch.equal(parameter, expected_parameters[name])
    assert not list(tmp_path.glob(".*.tmp"))


def test_checkpoint_restores_rng_state_for_current_distributed_rank(
    tmp_path, monkeypatch
):
    model = nn.Linear(2, 1)
    checkpoint_path = tmp_path / "distributed-rng.pth"

    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    rank_zero_state = capture_rng_state()

    random.seed(202)
    np.random.seed(202)
    torch.manual_seed(202)
    rank_one_state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))

    torch.save(
        {
            "model": model.state_dict(),
            "rng_state": rank_zero_state,
            "rng_state_by_rank": [rank_zero_state, rank_one_state],
        },
        checkpoint_path,
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    monkeypatch.setattr(checkpoint_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoint_utils, "get_rank", lambda: 1)

    load_checkpoint(
        str(checkpoint_path),
        model,
        restore_rng=True,
        map_location="cpu",
    )

    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])


def test_checkpointer_writes_canonical_training_schema(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.5,
        total_iters=2,
    )
    trainer = SimpleNamespace(
        cfg={"architecture": "TinyModel", "device": torch.device("cpu")},
        save_dir=str(tmp_path),
        end_epoch=2,
        global_step=3,
        model=model,
        optimizer=optimizer,
        lr=scheduler,
        scaler=None,
        use_ema=False,
        ema=None,
        _convert_cfg_to_dict=lambda cfg: convert_to_dict(cfg),
    )

    Checkpointer(trainer).on_epoch_end({"mode": "train", "epoch_id": 0, "loss": 1.25})

    checkpoint = torch.load(
        tmp_path / "epoch_1.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["format_version"] == 1
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 3
    assert checkpoint["sampler_epoch"] == 1
    assert "model" in checkpoint
    assert "optimizer" in checkpoint
    assert "scheduler" in checkpoint
    assert "rng_state" in checkpoint
    assert "model_state_dict" not in checkpoint
    assert checkpoint["config"] == {
        "architecture": "TinyModel",
        "device": "cpu",
    }


def test_checkpointer_honors_interval_and_always_saves_final(tmp_path):
    model = nn.Linear(2, 1)
    trainer = SimpleNamespace(
        cfg={"architecture": "TinyModel"},
        save_dir=str(tmp_path),
        end_epoch=4,
        global_step=0,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        lr=None,
        scaler=None,
        use_ema=False,
        ema=None,
        _convert_cfg_to_dict=lambda cfg: convert_to_dict(cfg),
    )
    checkpointer = Checkpointer(trainer, save_interval=3)

    for epoch_id in range(4):
        checkpointer.on_epoch_end({"mode": "train", "epoch_id": epoch_id})

    assert sorted(path.name for path in tmp_path.glob("*.pth")) == [
        "epoch_3.pth",
        "model_final.pth",
    ]


def test_trainer_converts_mapping_config_to_plain_serializable_data():
    trainer = Trainer.__new__(Trainer)
    config = {
        "architecture": "RTDETRV3",
        "device": torch.device("cuda"),
        "reader": {"sizes": (480, 640)},
        "seed": np.int64(2026),
    }

    converted = trainer._convert_cfg_to_dict(config)

    assert converted == {
        "architecture": "RTDETRV3",
        "device": "cuda",
        "reader": {"sizes": [480, 640]},
        "seed": 2026,
    }


def test_trainer_resume_restores_progress_from_canonical_checkpoint(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.5,
        total_iters=4,
    )
    checkpoint_path = tmp_path / "trainer-resume.pth"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        iteration=7,
        save_path=str(checkpoint_path),
        scheduler=scheduler,
        sampler_epoch=2,
    )

    resumed_model = nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(
        resumed_model.parameters(), lr=0.1, momentum=0.9
    )
    resumed_scheduler = torch.optim.lr_scheduler.LinearLR(
        resumed_optimizer,
        start_factor=0.5,
        total_iters=4,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.model = resumed_model
    trainer.optimizer = resumed_optimizer
    trainer.lr = resumed_scheduler
    trainer.scaler = None
    trainer.use_ema = False
    trainer.ema = None
    trainer.status = {}
    trainer.is_loaded_weights = False

    trainer.resume_weights(str(checkpoint_path))

    assert trainer.start_epoch == 2
    assert trainer.global_step == 7
    assert trainer.status["global_step"] == 7
    assert trainer.is_loaded_weights is True
    for expected, actual in zip(model.parameters(), resumed_model.parameters()):
        assert torch.equal(expected, actual)

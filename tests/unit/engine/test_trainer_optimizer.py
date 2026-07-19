import subprocess
import sys
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from ppdet_pytorch.engine.callbacks import Checkpointer, LogPrinter
from ppdet_pytorch.engine.trainer import Trainer


def test_trainer_import_registers_data_components_in_clean_process():
    code = (
        "from ppdet_pytorch.core.workspace import global_config; "
        "from ppdet_pytorch.engine.trainer import Trainer; "
        "assert {'COCODataSet', 'TrainReader', 'EvalReader'} <= set(global_config)"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


class _LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return {"loss": self.weight.square()}


class _PrintableModel(nn.Linear):
    def load_meanstd(self, sample_transforms):
        self.sample_transforms = sample_transforms


class _AttrConfig(dict):
    def __getattr__(self, name):
        return self[name]


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, parameters, events):
        super().__init__(parameters, lr=0.1)
        self.events = events

    def step(self, closure=None):
        self.events.append("optimizer")
        return super().step(closure)


class _NoSyncModel(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.events = events

    def forward(self, batch):
        return {"loss": (self.weight * batch["value"]).square()}

    @contextmanager
    def no_sync(self):
        self.events.append("no_sync_enter")
        yield
        self.events.append("no_sync_exit")


class _StatusRecorder:
    def __init__(self):
        self.steps = []

    def on_step_begin(self, status):
        pass

    def on_step_end(self, status):
        self.steps.append(dict(status))


class _SkippingScaler:
    def __init__(self, events):
        self.events = events
        self.current_scale = 65536.0

    def get_scale(self):
        return self.current_scale

    def scale(self, loss):
        self.events.append("scale")
        return loss

    def unscale_(self, optimizer):
        self.events.append("unscale")

    def step(self, optimizer):
        self.events.append("skip_optimizer")

    def update(self):
        self.events.append("update")
        self.current_scale /= 2


def test_log_printer_uses_configured_interval():
    trainer = type("TrainerStub", (), {"log_interval": 20})()

    assert LogPrinter(trainer).log_iter == 20


def test_log_printer_uses_global_average_batch_time_for_eta(monkeypatch):
    trainer = type(
        "TrainerStub",
        (),
        {"log_interval": 1, "end_epoch": 1},
    )()
    messages = []
    monkeypatch.setattr(
        "ppdet_pytorch.engine.callbacks.logger.info",
        messages.append,
    )
    log_printer = LogPrinter(trainer)
    status = {
        "mode": "train",
        "epoch_id": 0,
        "steps_per_epoch": 10,
        "loss": 1.0,
        "learning_rate": 0.1,
        "data_time": 0.0,
        "batch_size": 1,
    }

    log_printer.on_step_end({**status, "step_id": 0, "batch_time": 1.0})
    log_printer.on_step_end({**status, "step_id": 1, "batch_time": 3.0})

    assert "eta: 0:00:18" in messages[-1]


def test_trainer_callbacks_use_configured_snapshot_interval(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.mode = "train"
    trainer.cfg = {"snapshot_epoch": 3}
    trainer.log_interval = 20
    trainer.save_dir = str(tmp_path)

    trainer._init_callbacks()

    checkpointer = next(
        callback
        for callback in trainer._callbacks
        if isinstance(callback, Checkpointer)
    )
    assert checkpointer.save_interval == 3


def test_checkpointer_rejects_invalid_snapshot_interval(tmp_path):
    trainer = type("TrainerStub", (), {"save_dir": str(tmp_path)})()

    with pytest.raises(ValueError, match="save_interval"):
        Checkpointer(trainer, save_interval=0)


def test_trainer_accepts_paddle_style_nested_base_lr():
    trainer = Trainer.__new__(Trainer)
    trainer.mode = "train"
    trainer.loader = [None]
    trainer.model = nn.Linear(2, 1)

    trainer._build_optimizer(
        {
            "LearningRate": {
                "base_lr": 0.0004,
                "schedulers": [
                    {
                        "name": "PiecewiseDecay",
                        "gamma": [0.1],
                        "milestones": [1],
                        "use_warmup": False,
                    }
                ],
            },
            "OptimizerBuilder": {
                "regularizer": False,
                "optimizer": {"type": "SGD"},
            },
        }
    )

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0004)


def test_build_model_logs_python_parameter_count(monkeypatch):
    model = _PrintableModel(2, 1)
    cfg = _AttrConfig(
        architecture="PrintableModel",
        PrintableModel={},
        TestReader={"sample_transforms": ["normalize"]},
        print_params=True,
    )
    messages = []
    monkeypatch.setattr(
        "ppdet_pytorch.engine.trainer.create",
        lambda config: model,
    )
    monkeypatch.setattr(
        "ppdet_pytorch.engine.trainer.logger.info",
        messages.append,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.mode = "train"
    trainer.is_loaded_weights = False

    trainer._build_model(cfg)

    assert model.sample_transforms == ["normalize"]
    assert "Model Params : 3e-06 M." in messages


def test_trainer_collects_mot_images_in_deterministic_order(tmp_path):
    data_root = tmp_path / "mot"
    first_sequence = data_root / "sequence_a"
    second_sequence = data_root / "sequence_b"
    first_sequence.mkdir(parents=True)
    second_sequence.mkdir()
    (first_sequence / "0002.PNG").touch()
    (first_sequence / "0001.jpg").touch()
    (second_sequence / "0001.bmp").touch()

    trainer = Trainer.__new__(Trainer)
    images = trainer.parse_mot_images(
        {
            "EvalMOTDataset": {
                "dataset_dir": str(tmp_path),
                "data_root": "mot",
            }
        }
    )

    assert images == [
        str(first_sequence / "0001.jpg"),
        str(first_sequence / "0002.PNG"),
        str(second_sequence / "0001.bmp"),
    ]


def test_reset_norm_param_attr_uses_pytorch_norm_apis_and_preserves_state():
    module = nn.ModuleDict(
        {
            "batch": nn.BatchNorm2d(2, eps=1e-3),
            "layer": nn.LayerNorm((2, 3), eps=2e-3),
            "group": nn.GroupNorm(1, 2, eps=3e-3),
        }
    )
    with torch.no_grad():
        module["batch"].weight.fill_(1.5)
        module["layer"].bias.fill_(2.5)
        module["group"].weight.fill_(3.5)
    module.eval()
    original_layers = dict(module.items())
    original_states = {
        name: {key: value.clone() for key, value in layer.state_dict().items()}
        for name, layer in module.items()
    }

    trainer = Trainer.__new__(Trainer)
    result = trainer.reset_norm_param_attr(
        module,
        weight_attr=None,
        bias_attr=None,
    )

    assert result is module
    assert all(module[name] is not original_layers[name] for name in module)
    assert module["batch"].eps == pytest.approx(1e-3)
    assert module["layer"].eps == pytest.approx(2e-3)
    assert module["group"].eps == pytest.approx(3e-3)
    assert all(not layer.training for layer in module.values())
    for name, layer in module.items():
        for key, value in layer.state_dict().items():
            assert torch.equal(value, original_states[name][key])


def test_training_step_orders_clip_optimizer_scheduler_and_ema():
    events = []
    trainer = Trainer.__new__(Trainer)
    trainer.model = _LossModel()
    trainer.optimizer = _RecordingSGD(trainer.model.parameters(), events)
    trainer.lr = type(
        "Scheduler",
        (),
        {"step": lambda self: events.append("scheduler")},
    )()
    trainer.ema = type(
        "EMA",
        (),
        {"update": lambda self, model: events.append("ema")},
    )()
    trainer.loader = [{"image": torch.ones(1, 1)}]
    trainer.cfg = {}
    trainer.status = {}
    trainer.global_step = 0
    trainer.use_amp = False
    trainer.use_ema = True
    trainer._compose_callback = None
    trainer._clip_gradients = lambda: events.append("clip") or torch.tensor(1.0)

    trainer._train_epoch(epoch_id=0)

    assert events == ["clip", "optimizer", "scheduler", "ema"]
    assert trainer.global_step == 1


def test_amp_overflow_does_not_advance_optimizer_dependent_state():
    events = []
    trainer = Trainer.__new__(Trainer)
    trainer.model = _LossModel()
    trainer.optimizer = _RecordingSGD(trainer.model.parameters(), events)
    trainer.lr = type(
        "Scheduler",
        (),
        {"step": lambda self: events.append("scheduler")},
    )()
    trainer.ema = type(
        "EMA",
        (),
        {"update": lambda self, model: events.append("ema")},
    )()
    trainer.loader = [{"image": torch.ones(1, 1)}]
    trainer.cfg = {}
    trainer.status = {}
    trainer.global_step = 0
    trainer.use_amp = True
    trainer.scaler = _SkippingScaler(events)
    trainer.use_ema = True
    trainer._compose_callback = None
    trainer._clip_gradients = lambda: (
        events.append("clip") or torch.tensor(float("inf"))
    )

    trainer._train_epoch(epoch_id=0)

    assert events == ["scale", "unscale", "clip", "skip_optimizer", "update"]
    assert trainer.global_step == 0
    assert trainer.status["optimizer_step_skipped"] is True


def test_gradient_accumulation_uses_no_sync_only_before_update_boundary(
    monkeypatch,
):
    events = []
    monkeypatch.setattr("ppdet_pytorch.engine.trainer.DDP", _NoSyncModel)
    trainer = Trainer.__new__(Trainer)
    trainer.model = _NoSyncModel(events)
    trainer.optimizer = _RecordingSGD(trainer.model.parameters(), events)
    trainer.lr = type(
        "Scheduler",
        (),
        {"step": lambda self: events.append("scheduler")},
    )()
    trainer.ema = None
    trainer.loader = [
        {"value": torch.tensor([1.0])},
        {"value": torch.tensor([3.0])},
        {"value": torch.tensor([2.0])},
    ]
    trainer.cfg = {"accumulate_steps": 2}
    trainer.status = {}
    trainer.global_step = 0
    trainer.accumulate_steps = 2
    trainer.use_amp = False
    trainer.use_ema = False
    recorder = _StatusRecorder()
    trainer._compose_callback = recorder
    trainer._clip_gradients = lambda: torch.tensor(1.0)

    trainer._train_epoch(epoch_id=0)

    assert events == [
        "no_sync_enter",
        "no_sync_exit",
        "optimizer",
        "scheduler",
        "optimizer",
        "scheduler",
    ]
    assert trainer.global_step == 2
    assert [step["optimizer_step"] for step in recorder.steps] == [
        False,
        True,
        True,
    ]
    assert [step["accumulation_steps"] for step in recorder.steps] == [2, 2, 1]

    reference = _NoSyncModel([])
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    first_group_loss = (
        sum(
            (reference.weight * value).square()
            for value in (torch.tensor(1.0), torch.tensor(3.0))
        )
        / 2
    )
    first_group_loss.backward()
    reference_optimizer.step()
    reference_optimizer.zero_grad()
    (reference.weight * torch.tensor(2.0)).square().backward()
    reference_optimizer.step()

    assert trainer.model.weight.item() == pytest.approx(
        reference.weight.item(), abs=1e-7
    )


def test_distributed_reported_loss_is_world_size_mean(monkeypatch):
    monkeypatch.setattr(
        "ppdet_pytorch.engine.trainer.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr("ppdet_pytorch.engine.trainer.dist.get_world_size", lambda: 2)

    def add_remote_rank_loss(value, op):
        assert op == torch.distributed.ReduceOp.SUM
        value.add_(3.0)

    monkeypatch.setattr(
        "ppdet_pytorch.engine.trainer.dist.all_reduce", add_remote_rank_loss
    )

    local_loss = torch.tensor(1.0, requires_grad=True)
    reported_loss = Trainer._reduce_loss_for_logging(local_loss)

    assert reported_loss.item() == pytest.approx(2.0)
    assert reported_loss.requires_grad is False

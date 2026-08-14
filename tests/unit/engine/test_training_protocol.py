import torch
from torch import nn

from ppdet_pytorch.engine.trainer import Trainer
from ppdet_pytorch.engine.training_protocol import TrainingProtocol


class _LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return {"loss": (self.weight * batch["value"]).square().mean()}


class _Protocol(TrainingProtocol):
    def __init__(self, events):
        self.events = events
        self.successes = []

    def after_backward(self, gradients, status):
        self.events.append(("after_backward", float(gradients["weight"])))
        gradients["weight"].zero_()
        return {"gradient": gradients["weight"] + 3}

    def after_successful_optimizer_step(self, observation, status):
        self.events.append(("successful", status["global_step"]))
        self.successes.append(observation)


class _Scaler:
    def __init__(self, events, overflow=False):
        self.events = events
        self.scale_value = 8.0
        self.overflow = overflow

    def scale(self, loss):
        self.events.append("scale")
        return loss

    def unscale_(self, optimizer):
        self.events.append("unscale")

    def step(self, optimizer):
        if self.overflow:
            self.events.append("overflow")
        else:
            optimizer.step()

    def update(self):
        self.events.append("update")
        if self.overflow:
            self.scale_value /= 2

    def get_scale(self):
        return self.scale_value


def _trainer(events, protocol, *, amp=False, overflow=False):
    trainer = Trainer.__new__(Trainer)
    trainer.model = _LossModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr = type(
        "Scheduler", (), {"step": lambda self: events.append("scheduler")}
    )()
    trainer.loader = [
        {"value": torch.tensor([1.0])},
        {"value": torch.tensor([2.0])},
        {"value": torch.tensor([3.0])},
    ]
    trainer.cfg = {"accumulate_steps": 2}
    trainer.accumulate_steps = 2
    trainer.status = {}
    trainer.global_step = 0
    trainer.use_amp = amp
    trainer.scaler = _Scaler(events, overflow=overflow) if amp else None
    trainer.use_ema = False
    trainer.ema = None
    trainer.training_protocol = protocol
    trainer._compose_callback = None
    original_clip = trainer._clip_gradients

    def clip():
        events.append("clip")
        return original_clip()

    trainer._clip_gradients = clip
    return trainer


def test_after_backward_runs_only_at_final_microbatch_before_clip():
    events = []
    protocol = _Protocol(events)
    trainer = _trainer(events, protocol)

    trainer._train_epoch(0)

    assert [event[0] for event in events if isinstance(event, tuple)] == [
        "after_backward",
        "successful",
        "after_backward",
        "successful",
    ]
    assert events.index("clip") > next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "after_backward"
    )
    assert trainer.model.weight.grad is None
    assert len(protocol.successes) == 2


def test_after_backward_order_amp_accumulation_and_overflow_skip():
    events = []
    protocol = _Protocol(events)
    trainer = _trainer(events, protocol, amp=True, overflow=True)

    trainer._train_epoch(0)

    first_unscale = events.index("unscale")
    first_hook = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "after_backward"
    )
    assert first_unscale < first_hook < events.index("clip")
    assert not protocol.successes
    assert trainer.global_step == 0


def test_protocol_observes_gradient_clone_not_live_gradient():
    events = []
    protocol = _Protocol(events)
    trainer = _trainer(events, protocol)
    original = trainer.model.weight.detach().clone()

    trainer._train_epoch(0)

    assert not torch.equal(trainer.model.weight, original)


def test_validation_metrics_are_forwarded_without_live_mapping_access():
    observed = []

    class Protocol(TrainingProtocol):
        def after_validation(self, metrics):
            metrics["changed"] = True
            observed.append(metrics)

    trainer = Trainer.__new__(Trainer)
    trainer.training_protocol = Protocol()
    metrics = {"bbox_ap": 0.5}

    trainer.notify_validation(metrics)

    assert observed == [{"bbox_ap": 0.5, "changed": True}]
    assert metrics == {"bbox_ap": 0.5}

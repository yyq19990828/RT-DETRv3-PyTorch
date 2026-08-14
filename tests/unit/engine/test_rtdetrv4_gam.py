import pytest
import torch
from torch import nn

from ppdet_pytorch.engine.trainer import Trainer
from ppdet_pytorch.engine.training_protocol import TwoStageDetectionProtocol
from tests.unit.engine.test_training_protocol import _trainer


def _protocol(*, current=5.0, rho=11.0, delta=1.0, default=20.0, stop=50):
    return TwoStageDetectionProtocol(
        family="rtdetrv4",
        stop_epoch=stop,
        ema_restart_decay=0.9998,
        current_gam_weight=current,
        gam_rho=rho,
        gam_delta=delta,
        gam_default_weight=default,
    )


def test_gam_observes_only_aifi_encoder_gradient_l1_for_ddp_and_plain_names():
    protocol = _protocol()
    gradients = {
        "backbone.weight": torch.tensor([-2.0]),
        "encoder.encoder.0.weight": torch.tensor([-1.0, 3.0]),
        "module.encoder.feature_projector.0.weight": torch.tensor([5.0]),
        "module.encoder.encoder.1.weight": torch.tensor([-4.0]),
        "decoder.weight": torch.tensor([1.0]),
    }

    observation = protocol.after_backward(gradients, {})

    assert observation["encoder_l1"] == 8
    assert observation["total_l1"] == 16


@pytest.mark.parametrize(
    ("percentage", "epoch", "expected"),
    [
        (0.0, 0, 20.0),
        (10.0, 0, 5.0),
        (11.0, 0, 5.0),
        (12.0, 0, 5.0),
        (11.0, 50, 20.0),
        (5.0, 0, 5.0 * ((0.12 * 0.95) / (0.05 * 0.88))),
        (30.0, 0, 5.0 * ((0.10 * 0.70) / (0.30 * 0.90))),
    ],
)
def test_gam_transition_matches_official_formula(percentage, epoch, expected):
    protocol = _protocol()
    assert protocol._next_gam_weight(epoch, percentage) == pytest.approx(expected)


def test_gam_epoch_averages_successful_global_update_percentages():
    protocol = _protocol()
    protocol.after_successful_optimizer_step(
        {"encoder_l1": torch.tensor(1.0), "total_l1": torch.tensor(20.0)}, {}
    )
    protocol.after_successful_optimizer_step(
        {"encoder_l1": torch.tensor(3.0), "total_l1": torch.tensor(20.0)}, {}
    )

    protocol.after_epoch(0, {})
    action = protocol.pop_actions()[0]

    assert action["name"] == "set_gam_weight"
    assert action["weight"] == pytest.approx(5.0)
    protocol.complete_action(action)
    assert protocol.current_gam_weight == pytest.approx(5.0)


def test_amp_skipped_update_does_not_contribute_to_gam_epoch():
    events = []
    protocol = _protocol()
    trainer = _trainer(events, protocol, amp=True, overflow=True)

    trainer._train_epoch(0)

    assert protocol._gam_observation_count == 0
    protocol.after_epoch(0, {})
    assert protocol.pop_actions()[0]["weight"] == 20.0


def test_transition_preserves_latest_gam_weight_across_historical_reload():
    protocol = _protocol(current=7.5, stop=2)
    protocol.companion_basename = "best_stg1.pth"
    protocol.companion_sha256 = "a" * 64

    protocol.before_epoch(2, {})
    action = protocol.pop_actions()[0]
    protocol.current_gam_weight = 1.0
    protocol.complete_action(action)

    assert protocol.current_gam_weight == 7.5


@pytest.mark.parametrize("weight", [None, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_gam_checkpoint_weight(weight):
    protocol = _protocol()
    state = dict(protocol.state_dict(), current_gam_weight=weight)
    with pytest.raises(ValueError, match="GAM checkpoint weight|current GAM weight"):
        protocol.load_state_dict(state)


def test_rejects_stale_gam_state_before_mutation():
    protocol = _protocol(current=7.5)
    state = dict(protocol.state_dict(), gam_rho=999.0, current_gam_weight=2.0)

    with pytest.raises(ValueError, match="GAM configuration mismatch"):
        protocol.load_state_dict(state)

    assert protocol.current_gam_weight == 7.5


class _Criterion:
    def __init__(self):
        self.weight = None

    def set_distillation_weight(self, weight):
        self.weight = weight


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(()))
        self.criterion = _Criterion()


def test_trainer_applies_protocol_weight_to_criterion():
    trainer = Trainer.__new__(Trainer)
    trainer.model = _Model()
    trainer.training_protocol = _protocol(current=7.5)

    trainer._synchronize_gam_weight(require_equal=True)

    assert trainer.model.criterion.weight == 7.5


def test_epoch_end_gam_action_runs_before_checkpoint_callback():
    events = []

    class Protocol(TwoStageDetectionProtocol):
        def after_epoch(self, epoch, status):
            events.append("after_epoch")

    class Callback:
        def on_train_begin(self, status):
            pass

        def on_epoch_begin(self, status):
            pass

        def on_epoch_end(self, status):
            events.append("checkpoint_callback")

        def on_train_end(self, status):
            pass

    trainer = Trainer.__new__(Trainer)
    trainer.start_epoch = 0
    trainer.end_epoch = 1
    trainer._nranks = 1
    trainer.loader = []
    trainer.status = {}
    trainer.training_protocol = Protocol("dfine", 2, 0.9998)
    trainer._compose_callback = Callback()
    trainer._validation_loader = None
    trainer._train_epoch = lambda epoch: None
    trainer._execute_protocol_actions = lambda: events.append("execute_action")

    trainer.train()

    assert events == [
        "execute_action",
        "after_epoch",
        "execute_action",
        "checkpoint_callback",
    ]


def test_rejects_partial_or_non_rtdetrv4_gam_configuration():
    with pytest.raises(ValueError, match="requires rho, delta"):
        TwoStageDetectionProtocol(
            "rtdetrv4", 50, 0.9998, current_gam_weight=5, gam_rho=11
        )
    with pytest.raises(ValueError, match="only supported"):
        TwoStageDetectionProtocol(
            "dfine",
            50,
            0.9998,
            current_gam_weight=5,
            gam_rho=11,
            gam_delta=1,
            gam_default_weight=20,
        )

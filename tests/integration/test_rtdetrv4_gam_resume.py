import pytest
import torch
from torch import nn

from detrs.engine.trainer import Trainer
from detrs.engine.training_protocol import TwoStageDetectionProtocol


class _Criterion:
    def __init__(self, weight):
        self.weight = weight

    def set_distillation_weight(self, weight):
        self.weight = float(weight)


class _Student(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.criterion = _Criterion(weight)


def _trainer(tmp_path, current=5.0):
    trainer = Trainer.__new__(Trainer)
    trainer.model = _Student(current)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1)
    trainer.scaler = None
    trainer.use_ema = False
    trainer.ema = None
    trainer.training_protocol = TwoStageDetectionProtocol(
        "rtdetrv4",
        stop_epoch=2,
        ema_restart_decay=0.9998,
        current_gam_weight=current,
        gam_rho=11,
        gam_delta=1,
        gam_default_weight=20,
    )
    trainer.cfg = {"architecture": "RTDETRV4"}
    trainer.save_dir = str(tmp_path)
    trainer.status = {"epoch_id": 0, "global_step": 0}
    trainer.global_step = 0
    trainer.is_loaded_weights = False
    return trainer


@pytest.mark.integration
def test_gam_resume_restores_protocol_and_criterion_weight(tmp_path):
    trainer = _trainer(tmp_path, current=7.5)
    checkpoint = tmp_path / "epoch_1.pth"
    trainer._save_protocol_checkpoint(str(checkpoint))

    resumed = _trainer(tmp_path, current=1.0)
    resumed.resume_weights(str(checkpoint))

    assert resumed.training_protocol.current_gam_weight == 7.5
    assert resumed.model.criterion.weight == 7.5


@pytest.mark.integration
def test_transition_reload_preserves_latest_gam_weight(tmp_path):
    trainer = _trainer(tmp_path, current=5.0)
    trainer.notify_validation({"bbox": 0.4})
    trainer.training_protocol.current_gam_weight = 7.5
    trainer.model.criterion.set_distillation_weight(7.5)
    trainer.status["epoch_id"] = 2

    trainer.training_protocol.before_epoch(2, {})
    trainer._execute_protocol_actions()

    assert trainer.training_protocol.stage == 2
    assert trainer.training_protocol.current_gam_weight == 7.5
    assert trainer.model.criterion.weight == 7.5
